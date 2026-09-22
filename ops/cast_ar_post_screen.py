#!/usr/bin/env python
"""Train and evaluate CAST-AR F1 on decoded clips from one real codec.

The experiment isolates the decoder-side mechanism: the exact same encoded
bitstream is used by the anchor and treatment, so bitrate cannot be hidden or
estimated.  A codec-specific temporal POST is trained on the train split and
evaluated once on a locked 208-clip validation subset with a held-out action
recogniser.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

# ``python ops/cast_ar_post_screen.py`` sets sys.path[0] to ``ops/`` rather
# than the repository root.  Kaggle invokes the file in exactly that form, so
# make the project packages importable without relying on an ambient
# PYTHONPATH.  This is intentionally before every ``src.*`` import below.
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.codecs.standard import StandardCodec, ffmpeg_available
from src.data.video_dataset import VideoClipDataset, collate_clips
from src.metrics.bd_rate import bd_metric, bd_rate
from src.models.cast_ar import CASTTemporalPost
from src.tasks.action_recognition import ActionRecognitionAnalyzer


QPS = (30, 35, 40, 45, 50)
PICTURE_NAMES = {0: "I", 1: "P", 2: "B", 3: "other"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_key(record: dict) -> str:
    seq = str(record.get("sequence_id", record.get("video_id", record.get("id", ""))))
    if seq:
        return seq
    parts = str(record["path"]).replace("\\", "/").rstrip("/").split("/")
    return "/".join(parts[-2:])


def locked_indices(records: list[dict], count: int, salt: str) -> list[int]:
    ranked = []
    for index, record in enumerate(records):
        key = sample_key(record)
        score = hashlib.sha256(f"{salt}\0{key}".encode()).hexdigest()
        ranked.append((score, key, index))
    ranked.sort()
    return [item[2] for item in ranked[: min(count, len(ranked))]]


def subset_fingerprint(records: list[dict], indices: Iterable[int]) -> str:
    keys = sorted(sample_key(records[i]) for i in indices)
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()[:16]


def action_forward(analyzer: ActionRecognitionAnalyzer, x: torch.Tensor):
    """Single analyzer traversal returning logits and a compact layer2 feature."""
    net = analyzer.net
    h = analyzer._prep(x)
    h = net.stem(h)
    h = net.layer1(h)
    h = net.layer2(h)
    feature = F.adaptive_avg_pool3d(h, 1).flatten(1)
    h = net.layer3(h)
    h = net.layer4(h)
    h = net.avgpool(h).flatten(1)
    return net.fc(h), feature


def freeze(analyzer: ActionRecognitionAnalyzer) -> ActionRecognitionAnalyzer:
    analyzer.eval()
    for parameter in analyzer.parameters():
        parameter.requires_grad_(False)
    return analyzer


def target_probability(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return logits.softmax(dim=1).gather(1, labels[:, None]).squeeze(1)


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train(args, device: torch.device, train_set: VideoClipDataset, out_dir: Path):
    indices = locked_indices(train_set.samples, args.train_clips, f"cast-ar-train-{args.seed}")
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        Subset(train_set, indices), batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=collate_clips, pin_memory=True,
        persistent_workers=args.workers > 0, generator=generator,
    )
    model = CASTTemporalPost(
        width=args.width, bases=args.bases, max_delta=args.max_delta
    ).to(device)
    analyzer = freeze(ActionRecognitionAnalyzer("r3d_18", clip_size=112).to(device))
    codec = StandardCodec(args.codec, preset=args.preset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    global_step = 0
    picture_counts: Counter[int] = Counter()

    model.train()
    for epoch in range(1, args.epochs + 1):
        sums: Counter[str] = Counter()
        seen = 0
        started = time.time()
        for clips_cpu, labels_cpu in loader:
            qp = QPS[global_step % len(QPS)]
            decoded_cpu, _, picture_types_cpu = codec.compress_decompress_items_with_metadata(
                clips_cpu, qp=qp
            )
            picture_counts.update(picture_types_cpu.flatten().tolist())
            clips = clips_cpu.to(device, non_blocking=True)
            decoded = decoded_cpu.to(device, non_blocking=True)
            labels = labels_cpu.to(device, non_blocking=True)
            picture_types = picture_types_cpu.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                clean_logits, clean_feature = action_forward(analyzer, clips)
                anchor_logits, anchor_feature = action_forward(analyzer, decoded)
                anchor_ce = F.cross_entropy(anchor_logits, labels, reduction="none")
                anchor_distance = 1.0 - F.cosine_similarity(
                    anchor_feature, clean_feature, dim=1
                )

            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                restored = model(decoded, qp, picture_types)
                post_logits, post_feature = action_forward(analyzer, restored)
                post_ce = F.cross_entropy(post_logits, labels, reduction="none")
                task_regret = F.relu(post_ce - anchor_ce).mean()
                post_distance = 1.0 - F.cosine_similarity(
                    post_feature, clean_feature, dim=1
                )
                feature_regret = F.relu(post_distance - anchor_distance).mean()
                temperature = 2.0
                distill = F.kl_div(
                    F.log_softmax(post_logits / temperature, dim=1),
                    F.softmax(clean_logits / temperature, dim=1),
                    reduction="batchmean",
                ) * (temperature * temperature)
                anchor_target = target_probability(anchor_logits, labels)
                post_target = target_probability(post_logits, labels)
                probability_regret = F.relu(anchor_target - post_target).mean()
                edit = (restored - decoded).abs().mean()
                loss = (
                    0.25 * post_ce.mean()
                    + args.regret_weight * task_regret
                    + args.feature_weight * feature_regret
                    + args.distill_weight * distill
                    + args.probability_weight * probability_regret
                    + args.edit_weight * edit
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            batch = labels.shape[0]
            seen += batch
            global_step += 1
            for name, value in {
                "loss": loss, "post_ce": post_ce.mean(), "task_regret": task_regret,
                "feature_regret": feature_regret, "distill": distill,
                "probability_regret": probability_regret, "edit_l1": edit,
            }.items():
                sums[name] += float(value.detach()) * batch
            if global_step % args.log_every == 0:
                print(
                    f"[train] codec={args.codec} epoch={epoch}/{args.epochs} "
                    f"step={global_step} qp={qp} loss={float(loss):.5f} "
                    f"regret={float(task_regret):.5f} edit={float(edit):.6f} "
                    f"strength={float(torch.tanh(model.post_strength)):.5f}",
                    flush=True,
                )
        row = {name: value / max(seen, 1) for name, value in sums.items()}
        row.update({
            "epoch": epoch, "global_step": global_step, "samples": seen,
            "seconds": time.time() - started,
            "post_strength": float(torch.tanh(model.post_strength.detach())),
        })
        history.append(row)
        print("[epoch] " + json.dumps(row, sort_keys=True), flush=True)

    checkpoint = out_dir / "cast_ar_post.pth"
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "epoch": args.epochs, "global_step": global_step,
        "codec": args.codec, "seed": args.seed, "qps": list(QPS),
        "train_count": len(indices),
        "train_fingerprint": subset_fingerprint(train_set.samples, indices),
        "history": history, "args": vars(args),
        "picture_type_counts": {
            PICTURE_NAMES.get(key, "other"): int(value)
            for key, value in sorted(picture_counts.items())
        },
    }, checkpoint)
    return model, checkpoint, history


@torch.no_grad()
def evaluate(args, device, model, val_set, out_dir: Path, checkpoint: Path):
    indices = locked_indices(val_set.samples, args.val_clips, args.val_salt)
    loader = DataLoader(
        Subset(val_set, indices), batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=collate_clips, pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    analyzers = {
        name: freeze(ActionRecognitionAnalyzer(name, clip_size=112).to(device))
        for name in ("r3d_18", "r2plus1d_18")
    }
    codec = StandardCodec(args.codec, preset=args.preset)
    model.eval()
    accumulator = {
        name: {
            qp: {"n": 0, "bpp": 0.0, "anchor_correct": 0, "post_correct": 0,
                 "anchor_prob": 0.0, "post_prob": 0.0}
            for qp in QPS
        }
        for name in analyzers
    }
    records = []
    picture_counts: Counter[int] = Counter()
    done = 0
    for clips_cpu, labels_cpu, metadata in loader:
        labels = labels_cpu.to(device, non_blocking=True)
        for qp in QPS:
            decoded_cpu, bpps, picture_types_cpu = codec.compress_decompress_items_with_metadata(
                clips_cpu, qp=qp
            )
            picture_counts.update(picture_types_cpu.flatten().tolist())
            decoded = decoded_cpu.to(device, non_blocking=True)
            picture_types = picture_types_cpu.to(device, non_blocking=True)
            restored = model(decoded, qp, picture_types)
            for analyzer_name, analyzer in analyzers.items():
                anchor_logits, _ = action_forward(analyzer, decoded)
                post_logits, _ = action_forward(analyzer, restored)
                anchor_prob = target_probability(anchor_logits, labels)
                post_prob = target_probability(post_logits, labels)
                anchor_correct = anchor_logits.argmax(1).eq(labels)
                post_correct = post_logits.argmax(1).eq(labels)
                cell = accumulator[analyzer_name][qp]
                cell["n"] += labels.shape[0]
                cell["bpp"] += float(sum(bpps))
                cell["anchor_correct"] += int(anchor_correct.sum())
                cell["post_correct"] += int(post_correct.sum())
                cell["anchor_prob"] += float(anchor_prob.sum())
                cell["post_prob"] += float(post_prob.sum())
                for i, meta in enumerate(metadata):
                    records.append({
                        "sequence_id": meta["sequence_id"], "analyzer": analyzer_name,
                        "codec": args.codec, "qp": qp, "bpp": float(bpps[i]),
                        "anchor_correct": bool(anchor_correct[i]),
                        "post_correct": bool(post_correct[i]),
                        "anchor_target_prob": float(anchor_prob[i]),
                        "post_target_prob": float(post_prob[i]),
                    })
        done += labels.shape[0]
        if done % 20 == 0 or done == len(indices):
            print(f"[eval] codec={args.codec} {done}/{len(indices)}", flush=True)

    curves, metrics = {}, {}
    for analyzer_name, by_qp in accumulator.items():
        rows = []
        for qp in QPS:
            cell = by_qp[qp]
            n = cell["n"]
            rows.append({
                "qp": qp, "n": n, "bpp": cell["bpp"] / n,
                "anchor_top1": cell["anchor_correct"] / n,
                "post_top1": cell["post_correct"] / n,
                "anchor_target_prob": cell["anchor_prob"] / n,
                "post_target_prob": cell["post_prob"] / n,
            })
        curves[analyzer_name] = rows
        rates = [row["bpp"] for row in rows]
        anchor_top1 = [row["anchor_top1"] for row in rows]
        post_top1 = [row["post_top1"] for row in rows]
        anchor_prob = [row["anchor_target_prob"] for row in rows]
        post_prob = [row["post_target_prob"] for row in rows]
        metrics[analyzer_name] = {
            "bd_rate_top1_pct": finite_or_none(bd_rate(rates, anchor_top1, rates, post_top1)),
            "bd_accuracy_top1": finite_or_none(bd_metric(rates, anchor_top1, rates, post_top1)),
            "bd_rate_target_prob_pct": finite_or_none(bd_rate(rates, anchor_prob, rates, post_prob)),
            "bd_target_probability": finite_or_none(bd_metric(rates, anchor_prob, rates, post_prob)),
            "min_top1_gap": min(p - a for p, a in zip(post_top1, anchor_top1)),
            "mean_top1_gap": float(np.mean(np.asarray(post_top1) - np.asarray(anchor_top1))),
        }

    heldout = metrics["r2plus1d_18"]
    gate_pass = (
        heldout["bd_rate_target_prob_pct"] is not None
        and heldout["bd_rate_target_prob_pct"] <= -5.0
        and heldout["bd_accuracy_top1"] is not None
        and heldout["bd_accuracy_top1"] >= 0.0
        and heldout["min_top1_gap"] >= -0.005
    )
    result = {
        "experiment": "cast_ar_f1_real_codec_post_screen",
        "codec": args.codec, "seed": args.seed, "qps": list(QPS),
        "preset": args.preset, "train_clips": args.train_clips,
        "val_clips": len(indices), "val_salt": args.val_salt,
        "val_fingerprint": subset_fingerprint(val_set.samples, indices),
        "checkpoint_sha256": sha256_file(checkpoint),
        "curves": curves, "metrics": metrics,
        "heldout_gate_pass": gate_pass,
        "gate_rule": {
            "heldout_analyzer": "r2plus1d_18",
            "bd_rate_target_prob_pct_max": -5.0,
            "bd_accuracy_top1_min": 0.0,
            "min_top1_gap_min": -0.005,
        },
        "picture_type_counts": {
            PICTURE_NAMES.get(key, "other"): int(value)
            for key, value in sorted(picture_counts.items())
        },
        "model_diagnostics": model.diagnostics(
            decoded[:1], QPS[-1], picture_types[:1]
        ),
    }
    (out_dir / "screen_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out_dir / "per_clip_records.json").write_text(
        json.dumps(records), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--codec", choices=("h264", "h265"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--train-clips", type=int, default=1200)
    parser.add_argument("--val-clips", type=int, default=208)
    parser.add_argument("--val-salt", default="cast-ar-f1-val-v1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--bases", type=int, default=4)
    parser.add_argument("--max-delta", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--regret-weight", type=float, default=1.0)
    parser.add_argument("--feature-weight", type=float, default=0.5)
    parser.add_argument("--distill-weight", type=float, default=0.25)
    parser.add_argument("--probability-weight", type=float, default=0.5)
    parser.add_argument("--edit-weight", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not ffmpeg_available():
        raise SystemExit("ffmpeg and ffprobe are required")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[cast-ar] device={device} codec={args.codec} seed={args.seed}", flush=True)
    train_set = VideoClipDataset(
        args.index, split="train", num_frames=16, frame_size=128,
        temporal_stride=2, train=True,
    )
    val_set = VideoClipDataset(
        args.index, split="val", num_frames=16, frame_size=128,
        temporal_stride=2, train=False, return_metadata=True,
    )
    model, checkpoint, _ = train(args, device, train_set, out_dir)
    evaluate(args, device, model, val_set, out_dir, checkpoint)


if __name__ == "__main__":
    main()
