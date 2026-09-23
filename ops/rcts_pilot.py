#!/usr/bin/env python
"""Real-codec supervised PRE selector, with a paired held-out AR pilot.

Ground-truth labels select safe actions on TRAIN only.  The deployed policy
sees source pixels, source-model confidence, codec and QP; it never sees a
validation label or a candidate's encoded outcome before choosing the action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.codecs.standard import StandardCodec, ffmpeg_available
from src.data.video_dataset import VideoClipDataset
from src.metrics.bd_rate import bd_metric, bd_rate
from src.models.rcts import ACTIONS, make_candidates, select_teacher_action
from src.models.task_mask import task_saliency
from src.tasks.action_recognition import ActionRecognitionAnalyzer

QPS = (30, 35, 40, 45, 50)
FEATURES = (
    "luma_mean", "luma_std", "edge", "frame_diff", "flow_mean",
    "warp_confidence", "protected_fraction", "saliency_mean", "saliency_std",
    "source_confidence", "source_margin", "qp_scaled",
)
CONTROLS = ("identity", "uniform_s20", "spatial_s25", "temporal_t20", "joint_s25_t15")


def clip_id(record: dict) -> str:
    path = str(record["path"]).replace("\\", "/").rstrip("/").split("/")
    return "/".join(path[-2:])


def balanced_indices(records: list[dict], count: int, salt: str) -> list[int]:
    """Deterministic, approximately class-balanced source video selection."""
    by_class: dict[int, list[tuple[str, int]]] = {}
    for index, record in enumerate(records):
        key = clip_id(record)
        rank = hashlib.sha256(f"{salt}\0{key}".encode()).hexdigest()
        by_class.setdefault(int(record["label"]), []).append((rank, index))
    for rows in by_class.values():
        rows.sort()
    class_order = sorted(
        by_class,
        key=lambda cls: hashlib.sha256(f"{salt}\0class\0{cls}".encode()).hexdigest(),
    )
    chosen: list[int] = []
    depth = 0
    while len(chosen) < min(count, len(records)):
        advanced = False
        for cls in class_order:
            if depth < len(by_class[cls]):
                chosen.append(by_class[cls][depth][1])
                advanced = True
                if len(chosen) == count:
                    break
        if not advanced:
            break
        depth += 1
    return chosen


def fingerprint(records: list[dict], indices: list[int]) -> str:
    keys = sorted(clip_id(records[i]) for i in indices)
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()[:16]


def feature_vector(info: dict[str, float], source_logits: torch.Tensor, qp: int) -> np.ndarray:
    probabilities = source_logits.float().softmax(dim=1).topk(2, dim=1).values[0]
    values = dict(info)
    values.update({
        "source_confidence": float(probabilities[0].item()),
        "source_margin": float((probabilities[0] - probabilities[1]).item()),
        "qp_scaled": (qp - 30) / 20.0,
    })
    return np.asarray([values[key] for key in FEATURES], dtype=np.float32)


class Policy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(len(FEATURES), 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, len(ACTIONS)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_policy(
    examples: list[tuple[str, np.ndarray, int]], seed: int, out: Path,
) -> tuple[Policy, np.ndarray, np.ndarray, dict]:
    groups = sorted({key for key, _, _ in examples})
    held = set(sorted(groups, key=lambda key: hashlib.sha256(
        f"rcts-internal\0{key}".encode()).hexdigest()
    )[:max(1, len(groups) // 5)])
    x = np.stack([row[1] for row in examples])
    y = np.asarray([row[2] for row in examples], dtype=np.int64)
    is_valid = np.asarray([row[0] in held for row in examples])
    if not np.any(~is_valid):
        raise ValueError("too few training clips for an internal holdout")
    mean = x[~is_valid].mean(axis=0)
    std = np.maximum(x[~is_valid].std(axis=0), 1e-4)
    x = np.clip((x - mean) / std, -8, 8)
    x_train = torch.from_numpy(x[~is_valid])
    y_train = torch.from_numpy(y[~is_valid])
    x_valid = torch.from_numpy(x[is_valid])
    y_valid = torch.from_numpy(y[is_valid])
    torch.manual_seed(seed)
    model = Policy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    generator = torch.Generator().manual_seed(seed)
    best_loss, best_state, best_epoch, patience = math.inf, None, 0, 0
    for epoch in range(1, 81):
        model.train()
        for rows in torch.randperm(len(x_train), generator=generator).split(64):
            loss = F.cross_entropy(model(x_train[rows]), y_train[rows])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_loss = float(F.cross_entropy(model(x_valid), y_valid).item())
        if valid_loss < best_loss - 1e-4:
            best_loss = valid_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
        if patience >= 12:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        validation_accuracy = float((model(x_valid).argmax(1) == y_valid).float().mean().item())
    counts = Counter(ACTIONS[index].name for index in y.tolist())
    summary = {
        "examples": len(examples), "source_clips": len(groups),
        "internal_holdout_clips": len(held), "internal_holdout_accuracy": validation_accuracy,
        "best_epoch": best_epoch, "best_validation_loss": best_loss,
        "teacher_choice_counts": dict(counts),
    }
    torch.save({
        "state_dict": best_state, "mean": mean, "std": std,
        "actions": [action.name for action in ACTIONS], "features": list(FEATURES),
        "seed": seed, "summary": summary,
    }, out / "policy.pth")
    return model, mean, std, summary


@torch.no_grad()
def predict_in_chunks(analyzer: ActionRecognitionAnalyzer, videos: torch.Tensor) -> torch.Tensor:
    device = next(analyzer.parameters()).device
    return torch.cat([
        analyzer.predict(part.to(device)).float().cpu()
        for part in videos.split(2)
    ])


def to_tensor(variants: dict[str, np.ndarray], names: list[str]) -> torch.Tensor:
    arr = np.stack([variants[name].transpose(3, 0, 1, 2) for name in names])
    return torch.from_numpy(arr.copy()).float().div_(255.0)


def prepare_clip(
    dataset: VideoClipDataset, index: int, teacher: ActionRecognitionAnalyzer,
) -> tuple[str, int, dict[str, np.ndarray], dict[str, float], torch.Tensor]:
    source, label, meta = dataset[index]
    source = source.unsqueeze(0)
    source_logits = predict_in_chunks(teacher, source)
    pseudo_label = source_logits.argmax(1).to(next(teacher.parameters()).device)
    saliency = task_saliency(
        teacher, source.to(pseudo_label.device), pseudo_label, blur=5
    )[0, 0].cpu().numpy()
    rgb = (source[0].permute(1, 2, 3, 0).numpy() * 255).round().astype(np.uint8)
    variants, info = make_candidates(rgb, saliency)
    return str(meta["sequence_id"]), label, variants, info, source_logits


def assess(logits: torch.Tensor, label: int, bpp: float, name: str) -> dict:
    logits = logits[None]
    target = torch.tensor([label])
    return {
        "name": name, "bpp": float(bpp),
        "correct": bool(logits.argmax(1).item() == label),
        "ce": float(F.cross_entropy(logits, target).item()),
        "target_prob": float(logits.softmax(1)[0, label].item()),
    }


def collect_teacher(
    args: argparse.Namespace, train_set: VideoClipDataset,
    teacher: ActionRecognitionAnalyzer, codec: StandardCodec, out: Path,
) -> tuple[list[tuple[str, np.ndarray, int]], dict]:
    indices = balanced_indices(train_set.samples, args.train_clips, "rcts-train-v1")
    examples: list[tuple[str, np.ndarray, int]] = []
    oracle_rates, identity_rates = [], []
    path = out / "train_teacher.jsonl"
    names = [action.name for action in ACTIONS]
    with path.open("w", encoding="utf-8") as handle:
        for number, index in enumerate(indices, 1):
            sequence_id, label, variants, info, source_logits = prepare_clip(
                train_set, index, teacher
            )
            videos = to_tensor(variants, names)
            for qp in QPS:
                reconstructed, rates = codec.compress_decompress_items(videos, qp=qp)
                scores = predict_in_chunks(teacher, reconstructed)
                candidates = [
                    assess(scores[i], label, rates[i], name)
                    for i, name in enumerate(names)
                ]
                best = select_teacher_action(candidates, max_ce_regret=args.ce_regret)
                examples.append((sequence_id, feature_vector(info, source_logits, qp), best))
                oracle_rates.append(candidates[best]["bpp"])
                identity_rates.append(candidates[0]["bpp"])
                handle.write(json.dumps({
                    "sequence_id": sequence_id, "codec": args.codec, "qp": qp,
                    "teacher_choice": names[best], "candidates": candidates,
                }) + "\n")
            if number % 10 == 0 or number == len(indices):
                handle.flush()
                print(f"[teacher] {number}/{len(indices)} codec={args.codec}", flush=True)
    summary = {
        "train_clips": len(indices),
        "train_fingerprint": fingerprint(train_set.samples, indices),
        "train_only_oracle_mean_rate_ratio": float(np.mean(
            np.asarray(oracle_rates) / np.asarray(identity_rates)
        )),
        "warning": "The oracle uses train labels and is not a deployable result.",
    }
    return examples, summary


def choose_policy(
    model: Policy, features: np.ndarray, mean: np.ndarray, std: np.ndarray,
    confidence: float,
) -> tuple[str, float]:
    x = np.clip((features - mean) / std, -8, 8)
    with torch.no_grad():
        probabilities = model(torch.from_numpy(x)[None]).softmax(1)[0]
    probability, index = probabilities.max(0)
    name = ACTIONS[int(index.item())].name if float(probability) >= confidence else "identity"
    return name, float(probability.item())


def evaluate(
    args: argparse.Namespace, val_set: VideoClipDataset,
    teacher: ActionRecognitionAnalyzer, codec: StandardCodec,
    model: Policy, mean: np.ndarray, std: np.ndarray, out: Path,
) -> dict:
    indices = balanced_indices(val_set.samples, args.val_clips, "rcts-val-v1")
    heldout = ActionRecognitionAnalyzer("r2plus1d_18", clip_size=112).freeze().to(
        next(teacher.parameters()).device
    )
    analyzers = {"r3d_18": teacher, "r2plus1d_18": heldout}
    # Each row has one actual codec encode; policy can reuse a control's encode.
    records: list[dict] = []
    arm_names = (*CONTROLS, "policy")
    chosen_counts: Counter[str] = Counter()
    with (out / "eval_records.jsonl").open("w", encoding="utf-8") as handle:
        for number, index in enumerate(indices, 1):
            sequence_id, label, variants, info, source_logits = prepare_clip(
                val_set, index, teacher
            )
            for qp in QPS:
                features = feature_vector(info, source_logits, qp)
                chosen, certainty = choose_policy(
                    model, features, mean, std, args.policy_confidence
                )
                chosen_counts[chosen] += 1
                names = list(dict.fromkeys((*CONTROLS, chosen)))
                videos = to_tensor(variants, names)
                reconstructed, rates = codec.compress_decompress_items(videos, qp=qp)
                logits = {
                    name: predict_in_chunks(analyzer, reconstructed)
                    for name, analyzer in analyzers.items()
                }
                by_name = {name: i for i, name in enumerate(names)}
                for arm in arm_names:
                    actual = chosen if arm == "policy" else arm
                    row = by_name[actual]
                    record = {
                        "sequence_id": sequence_id, "codec": args.codec, "qp": qp,
                        "arm": arm, "actual_action": actual,
                        "policy_confidence": certainty if arm == "policy" else None,
                        "bpp": float(rates[row]),
                    }
                    for analyzer_name, output in logits.items():
                        record[analyzer_name] = assess(
                            output[row], label, rates[row], actual
                        )
                    records.append(record)
                    handle.write(json.dumps(record) + "\n")
            if number % 10 == 0 or number == len(indices):
                handle.flush()
                print(f"[eval] {number}/{len(indices)} codec={args.codec}", flush=True)
    curves: dict[str, dict] = {}
    metrics: dict[str, dict] = {}
    for analyzer_name in analyzers:
        curves[analyzer_name] = {}
        metrics[analyzer_name] = {}
        for arm in arm_names:
            rows = []
            for qp in QPS:
                chosen_rows = [
                    row for row in records if row["arm"] == arm and row["qp"] == qp
                ]
                rows.append({
                    "qp": qp, "n": len(chosen_rows),
                    "bpp": float(np.mean([row["bpp"] for row in chosen_rows])),
                    "top1": float(np.mean([
                        row[analyzer_name]["correct"] for row in chosen_rows
                    ])),
                    "target_prob": float(np.mean([
                        row[analyzer_name]["target_prob"] for row in chosen_rows
                    ])),
                })
            curves[analyzer_name][arm] = rows
            if arm == "identity":
                continue
            anchor = curves[analyzer_name]["identity"]
            bd = bd_rate(
                [row["bpp"] for row in anchor], [row["top1"] for row in anchor],
                [row["bpp"] for row in rows], [row["top1"] for row in rows],
            )
            bd_acc = bd_metric(
                [row["bpp"] for row in anchor], [row["top1"] for row in anchor],
                [row["bpp"] for row in rows], [row["top1"] for row in rows],
            )
            metrics[analyzer_name][arm] = {
                "bd_rate_top1_pct": float(bd) if math.isfinite(bd) else None,
                "bd_accuracy_top1": float(bd_acc) if math.isfinite(bd_acc) else None,
                "mean_same_qp_rate_ratio": float(np.mean([
                    row["bpp"] / base["bpp"] for row, base in zip(rows, anchor)
                ])),
                "min_same_qp_top1_gap": min(
                    row["top1"] - base["top1"] for row, base in zip(rows, anchor)
                ),
            }
    result = {
        "experiment": "rcts_real_codec_pre_pilot_v1",
        "codec": args.codec, "seed": args.seed, "qps": list(QPS),
        "preset": args.preset, "val_clips": len(indices),
        "val_fingerprint": fingerprint(val_set.samples, indices),
        "selector_choice_counts": dict(chosen_counts),
        "curves": curves, "metrics": metrics,
        "interpretation": "Exploratory pilot; validation is reused development data.",
    }
    (out / "pilot_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({
        "codec": args.codec, "seed": args.seed,
        "heldout_metrics": metrics["r2plus1d_18"],
    }, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--codec", choices=("h264", "h265"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-clips", type=int, default=80)
    parser.add_argument("--val-clips", type=int, default=104)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--ce-regret", type=float, default=0.02)
    parser.add_argument("--policy-confidence", type=float, default=0.35)
    args = parser.parse_args()
    if not ffmpeg_available():
        raise SystemExit("ffmpeg and ffprobe are required")
    if args.train_clips < 10 or args.val_clips < 10:
        raise ValueError("pilot needs at least 10 train and 10 validation clips")
    if not 0 <= args.ce_regret or not 0 <= args.policy_confidence <= 1:
        raise ValueError("invalid regret or confidence threshold")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[rcts] codec={args.codec} seed={args.seed} device={device}", flush=True)
    train_set = VideoClipDataset(
        args.index, split="train", num_frames=16, frame_size=128,
        temporal_stride=2, train=False, return_metadata=True,
    )
    val_set = VideoClipDataset(
        args.index, split="val", num_frames=16, frame_size=128,
        temporal_stride=2, train=False, return_metadata=True,
    )
    teacher = ActionRecognitionAnalyzer("r3d_18", clip_size=112).freeze().to(device)
    codec = StandardCodec(args.codec, preset=args.preset)
    examples, teacher_summary = collect_teacher(args, train_set, teacher, codec, out)
    model, mean, std, policy_summary = train_policy(examples, args.seed, out)
    (out / "train_summary.json").write_text(json.dumps({
        "teacher": teacher_summary, "policy": policy_summary,
        "ce_regret": args.ce_regret, "policy_confidence": args.policy_confidence,
    }, indent=2), encoding="utf-8")
    evaluate(args, val_set, teacher, codec, model, mean, std, out)


if __name__ == "__main__":
    main()
