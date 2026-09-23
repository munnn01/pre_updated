#!/usr/bin/env python
"""Real-codec, label-free action selection with a shared 128-pixel rate denominator.

Only TRAIN labels calibrate fixed task-distance tolerances. At evaluation the
encoder may inspect the source and its own decoded candidates, but never a label.
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
import torch.nn.functional as F

from ops.rcts_pilot import QPS, balanced_indices, fingerprint
from src.codecs.standard import StandardCodec, ffmpeg_available
from src.data.video_dataset import VideoClipDataset
from src.metrics.bd_rate import bd_metric, bd_rate
from src.models.codec_search import CANDIDATES, choose_action, make_candidates, normalized_bpp
from src.tasks.action_recognition import ActionRecognitionAnalyzer


@torch.no_grad()
def predict_and_feature(analyzer: ActionRecognitionAnalyzer, video: torch.Tensor):
    """One frozen-network pass gives both logits and layer-2 semantics."""
    device = next(analyzer.parameters()).device
    net = analyzer.net
    x = net.stem(analyzer._prep(video.to(device)))
    x = net.layer1(x)
    x = net.layer2(x)
    feature = F.adaptive_avg_pool3d(x, (1, 1, 1)).flatten(1)
    x = net.layer4(net.layer3(x))
    logits = net.fc(net.avgpool(x).flatten(1))
    return logits.float().cpu(), feature.float().cpu()


def as_video(rgb: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(rgb.transpose(3, 0, 1, 2).copy())[None].float().div_(255.0)


def case_for_clip(dataset, index, analyzer, codec, cross=None):
    source, label, meta = dataset[index]
    source = source.unsqueeze(0)
    rgb = (source[0].permute(1, 2, 3, 0).numpy() * 255).round().astype(np.uint8)
    variants = make_candidates(rgb)
    source_logits, source_feature = predict_and_feature(analyzer, source)
    source_probs = source_logits.softmax(1)
    source_label = int(source_logits.argmax(1).item())
    source_confidence = float(source_probs.max().item())
    clean = {}
    for name, candidate in variants.items():
        logits, _ = predict_and_feature(analyzer, as_video(candidate))
        clean[name] = bool(logits.argmax(1).item() == label)
    measurements = []
    for qp in QPS:
        candidates = []
        for name, candidate in variants.items():
            t, h, w, _ = candidate.shape
            reconstructed, native_bpp = codec._encode_decode_clip(candidate, qp=qp)
            rate = normalized_bpp(native_bpp, h, w)
            logits, feature = predict_and_feature(analyzer, as_video(reconstructed))
            row = {
                "name": name, "bpp": rate,
                "coded_bytes": int(round(native_bpp * t * h * w / 8)),
                "encoded_size": [h, w],
                "kl_source": float(F.kl_div(logits.log_softmax(1), source_probs, reduction="batchmean").item()),
                "feature_distance": float((1 - F.cosine_similarity(feature, source_feature).item())),
                "source_confidence": source_confidence,
                "source_top1_agrees": bool(logits.argmax(1).item() == source_label),
                "correct": bool(logits.argmax(1).item() == label),
                "target_prob": float(logits.softmax(1)[0, label].item()),
            }
            if cross is not None:
                other, _ = predict_and_feature(cross, as_video(reconstructed))
                row["cross_correct"] = bool(other.argmax(1).item() == label)
                row["cross_target_prob"] = float(other.softmax(1)[0, label].item())
            candidates.append(row)
        measurements.append({"qp": qp, "candidates": candidates})
    return {
        "schema": 1, "sequence_id": str(meta["sequence_id"]),
        "codec": codec.codec, "source_correct": bool(source_label == label),
        "source_confidence": source_confidence, "clean_correct": clean,
        "measurements": measurements,
    }


def collect(dataset, indices, analyzer, codec, out, stage, cross=None):
    directory = out / f"{stage}_cache"
    directory.mkdir(exist_ok=True)
    rows = []
    for number, index in enumerate(indices, 1):
        path = directory / f"clip_{index:05d}.json"
        if path.exists():
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("schema") != 1 or row.get("codec") != codec.codec:
                raise ValueError(f"stale cache entry: {path}")
        else:
            row = case_for_clip(dataset, index, analyzer, codec, cross=cross)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(row, allow_nan=False), encoding="utf-8")
            temporary.replace(path)
        if len(row["measurements"]) != len(QPS) or any(
            [candidate["name"] for candidate in m["candidates"]] != list(CANDIDATES)
            for m in row["measurements"]
        ):
            raise ValueError(f"incomplete cached measurements: {path}")
        rows.append(row)
        if number % 10 == 0 or number == len(indices):
            print(f"[{stage}] {number}/{len(indices)} codec={codec.codec}", flush=True)
    return rows


def flatten(rows):
    return [measurement for row in rows for measurement in row["measurements"]]


def summarize(rows, selector, field="correct"):
    flat = flatten(rows)
    curve = {}
    chosen_names = Counter()
    for qp in QPS:
        picked = [m["candidates"][selector(m["candidates"])] for m in flat if m["qp"] == qp]
        chosen_names.update(row["name"] for row in picked)
        curve[str(qp)] = {
            "n": len(picked),
            "bpp": float(np.mean([r["bpp"] for r in picked])),
            "top1": float(np.mean([r[field] for r in picked])),
            "target_prob": float(np.mean([
                r["target_prob" if field == "correct" else "cross_target_prob"] for r in picked
            ])),
        }
    return curve, dict(chosen_names)


def compare(base, trial):
    a = [base[str(q)] for q in QPS]
    b = [trial[str(q)] for q in QPS]
    bd = bd_rate([r["bpp"] for r in a], [r["top1"] for r in a],
                 [r["bpp"] for r in b], [r["top1"] for r in b])
    bda = bd_metric([r["bpp"] for r in a], [r["top1"] for r in a],
                    [r["bpp"] for r in b], [r["top1"] for r in b])
    return {
        "bd_rate_top1_pct": float(bd) if math.isfinite(bd) else None,
        "bd_accuracy_top1": float(bda) if math.isfinite(bda) else None,
        "mean_same_qp_rate_ratio": float(np.mean([x["bpp"] / y["bpp"] for x, y in zip(b, a)])),
        "min_same_qp_top1_gap": float(min(x["top1"] - y["top1"] for x, y in zip(b, a))),
    }


def calibrate(train_rows):
    anchor, _ = summarize(train_rows, lambda _: 0)
    grid = []
    for kl_slack in (0.0, .005, .02, .05, .10):
        for feature_slack in (0.0, .005, .02, .05):
            curve, choices = summarize(
                train_rows, lambda candidates: choose_action(candidates, kl_slack, feature_slack)
            )
            metric = compare(anchor, curve)
            metric.update({"kl_slack": kl_slack, "feature_slack": feature_slack,
                           "non_identity": sum(n for key, n in choices.items() if key != "identity128")})
            metric["eligible"] = bool(
                metric["bd_rate_top1_pct"] is not None and metric["bd_rate_top1_pct"] < 0
                and metric["bd_accuracy_top1"] is not None and metric["bd_accuracy_top1"] >= 0
                and metric["min_same_qp_top1_gap"] >= -.01
                and metric["mean_same_qp_rate_ratio"] < 1.0
            )
            grid.append(metric)
    eligible = [row for row in grid if row["eligible"]]
    best = min(eligible, key=lambda r: r["bd_rate_top1_pct"]) if eligible else None
    return ({"active": True, "kl_slack": best["kl_slack"],
             "feature_slack": best["feature_slack"]} if best else
            {"active": False, "kl_slack": 0.0, "feature_slack": 0.0}), grid


def selected(measurements, limits):
    return choose_action(measurements, limits["kl_slack"], limits["feature_slack"]) if limits["active"] else 0


def evaluate(rows, limits, args, train_fingerprint, val_fingerprint, out):
    selectors = {name: (lambda _, i=i: i) for i, name in enumerate(CANDIDATES)}
    selectors["policy"] = lambda candidates: selected(candidates, limits)
    report = {"experiment": "codec_search_ar_v1", "codec": args.codec,
              "qps": list(QPS), "train_clips": args.train_clips, "val_clips": len(rows),
              "train_fingerprint": train_fingerprint, "val_fingerprint": val_fingerprint,
              "limits": limits, "analyzers": {}, "clean_top1": {},
              "interpretation": "Development validation; label-free selection, fixed trained threshold."}
    for name in CANDIDATES:
        report["clean_top1"][name] = float(np.mean([r["clean_correct"][name] for r in rows]))
    for model, field in (("r2plus1d_18", "correct"), ("r3d_18", "cross_correct")):
        curves, choices = {}, {}
        for name, selector in selectors.items():
            curves[name], choices[name] = summarize(rows, selector, field=field)
        report["analyzers"][model] = {
            "curves": curves, "choices": choices,
            "metrics": {name: compare(curves["identity128"], curves[name])
                        for name in selectors if name != "identity128"},
        }
    with (out / "eval_records.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            for measurement in row["measurements"]:
                candidates = measurement["candidates"]
                for arm, selector in selectors.items():
                    chosen = candidates[selector(candidates)]
                    handle.write(json.dumps({"sequence_id": row["sequence_id"],
                        "codec": args.codec, "qp": measurement["qp"], "arm": arm,
                        "candidate": chosen}) + "\n")
    (out / "search_result.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"codec": args.codec, "policy_matched":
                      report["analyzers"]["r2plus1d_18"]["metrics"]["policy"],
                      "policy_cross": report["analyzers"]["r3d_18"]["metrics"]["policy"]},
                     indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--codec", choices=("h264", "h265"), required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-clips", type=int, default=400)
    parser.add_argument("--val-clips", type=int, default=104)
    parser.add_argument("--preset", default="medium")
    args = parser.parse_args()
    if not ffmpeg_available():
        raise SystemExit("ffmpeg and ffprobe are required")
    if args.train_clips < 40 or args.val_clips < 20:
        raise ValueError("insufficient calibration or validation clips")
    random.seed(53)
    np.random.seed(53)
    torch.manual_seed(53)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[codec-search] codec={args.codec} device={device}", flush=True)
    train_set = VideoClipDataset(args.index, split="train", num_frames=16, frame_size=128,
                                 temporal_stride=2, train=False, return_metadata=True)
    val_set = VideoClipDataset(args.index, split="val", num_frames=16, frame_size=128,
                               temporal_stride=2, train=False, return_metadata=True)
    train_indices = balanced_indices(train_set.samples, args.train_clips, "codec-search-train-v1")
    val_indices = balanced_indices(val_set.samples, args.val_clips, "rcts-val-v1")
    analyzer = ActionRecognitionAnalyzer("r2plus1d_18", clip_size=112).freeze().to(device)
    codec = StandardCodec(args.codec, preset=args.preset)
    train_rows = collect(train_set, train_indices, analyzer, codec, out, "train")
    limits, grid = calibrate(train_rows)
    (out / "calibration.json").write_text(json.dumps({
        "limits": limits, "grid": grid,
        "train_fingerprint": fingerprint(train_set.samples, train_indices),
        "n_train": len(train_rows),
        "rule": "calibration uses train labels; evaluation selector reads only bytes, source KL and feature distance",
    }, indent=2, allow_nan=False), encoding="utf-8")
    print(f"[calibration] {json.dumps(limits)}", flush=True)
    cross = ActionRecognitionAnalyzer("r3d_18", clip_size=112).freeze().to(device)
    val_rows = collect(val_set, val_indices, analyzer, codec, out, "val", cross=cross)
    evaluate(val_rows, limits, args, fingerprint(train_set.samples, train_indices),
             fingerprint(val_set.samples, val_indices), out)


if __name__ == "__main__":
    main()
