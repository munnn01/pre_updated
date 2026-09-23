#!/usr/bin/env python
"""400-clip RCTS risk-screen: real-codec targets, clip-level calibration, held-out AR.

The validation labels never train/calibrate the selector. The development
validation set is exploratory; a new source-disjoint test is required for a
final scientific claim.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from ops.rcts_pilot import QPS, assess, balanced_indices, fingerprint, predict_in_chunks, to_tensor
from src.codecs.standard import StandardCodec, ffmpeg_available
from src.data.video_dataset import VideoClipDataset
from src.metrics.bd_rate import bd_metric, bd_rate
from src.models.rcts import ACTIONS, make_candidates
from src.models.rcts_risk import RiskPredictor, choose_action, source_feature
from src.models.task_mask import task_saliency
from src.tasks.action_recognition import ActionRecognitionAnalyzer

NAMES = [action.name for action in ACTIONS]
CONTROLS = ("identity", "uniform_s20", "spatial_s25", "temporal_t20", "joint_s25_t15")


def source_case(dataset, index, teacher):
    video, label, meta = dataset[index]
    video = video.unsqueeze(0)
    logits = predict_in_chunks(teacher, video)
    pseudo = logits.argmax(1).to(next(teacher.parameters()).device)
    saliency = task_saliency(teacher, video.to(pseudo.device), pseudo, blur=5)[0, 0].cpu().numpy()
    feature = source_feature(teacher, video, saliency)
    rgb = (video[0].permute(1, 2, 3, 0).numpy() * 255).round().astype(np.uint8)
    variants, _ = make_candidates(rgb, saliency)
    return str(meta["sequence_id"]), int(label), feature, variants


def collect(args, dataset, teacher, second, codec, out):
    indices = balanced_indices(dataset.samples, args.train_clips, "rcts-risk-train-v1")
    cache = out / "teacher_cache"
    cache.mkdir(exist_ok=True)
    examples = []
    with (out / "teacher_records.jsonl").open("w", encoding="utf-8") as manifest:
        for number, index in enumerate(indices, 1):
            path = cache / f"clip_{index:05d}.pt"
            if path.exists():
                row = torch.load(path, map_location="cpu", weights_only=False)
            else:
                sequence, label, feature, variants = source_case(dataset, index, teacher)
                videos = to_tensor(variants, NAMES)
                measurements = []
                targets = []
                for qp in QPS:
                    reconstructed, rates = codec.compress_decompress_items(videos, qp=qp)
                    r3d = predict_in_chunks(teacher, reconstructed)
                    mc3 = predict_in_chunks(second, reconstructed)
                    teachers = {
                        "r3d_18": [assess(r3d[i], label, rates[i], NAMES[i]) for i in range(len(NAMES))],
                        "mc3_18": [assess(mc3[i], label, rates[i], NAMES[i]) for i in range(len(NAMES))],
                    }
                    measurements.append({"qp": qp, "teachers": teachers})
                    target = []
                    for action in range(len(NAMES)):
                        rate = math.log(max(float(rates[action]), 1e-9) / max(float(rates[0]), 1e-9))
                        regret = max(
                            teachers[name][action]["ce"] - teachers[name][0]["ce"]
                            for name in teachers
                        )
                        harm = any(
                            teachers[name][0]["correct"] and not teachers[name][action]["correct"]
                            for name in teachers
                        )
                        target.append([rate, regret, float(harm)])
                    targets.append(target)
                row = {
                    "sequence_id": sequence, "label": label, "feature": feature,
                    "targets": torch.tensor(targets, dtype=torch.float32),
                    "measurements": measurements,
                }
                # Per-clip atomic replacement preserves completed work across interruptions.
                temporary = path.with_suffix(".tmp")
                torch.save(row, temporary)
                temporary.replace(path)
            examples.append(row)
            manifest.write(json.dumps({
                "sequence_id": row["sequence_id"], "codec": args.codec,
                "measurements": row["measurements"],
            }) + "\n")
            if number % 10 == 0 or number == len(indices):
                manifest.flush()
                print(f"[teacher] {number}/{len(indices)} codec={args.codec}", flush=True)
    return examples, fingerprint(dataset.samples, indices)


def split_examples(rows):
    # Three disjoint source sets: fit, early-stop, safety calibration.
    groups = sorted({row["sequence_id"] for row in rows})
    order = sorted(groups, key=lambda key: hashlib.sha256(f"risk-cal-v1\0{key}".encode()).hexdigest())
    n_cal = max(1, len(order) // 5)
    n_stop = max(1, len(order) // 10)
    cal_ids = set(order[:n_cal])
    stop_ids = set(order[n_cal:n_cal + n_stop])
    fit = [row for row in rows if row["sequence_id"] not in cal_ids | stop_ids]
    stop = [row for row in rows if row["sequence_id"] in stop_ids]
    cal = [row for row in rows if row["sequence_id"] in cal_ids]
    if not fit or not stop or not cal:
        raise ValueError("empty fit, early-stop or calibration split")
    return fit, stop, cal


def as_batch(rows, device):
    features = torch.stack([row["feature"] for row in rows]).to(device)
    targets = torch.stack([row["targets"] for row in rows]).to(device)
    features = features.repeat_interleave(len(QPS), 0)
    qp = torch.tensor([(q - 30) / 20 for _ in rows for q in QPS], device=device)
    return features, qp, targets.reshape(-1, len(NAMES), 3)


def loss_fn(pred, target, positive_weight):
    rate = F.smooth_l1_loss(pred[:, 1:, 0], target[:, 1:, 0] / .25)
    regret = F.smooth_l1_loss(pred[:, 1:, 1], target[:, 1:, 1] / .2)
    harm = F.binary_cross_entropy_with_logits(
        pred[:, 1:, 2], target[:, 1:, 2], pos_weight=positive_weight
    )
    # Pairwise rate ordering only among actions with no observed harmful flip.
    safe = target[:, 1:, 2] == 0
    delta = target[:, 1:, 0, None] - target[:, None, 1:, 0]
    predicted = pred[:, 1:, 0, None] - pred[:, None, 1:, 0]
    pair = safe[:, :, None] & safe[:, None, :] & (delta.abs() > .02)
    ranking = F.softplus(-predicted[pair] * delta[pair].sign()).mean() if pair.any() else pred.sum() * 0
    return rate + regret + .5 * harm + .05 * ranking


def train(rows, seed, out, device):
    fit, stop, calibration = split_examples(rows)
    x, qp, target = as_batch(fit, device)
    positive = target[:, 1:, 2].sum()
    negative = target[:, 1:, 2].numel() - positive
    positive_weight = (negative / positive.clamp_min(1)).clamp(1, 8)
    torch.manual_seed(seed)
    model = RiskPredictor(len(NAMES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    generator = torch.Generator().manual_seed(seed)
    best, best_state, best_epoch, patience = math.inf, None, 0, 0
    for epoch in range(1, 81):
        model.train()
        for subset in torch.randperm(len(x), generator=generator).split(32):
            prediction = model(x[subset], qp[subset])
            loss = loss_fn(prediction, target[subset], positive_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            cx, cq, cy = as_batch(stop, device)
            validation = float(loss_fn(model(cx, cq), cy, positive_weight).item())
        print(f"[policy] epoch={epoch} calibration_loss={validation:.5f}", flush=True)
        if validation < best - 1e-4:
            best, best_epoch, patience = validation, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if patience >= 12:
            break
    model.load_state_dict(best_state)
    model.eval()
    limits, table = calibrate(model, calibration, device)
    summary = {
        "fit_clips": len(fit), "early_stop_clips": len(stop),
        "calibration_clips": len(calibration),
        "fit_fingerprint": hashlib.sha256("\n".join(sorted(r["sequence_id"] for r in fit)).encode()).hexdigest()[:16],
        "early_stop_fingerprint": hashlib.sha256("\n".join(sorted(r["sequence_id"] for r in stop)).encode()).hexdigest()[:16],
        "calibration_fingerprint": hashlib.sha256("\n".join(sorted(r["sequence_id"] for r in calibration)).encode()).hexdigest()[:16],
        "best_epoch": best_epoch, "best_calibration_loss": best,
        "limits": limits, "calibration_grid": table,
        "warning": "Calibration uses TRAIN-only clips; validation labels are untouched.",
    }
    torch.save({"state_dict": best_state, "actions": NAMES, "qps": QPS,
                "limits": limits, "seed": seed, "summary": summary}, out / "policy.pth")
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return model, limits, summary


@torch.no_grad()
def infer(model, feature, qp, device):
    x = feature[None].to(device)
    p = torch.tensor([(qp - 30) / 20], device=device)
    output = model(x, p)[0].cpu().numpy()
    output[:, 0] *= .25
    output[:, 1] *= .2
    return output


def calibrate(model, rows, device):
    cached = []
    for row in rows:
        for k, qp in enumerate(QPS):
            cached.append((qp, infer(model, row["feature"], qp, device), row["targets"][k].numpy()))
    table = []
    for ce in (.0, .01, .02, .04, .08):
        for risk in (.05, .1, .2, .35, .5, .7):
            choices = [(qp, choose_action(pred, ce, risk), truth) for qp, pred, truth in cached]
            harms = [truth[action, 2] for _, action, truth in choices]
            regrets = [truth[action, 1] for _, action, truth in choices]
            rates = [truth[action, 0] for _, action, truth in choices]
            worst_qp_harm = max(np.mean([truth[action, 2] for q, action, truth in choices if q == qp]) for qp in QPS)
            row = {"ce_limit": ce, "risk_limit": risk,
                   "harm_rate": float(np.mean(harms)), "worst_qp_harm_rate": float(worst_qp_harm),
                   "mean_ce_regret": float(np.mean(regrets)), "mean_log_rate": float(np.mean(rates)),
                   "non_identity": int(sum(action != 0 for _, action, _ in choices))}
            row["eligible"] = bool(row["harm_rate"] <= .01 and worst_qp_harm <= .025 and row["mean_ce_regret"] <= .02)
            table.append(row)
    eligible = [r for r in table if r["eligible"] and r["mean_log_rate"] < -.01]
    chosen = min(eligible, key=lambda r: r["mean_log_rate"]) if eligible else None
    return ({"ce_limit": chosen["ce_limit"], "risk_limit": chosen["risk_limit"]}
            if chosen else {"ce_limit": -math.inf, "risk_limit": 0.0}), table


def evaluate(args, dataset, teacher, codec, model, limits, out, device):
    indices = balanced_indices(dataset.samples, args.val_clips, "rcts-val-v1")
    heldout = ActionRecognitionAnalyzer("r2plus1d_18", clip_size=112).freeze().to(device)
    analyzers = {"r3d_18": teacher, "r2plus1d_18": heldout}
    arms = (*CONTROLS, "policy")
    records, choices = [], Counter()
    with (out / "eval_records.jsonl").open("w", encoding="utf-8") as handle:
        for number, index in enumerate(indices, 1):
            sequence, label, feature, variants = source_case(dataset, index, teacher)
            for qp in QPS:
                pred = infer(model, feature, qp, device)
                choice = NAMES[choose_action(pred, **limits)]
                choices[choice] += 1
                names = list(dict.fromkeys((*CONTROLS, choice)))
                reconstructed, rates = codec.compress_decompress_items(to_tensor(variants, names), qp=qp)
                scores = {key: predict_in_chunks(net, reconstructed) for key, net in analyzers.items()}
                for arm in arms:
                    action = choice if arm == "policy" else arm
                    k = names.index(action)
                    row = {"sequence_id": sequence, "qp": qp, "arm": arm,
                           "actual_action": action, "bpp": float(rates[k]),
                           "predicted_log_rate": float(pred[NAMES.index(action), 0]) if arm == "policy" else None,
                           "predicted_ce_regret": float(pred[NAMES.index(action), 1]) if arm == "policy" else None,
                           "predicted_harm_risk": float(1 / (1 + np.exp(-np.clip(pred[NAMES.index(action), 2], -30, 30)))) if arm == "policy" else None}
                    row.update({key: assess(value[k], label, rates[k], action) for key, value in scores.items()})
                    records.append(row)
                    handle.write(json.dumps(row) + "\n")
            if number % 10 == 0 or number == len(indices):
                handle.flush()
                print(f"[eval] {number}/{len(indices)} codec={args.codec}", flush=True)
    curves, metrics = {}, {}
    for backbone in analyzers:
        curves[backbone], metrics[backbone] = {}, {}
        for arm in arms:
            curve = []
            for qp in QPS:
                subset = [r for r in records if r["arm"] == arm and r["qp"] == qp]
                curve.append({"qp": qp, "n": len(subset),
                              "bpp": float(np.mean([r["bpp"] for r in subset])),
                              "top1": float(np.mean([r[backbone]["correct"] for r in subset])),
                              "target_prob": float(np.mean([r[backbone]["target_prob"] for r in subset]))})
            curves[backbone][arm] = curve
            if arm == "identity":
                continue
            base = curves[backbone]["identity"]
            bd = bd_rate([r["bpp"] for r in base], [r["top1"] for r in base],
                         [r["bpp"] for r in curve], [r["top1"] for r in curve])
            bd_acc = bd_metric([r["bpp"] for r in base], [r["top1"] for r in base],
                               [r["bpp"] for r in curve], [r["top1"] for r in curve])
            metrics[backbone][arm] = {
                "bd_rate_top1_pct": float(bd) if math.isfinite(bd) else None,
                "bd_accuracy_top1": float(bd_acc) if math.isfinite(bd_acc) else None,
                "mean_same_qp_rate_ratio": float(np.mean([r["bpp"] / b["bpp"] for r, b in zip(curve, base)])),
                "min_same_qp_top1_gap": min(r["top1"] - b["top1"] for r, b in zip(curve, base)),
            }
    result = {"experiment": "rcts_risk_400_v1", "codec": args.codec,
              "seed": args.seed, "qps": list(QPS), "train_clips": args.train_clips,
              "val_clips": len(indices), "val_fingerprint": fingerprint(dataset.samples, indices),
              "selection_counts": dict(choices), "limits": limits,
              "curves": curves, "metrics": metrics,
              "interpretation": "Exploratory development validation; not final held-out test."}
    (out / "risk_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"codec": args.codec, "heldout_metrics": metrics["r2plus1d_18"]}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--codec", choices=("h264", "h265"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-clips", type=int, default=400)
    parser.add_argument("--val-clips", type=int, default=104)
    parser.add_argument("--preset", default="medium")
    args = parser.parse_args()
    if not ffmpeg_available():
        raise SystemExit("ffmpeg and ffprobe are required")
    if args.train_clips < 20 or args.val_clips < 10:
        raise ValueError("insufficient train or evaluation clips")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[rcts-risk] codec={args.codec} seed={args.seed} device={device}", flush=True)
    train_set = VideoClipDataset(args.index, split="train", num_frames=16, frame_size=128,
                                 temporal_stride=2, train=False, return_metadata=True)
    val_set = VideoClipDataset(args.index, split="val", num_frames=16, frame_size=128,
                               temporal_stride=2, train=False, return_metadata=True)
    teacher = ActionRecognitionAnalyzer("r3d_18", clip_size=112).freeze().to(device)
    second = ActionRecognitionAnalyzer("mc3_18", clip_size=112).freeze().to(device)
    codec = StandardCodec(args.codec, preset=args.preset)
    rows, train_fingerprint = collect(args, train_set, teacher, second, codec, out)
    model, limits, summary = train(rows, args.seed, out, device)
    summary["train_fingerprint"] = train_fingerprint
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    del second
    if device.type == "cuda":
        torch.cuda.empty_cache()
    evaluate(args, val_set, teacher, codec, model, limits, out, device)


if __name__ == "__main__":
    main()
