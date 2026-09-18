#!/usr/bin/env python
"""Encoder-side action-saliency suppression probe.

The encoder obtains a pseudo-label from a frozen action teacher, differentiates
that decision with respect to the source clip, and protects a fixed saliency
budget exactly.  Only the remaining background is blurred and temporally
stabilised before the real codec.  No label or mask is transmitted, and all
arms are evaluated against the same frozen analyzer and source clips.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.codecs.standard import StandardCodec, ffmpeg_available  # noqa: E402
from src.data.video_dataset import VideoClipDataset, collate_clips  # noqa: E402
from src.metrics.bd_rate import bd_metric, bd_rate  # noqa: E402
from src.models.importance_tube import (  # noqa: E402
    ImportanceTubeSuppress,
    expand_importance_tube,
    feather_protection,
)
from src.models.task_mask import hard_saliency_mask, task_saliency  # noqa: E402
from src.tasks.action_recognition import ActionRecognitionAnalyzer  # noqa: E402


def _float_grid(raw: str, name: str) -> list[float]:
    try:
        values = [float(value) for value in raw.split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{name} must be a non-empty unique grid")
    return values


def _protect_grid(raw: str) -> list[float]:
    values = _float_grid(raw, "protect fractions")
    if any(not 0.0 < value < 1.0 for value in values):
        raise ValueError("protect fractions must be in (0,1)")
    return values


def _mode_grid(raw: str) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("temporal modes must be a non-empty unique grid")
    if set(values) - {"clip", "tube"}:
        raise ValueError("temporal modes must be a subset of clip,tube")
    return values


def _codec_grid(raw: str) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("codecs must be a non-empty unique grid")
    if set(values) - {"h264", "h265"}:
        raise ValueError("codecs must be a subset of h264,h265")
    return values


def _arm(fraction: float, mode: str, sigma: float, strength: float) -> str:
    return (
        f"sal{fraction * 100:g}_{mode}_s{sigma:g}_t{strength:g}"
    )


def _finite(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


@torch.no_grad()
def _predict_chunks(
    analyzer: ActionRecognitionAnalyzer,
    videos: torch.Tensor,
    chunk_size: int = 2,
) -> torch.Tensor:
    return torch.cat(
        [analyzer.predict(part) for part in videos.split(chunk_size, dim=0)], dim=0
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--n-clips", type=int, default=200)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--qps", default="30,35,40,45,50")
    parser.add_argument("--codecs", default="h264,h265")
    parser.add_argument("--protect-fractions", default="0.15,0.25,0.4")
    parser.add_argument("--temporal-modes", default="clip,tube")
    parser.add_argument("--sigma", type=float, default=8.0)
    parser.add_argument("--temporal-strength", type=float, default=0.75)
    parser.add_argument("--temporal-radius", type=int, default=0)
    parser.add_argument("--feather", type=int, default=1)
    parser.add_argument("--motion-tau", type=float, default=0.05)
    parser.add_argument("--saliency-blur", type=int, default=5)
    parser.add_argument("--saliency-teacher", default="r3d_18")
    parser.add_argument("--eval-backbone", default="r3d_18")
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--predict-chunk", type=int, default=2)
    parser.add_argument("--out", default="outputs/ar_saliency")
    args = parser.parse_args()

    if not ffmpeg_available():
        raise SystemExit("ffmpeg and ffprobe are required for real-codec evaluation")
    if args.n_clips <= 0 or args.predict_chunk <= 0:
        raise ValueError("n_clips and predict_chunk must be positive")
    if args.sigma <= 0 or args.motion_tau <= 0:
        raise ValueError("sigma and motion_tau must be positive")
    if not 0.0 <= args.temporal_strength <= 1.0:
        raise ValueError("temporal_strength must be in [0,1]")
    if args.temporal_radius < 0 or args.feather < 0:
        raise ValueError("temporal_radius and feather must be non-negative")

    qps = [int(value) for value in args.qps.split(",") if value.strip()]
    codecs = _codec_grid(args.codecs)
    fractions = _protect_grid(args.protect_fractions)
    modes = _mode_grid(args.temporal_modes)
    arm_cfg = {
        _arm(fraction, mode, args.sigma, args.temporal_strength): (fraction, mode)
        for fraction in fractions for mode in modes
    }
    arms = ["anchor", *arm_cfg]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = VideoClipDataset(
        args.index,
        split=args.split,
        num_frames=args.num_frames,
        frame_size=args.size,
        temporal_stride=args.stride,
        train=False,
        return_metadata=True,
    )
    dataset = Subset(dataset, range(min(args.n_clips, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_clips,
    )
    teacher = ActionRecognitionAnalyzer(args.saliency_teacher).freeze().to(device)
    evaluator = (
        teacher
        if args.eval_backbone == args.saliency_teacher
        else ActionRecognitionAnalyzer(args.eval_backbone).freeze().to(device)
    )
    transforms = {
        arm: ImportanceTubeSuppress(
            sigma=args.sigma,
            temporal_radius=args.temporal_radius,
            feather=args.feather,
            temporal_strength=args.temporal_strength,
            motion_tau=args.motion_tau,
        ).to(device)
        for arm in arm_cfg
    }

    store = {
        arm: {
            (codec, qp): {"bpp": [], "correct": [], "target_prob": []}
            for codec in codecs for qp in qps
        }
        for arm in arms
    }
    source_store = {
        arm: {"correct": [], "target_prob": []} for arm in arms
    }
    mask_store = {
        arm: {"core_fraction": [], "effective_fraction": []} for arm in arm_cfg
    }
    clip_ids: list[str] = []

    def append_metrics(slot: dict, logits: torch.Tensor, label: torch.Tensor) -> None:
        probs = logits.softmax(dim=1)
        slot["correct"].append(float(logits.argmax(dim=1).eq(label).item()))
        slot["target_prob"].append(
            float(probs.gather(1, label[:, None]).item())
        )

    for clips, labels, metadata in tqdm(loader, desc="probe-action-saliency"):
        clips, labels = clips.to(device), labels.to(device)
        clip_ids.append(str(metadata[0]["sequence_id"]))
        with torch.no_grad():
            pseudo_labels = teacher.predict(clips).argmax(dim=1)
        saliency = task_saliency(
            teacher, clips, pseudo_labels, blur=args.saliency_blur
        )

        variants = {"anchor": clips}
        for arm, (fraction, mode) in arm_cfg.items():
            core = hard_saliency_mask(saliency, fraction, mode)
            effective = feather_protection(
                expand_importance_tube(core, args.temporal_radius), args.feather
            )
            mask_store[arm]["core_fraction"].append(float(core.mean().item()))
            mask_store[arm]["effective_fraction"].append(
                float(effective.mean().item())
            )
            variants[arm] = transforms[arm](clips, core)

        names = list(variants)
        videos = torch.cat([variants[name] for name in names], dim=0)
        source_logits = _predict_chunks(evaluator, videos, args.predict_chunk)
        for row, arm in zip(source_logits.split(1, dim=0), names):
            append_metrics(source_store[arm], row, labels)

        for codec_name in codecs:
            for qp in qps:
                standard = StandardCodec(codec_name, qp, args.preset)
                recon, bpps = standard.compress_decompress_items(videos)
                logits = _predict_chunks(
                    evaluator, recon.to(device), args.predict_chunk
                )
                for row, bpp, arm in zip(logits.split(1, dim=0), bpps, names):
                    slot = store[arm][(codec_name, qp)]
                    slot["bpp"].append(float(bpp))
                    append_metrics(slot, row, labels)

    result = {
        "task": "action_recognition",
        "method": "encoder_action_saliency_suppression",
        "split": args.split,
        "saliency_teacher": args.saliency_teacher,
        "eval_backbone": args.eval_backbone,
        "n_clips": len(dataset),
        "num_frames": args.num_frames,
        "size": args.size,
        "qps": qps,
        "codecs": codecs,
        "protect_fractions": fractions,
        "temporal_modes": modes,
        "sigma": args.sigma,
        "temporal_strength": args.temporal_strength,
        "temporal_radius": args.temporal_radius,
        "feather": args.feather,
        "motion_tau": args.motion_tau,
        "source_metrics": {},
        "mask_stats": {},
        "curves": {},
        "bd_vs_anchor": {},
    }
    flat: dict[str, np.ndarray] = {"clip_id": np.asarray(clip_ids, dtype=str)}
    for arm in arms:
        result["source_metrics"][arm] = {
            metric: float(np.mean(values))
            for metric, values in source_store[arm].items()
        }
        for metric, values in source_store[arm].items():
            flat[f"source_{arm}_{metric}"] = np.asarray(values, dtype=np.float32)
    for arm, metrics in mask_store.items():
        result["mask_stats"][arm] = {
            metric: float(np.mean(values)) for metric, values in metrics.items()
        }
        for metric, values in metrics.items():
            flat[f"{arm}_{metric}"] = np.asarray(values, dtype=np.float32)

    for codec_name in codecs:
        curves = {}
        for arm in arms:
            slots = [store[arm][(codec_name, qp)] for qp in qps]
            curves[arm] = {
                "bpp": [float(np.mean(slot["bpp"])) for slot in slots],
                "top1": [float(np.mean(slot["correct"])) for slot in slots],
                "target_prob": [
                    float(np.mean(slot["target_prob"])) for slot in slots
                ],
            }
            for qp, slot in zip(qps, slots):
                tag = f"{arm}_{codec_name}_{qp}"
                for metric, values in slot.items():
                    flat[f"{tag}_{metric}"] = np.asarray(values, dtype=np.float32)
        result["curves"][codec_name] = curves
        anchor = curves["anchor"]
        result["bd_vs_anchor"][codec_name] = {}
        for arm in arms[1:]:
            curve = curves[arm]
            result["bd_vs_anchor"][codec_name][arm] = {
                "bd_rate_top1_pct": _finite(
                    bd_rate(
                        anchor["bpp"], anchor["top1"], curve["bpp"], curve["top1"]
                    )
                ),
                "bd_accuracy_top1": _finite(
                    bd_metric(
                        anchor["bpp"], anchor["top1"], curve["bpp"], curve["top1"]
                    )
                ),
                "bd_rate_target_prob_pct": _finite(
                    bd_rate(
                        anchor["bpp"],
                        anchor["target_prob"],
                        curve["bpp"],
                        curve["target_prob"],
                    )
                ),
            }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "per_clip_records.npz", **flat)
    (out / "probe_action_saliency.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result["bd_vs_anchor"], indent=2))
    print(f"[action-saliency] wrote {out}")


if __name__ == "__main__":
    main()
