#!/usr/bin/env python
"""Motion-preserving decoder post-filter probe for action recognition.

The previous whole-frame Gaussian POST arm reduced both target-class
probability and top-1 accuracy.  This follow-up keeps the same zero-bit,
decoder-only design but blurs only temporally static pixels.  Motion-bearing
regions and a small spatiotemporal context are copied exactly from the decoded
anchor.  A whole-frame Gaussian arm is retained as an internal negative
control; every arm shares the exact same encoded bitstream and bitrate.
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
from src.models.mask_suppress import (  # noqa: E402
    gaussian_filter,
    motion_preserving_gaussian,
)
from src.tasks.action_recognition import ActionRecognitionAnalyzer  # noqa: E402


def _codec_grid(raw: str) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("codecs must be a non-empty unique list")
    if set(values) - {"h264", "h265"}:
        raise ValueError("codecs must be a subset of h264,h265")
    return values


def _quantile_grid(raw: str) -> list[float]:
    values = [float(value) for value in raw.split(",") if value.strip()]
    if not values or any(not 0.0 < value < 1.0 for value in values):
        raise ValueError("motion quantiles must be strictly between zero and one")
    return list(dict.fromkeys(values))


def _global_arm(sigma: float, min_qp: int) -> str:
    return f"global{sigma:g}q{min_qp}"


def _motion_arm(quantile: float, sigma: float, min_qp: int) -> str:
    return f"motionq{quantile * 100:g}_s{sigma:g}q{min_qp}"


def _finite(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


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
    parser.add_argument("--motion-quantiles", default="0.5,0.75")
    parser.add_argument("--spatial-sigma", type=float, default=1.0)
    parser.add_argument("--post-min-qp", type=int, default=45)
    parser.add_argument("--motion-dilation", type=int, default=2)
    parser.add_argument("--motion-feather", type=int, default=2)
    parser.add_argument("--ar-backbone", default="r3d_18")
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--out", default="outputs/ar_motion_post")
    args = parser.parse_args()

    if not ffmpeg_available():
        raise SystemExit("ffmpeg and ffprobe are required for real-codec evaluation")
    if args.n_clips <= 0:
        raise ValueError("n_clips must be positive")
    if args.spatial_sigma <= 0:
        raise ValueError("spatial_sigma must be positive")
    if args.post_min_qp < 0:
        raise ValueError("post_min_qp must be non-negative")
    if args.motion_dilation < 0 or args.motion_feather < 0:
        raise ValueError("motion dilation and feather must be non-negative")

    qps = [int(value) for value in args.qps.split(",") if value.strip()]
    codecs = _codec_grid(args.codecs)
    quantiles = _quantile_grid(args.motion_quantiles)
    global_arm = _global_arm(args.spatial_sigma, args.post_min_qp)
    motion_arms = [
        _motion_arm(q, args.spatial_sigma, args.post_min_qp) for q in quantiles
    ]
    arms = ["anchor", global_arm, *motion_arms]
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
    analyzer = ActionRecognitionAnalyzer(args.ar_backbone).freeze().to(device)

    store = {
        arm: {
            (codec, qp): {
                "bpp": [],
                "correct": [],
                "target_prob": [],
                "protected_fraction": [],
            }
            for codec in codecs for qp in qps
        }
        for arm in arms
    }
    clip_ids: list[str] = []

    def append(
        arm: str,
        codec: str,
        qp: int,
        bpp: float,
        logits: torch.Tensor,
        labels: torch.Tensor,
        protected_fraction: float,
    ) -> None:
        probs = logits.softmax(dim=1)
        slot = store[arm][(codec, qp)]
        slot["bpp"].append(float(bpp))
        slot["correct"].append(float(logits.argmax(dim=1).eq(labels).item()))
        slot["target_prob"].append(float(probs.gather(1, labels[:, None]).item()))
        slot["protected_fraction"].append(float(protected_fraction))

    for clips, labels, metadata in tqdm(loader, desc="probe-action-motion-post"):
        clips, labels = clips.to(device), labels.to(device)
        clip_ids.append(str(metadata[0]["sequence_id"]))
        for codec_name in codecs:
            for qp in qps:
                standard = StandardCodec(codec_name, qp, args.preset)
                recon, bpps = standard.compress_decompress_items(clips)
                anchor_logits = analyzer.predict(recon)
                append("anchor", codec_name, qp, bpps[0], anchor_logits, labels, 1.0)

                if qp < args.post_min_qp:
                    for arm in [global_arm, *motion_arms]:
                        append(arm, codec_name, qp, bpps[0], anchor_logits, labels, 1.0)
                    continue

                variants = [gaussian_filter(recon, args.spatial_sigma)]
                fractions = [0.0]
                for quantile in quantiles:
                    filtered, mask = motion_preserving_gaussian(
                        recon,
                        args.spatial_sigma,
                        motion_quantile=quantile,
                        dilation=args.motion_dilation,
                        feather=args.motion_feather,
                    )
                    variants.append(filtered)
                    fractions.append(float(mask.mean().item()))
                logits = analyzer.predict(torch.cat(variants, dim=0))
                for row, arm, fraction in zip(
                    logits.split(1, dim=0), [global_arm, *motion_arms], fractions
                ):
                    append(arm, codec_name, qp, bpps[0], row, labels, fraction)

    result = {
        "task": "action_recognition",
        "method": "motion_preserving_decoder_post",
        "split": args.split,
        "ar_backbone": args.ar_backbone,
        "n_clips": len(dataset),
        "num_frames": args.num_frames,
        "size": args.size,
        "qps": qps,
        "codecs": codecs,
        "motion_quantiles": quantiles,
        "spatial_sigma": args.spatial_sigma,
        "post_min_qp": args.post_min_qp,
        "motion_dilation": args.motion_dilation,
        "motion_feather": args.motion_feather,
        "global_control": global_arm,
        "curves": {},
        "bd_vs_anchor": {},
    }
    flat: dict[str, np.ndarray] = {
        "clip_id": np.asarray(clip_ids, dtype=str),
    }
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
                "protected_fraction": [
                    float(np.mean(slot["protected_fraction"])) for slot in slots
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
    (out / "probe_action_motion_post.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result["bd_vs_anchor"], indent=2))
    print(f"[action-motion-post] wrote {out}")


if __name__ == "__main__":
    main()
