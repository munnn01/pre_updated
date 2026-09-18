#!/usr/bin/env python
"""Zero-bit, QP-gated decoder Gaussian POST probe for action recognition.

The encoded bitstream is shared by every arm.  POST is applied only after the
real x264/x265 decoder, so each arm has exactly the anchor bitrate.  This tests
whether the high-QP denoising mechanism confirmed for object detection also
transfers to a frozen Kinetics-400 video analyzer.
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
from src.models.mask_suppress import gaussian_filter  # noqa: E402
from src.tasks.action_recognition import ActionRecognitionAnalyzer  # noqa: E402


def _float_grid(raw: str) -> list[float]:
    values = [float(value) for value in raw.split(",") if value.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError("post sigmas must be positive")
    return list(dict.fromkeys(values))


def _codec_grid(raw: str) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("codecs must be a non-empty unique list")
    if set(values) - {"h264", "h265"}:
        raise ValueError("codecs must be a subset of h264,h265")
    return values


def _arm(sigma: float, min_qp: int) -> str:
    return f"post{sigma:g}q{min_qp}"


def _finite(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-clips", type=int, default=200)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--qps", default="30,35,40,45,50")
    parser.add_argument("--codecs", default="h264,h265")
    parser.add_argument("--post-sigmas", default="1")
    parser.add_argument("--post-min-qp", type=int, default=45)
    parser.add_argument("--ar-backbone", default="r3d_18")
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--out", default="outputs/ar_decoder_post")
    args = parser.parse_args()

    if not ffmpeg_available():
        raise SystemExit("ffmpeg and ffprobe are required for real-codec evaluation")
    if args.n_clips <= 0:
        raise ValueError("n_clips must be positive")
    if args.post_min_qp < 0:
        raise ValueError("post_min_qp must be non-negative")

    qps = [int(value) for value in args.qps.split(",") if value.strip()]
    codecs = _codec_grid(args.codecs)
    sigmas = _float_grid(args.post_sigmas)
    arms = ["anchor", *[_arm(sigma, args.post_min_qp) for sigma in sigmas]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = VideoClipDataset(
        args.index,
        split=args.split,
        num_frames=args.num_frames,
        frame_size=args.size,
        temporal_stride=args.stride,
        train=False,
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
            (codec, qp): {"bpp": [], "correct": [], "target_prob": []}
            for codec in codecs for qp in qps
        }
        for arm in arms
    }

    def append(arm: str, codec: str, qp: int, bpp: float,
               logits: torch.Tensor, labels: torch.Tensor) -> None:
        probs = logits.softmax(dim=1)
        slot = store[arm][(codec, qp)]
        slot["bpp"].append(float(bpp))
        slot["correct"].append(float(logits.argmax(dim=1).eq(labels).item()))
        slot["target_prob"].append(float(probs.gather(1, labels[:, None]).item()))

    for clips, labels in tqdm(loader, desc="probe-action-post"):
        clips, labels = clips.to(device), labels.to(device)
        for codec_name in codecs:
            for qp in qps:
                standard = StandardCodec(codec_name, qp, args.preset)
                recon, bpps = standard.compress_decompress_items(clips)
                anchor_logits = analyzer.predict(recon)
                append("anchor", codec_name, qp, bpps[0], anchor_logits, labels)
                for sigma in sigmas:
                    arm = _arm(sigma, args.post_min_qp)
                    logits = (anchor_logits if qp < args.post_min_qp else
                              analyzer.predict(gaussian_filter(recon, sigma)))
                    append(arm, codec_name, qp, bpps[0], logits, labels)

    result = {
        "task": "action_recognition",
        "method": "decoder_gaussian_post",
        "ar_backbone": args.ar_backbone,
        "n_clips": len(dataset),
        "num_frames": args.num_frames,
        "size": args.size,
        "qps": qps,
        "codecs": codecs,
        "post_sigmas": sigmas,
        "post_min_qp": args.post_min_qp,
        "curves": {},
        "bd_vs_anchor": {},
    }
    flat = {}
    for codec_name in codecs:
        curves = {}
        for arm in arms:
            slots = [store[arm][(codec_name, qp)] for qp in qps]
            curves[arm] = {
                "bpp": [float(np.mean(slot["bpp"])) for slot in slots],
                "top1": [float(np.mean(slot["correct"])) for slot in slots],
                "target_prob": [float(np.mean(slot["target_prob"])) for slot in slots],
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
                "bd_rate_top1_pct": _finite(bd_rate(
                    anchor["bpp"], anchor["top1"], curve["bpp"], curve["top1"]
                )),
                "bd_accuracy_top1": _finite(bd_metric(
                    anchor["bpp"], anchor["top1"], curve["bpp"], curve["top1"]
                )),
                "bd_rate_target_prob_pct": _finite(bd_rate(
                    anchor["bpp"], anchor["target_prob"],
                    curve["bpp"], curve["target_prob"]
                )),
            }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "per_clip_records.npz", **flat)
    (out / "probe_action_post.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result["bd_vs_anchor"], indent=2))
    print(f"[action-post] wrote {out}")


if __name__ == "__main__":
    main()
