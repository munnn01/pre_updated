#!/usr/bin/env python
"""Parameter-free OD-to-AR bridge: detector tubes + background suppression.

The encoder-side detector runs once on every source frame.  Its detections form
short importance tubes; object cores remain byte-identical while static
background is spatially simplified and temporally stabilised.  The probe then
measures real x264/x265 BD-rate on a frozen Kinetics analyzer.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_detection import Detector  # noqa: E402
from src.codecs.standard import StandardCodec, ffmpeg_available  # noqa: E402
from src.data.video_dataset import VideoClipDataset, collate_clips  # noqa: E402
from src.metrics.bd_rate import bd_metric, bd_rate  # noqa: E402
from src.models.importance_tube import (  # noqa: E402
    ImportanceTubeSuppress,
    masks_from_frame_detections,
)
from src.tasks.action_recognition import ActionRecognitionAnalyzer  # noqa: E402


def _finite(value):
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
    parser.add_argument("--sigmas", default="4,8")
    parser.add_argument("--temporal-strengths", default="0,0.5")
    parser.add_argument("--tube-radius", type=int, default=1)
    parser.add_argument("--feather", type=int, default=4)
    parser.add_argument("--score", type=float, default=0.5)
    parser.add_argument("--dilate", type=float, default=0.15)
    parser.add_argument("--motion-tau", type=float, default=0.05)
    parser.add_argument("--mask-backbone", default="fasterrcnn_mobilenet_v3_large_fpn")
    parser.add_argument("--ar-backbone", default="r2plus1d_18")
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--out", default="outputs/probe_action_tubes")
    args = parser.parse_args()

    if not ffmpeg_available():
        raise SystemExit("ffmpeg and ffprobe are required for real-codec evaluation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qps = [int(v) for v in args.qps.split(",")]
    sigmas = [float(v) for v in args.sigmas.split(",")]
    temporal_strengths = [float(v) for v in args.temporal_strengths.split(",")]

    dataset = VideoClipDataset(
        args.index,
        split=args.split,
        num_frames=args.num_frames,
        frame_size=args.size,
        temporal_stride=args.stride,
        train=False,
    )
    if args.n_clips:
        dataset = Subset(dataset, range(min(args.n_clips, len(dataset))))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=collate_clips)
    mask_detector = Detector(device, score_thresh=args.score, backbone=args.mask_backbone)
    analyzer = ActionRecognitionAnalyzer(args.ar_backbone).freeze().to(device)

    arm_cfg = {
        f"tube_s{sigma:g}_t{strength:g}": (sigma, strength)
        for sigma in sigmas for strength in temporal_strengths
    }
    arms = ["anchor", *arm_cfg]
    store = {
        arm: {(codec, qp): {"bpp": 0.0, "correct": 0, "prob": 0.0, "n": 0}
              for codec in ("h264", "h265") for qp in qps}
        for arm in arms
    }
    cover = []

    for clips, labels in tqdm(loader, desc="probe-action-tubes"):
        clips, labels = clips.to(device), labels.to(device)
        frames = clips[0].permute(1, 0, 2, 3)  # [T,3,H,W]
        detections = mask_detector.predict(frames)
        mask = masks_from_frame_detections(
            detections, args.size, score_thresh=args.score, dilate=args.dilate
        )
        cover.append(float(mask.mean()))
        variants = {"anchor": clips}
        for arm, (sigma, strength) in arm_cfg.items():
            variants[arm] = ImportanceTubeSuppress(
                sigma=sigma,
                temporal_radius=args.tube_radius,
                feather=args.feather,
                temporal_strength=strength,
                motion_tau=args.motion_tau,
            )(clips, mask)

        for codec_name in ("h264", "h265"):
            for qp in qps:
                standard = StandardCodec(codec_name, qp, args.preset)
                for arm, video in variants.items():
                    recon, bpps = standard.compress_decompress_items(video)
                    logits = analyzer.predict(recon.to(device))
                    probs = logits.softmax(dim=1)
                    slot = store[arm][(codec_name, qp)]
                    slot["bpp"] += float(bpps[0])
                    slot["correct"] += int(logits.argmax(dim=1).eq(labels).sum())
                    slot["prob"] += float(probs.gather(1, labels[:, None]).sum())
                    slot["n"] += int(labels.numel())

    result = {
        "task": "action_recognition",
        "mask_backbone": args.mask_backbone,
        "ar_backbone": args.ar_backbone,
        "n_clips": len(dataset),
        "mean_mask_cover": float(np.mean(cover)) if cover else None,
        "curves": {},
        "bd_vs_anchor": {},
    }
    for codec_name in ("h264", "h265"):
        codec_curves = {}
        for arm in arms:
            slots = [store[arm][(codec_name, qp)] for qp in qps]
            codec_curves[arm] = {
                "qp": qps,
                "bpp": [s["bpp"] / max(s["n"], 1) for s in slots],
                "top1": [s["correct"] / max(s["n"], 1) for s in slots],
                "target_prob": [s["prob"] / max(s["n"], 1) for s in slots],
            }
        result["curves"][codec_name] = codec_curves
        anchor = codec_curves["anchor"]
        result["bd_vs_anchor"][codec_name] = {}
        for arm in arm_cfg:
            curve = codec_curves[arm]
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
    path = out / "probe_action_tubes.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["bd_vs_anchor"], indent=2))
    print(f"[action-tubes] wrote {path}")


if __name__ == "__main__":
    main()
