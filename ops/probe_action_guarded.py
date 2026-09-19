#!/usr/bin/env python
"""Guarded saliency-motion prefilter for action-recognition VCM.

The previous hard saliency screen saved bits but destroyed action evidence before
the codec.  This probe keeps a stable union of action saliency and motion, blends
background simplification gradually, and applies a label-free source guard.  For
each clip the guard chooses the strongest candidate that preserves the frozen
teacher's pseudo-label and a fixed fraction of its source confidence; otherwise
the arm falls back to the unmodified clip.

Ground-truth labels are used only after selection for evaluation.  No mask,
pseudo-label, or guard decision is transmitted to the decoder.
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
    feather_protection,
)
from src.models.task_mask import hard_saliency_mask, task_saliency  # noqa: E402
from src.tasks.action_recognition import ActionRecognitionAnalyzer  # noqa: E402


def _float_grid(raw: str, name: str, *, open_unit: bool = False) -> list[float]:
    try:
        values = [float(value) for value in raw.split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{name} must be a non-empty unique grid")
    if open_unit and any(not 0.0 < value < 1.0 for value in values):
        raise ValueError(f"{name} must contain values in (0,1)")
    return values


def _codec_grid(raw: str) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("codecs must be a non-empty unique grid")
    if set(values) - {"h264", "h265"}:
        raise ValueError("codecs must be a subset of h264,h265")
    return values


def _arm(
    protect: float,
    motion: float,
    sigma: float,
    blend: float,
    retention: float,
    temporal: float,
) -> str:
    return (
        f"guard_p{protect * 100:g}_m{motion * 100:g}_s{sigma:g}"
        f"_a{blend:g}_r{retention:g}_t{temporal:g}"
    )


def _blend_ladder(max_blend: float, steps: int) -> list[float]:
    if not 0.0 < max_blend <= 1.0:
        raise ValueError("max blend must be in (0,1]")
    if steps <= 0:
        raise ValueError("blend steps must be positive")
    return [max_blend * level / steps for level in range(steps, 0, -1)]


def _top_fraction_mask(score: torch.Tensor, fraction: float) -> torch.Tensor:
    """Return an exact-budget binary mask for ``[B,1,1,H,W]`` scores."""
    if score.ndim != 5 or score.shape[1] != 1 or score.shape[2] != 1:
        raise ValueError("score must have shape [B,1,1,H,W]")
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0,1)")
    flat = score.reshape(score.shape[0], -1)
    count = max(1, int(round(flat.shape[1] * fraction)))
    indices = flat.topk(count, dim=1, largest=True, sorted=False).indices
    mask = torch.zeros_like(flat).scatter_(1, indices, 1.0)
    return mask.reshape_as(score).detach()


def motion_tube_mask(video: torch.Tensor, fraction: float) -> torch.Tensor:
    """Protect the most dynamic spatial locations with one stable temporal tube."""
    if video.ndim != 5 or video.shape[1] != 3:
        raise ValueError("video must have shape [B,3,T,H,W]")
    if video.shape[2] < 2:
        score = torch.zeros(
            video.shape[0], 1, 1, video.shape[3], video.shape[4],
            device=video.device, dtype=video.dtype,
        )
    else:
        delta = (video[:, :, 1:] - video[:, :, :-1]).abs().mean(dim=1, keepdim=True)
        score = delta.amax(dim=2, keepdim=True)
    return _top_fraction_mask(score, fraction).expand(
        -1, -1, video.shape[2], -1, -1
    )


def guard_choices(
    source_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    retention: float,
) -> torch.Tensor:
    """Choose the first (strongest) passing candidate, or ``-1`` for identity.

    ``candidate_logits`` is ``[S,B,K]`` and candidates must be ordered from the
    strongest to the weakest blend.  The guard never receives a ground-truth
    label; it protects the source teacher's own decision and confidence.
    """
    if source_logits.ndim != 2 or candidate_logits.ndim != 3:
        raise ValueError("expected source [B,K] and candidates [S,B,K]")
    if candidate_logits.shape[1:] != source_logits.shape:
        raise ValueError("candidate/source logit shapes are incompatible")
    if not 0.0 < retention <= 1.0:
        raise ValueError("retention must be in (0,1]")
    labels = source_logits.argmax(dim=1)
    source_prob = source_logits.softmax(dim=1).gather(1, labels[:, None]).squeeze(1)
    probs = candidate_logits.softmax(dim=2)
    wanted = labels[None, :, None].expand(candidate_logits.shape[0], -1, 1)
    candidate_prob = probs.gather(2, wanted).squeeze(2)
    passing = (
        candidate_logits.argmax(dim=2).eq(labels[None, :])
        & candidate_prob.ge(retention * source_prob[None, :])
    )
    choices = torch.full(
        (source_logits.shape[0],), -1, dtype=torch.long, device=source_logits.device
    )
    for index in range(candidate_logits.shape[0]):
        take = choices.lt(0) & passing[index]
        choices[take] = index
    return choices


def _finite(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


@torch.no_grad()
def _predict_chunks(
    analyzer: ActionRecognitionAnalyzer,
    videos: torch.Tensor,
    chunk_size: int,
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
    parser.add_argument("--protect-fractions", default="0.65,0.8")
    parser.add_argument("--motion-fractions", default="0.5")
    parser.add_argument("--max-blends", default="0.25,0.4")
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--temporal-strength", type=float, default=0.1)
    parser.add_argument("--motion-tau", type=float, default=0.05)
    parser.add_argument("--feather", type=int, default=1)
    parser.add_argument("--guard-retention", type=float, default=0.97)
    parser.add_argument("--blend-steps", type=int, default=4)
    parser.add_argument("--saliency-blur", type=int, default=5)
    parser.add_argument("--saliency-teacher", default="r3d_18")
    parser.add_argument("--eval-backbone", default="r3d_18")
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--predict-chunk", type=int, default=4)
    parser.add_argument("--out", default="outputs/ar_guarded")
    args = parser.parse_args()

    if not ffmpeg_available():
        raise SystemExit("ffmpeg and ffprobe are required for real-codec evaluation")
    if args.n_clips <= 0 or args.predict_chunk <= 0:
        raise ValueError("n_clips and predict_chunk must be positive")
    if args.sigma <= 0 or args.motion_tau <= 0:
        raise ValueError("sigma and motion_tau must be positive")
    if not 0.0 <= args.temporal_strength <= 1.0:
        raise ValueError("temporal_strength must be in [0,1]")
    if args.feather < 0:
        raise ValueError("feather must be non-negative")
    if not 0.0 < args.guard_retention <= 1.0:
        raise ValueError("guard retention must be in (0,1]")
    if args.blend_steps <= 0:
        raise ValueError("blend steps must be positive")

    qps = [int(value) for value in args.qps.split(",") if value.strip()]
    if not qps or len(qps) != len(set(qps)):
        raise ValueError("qps must be a non-empty unique grid")
    codecs = _codec_grid(args.codecs)
    protects = _float_grid(args.protect_fractions, "protect fractions", open_unit=True)
    motions = _float_grid(args.motion_fractions, "motion fractions", open_unit=True)
    blends = _float_grid(args.max_blends, "max blends", open_unit=True)
    ladders = {blend: _blend_ladder(blend, args.blend_steps) for blend in blends}
    arm_cfg = {
        _arm(
            protect, motion, args.sigma, blend,
            args.guard_retention, args.temporal_strength,
        ): (protect, motion, blend)
        for protect in protects for motion in motions for blend in blends
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
        dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_clips
    )
    teacher = ActionRecognitionAnalyzer(args.saliency_teacher).freeze().to(device)
    evaluator = (
        teacher if args.eval_backbone == args.saliency_teacher
        else ActionRecognitionAnalyzer(args.eval_backbone).freeze().to(device)
    )
    transform = ImportanceTubeSuppress(
        sigma=args.sigma,
        temporal_radius=0,
        feather=args.feather,
        temporal_strength=args.temporal_strength,
        motion_tau=args.motion_tau,
    ).to(device)

    store = {
        arm: {
            (codec, qp): {"bpp": [], "correct": [], "target_prob": []}
            for codec in codecs for qp in qps
        }
        for arm in arms
    }
    source_store = {arm: {"correct": [], "target_prob": []} for arm in arms}
    guard_store = {
        arm: {
            "accepted": [], "selected_blend": [],
            "core_fraction": [], "effective_fraction": [],
        }
        for arm in arm_cfg
    }
    clip_ids: list[str] = []

    def append_metrics(slot: dict, logits: torch.Tensor, label: torch.Tensor) -> None:
        probs = logits.softmax(dim=1)
        slot["correct"].append(float(logits.argmax(dim=1).eq(label).item()))
        slot["target_prob"].append(float(probs.gather(1, label[:, None]).item()))

    for clips, labels, metadata in tqdm(loader, desc="probe-action-guarded"):
        clips, labels = clips.to(device), labels.to(device)
        clip_ids.append(str(metadata[0]["sequence_id"]))
        with torch.no_grad():
            teacher_source = teacher.predict(clips)
            pseudo_labels = teacher_source.argmax(dim=1)
        saliency = task_saliency(
            teacher, clips, pseudo_labels, blur=args.saliency_blur
        )

        bases: dict[tuple[float, float], tuple[torch.Tensor, float, float]] = {}
        for protect in protects:
            saliency_core = hard_saliency_mask(saliency, protect, "tube")
            for motion in motions:
                core = torch.maximum(saliency_core, motion_tube_mask(clips, motion))
                effective = feather_protection(core, args.feather)
                bases[(protect, motion)] = (
                    transform(clips, core),
                    float(core.mean().item()),
                    float(effective.mean().item()),
                )

        variants: dict[str, torch.Tensor] = {"anchor": clips}
        for arm, (protect, motion, max_blend) in arm_cfg.items():
            full, core_fraction, effective_fraction = bases[(protect, motion)]
            ladder = ladders[max_blend]
            candidates = torch.cat(
                [clips + alpha * (full - clips) for alpha in ladder], dim=0
            ).clamp(0.0, 1.0)
            candidate_logits = _predict_chunks(
                teacher, candidates, args.predict_chunk
            ).reshape(len(ladder), clips.shape[0], -1)
            choice = guard_choices(
                teacher_source, candidate_logits, args.guard_retention
            )
            selected = clips.clone()
            selected_blend = 0.0
            if int(choice[0].item()) >= 0:
                index = int(choice[0].item())
                selected = candidates[index:index + 1]
                selected_blend = ladder[index]
            variants[arm] = selected
            guard_store[arm]["accepted"].append(float(selected_blend > 0.0))
            guard_store[arm]["selected_blend"].append(float(selected_blend))
            guard_store[arm]["core_fraction"].append(core_fraction)
            guard_store[arm]["effective_fraction"].append(effective_fraction)

        names = list(variants)
        videos = torch.cat([variants[name] for name in names], dim=0)
        source_logits = _predict_chunks(evaluator, videos, args.predict_chunk)
        for row, arm in zip(source_logits.split(1, dim=0), names):
            append_metrics(source_store[arm], row, labels)

        for codec_name in codecs:
            for qp in qps:
                standard = StandardCodec(codec_name, qp, args.preset)
                recon, bpps = standard.compress_decompress_items(videos)
                logits = _predict_chunks(evaluator, recon.to(device), args.predict_chunk)
                for row, bpp, arm in zip(logits.split(1, dim=0), bpps, names):
                    slot = store[arm][(codec_name, qp)]
                    slot["bpp"].append(float(bpp))
                    append_metrics(slot, row, labels)

    result = {
        "task": "action_recognition",
        "method": "guarded_saliency_motion_prefilter",
        "split": args.split,
        "saliency_teacher": args.saliency_teacher,
        "eval_backbone": args.eval_backbone,
        "n_clips": len(dataset),
        "num_frames": args.num_frames,
        "size": args.size,
        "qps": qps,
        "codecs": codecs,
        "protect_fractions": protects,
        "motion_fractions": motions,
        "max_blends": blends,
        "sigma": args.sigma,
        "temporal_strength": args.temporal_strength,
        "guard_retention": args.guard_retention,
        "blend_steps": args.blend_steps,
        "source_metrics": {},
        "guard_stats": {},
        "curves": {},
        "bd_vs_anchor": {},
        "decision": {},
    }
    flat: dict[str, np.ndarray] = {"clip_id": np.asarray(clip_ids, dtype=str)}
    for arm in arms:
        result["source_metrics"][arm] = {
            metric: float(np.mean(values)) for metric, values in source_store[arm].items()
        }
        for metric, values in source_store[arm].items():
            flat[f"source_{arm}_{metric}"] = np.asarray(values, dtype=np.float32)
    for arm, metrics in guard_store.items():
        result["guard_stats"][arm] = {
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
                    bd_rate(anchor["bpp"], anchor["top1"], curve["bpp"], curve["top1"])
                ),
                "bd_accuracy_top1": _finite(
                    bd_metric(anchor["bpp"], anchor["top1"], curve["bpp"], curve["top1"])
                ),
                "bd_rate_target_prob_pct": _finite(
                    bd_rate(
                        anchor["bpp"], anchor["target_prob"],
                        curve["bpp"], curve["target_prob"],
                    )
                ),
            }

    anchor_source = result["source_metrics"]["anchor"]["correct"]
    for arm in arms[1:]:
        source_gap = result["source_metrics"][arm]["correct"] - anchor_source
        codec_decisions = {}
        for codec_name in codecs:
            anchor_top1 = result["curves"][codec_name]["anchor"]["top1"]
            arm_top1 = result["curves"][codec_name][arm]["top1"]
            min_gap = min(a - b for a, b in zip(arm_top1, anchor_top1))
            bd_value = result["bd_vs_anchor"][codec_name][arm]["bd_rate_top1_pct"]
            codec_decisions[codec_name] = {
                "bd_rate_top1_pct": bd_value,
                "min_qp_top1_gap": min_gap,
                "passes_bd_target": bd_value is not None and bd_value <= -15.0,
                "passes_qp_gap": min_gap >= -0.05,
            }
        result["decision"][arm] = {
            "source_top1_gap": source_gap,
            "passes_source_gap": source_gap >= -0.02,
            "codecs": codec_decisions,
            "passes_all_local_gates": (
                source_gap >= -0.02
                and all(
                    item["passes_bd_target"] and item["passes_qp_gap"]
                    for item in codec_decisions.values()
                )
            ),
        }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "per_clip_records.npz", **flat)
    (out / "probe_action_guarded.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result["decision"], indent=2))
    print(f"[action-guarded] wrote {out}")


if __name__ == "__main__":
    main()
