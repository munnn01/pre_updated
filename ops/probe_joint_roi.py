#!/usr/bin/env python
"""V3 codec-native ROI feasibility gate and action-recognition screen."""

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

from src.codecs.roi import ROICodec, ROIRect, codec_capabilities  # noqa: E402
from src.data.video_dataset import VideoClipDataset, collate_clips  # noqa: E402
from src.metrics.bd_rate import bd_metric, bd_rate  # noqa: E402
from src.models.task_mask import task_saliency  # noqa: E402
from src.models.task_roi import (  # noqa: E402
    action_importance_score,
    importance_window,
    spatial_roi_regions,
)
from src.tasks.action_recognition import ActionRecognitionAnalyzer  # noqa: E402

ARM_CONFIG = {
    "roi50_bg3": (0.50, 0, 3),
    "roi50_bg6": (0.50, 0, 6),
    "roi65_bg6": (0.65, 0, 6),
    "roi65_bg9": (0.65, 0, 9),
    "roi50m2_bg6": (0.50, -2, 6),
    "roi65m2_bg9": (0.65, -2, 9),
}


def _grid(raw: str, name: str) -> list[int]:
    try:
        values = [int(item) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated integer grid") from exc
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{name} must be a non-empty unique grid")
    return values


def _codec_grid(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("codecs must be a non-empty unique grid")
    if set(values) - {"h264", "h265"}:
        raise ValueError("codecs must be a subset of h264,h265")
    return values


def _finite(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _json_text(value: object) -> str:
    """Pretty JSON that safely converts NumPy scalar diagnostics."""

    def convert(item: object) -> object:
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Object of type {type(item).__name__} is not JSON serializable")

    return json.dumps(value, indent=2, default=convert)


def _synthetic_clip(frames: int = 16, size: int = 128) -> np.ndarray:
    """Deterministic textured clip that exposes spatial QP differences."""
    rng = np.random.default_rng(20260920)
    base = rng.integers(0, 256, (frames, size, size, 3), dtype=np.uint8)
    yy, xx = np.indices((size, size))
    checker = (((xx // 2 + yy // 2) % 2) * 210 + 22).astype(np.uint8)
    for index in range(frames):
        base[index, 32:96, 32:96, 0] = checker[32:96, 32:96]
        base[index, 32:96, 32:96, 1] = np.roll(checker[32:96, 32:96], index, axis=1)
        base[index, 32:96, 32:96, 2] = 255 - checker[32:96, 32:96]
    return base


def _mse(source: np.ndarray, reconstruction: np.ndarray, mask: np.ndarray) -> float:
    error = (source.astype(np.float32) - reconstruction.astype(np.float32)) ** 2
    return float(error[:, mask, :].mean())


def run_f0(args: argparse.Namespace) -> bool:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    capabilities = codec_capabilities()
    codecs = _codec_grid(args.codecs)
    crfs = _grid(args.f0_crfs, "f0 crfs")
    availability = capabilities["addroi"] and all(capabilities[f"libx{codec[-3:]}"] for codec in codecs)
    result: dict = {
        "phase": "F0",
        "protocol": "crf_aq_roi_v3",
        "capabilities": capabilities,
        "codecs": codecs,
        "crfs": crfs,
        "cases": {},
        "checks": {"capabilities": availability},
        "pass": False,
    }
    if not availability:
        (out / "codec_capability.json").write_text(_json_text(result), encoding="utf-8")
        print(_json_text(result))
        return False

    clip = _synthetic_clip(size=args.size)
    height, width = clip.shape[1:3]
    protected = ROIRect(width // 4, height // 4, width // 2, height // 2, 0)
    full_zero = ROIRect(0, 0, width, height, 0)
    full_bg = ROIRect(0, 0, width, height, args.f0_background_delta)
    region_mask = np.zeros((height, width), dtype=bool)
    region_mask[
        protected.y : protected.y + protected.height,
        protected.x : protected.x + protected.width,
    ] = True
    controls = {
        "anchor": [],
        "sham_roi": [protected, full_zero],
        "global_offset": [full_bg],
        "spatial_roi": [protected, full_bg],
    }
    all_decoded = True
    all_changed = True
    all_local = True
    no_rejection = True
    for codec_name in codecs:
        codec_rows: dict = {}
        for crf in crfs:
            encoded = {}
            try:
                for name, regions in controls.items():
                    encoded[name] = ROICodec(
                        codec_name,
                        crf,
                        args.preset,
                        aq_mode=args.aq_mode,
                        aq_strength=args.aq_strength,
                        gop=args.gop,
                    )._encode_decode_clip(clip, regions)
            except RuntimeError as exc:
                no_rejection = False
                codec_rows[str(crf)] = {"error": str(exc)}
                continue
            anchor = encoded["anchor"].reconstruction
            spatial = encoded["spatial_roi"].reconstruction
            global_rec = encoded["global_offset"].reconstruction
            decoded = bool(all(item.reconstruction.shape == clip.shape for item in encoded.values()))
            changed = bool(
                encoded["spatial_roi"].coded_bytes != encoded["anchor"].coded_bytes
                and np.any(spatial != anchor)
                and np.any(spatial != global_rec)
            )
            inside_spatial = _mse(clip, spatial, region_mask)
            inside_global = _mse(clip, global_rec, region_mask)
            outside_spatial = _mse(clip, spatial, ~region_mask)
            outside_anchor = _mse(clip, anchor, ~region_mask)
            local_direction = bool(
                inside_spatial <= inside_global * 1.10 and outside_spatial >= outside_anchor * 0.90
            )
            all_decoded &= decoded
            all_changed &= changed
            all_local &= local_direction
            codec_rows[str(crf)] = {
                name: {
                    "coded_bytes": item.coded_bytes,
                    "bpp": item.bpp,
                    "stderr": item.stderr,
                    "command": item.command,
                    "regions": item.regions,
                }
                for name, item in encoded.items()
            }
            codec_rows[str(crf)]["diagnostics"] = {
                "decoded": decoded,
                "bitstream_and_reconstruction_changed": changed,
                "inside_mse_spatial": inside_spatial,
                "inside_mse_global": inside_global,
                "outside_mse_spatial": outside_spatial,
                "outside_mse_anchor": outside_anchor,
                "local_direction": local_direction,
            }
        result["cases"][codec_name] = codec_rows
    result["checks"].update(
        {
            "decode_without_side_file": all_decoded,
            "roi_not_rejected": no_rejection,
            "roi_changes_bitstream_and_reconstruction": all_changed,
            "block_level_direction": all_local,
        }
    )
    result["pass"] = bool(all(result["checks"].values()))
    (out / "codec_capability.json").write_text(_json_text(result), encoding="utf-8")
    print(_json_text({"F0": result["checks"], "pass": result["pass"]}))
    return bool(result["pass"])


@torch.no_grad()
def _predict_chunks(
    analyzer: ActionRecognitionAnalyzer, videos: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    return torch.cat([analyzer.predict(part) for part in videos.split(chunk_size, dim=0)], dim=0)


def run_d1(args: argparse.Namespace) -> None:
    if not args.index:
        raise ValueError("--index is required for D1")
    codecs = _codec_grid(args.codecs)
    if len(codecs) != 1:
        raise ValueError("each D1 shard must own exactly one codec")
    codec_name = codecs[0]
    crfs = _grid(args.crfs, "crfs")
    if any(not 0 <= value <= 51 for value in crfs):
        raise ValueError("CRF values must be in [0,51]")
    if args.n_clips <= 0 or args.predict_chunk <= 0:
        raise ValueError("n_clips and predict_chunk must be positive")
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
    global_arms = sorted({f"global_bg{config[2]}" for config in ARM_CONFIG.values()})
    arms = ["anchor", "sham_roi", *global_arms, *ARM_CONFIG]
    store = {
        arm: {crf: {"bpp": [], "coded_bytes": [], "correct": [], "target_prob": []} for crf in crfs}
        for arm in arms
    }
    clip_ids: list[str] = []
    roi_records: dict[str, list[list[int]]] = {"budget50": [], "budget65": []}
    first_commands: dict[str, dict[str, list[str]]] = {}
    source_correct: list[float] = []

    def append_metrics(slot: dict, logits: torch.Tensor, label: torch.Tensor) -> None:
        probability = logits.softmax(dim=1).gather(1, label[:, None]).item()
        slot["correct"].append(float(logits.argmax(dim=1).eq(label).item()))
        slot["target_prob"].append(float(probability))

    for clips, labels, metadata in tqdm(loader, desc=f"roi-v3-{codec_name}-{args.eval_backbone}"):
        clips, labels = clips.to(device), labels.to(device)
        clip_ids.append(str(metadata[0]["sequence_id"]))
        with torch.no_grad():
            teacher_logits = teacher.predict(clips)
            pseudo_labels = teacher_logits.argmax(dim=1)
            source_logits = evaluator.predict(clips)
        source_correct.append(float(source_logits.argmax(dim=1).eq(labels).item()))
        saliency = task_saliency(teacher, clips, pseudo_labels, blur=args.saliency_blur)
        score = action_importance_score(saliency, clips, saliency_weight=args.saliency_weight)
        windows = {
            0.50: importance_window(score, 0.50, block=args.block),
            0.65: importance_window(score, 0.65, block=args.block),
        }
        for budget, window in windows.items():
            roi_records[f"budget{int(budget * 100)}"].append(
                [window.x, window.y, window.width, window.height]
            )
        height, width = clips.shape[-2:]
        region_sets: dict[str, list[ROIRect]] = {
            "anchor": [],
            "sham_roi": spatial_roi_regions(windows[0.50], width, height, background_delta_qp=0),
        }
        for name in global_arms:
            delta = int(name.removeprefix("global_bg"))
            region_sets[name] = [ROIRect(0, 0, width, height, delta)]
        for name, (budget, roi_delta, bg_delta) in ARM_CONFIG.items():
            protected = windows[budget]
            protected = ROIRect(protected.x, protected.y, protected.width, protected.height, roi_delta)
            region_sets[name] = spatial_roi_regions(protected, width, height, background_delta_qp=bg_delta)

        clip_np = clips[0].detach().clamp(0, 1).mul(255).round().byte().cpu().numpy().transpose(1, 2, 3, 0)
        for crf in crfs:
            reconstructions = []
            for arm in arms:
                result = ROICodec(
                    codec_name,
                    crf,
                    args.preset,
                    aq_mode=args.aq_mode,
                    aq_strength=args.aq_strength,
                    gop=args.gop,
                )._encode_decode_clip(clip_np, region_sets[arm])
                slot = store[arm][crf]
                slot["bpp"].append(result.bpp)
                slot["coded_bytes"].append(float(result.coded_bytes))
                reconstructions.append(
                    torch.from_numpy(result.reconstruction.transpose(3, 0, 1, 2).copy()).float().div_(255.0)
                )
                if arm not in first_commands:
                    first_commands[arm] = {}
                first_commands[arm].setdefault(str(crf), result.command)
            batch = torch.stack(reconstructions).to(device)
            logits = _predict_chunks(evaluator, batch, args.predict_chunk)
            for arm, row in zip(arms, logits.split(1, dim=0)):
                append_metrics(store[arm][crf], row, labels)

    curves: dict[str, dict] = {}
    flat: dict[str, np.ndarray] = {
        "clip_id": np.asarray(clip_ids, dtype=str),
        "source_correct": np.asarray(source_correct, dtype=np.float32),
    }
    accounting: dict[str, dict] = {}
    for arm in arms:
        slots = [store[arm][crf] for crf in crfs]
        curves[arm] = {
            "bpp": [float(np.mean(slot["bpp"])) for slot in slots],
            "coded_bytes": [float(np.mean(slot["coded_bytes"])) for slot in slots],
            "top1": [float(np.mean(slot["correct"])) for slot in slots],
            "target_prob": [float(np.mean(slot["target_prob"])) for slot in slots],
        }
        accounting[arm] = {
            str(crf): {
                "mean_bpp": curves[arm]["bpp"][index],
                "mean_coded_bytes": curves[arm]["coded_bytes"][index],
            }
            for index, crf in enumerate(crfs)
        }
        for crf in crfs:
            for metric, values in store[arm][crf].items():
                flat[f"{arm}_{codec_name}_crf{crf}_{metric}"] = np.asarray(values, dtype=np.float32)
    for name, records in roi_records.items():
        flat[f"roi_{name}_xywh"] = np.asarray(records, dtype=np.int16)

    anchor = curves["anchor"]
    comparisons: dict[str, dict] = {}
    decisions: dict[str, dict] = {}
    for arm in arms[1:]:
        curve = curves[arm]
        comparison = {
            "bd_rate_top1_pct": _finite(bd_rate(anchor["bpp"], anchor["top1"], curve["bpp"], curve["top1"])),
            "bd_accuracy_top1": _finite(
                bd_metric(anchor["bpp"], anchor["top1"], curve["bpp"], curve["top1"])
            ),
            "bd_rate_target_prob_pct": _finite(
                bd_rate(
                    anchor["bpp"],
                    anchor["target_prob"],
                    curve["bpp"],
                    curve["target_prob"],
                )
            ),
            "min_crf_top1_gap": min(test - base for test, base in zip(curve["top1"], anchor["top1"])),
        }
        comparisons[arm] = comparison
        if arm in ARM_CONFIG:
            global_name = f"global_bg{ARM_CONFIG[arm][2]}"
            global_curve = curves[global_name]
            global_bd = bd_rate(
                anchor["bpp"],
                anchor["top1"],
                global_curve["bpp"],
                global_curve["top1"],
            )
            value = comparison["bd_rate_top1_pct"]
            decisions[arm] = {
                **comparison,
                "matched_global_control": global_name,
                "global_control_bd_rate_top1_pct": _finite(global_bd),
                "beats_global_control": bool(
                    value is not None and np.isfinite(global_bd) and value < global_bd
                ),
                "passes_local_gate": bool(
                    value is not None
                    and value <= -15.0
                    and comparison["min_crf_top1_gap"] >= -0.05
                    and np.isfinite(global_bd)
                    and value < global_bd
                ),
            }

    result = {
        "phase": "D1",
        "task": "action_recognition",
        "method": "codec_native_semantic_roi_v3",
        "rate_control": "CRF+AQ",
        "split": args.split,
        "n_clips": len(dataset),
        "num_frames": args.num_frames,
        "size": args.size,
        "codec": codec_name,
        "crfs": crfs,
        "aq_mode": args.aq_mode,
        "aq_strength": args.aq_strength,
        "preset": args.preset,
        "gop": args.gop,
        "saliency_teacher": args.saliency_teacher,
        "eval_backbone": args.eval_backbone,
        "source_top1": float(np.mean(source_correct)),
        "arms": ARM_CONFIG,
        "controls": ["anchor", "sham_roi", *global_arms],
        "curves": curves,
        "bd_vs_anchor": comparisons,
        "decision": decisions,
        "first_encoder_commands": first_commands,
        "note": "requested delta-QP is not claimed as realised per-block QP under AQ",
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "per_clip_records.npz", **flat)
    (out / "probe_joint_roi.json").write_text(_json_text(result), encoding="utf-8")
    (out / "roi_bit_accounting.json").write_text(_json_text(accounting), encoding="utf-8")
    print(_json_text(result["decision"]))
    print(f"[roi-v3] wrote {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["f0", "d1", "f0-d1"], default="f0")
    parser.add_argument("--index", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--n-clips", type=int, default=200)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--codecs", default="h264,h265")
    parser.add_argument("--crfs", default="24,30,36,42,48")
    parser.add_argument("--f0-crfs", default="30,38,46")
    parser.add_argument("--f0-background-delta", type=int, default=6)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--aq-mode", type=int, default=2)
    parser.add_argument("--aq-strength", type=float, default=1.0)
    parser.add_argument("--gop", type=int, default=32)
    parser.add_argument("--block", type=int, default=16)
    parser.add_argument("--saliency-teacher", default="r3d_18")
    parser.add_argument("--eval-backbone", default="r3d_18")
    parser.add_argument("--saliency-blur", type=int, default=5)
    parser.add_argument("--saliency-weight", type=float, default=0.65)
    parser.add_argument("--predict-chunk", type=int, default=4)
    parser.add_argument("--out", default="outputs/roi_v3")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase in {"f0", "f0-d1"} and not run_f0(args):
        raise SystemExit("F0 codec-native ROI gate failed; D1 was not run")
    if args.phase in {"d1", "f0-d1"}:
        run_d1(args)


if __name__ == "__main__":
    main()
