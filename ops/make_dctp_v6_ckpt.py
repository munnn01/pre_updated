#!/usr/bin/env python
"""Convert a QPC Stage-1 checkpoint into a fixed DCTP-V6 evaluation arm."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.engine import _load_state_compat  # noqa: E402
from src.models.dct_projector import DCTProjectedAdditivePreprocessor  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(
    source_path: Path,
    out_path: Path,
    expected_sha: str,
    *,
    residual_scale: float,
    dct_strength: float,
    dct_threshold: float,
    temporal_strength: float,
    seed: int,
    dct_band_start: int = 4,
    qp_slope: float = 0.65,
    semantic_protect_area: float = 0.0,
) -> dict:
    actual_sha = file_sha256(source_path)
    if actual_sha != expected_sha:
        raise ValueError(f"source SHA mismatch expected={expected_sha} actual={actual_sha}")
    torch.manual_seed(seed)
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    source_state = source.get("model", source)
    model = DCTProjectedAdditivePreprocessor(
        cond_dim=3,
        residual_scale=residual_scale,
        dct_strength=dct_strength,
        dct_threshold=dct_threshold,
        dct_band_start=dct_band_start,
        temporal_strength=temporal_strength,
        qp_slope=qp_slope,
        semantic_protect_area=semantic_protect_area,
    )
    missing = _load_state_compat(model, source_state)
    if missing:
        raise ValueError(f"canonical checkpoint conversion left missing keys: {missing}")

    cfg = copy.deepcopy(source.get("cfg") or {})
    cfg["seed"] = seed
    cfg["model"] = dict(cfg.get("model") or {})
    cfg["model"].update(
        {
            "arch": "additive_dct",
            "temporal_frames": 8,
            "strength": 1.0,
            "cond_dim": 3,
            "residual_scale": float(residual_scale),
            "dct_strength": float(dct_strength),
            "dct_threshold": float(dct_threshold),
            "dct_softness": 0.25,
            "dct_block": 8,
            "dct_band_start": int(dct_band_start),
            "temporal_strength": float(temporal_strength),
            "motion_tau": 0.05,
            "qp_slope": float(qp_slope),
            "h264_scale": 1.0,
            "h265_scale": 1.0,
            "semantic_protect_area": float(semantic_protect_area),
            "gate": False,
            "qp_ref": [20, 51],
        }
    )
    checkpoint = {
        "model": model.state_dict(),
        "opt": None,
        "sched": None,
        "cfg": cfg,
        "epoch": 0,
        "global_step": 0,
        "best_val": None,
        "no_improve": 0,
        "lineage": {
            "source_sha256": actual_sha,
            "source_epoch": source.get("epoch"),
            "source_global_step": source.get("global_step"),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out_path)
    return {
        "source_sha256": actual_sha,
        "output": str(out_path),
        "residual_scale": residual_scale,
        "dct_strength": dct_strength,
        "dct_threshold": dct_threshold,
        "temporal_strength": temporal_strength,
        "dct_band_start": dct_band_start,
        "qp_slope": qp_slope,
        "semantic_protect_area": semantic_protect_area,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--residual-scale", type=float, required=True)
    parser.add_argument("--dct-strength", type=float, required=True)
    parser.add_argument("--dct-threshold", type=float, default=1.0)
    parser.add_argument("--temporal-strength", type=float, required=True)
    parser.add_argument("--dct-band-start", type=int, default=4)
    parser.add_argument("--qp-slope", type=float, default=0.65)
    parser.add_argument("--semantic-protect-area", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=220921)
    args = parser.parse_args()
    result = convert(
        args.source.resolve(),
        args.out.resolve(),
        args.expected_sha,
        residual_scale=args.residual_scale,
        dct_strength=args.dct_strength,
        dct_threshold=args.dct_threshold,
        temporal_strength=args.temporal_strength,
        seed=args.seed,
        dct_band_start=args.dct_band_start,
        qp_slope=args.qp_slope,
        semantic_protect_area=args.semantic_protect_area,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
