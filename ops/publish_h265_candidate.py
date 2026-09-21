#!/usr/bin/env python
"""Publish the immutable CRC-V5 H.265 -24.26% candidate checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
EXPECTED_SHA = "a3580b32e1554071811888238ea249ce9ae26e7b32392a2d975e4bd6cd0a9c8e"
EXPECTED_H265_BD_RATE = -24.2623162235382
EXPECTED_H265_BD_ACCURACY = 0.0318284840950696
EXPECTED_N_EVAL = 44


def kaggle_command() -> list[str]:
    executable = shutil.which("kaggle")
    if executable:
        return [executable]
    return [sys.executable, "-c", "from kaggle.cli import main; main()"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(checkpoint: Path, results: Path) -> dict:
    sha = file_sha256(checkpoint)
    if sha != EXPECTED_SHA:
        raise ValueError(f"candidate SHA mismatch: {sha} != {EXPECTED_SHA}")

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = state.get("cfg") or {}
    model = cfg.get("model") or {}
    rate = (cfg.get("loss") or {}).get("rate_constraint") or {}
    if model.get("arch") != "additive_cond" or int(model.get("cond_dim", -1)) != 3:
        raise ValueError(f"not the CRC-V5 conditional checkpoint: {model}")
    if int(state.get("global_step", -1)) != 500:
        raise ValueError(f"candidate is not the 500-step checkpoint: {state.get('global_step')}")
    if float(rate.get("target_ratio", math.nan)) != 0.0:
        raise ValueError(f"candidate target_ratio is not 0: {rate}")
    if float(rate.get("dual_lr", math.nan)) != 0.005:
        raise ValueError(f"candidate dual_lr is not 0.005: {rate}")

    report = json.loads(results.read_text(encoding="utf-8"))
    metric = report["bd_prep_gain"]["prep+h265 vs h265"]
    if int(report.get("n_eval", -1)) != EXPECTED_N_EVAL:
        raise ValueError(f"candidate n_eval is not {EXPECTED_N_EVAL}")
    if not math.isclose(
        float(metric["bd_rate_pct"]), EXPECTED_H265_BD_RATE, abs_tol=1e-9
    ):
        raise ValueError(f"candidate H.265 BD-rate mismatch: {metric}")
    if not math.isclose(
        float(metric["bd_accuracy"]), EXPECTED_H265_BD_ACCURACY, abs_tol=1e-9
    ):
        raise ValueError(f"candidate H.265 BD-accuracy mismatch: {metric}")

    return {
        "sha256": sha,
        "source_kernel": "baooo25r/preupd-crc-v5-s23eval-t0-lr5",
        "source_kernel_version": 1,
        "source_output": "crc_v5_stage2_t0_lr5_t0_lr5/checkpoints/preprocessor.pth",
        "global_step": int(state["global_step"]),
        "epoch": int(state["epoch"]),
        "h265_bd_rate_pct": float(metric["bd_rate_pct"]),
        "h265_bd_accuracy": float(metric["bd_accuracy"]),
        "n_eval": int(report["n_eval"]),
        "evaluation_split": "validation shard 0/20",
        "evaluation_salt": "crc-v5-val-v1",
        "qp_list": [30, 35, 40, 45, 50],
        "scientific_status": "screening candidate; not a full-validation claim",
    }


def prepare(
    account: str,
    dataset_slug: str,
    checkpoint: Path,
    results: Path,
) -> tuple[Path, dict]:
    manifest = validate(checkpoint, results)
    target = REPO / "ops" / "_push" / account / f"_{dataset_slug}"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, target / "preprocessor.pth")
    shutil.copy2(results, target / "source_results.json")
    (target / "checkpoint_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    metadata = {
        "id": f"{account}/{dataset_slug}",
        "title": "CRC V5 H265 minus24 candidate",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (target / "dataset-metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return target, manifest


def dataset_exists(handle: str) -> bool:
    result = subprocess.run(
        kaggle_command() + ["datasets", "files", handle],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and "preprocessor.pth" in result.stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--account", default="baooo25r")
    parser.add_argument(
        "--dataset-slug", default="crc-v5-h265-minus24-candidate-v1"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--write-only", action="store_true")
    args = parser.parse_args()

    target, manifest = prepare(
        args.account,
        args.dataset_slug,
        args.checkpoint.resolve(),
        args.results.resolve(),
    )
    handle = f"{args.account}/{args.dataset_slug}"
    print(f"[h265-candidate] prepared {target} sha256={manifest['sha256']}")
    if args.write_only:
        return
    if dataset_exists(handle):
        print(f"[h265-candidate] exists: {handle}")
        return
    command = kaggle_command() + ["datasets", "create", "-p", str(target), "-u", "-q"]
    raise SystemExit(subprocess.run(command, text=True).returncode)


if __name__ == "__main__":
    main()
