#!/usr/bin/env python
"""Publish a verified account-local CRC Stage-1 checkpoint as a private dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]


def kaggle_command() -> list[str]:
    executable = shutil.which("kaggle")
    if executable:
        return [executable]
    return [sys.executable, "-c", "from kaggle.cli import main; main()"]


def checkpoint_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint(path: Path, arm: str, seed: int) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = state.get("cfg") or {}
    model = cfg.get("model") or {}
    if model.get("arch") != "additive_cond" or int(model.get("cond_dim", -1)) != 1:
        raise ValueError(f"not a QPC Stage-1 checkpoint: {model}")
    if int(cfg.get("seed", -1)) != seed:
        raise ValueError(f"checkpoint seed {cfg.get('seed')} != expected {seed}")
    out_dir = str(cfg.get("out_dir", ""))
    if f"crc_v5_{arm}_stage1" not in out_dir:
        raise ValueError(f"checkpoint out_dir does not match arm {arm}: {out_dir}")
    step = int(state.get("global_step", -1))
    epoch = int(state.get("epoch", -1))
    if step <= 0 or epoch <= 0:
        raise ValueError(f"incomplete checkpoint epoch={epoch} step={step}")
    return {
        "sha256": checkpoint_sha(path),
        "arm": arm,
        "seed": seed,
        "epoch": epoch,
        "global_step": step,
        "best_val": float(state.get("best_val", float("nan"))),
        "source_out_dir": out_dir,
    }


def prepare(
    account: str,
    dataset_slug: str,
    checkpoint: Path,
    arm: str,
    seed: int,
    source_kernel: str,
    source_version: int,
) -> tuple[Path, dict]:
    audit = validate_checkpoint(checkpoint, arm, seed)
    audit.update(
        {
            "account": account,
            "source_kernel": source_kernel,
            "source_version": source_version,
        }
    )
    target = REPO / "ops" / "_push" / account / f"_{dataset_slug}"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, target / "preprocessor.pth")
    (target / "checkpoint_manifest.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    metadata = {
        "id": f"{account}/{dataset_slug}",
        "title": f"CRC V5 Stage 1 {arm}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (target / "dataset-metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return target, audit


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
    parser.add_argument("--account", required=True)
    parser.add_argument("--dataset-slug", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, default=260920)
    parser.add_argument("--source-kernel", required=True)
    parser.add_argument("--source-version", type=int, required=True)
    parser.add_argument("--write-only", action="store_true")
    args = parser.parse_args()

    target, audit = prepare(
        args.account,
        args.dataset_slug,
        args.checkpoint.resolve(),
        args.arm,
        args.seed,
        args.source_kernel,
        args.source_version,
    )
    handle = f"{args.account}/{args.dataset_slug}"
    print(f"[crc-stage1] prepared {target} sha256={audit['sha256']}")
    if args.write_only:
        return
    if dataset_exists(handle):
        print(f"[crc-stage1] exists: {handle}")
        return
    command = kaggle_command() + ["datasets", "create", "-p", str(target), "-q"]
    raise SystemExit(subprocess.run(command, text=True).returncode)


if __name__ == "__main__":
    main()

