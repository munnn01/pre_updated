#!/usr/bin/env python
"""Generate and push a frozen-H265 / independent-H264 V10 Kaggle job."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "kaggle" / "dual_checkpoint_v10_cell.sh"
KINETICS = "qktttttttttt/kineticscleaned"
H265_DATASET = "baooo25r/crc-v5-h265-minus24-candidate-v1"
ARMS = {"t0_lr5", "t0_lr15", "tm5_lr5", "tm5_lr15", "tm25_lr10"}


def kaggle_command() -> list[str]:
    executable = shutil.which("kaggle")
    if executable:
        return [executable]
    return [sys.executable, "-c", "from kaggle.cli import main; main()"]


def resolve_local_commit(ref: str) -> str:
    result = subprocess.run(
        [
            "git", "-c", f"safe.directory={REPO.as_posix()}",
            "-C", str(REPO), "rev-parse", "--verify", f"{ref}^{{commit}}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    resolved = result.stdout.strip()
    if result.returncode or not re.fullmatch(r"[0-9a-f]{40}", resolved):
        raise ValueError(f"commit does not resolve locally: {ref}")
    return resolved


def validate_arm_order(value: str) -> str:
    ordered = value.split(",")
    if len(ordered) != len(ARMS) or set(ordered) != ARMS:
        raise ValueError(f"arm order must contain each of {sorted(ARMS)} exactly once")
    return value


def render_cell(ref: str, seed: int, arm_order: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", ref):
        raise ValueError("ref must be a lowercase 40-character commit SHA")
    validate_arm_order(arm_order)
    values = {
        "__REF__": ref,
        "__SEED__": str(seed),
        "__ARM_ORDER__": arm_order,
    }
    cell = TEMPLATE.read_text(encoding="utf-8")
    for source, target in values.items():
        cell = cell.replace(source, target)
    return cell


def notebook(cell: str) -> dict:
    return {
        "cells": [{
            "id": "dual-checkpoint-v10",
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ("%%bash\n" + cell).splitlines(keepends=True),
        }],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3", "language": "python", "name": "python3"
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def metadata(account: str, slug: str) -> dict:
    return {
        "id": f"{account}/{slug}",
        "title": slug,
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [KINETICS, H265_DATASET],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }


def target_is_active(handle: str) -> bool:
    result = subprocess.run(
        kaggle_command() + ["kernels", "status", handle],
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout.lower()
    return result.returncode == 0 and any(
        status in output for status in ("running", "queued", "pending")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arm-order", required=True)
    parser.add_argument("--accelerator", default="NvidiaTeslaT4")
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--write-only", action="store_true")
    args = parser.parse_args()

    commit = resolve_local_commit(args.commit)
    order = validate_arm_order(args.arm_order)
    push_dir = REPO / "ops" / "_push" / args.account / args.slug
    push_dir.mkdir(parents=True, exist_ok=True)
    cell = render_cell(commit, args.seed, order)
    (push_dir / "notebook.ipynb").write_text(
        json.dumps(notebook(cell)), encoding="utf-8"
    )
    (push_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata(args.account, args.slug)), encoding="utf-8"
    )
    print(f"[dual-checkpoint-v10] generated {push_dir} at {commit}")
    if args.write_only:
        return

    handle = f"{args.account}/{args.slug}"
    if target_is_active(handle):
        raise SystemExit(f"refusing duplicate push: {handle} is already active")
    command = kaggle_command() + ["kernels", "push", "-p", str(push_dir)]
    if args.timeout:
        command += ["--timeout", str(args.timeout)]
    if args.accelerator:
        command += ["--accelerator", args.accelerator]
    print(
        f"[dual-checkpoint-v10] pushing {handle} "
        f"(timeout={'unset' if not args.timeout else args.timeout})"
    )
    raise SystemExit(subprocess.run(command, text=True).returncode)


if __name__ == "__main__":
    main()
