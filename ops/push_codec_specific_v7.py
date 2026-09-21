#!/usr/bin/env python
"""Generate and push one codec-specific CRC-V7 penalty experiment."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "kaggle" / "codec_specific_v7_cell.sh"
KINETICS = "qktttttttttt/kineticscleaned"
CHECKPOINT_DATASET = "dieulinhh/crc-v5-t0-lr1-stage1-v1"

ARMS = {
    "t0_lr5": {"target": "0.0", "dual_lr": "0.005"},
    "t0_lr15": {"target": "0.0", "dual_lr": "0.015"},
    "tm15_lr5": {"target": "-0.15", "dual_lr": "0.005"},
    "tm15_lr15": {"target": "-0.15", "dual_lr": "0.015"},
}


def kaggle_command() -> list[str]:
    executable = shutil.which("kaggle")
    return [executable] if executable else [sys.executable, "-c", "from kaggle.cli import main; main()"]


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


def render_cell(
    ref: str,
    expected_sha: str,
    codec: str,
    arm: str,
    seed: int,
) -> str:
    if codec not in {"h264", "h265"}:
        raise ValueError("codec must be h264 or h265")
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        raise ValueError("expected_sha must be a lowercase SHA-256")
    values = {
        "__REF__": ref,
        "__EXPECTED_SHA__": expected_sha,
        "__CODEC__": codec,
        "__ARM__": arm,
        "__TARGET__": ARMS[arm]["target"],
        "__DUAL_LR__": ARMS[arm]["dual_lr"],
        "__SEED__": str(seed),
    }
    cell = TEMPLATE.read_text(encoding="utf-8")
    for source, target in values.items():
        cell = cell.replace(source, target)
    return cell


def notebook(cell: str, codec: str, arm: str) -> dict:
    return {
        "cells": [{
            "id": f"codec-specific-v7-{codec}-{arm}",
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ("%%bash\n" + cell).splitlines(keepends=True),
        }],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
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
        "dataset_sources": [KINETICS, CHECKPOINT_DATASET],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--codec", choices=("h264", "h265"), required=True)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--accelerator", default="NvidiaTeslaT4")
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--write-only", action="store_true")
    args = parser.parse_args()

    commit = resolve_local_commit(args.commit)
    push_dir = REPO / "ops" / "_push" / args.account / args.slug
    push_dir.mkdir(parents=True, exist_ok=True)
    cell = render_cell(commit, args.expected_sha, args.codec, args.arm, args.seed)
    (push_dir / "notebook.ipynb").write_text(
        json.dumps(notebook(cell, args.codec, args.arm)), encoding="utf-8"
    )
    (push_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata(args.account, args.slug)), encoding="utf-8"
    )
    print(f"[codec-v7-push] generated {push_dir} at {commit}")
    if args.write_only:
        return
    command = kaggle_command() + ["kernels", "push", "-p", str(push_dir)]
    if args.timeout:
        command += ["--timeout", str(args.timeout)]
    if args.accelerator:
        command += ["--accelerator", args.accelerator]
    print(f"[codec-v7-push] pushing {args.account}/{args.slug}")
    raise SystemExit(subprocess.run(command, text=True).returncode)


if __name__ == "__main__":
    main()
