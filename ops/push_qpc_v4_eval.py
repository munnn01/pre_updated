#!/usr/bin/env python
"""Generate and push a real-codec validation job for QPC-V4."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "kaggle" / "qpc_v4_eval_cell.sh"
DATASET = "qktttttttttt/kineticscleaned"
ARMS = {
    "control": "additive",
    "uniform": "additive_cond",
}


def kaggle_command() -> list[str]:
    executable = shutil.which("kaggle")
    if executable:
        return [executable]
    return [sys.executable, "-c", "from kaggle.cli import main; main()"]


def resolve_local_commit(ref: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--verify", f"{ref}^{{commit}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    resolved = result.stdout.strip()
    if result.returncode or not re.fullmatch(r"[0-9a-f]{40}", resolved):
        raise ValueError(f"commit does not resolve locally: {ref}")
    return resolved


def render_cell(ref: str, arm: str, seed: int, shard_idx: int, num_shards: int) -> str:
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {sorted(ARMS)}")
    if seed not in (0, 1, 2):
        raise ValueError("seed must be 0, 1 or 2")
    if num_shards <= 0 or not 0 <= shard_idx < num_shards:
        raise ValueError("shard_idx must be within num_shards")
    values = {
        "__REF__": ref,
        "__ARM__": arm,
        "__ARCH__": ARMS[arm],
        "__SEED__": str(seed),
        "__SHARD_IDX__": str(shard_idx),
        "__NUM_SHARDS__": str(num_shards),
    }
    cell = TEMPLATE.read_text(encoding="utf-8")
    for source, target in values.items():
        cell = cell.replace(source, target)
    return cell


def notebook(cell: str) -> dict:
    return {
        "cells": [
            {
                "id": "qpc-v4-real-codec-validation",
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ("%%bash\n" + cell).splitlines(keepends=True),
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def metadata(account: str, slug: str, train_slug: str) -> dict:
    return {
        "id": f"{account}/{slug}",
        "title": slug,
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [DATASET],
        "kernel_sources": [f"{account}/{train_slug}"],
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
    return result.returncode == 0 and any(status in output for status in ("running", "queued", "pending"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--train-slug", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--shard-idx", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=5)
    parser.add_argument("--accelerator", default="NvidiaTeslaT4")
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--write-only", action="store_true")
    args = parser.parse_args()

    commit = resolve_local_commit(args.commit)
    push_dir = REPO / "ops" / "_push" / args.account / args.slug
    push_dir.mkdir(parents=True, exist_ok=True)
    cell = render_cell(commit, args.arm, args.seed, args.shard_idx, args.num_shards)
    (push_dir / "notebook.ipynb").write_text(json.dumps(notebook(cell)), encoding="utf-8")
    (push_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata(args.account, args.slug, args.train_slug)),
        encoding="utf-8",
    )
    print(f"[qpc-eval-push] generated {push_dir} at {commit}")
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
    print(f"[qpc-eval-push] pushing {handle} (timeout={'unset' if not args.timeout else args.timeout})")
    raise SystemExit(subprocess.run(command, text=True).returncode)


if __name__ == "__main__":
    main()
