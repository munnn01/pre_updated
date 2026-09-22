#!/usr/bin/env python
"""Push the two-arm V13 task-regret mechanism screen to Kaggle."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "kaggle" / "v13_task_regret_screen_cell.sh"
KINETICS = "qktttttttttt/kineticscleaned"
H265_SOURCE = "baooo25r/crc-v5-h265-minus24-candidate-v1"


def kaggle_command() -> list[str]:
    executable = shutil.which("kaggle")
    if executable:
        return [executable]
    return [sys.executable, "-c", "from kaggle.cli import main; main()"]


def resolve_local_commit(ref: str) -> str:
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={REPO.as_posix()}",
            "-C",
            str(REPO),
            "rev-parse",
            "--verify",
            f"{ref}^{{commit}}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    resolved = result.stdout.strip()
    if result.returncode or not re.fullmatch(r"[0-9a-f]{40}", resolved):
        raise ValueError(f"commit does not resolve locally: {ref}")
    return resolved


def render_cell(ref: str, codec: str, seed: int, expected_source_sha: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", ref):
        raise ValueError("ref must be a lowercase 40-character commit SHA")
    if codec not in {"h264", "h265"}:
        raise ValueError("codec must be h264 or h265")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_source_sha):
        raise ValueError("expected_source_sha must be a lowercase SHA-256")
    cell = TEMPLATE.read_text(encoding="utf-8")
    for source, target in {
        "__REF__": ref,
        "__CODEC__": codec,
        "__SEED__": str(int(seed)),
        "__EXPECTED_SOURCE_SHA__": expected_source_sha,
    }.items():
        cell = cell.replace(source, target)
    return cell


def notebook(cell: str, codec: str) -> dict:
    return {
        "cells": [
            {
                "id": f"v13-task-regret-screen-{codec}",
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


def metadata(
    account: str,
    slug: str,
    codec: str,
    source_dataset: str | None,
) -> dict:
    if codec == "h264":
        if not source_dataset:
            raise ValueError("H.264 V13 requires the account-local private source")
        if source_dataset.split("/", 1)[0] != account:
            raise ValueError("H.264 source dataset must be owned by target account")
        sources = [KINETICS, source_dataset]
    else:
        if source_dataset:
            raise ValueError("H.265 V13 uses the frozen public source")
        sources = [KINETICS, H265_SOURCE]
    return {
        "id": f"{account}/{slug}",
        "title": slug,
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": sources,
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
    parser.add_argument("--codec", choices=("h264", "h265"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--source-dataset")
    parser.add_argument("--accelerator", default="NvidiaTeslaT4")
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--write-only", action="store_true")
    args = parser.parse_args()

    commit = resolve_local_commit(args.commit)
    cell = render_cell(
        commit, args.codec, args.seed, args.expected_source_sha
    )
    meta = metadata(args.account, args.slug, args.codec, args.source_dataset)
    push_dir = REPO / "ops" / "_push" / args.account / args.slug
    push_dir.mkdir(parents=True, exist_ok=True)
    (push_dir / "notebook.ipynb").write_text(
        json.dumps(notebook(cell, args.codec)), encoding="utf-8"
    )
    (push_dir / "kernel-metadata.json").write_text(
        json.dumps(meta), encoding="utf-8"
    )
    print(f"[v13-task-regret] generated {push_dir} at {commit}")
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
        f"[v13-task-regret] pushing {handle} codec={args.codec} "
        f"(timeout={'unset' if not args.timeout else args.timeout})"
    )
    raise SystemExit(subprocess.run(command, text=True).returncode)


if __name__ == "__main__":
    main()
