#!/usr/bin/env python
"""Generate and push an attentive additive (Zhao 2512.15331) Kaggle kernel."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "kaggle" / "additive_attn_ar_cell.sh"
KINETICS = "qktttttttttt/kineticscleaned"
PRE_UPDATED_DATASET = "qktttttttttt/pre-updated-copy"


def kaggle_command() -> list[str]:
    executable = shutil.which("kaggle")
    if executable:
        return [executable]
    return [sys.executable, "-c", "from kaggle.cli import main; main()"]


def resolve_local_commit(ref: str) -> str:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={REPO.as_posix()}", "-C", str(REPO),
         "rev-parse", "--verify", f"{ref}^{{commit}}"],
        capture_output=True, text=True, check=False,
    )
    resolved = result.stdout.strip()
    if result.returncode or not re.fullmatch(r"[0-9a-f]{40}", resolved):
        raise ValueError(f"commit does not resolve locally: {ref}")
    return resolved


def render_cell(ref: str, seed: int) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", ref):
        raise ValueError("ref must be a lowercase 40-character commit SHA")
    cell = TEMPLATE.read_text(encoding="utf-8")
    for source, target in {
        "__REF__": ref, "__SEED__": str(int(seed)),
    }.items():
        cell = cell.replace(source, target)
    return cell


def notebook(cell: str, seed: int) -> dict:
    return {
        "cells": [{
            "id": f"additive-attn-ar-s{seed}", "cell_type": "code",
            "execution_count": None, "metadata": {}, "outputs": [],
            "source": ("%%bash\n" + cell).splitlines(keepends=True),
        }],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


def metadata(account: str, slug: str) -> dict:
    return {
        "id": f"{account}/{slug}", "title": slug,
        "code_file": "notebook.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [KINETICS, PRE_UPDATED_DATASET], "kernel_sources": [],
        "competition_sources": [], "model_sources": [],
    }


def target_is_active(handle: str) -> bool:
    result = subprocess.run(
        kaggle_command() + ["kernels", "status", handle],
        capture_output=True, text=True, check=False,
    )
    output = (result.stdout + result.stderr).lower()
    return result.returncode == 0 and any(
        status in output for status in ("running", "queued", "pending")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--accelerator", default="NvidiaTeslaT4")
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--write-only", action="store_true")
    args = parser.parse_args()

    commit = resolve_local_commit(args.commit)
    cell = render_cell(commit, args.seed)
    push_dir = REPO / "ops" / "_push" / args.account / args.slug
    push_dir.mkdir(parents=True, exist_ok=True)
    (push_dir / "notebook.ipynb").write_text(
        json.dumps(notebook(cell, args.seed)), encoding="utf-8"
    )
    (push_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata(args.account, args.slug)), encoding="utf-8"
    )
    print(f"[additive-attn-ar] generated {push_dir} at {commit}")
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
        f"[additive-attn-ar] pushing {handle} seed={args.seed} "
        f"(timeout={'unset' if not args.timeout else args.timeout})"
    )
    raise SystemExit(subprocess.run(command, text=True).returncode)


if __name__ == "__main__":
    main()
