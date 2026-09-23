#!/usr/bin/env python
"""Create and optionally push one commit-pinned RCTS Kaggle notebook."""

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
TEMPLATE = REPO / "kaggle" / "rcts_pilot_cell.sh"
KINETICS = "qktttttttttt/kineticscleaned"


def kaggle_command() -> list[str]:
    executable = shutil.which("kaggle")
    if executable:
        return [executable]
    return [sys.executable, "-c", "from kaggle.cli import main; main()"]


def render_cell(commit: str, codec: str, seed: int) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("commit must be a lowercase 40-character SHA")
    if codec not in {"h264", "h265"}:
        raise ValueError("codec must be h264 or h265")
    cell = TEMPLATE.read_text(encoding="utf-8")
    for source, target in {
        "__REF__": commit, "__CODEC__": codec, "__SEED__": str(int(seed)),
    }.items():
        cell = cell.replace(source, target)
    return cell


def notebook(cell: str, codec: str) -> dict:
    return {
        "cells": [{
            "id": f"rcts-pilot-{codec}", "cell_type": "code",
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
        "dataset_sources": [KINETICS], "kernel_sources": [],
        "competition_sources": [], "model_sources": [],
    }


def account_environment(pool: Path, account: str) -> dict[str, str]:
    credentials = json.loads(pool.read_text(encoding="utf-8-sig"))
    token = credentials.get(account)
    if not isinstance(token, str) or not token.startswith("KGAT_"):
        raise ValueError(f"no KGAT token for {account} in {pool}")
    environment = os.environ.copy()
    environment["KAGGLE_API_TOKEN"] = token
    return environment


def target_is_active(handle: str, environment: dict[str, str]) -> bool:
    result = subprocess.run(
        kaggle_command() + ["kernels", "status", handle],
        capture_output=True, text=True, check=False, env=environment,
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
    parser.add_argument("--codec", choices=("h264", "h265"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pool", type=Path, default=Path("D:/STUDY/LAB/pool.json"))
    parser.add_argument("--write-only", action="store_true")
    args = parser.parse_args()

    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", args.slug):
        raise ValueError("slug must contain only lowercase letters, numbers and hyphens")
    cell = render_cell(args.commit, args.codec, args.seed)
    push_dir = REPO / "ops" / "_push" / args.account / args.slug
    push_dir.mkdir(parents=True, exist_ok=True)
    (push_dir / "notebook.ipynb").write_text(
        json.dumps(notebook(cell, args.codec)), encoding="utf-8"
    )
    (push_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata(args.account, args.slug)), encoding="utf-8"
    )
    print(f"[rcts] generated {args.account}/{args.slug} at {args.commit}", flush=True)
    if args.write_only:
        return

    environment = account_environment(args.pool, args.account)
    handle = f"{args.account}/{args.slug}"
    if target_is_active(handle, environment):
        raise SystemExit(f"refusing duplicate push: {handle} is active")
    command = kaggle_command() + ["kernels", "push", "-p", str(push_dir)]
    result = subprocess.run(command, text=True, env=environment, check=False)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
