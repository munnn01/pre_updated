#!/usr/bin/env python
"""Push the commit-pinned RCTS risk experiment to one Kaggle account."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from push_rcts_pilot import account_environment, kaggle_command, metadata, target_is_active

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "kaggle" / "rcts_risk_cell.sh"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--codec", choices=("h264", "h265"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pool", type=Path, default=Path("D:/STUDY/LAB/pool.json"))
    parser.add_argument("--write-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.commit):
        raise ValueError("commit must be a lowercase 40-character SHA")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", args.slug):
        raise ValueError("invalid slug")
    cell = TEMPLATE.read_text(encoding="utf-8")
    for source, value in {"__REF__": args.commit, "__CODEC__": args.codec, "__SEED__": str(args.seed)}.items():
        cell = cell.replace(source, value)
    notebook = {
        "cells": [{"id": f"rcts-risk-{args.codec}", "cell_type": "code",
                   "execution_count": None, "metadata": {}, "outputs": [],
                   "source": ("%%bash\n" + cell).splitlines(keepends=True)}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    push_dir = ROOT / "ops" / "_push" / args.account / args.slug
    push_dir.mkdir(parents=True, exist_ok=True)
    (push_dir / "notebook.ipynb").write_text(json.dumps(notebook), encoding="utf-8")
    (push_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata(args.account, args.slug)), encoding="utf-8")
    print(f"[rcts-risk] generated {args.account}/{args.slug} at {args.commit}", flush=True)
    if args.write_only:
        return
    environment = account_environment(args.pool, args.account)
    handle = f"{args.account}/{args.slug}"
    if target_is_active(handle, environment):
        raise SystemExit(f"refusing duplicate push: {handle} is active")
    result = subprocess.run(kaggle_command() + ["kernels", "push", "-p", str(push_dir)],
                            capture_output=True, text=True, env=environment, check=False)
    response = (result.stdout + result.stderr).strip()
    if response:
        print(response, flush=True)
    if result.returncode or "successfully pushed" not in response.lower():
        raise SystemExit(result.returncode or 1)


if __name__ == "__main__":
    main()
