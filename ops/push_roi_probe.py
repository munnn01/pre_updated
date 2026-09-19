#!/usr/bin/env python
"""Generate and push a private Kaggle ROI-V3 F0 or D1 notebook."""

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
TEMPLATE = REPO / "kaggle" / "roi_v3_cell.sh"
BACKBONES = {"r3d_18", "mc3_18", "r2plus1d_18"}


def kaggle_command() -> list[str]:
    executable = shutil.which("kaggle")
    if executable:
        return [executable]
    return [sys.executable, "-c", "from kaggle.cli import main; main()"]


def resolve_local_commit(ref: str) -> str:
    """Resolve a local ref to a full commit SHA before generating a notebook."""
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


def render_cell(
    ref: str,
    phase: str,
    codecs: str,
    evaluator: str,
    n_clips: int,
    crfs: str,
) -> str:
    if not ref or any(char.isspace() for char in ref):
        raise ValueError("ref must be a non-empty immutable Git ref")
    if phase not in {"f0", "d1"}:
        raise ValueError("phase must be f0 or d1")
    values = [item.strip() for item in codecs.split(",") if item.strip()]
    if not values or set(values) - {"h264", "h265"}:
        raise ValueError("codecs must be a subset of h264,h265")
    if phase == "d1" and len(values) != 1:
        raise ValueError("D1 shards must own exactly one codec")
    if evaluator not in BACKBONES:
        raise ValueError("unsupported evaluator")
    if n_clips <= 0:
        raise ValueError("n_clips must be positive")
    if not re.fullmatch(r"\d+(?:,\d+)+", crfs):
        raise ValueError("crfs must be a comma-separated integer grid")
    replacements = {
        "__REF__": ref,
        "__PHASE__": phase,
        "__CODECS__": ",".join(values),
        "__EVAL_BACKBONE__": evaluator,
        "__N_CLIPS__": str(n_clips),
        "__CRFS__": crfs,
    }
    rendered = TEMPLATE.read_text(encoding="utf-8")
    for source, target in replacements.items():
        rendered = rendered.replace(source, target)
    return rendered


def notebook(cell: str) -> dict:
    return {
        "cells": [
            {
                "id": "roi-v3-probe",
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


def metadata(account: str, slug: str, phase: str) -> dict:
    return {
        "id": f"{account}/{slug}",
        "title": slug,
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": phase == "d1",
        "enable_internet": True,
        "dataset_sources": (["qktttttttttt/kineticscleaned"] if phase == "d1" else []),
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }


def target_is_active(handle: str) -> bool:
    result = subprocess.run(
        kaggle_command() + ["kernels", "status", handle],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    status = result.stdout.lower()
    return result.returncode == 0 and any(marker in status for marker in ("running", "queued", "pending"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--account", default=os.environ.get("KAGGLE_ACCOUNT", "wagur124705"))
    parser.add_argument("--slug", required=True)
    parser.add_argument("--phase", choices=["f0", "d1"], required=True)
    parser.add_argument("--codecs", default="h264,h265")
    parser.add_argument("--eval-backbone", choices=sorted(BACKBONES), default="r3d_18")
    parser.add_argument("--n-clips", type=int, default=200)
    parser.add_argument("--crfs", default="24,30,36,42,48")
    parser.add_argument("--accelerator", default="NvidiaTeslaT4")
    parser.add_argument("--timeout", type=int, default=0, help="0 leaves the local Kaggle CLI timeout unset")
    parser.add_argument("--write-only", action="store_true")
    parser.add_argument("--allow-active-update", action="store_true")
    args = parser.parse_args()
    commit = resolve_local_commit(args.commit)

    push_dir = REPO / "ops" / "_push" / args.account / args.slug
    push_dir.mkdir(parents=True, exist_ok=True)
    cell = render_cell(
        commit,
        args.phase,
        args.codecs,
        args.eval_backbone,
        args.n_clips,
        args.crfs,
    )
    (push_dir / "notebook.ipynb").write_text(json.dumps(notebook(cell)), encoding="utf-8")
    (push_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata(args.account, args.slug, args.phase)), encoding="utf-8"
    )
    print(f"[roi-push] generated {push_dir} at {commit}")
    if args.write_only:
        return
    handle = f"{args.account}/{args.slug}"
    if not args.allow_active_update and target_is_active(handle):
        raise SystemExit(f"refusing duplicate push: {handle} is already active")
    command = kaggle_command() + ["kernels", "push", "-p", str(push_dir)]
    if args.timeout:
        command += ["--timeout", str(args.timeout)]
    if args.phase == "d1" and args.accelerator:
        command += ["--accelerator", args.accelerator]
    print(f"[roi-push] pushing {handle} (timeout={'unset' if not args.timeout else args.timeout})")
    raise SystemExit(subprocess.run(command, text=True).returncode)


if __name__ == "__main__":
    main()
