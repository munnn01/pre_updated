#!/usr/bin/env python
"""Create and push the joint OD + action-recognition Kaggle notebook.

The generated notebook has one ``%%bash`` cell.  It attaches the two canonical
datasets, pins the repository to an immutable Git commit, and runs both real
codec probes.  ``quick`` is the safe default; use ``--profile confirmatory`` for
the preregistered 500-image/200-clip evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "kaggle" / "joint_probe_cell.sh"
DEFAULT_DATASETS = (
    "awsaf49/coco-2017-dataset",
    "qktttttttttt/kineticscleaned",
)


def render_cell(ref: str, profile: str = "quick") -> str:
    """Render the checked-in shell template as one Kaggle notebook cell."""
    if not ref or any(c.isspace() for c in ref):
        raise ValueError("ref must be a non-empty Git ref without whitespace")
    if profile not in {"quick", "confirmatory"}:
        raise ValueError("profile must be 'quick' or 'confirmatory'")
    shell = TEMPLATE.read_text(encoding="utf-8").replace("__REF__", ref)
    shell = shell.replace('PROFILE="${PROFILE:-quick}"', f'PROFILE="{profile}"')
    return "%%bash\n" + shell


def notebook(cell: str) -> dict:
    return {
        "cells": [{
            "id": "joint-od-ar-probe",
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": cell.splitlines(keepends=True),
        }],
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


def metadata(account: str, slug: str, datasets: tuple[str, ...]) -> dict:
    return {
        "id": f"{account}/{slug}",
        "title": slug,
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": list(datasets),
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }


def kaggle_command() -> list[str]:
    """Return a working Kaggle CLI prefix, including environments without an exe."""
    executable = shutil.which("kaggle")
    if executable:
        return [executable]
    # The current kaggle package exposes kaggle.cli but has no __main__.py, so
    # ``python -m kaggle`` fails even though the API package is installed.
    return [sys.executable, "-c", "from kaggle.cli import main; main()"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True, help="immutable Git commit to run")
    parser.add_argument(
        "--account",
        default=os.environ.get("KAGGLE_ACCOUNT", "hieusunday0412"),
    )
    parser.add_argument("--slug", default="pre-updated-joint-od-ar")
    parser.add_argument("--profile", choices=["quick", "confirmatory"], default="quick")
    parser.add_argument("--accelerator", default="NvidiaTeslaT4")
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="comma-separated Kaggle dataset handles",
    )
    parser.add_argument(
        "--write-only",
        action="store_true",
        help="generate notebook + metadata without invoking the Kaggle API",
    )
    args = parser.parse_args()

    datasets = tuple(x.strip() for x in args.datasets.split(",") if x.strip())
    push_dir = REPO / "ops" / "_push" / args.slug
    push_dir.mkdir(parents=True, exist_ok=True)
    (push_dir / "notebook.ipynb").write_text(
        json.dumps(notebook(render_cell(args.commit, args.profile))),
        encoding="utf-8",
    )
    (push_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata(args.account, args.slug, datasets)),
        encoding="utf-8",
    )
    print(f"[joint-push] generated {push_dir}")
    if args.write_only:
        return

    cmd = kaggle_command()
    cmd += ["kernels", "push", "-p", str(push_dir)]
    if args.accelerator:
        cmd += ["--accelerator", args.accelerator]
    print("[joint-push] pushing private GPU notebook")
    result = subprocess.run(cmd, text=True)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
