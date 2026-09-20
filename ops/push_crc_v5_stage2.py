#!/usr/bin/env python
"""Push CRC-V5 Stage-2/Stage-3 paired jobs from a private Stage-1 dataset."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "kaggle" / "crc_v5_stage2_cell.sh"
KINETICS = "qktttttttttt/kineticscleaned"
ARMS = {
    "t0_lr1": {"enabled": "true", "target": "0.0", "dual_lr": "0.001", "beta": "0.0"},
    "tm5_lr1": {"enabled": "true", "target": "-0.05", "dual_lr": "0.001", "beta": "0.0"},
    "t0_lr5": {"enabled": "true", "target": "0.0", "dual_lr": "0.005", "beta": "0.0"},
    "tm5_lr5": {"enabled": "true", "target": "-0.05", "dual_lr": "0.005", "beta": "0.0"},
}


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


def render_cell(ref: str, arm: str, expected_sha: str, seed: int = 260920) -> str:
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {sorted(ARMS)}")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        raise ValueError("expected_sha must be a lowercase SHA-256")
    values = {
        "__REF__": ref,
        "__ARM__": arm,
        "__SEED__": str(seed),
        "__EXPECTED_SHA__": expected_sha,
        "__ENABLED__": ARMS[arm]["enabled"],
        "__TARGET__": ARMS[arm]["target"],
        "__DUAL_LR__": ARMS[arm]["dual_lr"],
        "__BETA__": ARMS[arm]["beta"],
    }
    cell = TEMPLATE.read_text(encoding="utf-8")
    for source, target in values.items():
        cell = cell.replace(source, target)
    return cell


def notebook(cell: str) -> dict:
    return {
        "cells": [{
            "id": "crc-v5-stage2-stage3-eval",
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


def metadata(account: str, slug: str, dataset_slug: str) -> dict:
    return {
        "id": f"{account}/{slug}",
        "title": slug,
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [KINETICS, f"{account}/{dataset_slug}"],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--dataset-slug", required=True)
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--seed", type=int, default=260920)
    parser.add_argument("--accelerator", default="NvidiaTeslaT4")
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--write-only", action="store_true")
    args = parser.parse_args()

    commit = resolve_local_commit(args.commit)
    push_dir = REPO / "ops" / "_push" / args.account / args.slug
    push_dir.mkdir(parents=True, exist_ok=True)
    (push_dir / "notebook.ipynb").write_text(
        json.dumps(notebook(render_cell(commit, args.arm, args.expected_sha))),
        encoding="utf-8",
    )
    (push_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata(args.account, args.slug, args.dataset_slug)),
        encoding="utf-8",
    )
    print(f"[crc-stage2-3-push] generated {push_dir} at {commit}")
    if args.write_only:
        return
    command = kaggle_command() + ["kernels", "push", "-p", str(push_dir)]
    if args.timeout:
        command += ["--timeout", str(args.timeout)]
    if args.accelerator:
        command += ["--accelerator", args.accelerator]
    print(f"[crc-stage2-3-push] pushing {args.account}/{args.slug}")
    raise SystemExit(subprocess.run(command, text=True).returncode)


if __name__ == "__main__":
    main()
