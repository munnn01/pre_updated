#!/usr/bin/env python
"""Push a paired V7-checkpoint x DCTP residual response screen."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "kaggle" / "codec_dctp_v8_cell.sh"
KINETICS = "qktttttttttt/kineticscleaned"
SETTINGS = {
    "h264": {
        "rate_arm": "t0_lr15",
        "dct": "1.0",
        "temporal": "0.0",
        "residuals": "0.0 0.125 0.25 0.5",
    },
    "h265": {
        "rate_arm": "tm15_lr15",
        "dct": "0.6",
        "temporal": "0.35",
        "residuals": "0.0 0.0625 0.125 0.25",
    },
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


def render_cell(ref: str, codec: str, expected_sha: str) -> str:
    if codec not in SETTINGS:
        raise ValueError("codec must be h264 or h265")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        raise ValueError("expected_sha must be a lowercase SHA-256")
    setting = SETTINGS[codec]
    values = {
        "__REF__": ref,
        "__CODEC__": codec,
        "__RATE_ARM__": setting["rate_arm"],
        "__EXPECTED_SHA__": expected_sha,
        "__DCT__": setting["dct"],
        "__TEMPORAL__": setting["temporal"],
        "__RESIDUALS__": setting["residuals"],
    }
    cell = TEMPLATE.read_text(encoding="utf-8")
    for source, target in values.items():
        cell = cell.replace(source, target)
    return cell


def notebook(cell: str, codec: str) -> dict:
    return {
        "cells": [{
            "id": f"codec-dctp-v8-{codec}",
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


def metadata(account: str, slug: str, source_kernel: str) -> dict:
    return {
        "id": f"{account}/{slug}",
        "title": slug,
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [KINETICS],
        "kernel_sources": [source_kernel],
        "competition_sources": [],
        "model_sources": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--codec", choices=tuple(SETTINGS), required=True)
    parser.add_argument("--source-kernel", required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--accelerator", default="NvidiaTeslaT4")
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--write-only", action="store_true")
    args = parser.parse_args()

    commit = resolve_local_commit(args.commit)
    push_dir = REPO / "ops" / "_push" / args.account / args.slug
    push_dir.mkdir(parents=True, exist_ok=True)
    cell = render_cell(commit, args.codec, args.expected_sha)
    (push_dir / "notebook.ipynb").write_text(
        json.dumps(notebook(cell, args.codec)), encoding="utf-8"
    )
    (push_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata(args.account, args.slug, args.source_kernel)), encoding="utf-8"
    )
    print(f"[codec-dctp-v8-push] generated {push_dir} at {commit}")
    if args.write_only:
        return
    command = kaggle_command() + ["kernels", "push", "-p", str(push_dir)]
    if args.timeout:
        command += ["--timeout", str(args.timeout)]
    if args.accelerator:
        command += ["--accelerator", args.accelerator]
    print(f"[codec-dctp-v8-push] pushing {args.account}/{args.slug}")
    raise SystemExit(subprocess.run(command, text=True).returncode)


if __name__ == "__main__":
    main()
