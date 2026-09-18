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
import re
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


def render_cell(
    ref: str,
    profile: str = "quick",
    *,
    run_ar: bool = True,
    run_od: bool = True,
    n_od: int | None = None,
    od_sigmas: str | None = None,
    od_roi_sigmas: str | None = None,
    od_post_sigmas: str | None = None,
    od_post_min_qp: int | None = None,
    od_min_margin_px: float | None = None,
    od_mask_grid: int | None = None,
    od_seed: int | None = None,
    qps: str | None = None,
    bootstrap: int | None = None,
) -> str:
    """Render the checked-in shell template as one Kaggle notebook cell."""
    if not ref or any(c.isspace() for c in ref):
        raise ValueError("ref must be a non-empty Git ref without whitespace")
    if profile not in {"quick", "confirmatory"}:
        raise ValueError("profile must be 'quick' or 'confirmatory'")
    shell = TEMPLATE.read_text(encoding="utf-8").replace("__REF__", ref)
    shell = shell.replace('PROFILE="${PROFILE:-quick}"', f'PROFILE="{profile}"')
    shell = shell.replace('RUN_AR="${RUN_AR:-1}"', f'RUN_AR="{int(run_ar)}"')
    shell = shell.replace('RUN_OD="${RUN_OD:-1}"', f'RUN_OD="{int(run_od)}"')
    if n_od is not None:
        if n_od <= 0:
            raise ValueError("n_od must be positive")
        shell = re.sub(r'N_OD="\$\{N_OD:-\d+\}"', f'N_OD="{n_od}"', shell)
    if od_sigmas is not None:
        if not od_sigmas or any(c not in "0123456789.," for c in od_sigmas):
            raise ValueError("od_sigmas must be a comma-separated numeric grid")
        shell = re.sub(
            r'OD_SIGMAS="\$\{OD_SIGMAS:-[^}]+\}"',
            f'OD_SIGMAS="{od_sigmas}"',
            shell,
        )
    for value, argument, variable in (
        (od_roi_sigmas, "od_roi_sigmas", "OD_ROI_SIGMAS"),
        (od_post_sigmas, "od_post_sigmas", "OD_POST_SIGMAS"),
    ):
        if value is not None:
            if not value or any(c not in "0123456789.," for c in value):
                raise ValueError(f"{argument} must be a comma-separated numeric grid")
            shell = re.sub(
                rf'{variable}="\$\{{{variable}:-[^}}]+\}}"',
                f'{variable}="{value}"',
                shell,
            )
    if od_post_min_qp is not None:
        if od_post_min_qp < 0:
            raise ValueError("od_post_min_qp must be non-negative")
        shell = re.sub(
            r'OD_POST_MIN_QP="\$\{OD_POST_MIN_QP:-\d+\}"',
            f'OD_POST_MIN_QP="{od_post_min_qp}"',
            shell,
        )
    if od_min_margin_px is not None:
        if od_min_margin_px < 0:
            raise ValueError("od_min_margin_px must be non-negative")
        shell = re.sub(
            r'OD_MIN_MARGIN_PX="\$\{OD_MIN_MARGIN_PX:-[^}]+\}"',
            f'OD_MIN_MARGIN_PX="{od_min_margin_px:g}"',
            shell,
        )
    if od_mask_grid is not None:
        if od_mask_grid <= 0:
            raise ValueError("od_mask_grid must be positive")
        shell = re.sub(
            r'OD_MASK_GRID="\$\{OD_MASK_GRID:-\d+\}"',
            f'OD_MASK_GRID="{od_mask_grid}"',
            shell,
        )
    if od_seed is not None:
        if od_seed < 0:
            raise ValueError("od_seed must be non-negative")
        shell = re.sub(
            r'OD_SEED="\$\{OD_SEED:-\d+\}"',
            f'OD_SEED="{od_seed}"',
            shell,
        )
    if qps is not None:
        if not qps or any(c not in "0123456789," for c in qps):
            raise ValueError("qps must be a comma-separated integer grid")
        shell = re.sub(r'QPS="\$\{QPS:-[^}]+\}"', f'QPS="{qps}"', shell)
    if bootstrap is not None:
        if bootstrap < 0:
            raise ValueError("bootstrap must be non-negative")
        shell = re.sub(
            r'BOOTSTRAP="\$\{BOOTSTRAP:-\d+\}"',
            f'BOOTSTRAP="{bootstrap}"',
            shell,
        )
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
    parser.add_argument("--skip-ar", action="store_true")
    parser.add_argument("--skip-od", action="store_true")
    parser.add_argument("--n-od", type=int, default=None,
                        help="override the profile's COCO image count")
    parser.add_argument("--od-sigmas", default=None,
                        help="override the profile's comma-separated OD sigma grid")
    parser.add_argument("--od-roi-sigmas", default=None,
                        help="comma-separated Gaussian sigmas inside OD ROI")
    parser.add_argument("--od-post-sigmas", default=None,
                        help="comma-separated zero-bit post-decode sigmas")
    parser.add_argument("--od-post-min-qp", type=int, default=None)
    parser.add_argument("--od-min-margin-px", type=float, default=None)
    parser.add_argument("--od-mask-grid", type=int, default=None)
    parser.add_argument("--od-seed", type=int, default=None,
                        help="deterministic COCO shuffle seed")
    parser.add_argument("--qps", default=None,
                        help="override the profile's comma-separated QP grid")
    parser.add_argument("--bootstrap", type=int, default=None,
                        help="override paired image-bootstrap draws")
    parser.add_argument("--accelerator", default="NvidiaTeslaT4")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Kaggle run limit in seconds (quick default: one hour)")
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
        json.dumps(notebook(render_cell(
            args.commit,
            args.profile,
            run_ar=not args.skip_ar,
            run_od=not args.skip_od,
            n_od=args.n_od,
            od_sigmas=args.od_sigmas,
            od_roi_sigmas=args.od_roi_sigmas,
            od_post_sigmas=args.od_post_sigmas,
            od_post_min_qp=args.od_post_min_qp,
            od_min_margin_px=args.od_min_margin_px,
            od_mask_grid=args.od_mask_grid,
            od_seed=args.od_seed,
            qps=args.qps,
            bootstrap=args.bootstrap,
        ))),
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
    if args.timeout:
        cmd += ["--timeout", str(args.timeout)]
    if args.accelerator:
        cmd += ["--accelerator", args.accelerator]
    print("[joint-push] pushing private GPU notebook")
    result = subprocess.run(cmd, text=True)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
