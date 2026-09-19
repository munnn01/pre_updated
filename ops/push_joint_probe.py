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
OD_BACKBONES = {
    "fasterrcnn_resnet50_fpn",
    "fasterrcnn_mobilenet_v3_large_fpn",
    "retinanet_resnet50_fpn_v2",
    "fcos_resnet50_fpn",
}
AR_BACKBONES = {"r3d_18", "mc3_18", "r2plus1d_18"}


def render_cell(
    ref: str,
    profile: str = "quick",
    *,
    run_ar: bool = True,
    run_od: bool = True,
    n_od: int | None = None,
    n_ar: int | None = None,
    od_size: int | None = None,
    od_sigmas: str | None = None,
    od_roi_sigmas: str | None = None,
    od_post_sigmas: str | None = None,
    od_post_min_qp: int | None = None,
    od_min_margin_px: float | None = None,
    od_mask_grid: int | None = None,
    od_seed: int | None = None,
    od_codecs: str | None = None,
    od_mask_backbone: str | None = None,
    od_eval_backbone: str | None = None,
    ar_probe: str | None = None,
    ar_split: str | None = None,
    ar_backbone: str | None = None,
    ar_post_sigmas: str | None = None,
    ar_post_min_qp: int | None = None,
    ar_motion_quantiles: str | None = None,
    ar_motion_sigma: float | None = None,
    ar_motion_dilation: int | None = None,
    ar_motion_feather: int | None = None,
    ar_saliency_teacher: str | None = None,
    ar_protect_fractions: str | None = None,
    ar_saliency_modes: str | None = None,
    ar_saliency_sigma: float | None = None,
    ar_temporal_strength: float | None = None,
    ar_guard_protect_fractions: str | None = None,
    ar_guard_motion_fractions: str | None = None,
    ar_guard_max_blends: str | None = None,
    ar_guard_sigma: float | None = None,
    ar_guard_retention: float | None = None,
    ar_guard_blend_steps: int | None = None,
    ar_guard_temporal_strength: float | None = None,
    ar_guard_feather: int | None = None,
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
    if n_ar is not None:
        if n_ar <= 0:
            raise ValueError("n_ar must be positive")
        shell = re.sub(r'N_AR="\$\{N_AR:-\d+\}"', f'N_AR="{n_ar}"', shell)
    if od_size is not None:
        if od_size <= 0:
            raise ValueError("od_size must be positive")
        shell = re.sub(r'OD_SIZE="\$\{OD_SIZE:-\d+\}"', f'OD_SIZE="{od_size}"', shell)
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
    if od_codecs is not None:
        codecs = [value.strip() for value in od_codecs.split(",") if value.strip()]
        if not codecs or len(codecs) != len(set(codecs)) or any(
            value not in {"h264", "h265"} for value in codecs
        ):
            raise ValueError("od_codecs must be a unique comma-separated subset of h264,h265")
        shell = re.sub(
            r'OD_CODECS="\$\{OD_CODECS:-[^}]+\}"',
            f'OD_CODECS="{",".join(codecs)}"',
            shell,
        )
    for value, argument, variable in (
        (od_mask_backbone, "od_mask_backbone", "OD_MASK_BACKBONE"),
        (od_eval_backbone, "od_eval_backbone", "OD_EVAL_BACKBONE"),
    ):
        if value is not None:
            if value not in OD_BACKBONES:
                raise ValueError(f"{argument} is not supported")
            shell = re.sub(
                rf'{variable}="\$\{{{variable}:-[^}}]+\}}"',
                f'{variable}="{value}"',
                shell,
            )
    if ar_probe is not None:
        if ar_probe not in {"tubes", "post", "motion", "saliency", "guarded"}:
            raise ValueError(
                "ar_probe must be tubes, post, motion, saliency, or guarded"
            )
        shell = re.sub(
            r'AR_PROBE="\$\{AR_PROBE:-[^}]+\}"',
            f'AR_PROBE="{ar_probe}"',
            shell,
        )
    if ar_split is not None:
        if ar_split not in {"train", "val", "test"}:
            raise ValueError("ar_split must be train, val, or test")
        shell = re.sub(
            r'AR_SPLIT="\$\{AR_SPLIT:-[^}]+\}"',
            f'AR_SPLIT="{ar_split}"',
            shell,
        )
    if ar_backbone is not None:
        if ar_backbone not in AR_BACKBONES:
            raise ValueError("ar_backbone is not supported")
        shell = re.sub(
            r'AR_BACKBONE="\$\{AR_BACKBONE:-[^}]+\}"',
            f'AR_BACKBONE="{ar_backbone}"',
            shell,
        )
    if ar_post_sigmas is not None:
        if not ar_post_sigmas or any(c not in "0123456789.," for c in ar_post_sigmas):
            raise ValueError("ar_post_sigmas must be a comma-separated numeric grid")
        shell = re.sub(
            r'AR_POST_SIGMAS="\$\{AR_POST_SIGMAS:-[^}]+\}"',
            f'AR_POST_SIGMAS="{ar_post_sigmas}"',
            shell,
        )
    if ar_post_min_qp is not None:
        if ar_post_min_qp < 0:
            raise ValueError("ar_post_min_qp must be non-negative")
        shell = re.sub(
            r'AR_POST_MIN_QP="\$\{AR_POST_MIN_QP:-\d+\}"',
            f'AR_POST_MIN_QP="{ar_post_min_qp}"',
            shell,
        )
    if ar_motion_quantiles is not None:
        try:
            quantiles = [
                float(value) for value in ar_motion_quantiles.split(",") if value
            ]
        except ValueError as exc:
            raise ValueError("ar_motion_quantiles must be numeric") from exc
        if (
            not quantiles
            or len(quantiles) != len(set(quantiles))
            or any(not 0.0 < value < 1.0 for value in quantiles)
        ):
            raise ValueError("ar_motion_quantiles must be unique values in (0,1)")
        normalized = ",".join(f"{value:g}" for value in quantiles)
        shell = re.sub(
            r'AR_MOTION_QUANTILES="\$\{AR_MOTION_QUANTILES:-[^}]+\}"',
            f'AR_MOTION_QUANTILES="{normalized}"',
            shell,
        )
    if ar_motion_sigma is not None:
        if ar_motion_sigma <= 0:
            raise ValueError("ar_motion_sigma must be positive")
        shell = re.sub(
            r'AR_MOTION_SIGMA="\$\{AR_MOTION_SIGMA:-[^}]+\}"',
            f'AR_MOTION_SIGMA="{ar_motion_sigma:g}"',
            shell,
        )
    for value, argument, variable in (
        (ar_motion_dilation, "ar_motion_dilation", "AR_MOTION_DILATION"),
        (ar_motion_feather, "ar_motion_feather", "AR_MOTION_FEATHER"),
    ):
        if value is not None:
            if value < 0:
                raise ValueError(f"{argument} must be non-negative")
            shell = re.sub(
                rf'{variable}="\$\{{{variable}:-\d+\}}"',
                f'{variable}="{value}"',
                shell,
            )
    if ar_saliency_teacher is not None:
        if ar_saliency_teacher not in AR_BACKBONES:
            raise ValueError("ar_saliency_teacher is not supported")
        shell = re.sub(
            r'AR_SALIENCY_TEACHER="\$\{AR_SALIENCY_TEACHER:-[^}]+\}"',
            f'AR_SALIENCY_TEACHER="{ar_saliency_teacher}"',
            shell,
        )
    if ar_protect_fractions is not None:
        try:
            fractions = [
                float(value) for value in ar_protect_fractions.split(",") if value
            ]
        except ValueError as exc:
            raise ValueError("ar_protect_fractions must be numeric") from exc
        if (
            not fractions
            or len(fractions) != len(set(fractions))
            or any(not 0.0 < value < 1.0 for value in fractions)
        ):
            raise ValueError("ar_protect_fractions must be unique values in (0,1)")
        shell = re.sub(
            r'AR_PROTECT_FRACTIONS="\$\{AR_PROTECT_FRACTIONS:-[^}]+\}"',
            f'AR_PROTECT_FRACTIONS="{",".join(f"{value:g}" for value in fractions)}"',
            shell,
        )
    if ar_saliency_modes is not None:
        modes = [value for value in ar_saliency_modes.split(",") if value]
        if (
            not modes
            or len(modes) != len(set(modes))
            or set(modes) - {"clip", "tube"}
        ):
            raise ValueError("ar_saliency_modes must be a unique subset of clip,tube")
        shell = re.sub(
            r'AR_SALIENCY_MODES="\$\{AR_SALIENCY_MODES:-[^}]+\}"',
            f'AR_SALIENCY_MODES="{",".join(modes)}"',
            shell,
        )
    if ar_saliency_sigma is not None:
        if ar_saliency_sigma <= 0:
            raise ValueError("ar_saliency_sigma must be positive")
        shell = re.sub(
            r'AR_SALIENCY_SIGMA="\$\{AR_SALIENCY_SIGMA:-[^}]+\}"',
            f'AR_SALIENCY_SIGMA="{ar_saliency_sigma:g}"',
            shell,
        )
    if ar_temporal_strength is not None:
        if not 0.0 <= ar_temporal_strength <= 1.0:
            raise ValueError("ar_temporal_strength must be in [0,1]")
        shell = re.sub(
            r'AR_TEMPORAL_STRENGTH="\$\{AR_TEMPORAL_STRENGTH:-[^}]+\}"',
            f'AR_TEMPORAL_STRENGTH="{ar_temporal_strength:g}"',
            shell,
        )
    for value, argument, variable in (
        (
            ar_guard_protect_fractions,
            "ar_guard_protect_fractions",
            "AR_GUARD_PROTECT_FRACTIONS",
        ),
        (
            ar_guard_motion_fractions,
            "ar_guard_motion_fractions",
            "AR_GUARD_MOTION_FRACTIONS",
        ),
        (ar_guard_max_blends, "ar_guard_max_blends", "AR_GUARD_MAX_BLENDS"),
    ):
        if value is not None:
            try:
                grid = [float(item) for item in value.split(",") if item]
            except ValueError as exc:
                raise ValueError(f"{argument} must be numeric") from exc
            if (
                not grid
                or len(grid) != len(set(grid))
                or any(not 0.0 < item < 1.0 for item in grid)
            ):
                raise ValueError(f"{argument} must be unique values in (0,1)")
            normalized = ",".join(f"{item:g}" for item in grid)
            shell = re.sub(
                rf'{variable}="\$\{{{variable}:-[^}}]+\}}"',
                f'{variable}="{normalized}"',
                shell,
            )
    if ar_guard_sigma is not None:
        if ar_guard_sigma <= 0:
            raise ValueError("ar_guard_sigma must be positive")
        shell = re.sub(
            r'AR_GUARD_SIGMA="\$\{AR_GUARD_SIGMA:-[^}]+\}"',
            f'AR_GUARD_SIGMA="{ar_guard_sigma:g}"',
            shell,
        )
    if ar_guard_retention is not None:
        if not 0.0 < ar_guard_retention <= 1.0:
            raise ValueError("ar_guard_retention must be in (0,1]")
        shell = re.sub(
            r'AR_GUARD_RETENTION="\$\{AR_GUARD_RETENTION:-[^}]+\}"',
            f'AR_GUARD_RETENTION="{ar_guard_retention:g}"',
            shell,
        )
    if ar_guard_blend_steps is not None:
        if ar_guard_blend_steps <= 0:
            raise ValueError("ar_guard_blend_steps must be positive")
        shell = re.sub(
            r'AR_GUARD_BLEND_STEPS="\$\{AR_GUARD_BLEND_STEPS:-\d+\}"',
            f'AR_GUARD_BLEND_STEPS="{ar_guard_blend_steps}"',
            shell,
        )
    if ar_guard_temporal_strength is not None:
        if not 0.0 <= ar_guard_temporal_strength <= 1.0:
            raise ValueError("ar_guard_temporal_strength must be in [0,1]")
        shell = re.sub(
            r'AR_GUARD_TEMPORAL_STRENGTH="\$\{AR_GUARD_TEMPORAL_STRENGTH:-[^}]+\}"',
            f'AR_GUARD_TEMPORAL_STRENGTH="{ar_guard_temporal_strength:g}"',
            shell,
        )
    if ar_guard_feather is not None:
        if ar_guard_feather < 0:
            raise ValueError("ar_guard_feather must be non-negative")
        shell = re.sub(
            r'AR_GUARD_FEATHER="\$\{AR_GUARD_FEATHER:-\d+\}"',
            f'AR_GUARD_FEATHER="{ar_guard_feather}"',
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
    parser.add_argument("--n-ar", type=int, default=None,
                        help="override the profile's Kinetics clip count")
    parser.add_argument("--od-size", type=int, default=None)
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
    parser.add_argument("--od-codecs", default=None,
                        help="comma-separated OD codecs: h264,h265")
    parser.add_argument("--od-mask-backbone", choices=sorted(OD_BACKBONES), default=None)
    parser.add_argument("--od-eval-backbone", choices=sorted(OD_BACKBONES), default=None)
    parser.add_argument(
        "--ar-probe",
        choices=["tubes", "post", "motion", "saliency", "guarded"],
        default=None,
    )
    parser.add_argument("--ar-split", choices=["train", "val", "test"], default=None)
    parser.add_argument("--ar-backbone", choices=sorted(AR_BACKBONES), default=None)
    parser.add_argument("--ar-post-sigmas", default=None)
    parser.add_argument("--ar-post-min-qp", type=int, default=None)
    parser.add_argument("--ar-motion-quantiles", default=None)
    parser.add_argument("--ar-motion-sigma", type=float, default=None)
    parser.add_argument("--ar-motion-dilation", type=int, default=None)
    parser.add_argument("--ar-motion-feather", type=int, default=None)
    parser.add_argument("--ar-saliency-teacher", choices=sorted(AR_BACKBONES), default=None)
    parser.add_argument("--ar-protect-fractions", default=None)
    parser.add_argument("--ar-saliency-modes", default=None)
    parser.add_argument("--ar-saliency-sigma", type=float, default=None)
    parser.add_argument("--ar-temporal-strength", type=float, default=None)
    parser.add_argument("--ar-guard-protect-fractions", default=None)
    parser.add_argument("--ar-guard-motion-fractions", default=None)
    parser.add_argument("--ar-guard-max-blends", default=None)
    parser.add_argument("--ar-guard-sigma", type=float, default=None)
    parser.add_argument("--ar-guard-retention", type=float, default=None)
    parser.add_argument("--ar-guard-blend-steps", type=int, default=None)
    parser.add_argument("--ar-guard-temporal-strength", type=float, default=None)
    parser.add_argument("--ar-guard-feather", type=int, default=None)
    parser.add_argument("--qps", default=None,
                        help="override the profile's comma-separated QP grid")
    parser.add_argument("--bootstrap", type=int, default=None,
                        help="override paired image-bootstrap draws")
    parser.add_argument("--accelerator", default="NvidiaTeslaT4")
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="optional Kaggle run limit in seconds; 0 leaves it unset",
    )
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
            n_ar=args.n_ar,
            od_size=args.od_size,
            od_sigmas=args.od_sigmas,
            od_roi_sigmas=args.od_roi_sigmas,
            od_post_sigmas=args.od_post_sigmas,
            od_post_min_qp=args.od_post_min_qp,
            od_min_margin_px=args.od_min_margin_px,
            od_mask_grid=args.od_mask_grid,
            od_seed=args.od_seed,
            od_codecs=args.od_codecs,
            od_mask_backbone=args.od_mask_backbone,
            od_eval_backbone=args.od_eval_backbone,
            ar_probe=args.ar_probe,
            ar_split=args.ar_split,
            ar_backbone=args.ar_backbone,
            ar_post_sigmas=args.ar_post_sigmas,
            ar_post_min_qp=args.ar_post_min_qp,
            ar_motion_quantiles=args.ar_motion_quantiles,
            ar_motion_sigma=args.ar_motion_sigma,
            ar_motion_dilation=args.ar_motion_dilation,
            ar_motion_feather=args.ar_motion_feather,
            ar_saliency_teacher=args.ar_saliency_teacher,
            ar_protect_fractions=args.ar_protect_fractions,
            ar_saliency_modes=args.ar_saliency_modes,
            ar_saliency_sigma=args.ar_saliency_sigma,
            ar_temporal_strength=args.ar_temporal_strength,
            ar_guard_protect_fractions=args.ar_guard_protect_fractions,
            ar_guard_motion_fractions=args.ar_guard_motion_fractions,
            ar_guard_max_blends=args.ar_guard_max_blends,
            ar_guard_sigma=args.ar_guard_sigma,
            ar_guard_retention=args.ar_guard_retention,
            ar_guard_blend_steps=args.ar_guard_blend_steps,
            ar_guard_temporal_strength=args.ar_guard_temporal_strength,
            ar_guard_feather=args.ar_guard_feather,
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
