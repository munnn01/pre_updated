#!/usr/bin/env python
"""Publish an audited H.264 V10 resume pair as a private account dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def kaggle_command() -> list[str]:
    executable = shutil.which("kaggle")
    if executable:
        return [executable]
    return [sys.executable, "-c", "from kaggle.cli import main; main()"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_metadata(account: str, slug: str, seed: int) -> dict:
    return {
        "title": f"CRC-V5 H264 V12 private resume source seed {seed}",
        "id": f"{account}/{slug}",
        "licenses": [{"name": "CC0-1.0"}],
    }


def validate_checkpoint_pair(
    best: Path,
    last: Path,
    expected_best_sha: str,
    expected_last_sha: str,
) -> dict:
    if sha256(best) != expected_best_sha:
        raise ValueError("best checkpoint SHA-256 mismatch")
    if sha256(last) != expected_last_sha:
        raise ValueError("last checkpoint SHA-256 mismatch")
    import torch

    state = torch.load(last, map_location="cpu", weights_only=False)
    if state.get("epoch") != 1 or state.get("global_step") != 500:
        raise ValueError("resume checkpoint must be epoch=1, global_step=500")
    cfg = state["cfg"]
    rc = cfg["loss"]["rate_constraint"]["per_codec"]["h264"]
    if cfg["codec"]["ste_codec"] != "h264" or cfg["codec"]["ste_alternate"] is not False:
        raise ValueError("resume checkpoint is not H.264-only")
    if float(rc["target_ratio"]) != 0.0 or float(rc["dual_lr"]) != 0.015:
        raise ValueError("resume checkpoint is not the locked t0_lr15 arm")
    return {
        "epoch": int(state["epoch"]),
        "global_step": int(state["global_step"]),
        "rate_duals": {key: float(value) for key, value in state["rate_duals"].items()},
    }


def dataset_exists(handle: str) -> bool:
    result = subprocess.run(
        kaggle_command() + ["datasets", "files", handle],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--account", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--best", type=Path, required=True)
    parser.add_argument("--last", type=Path, required=True)
    parser.add_argument("--expected-best-sha", required=True)
    parser.add_argument("--expected-last-sha", required=True)
    parser.add_argument("--write-only", action="store_true")
    args = parser.parse_args()

    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", args.slug):
        raise ValueError("slug must be lowercase alphanumeric with optional hyphens")
    for name, value in {
        "expected_best_sha": args.expected_best_sha,
        "expected_last_sha": args.expected_last_sha,
    }.items():
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"{name} must be a lowercase SHA-256")

    audit = validate_checkpoint_pair(
        args.best,
        args.last,
        args.expected_best_sha,
        args.expected_last_sha,
    )
    target = REPO / "ops" / "_push_datasets" / args.account / args.slug
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.best, target / "preprocessor.pth")
    shutil.copy2(args.last, target / "preprocessor_last.pth")
    manifest = {
        "protocol": "longtrain-fullval-v12-source",
        "account": args.account,
        "seed": args.seed,
        "best_sha256": args.expected_best_sha,
        "last_sha256": args.expected_last_sha,
        **audit,
    }
    (target / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (target / "dataset-metadata.json").write_text(
        json.dumps(dataset_metadata(args.account, args.slug, args.seed)), encoding="utf-8"
    )
    handle = f"{args.account}/{args.slug}"
    print(
        f"[v12-source] prepared private dataset {handle} "
        f"best={args.expected_best_sha} last={args.expected_last_sha}"
    )
    if args.write_only:
        return
    if dataset_exists(handle):
        print(f"[v12-source] exists: {handle}")
        return
    command = kaggle_command() + ["datasets", "create", "-p", str(target), "-q"]
    raise SystemExit(subprocess.run(command, text=True).returncode)


if __name__ == "__main__":
    main()
