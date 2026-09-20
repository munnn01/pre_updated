#!/usr/bin/env python
"""Publish the verified CRC-V5 warm-start as a private per-account dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SLUG = "qpc-v4-uniform-s1-warmstart"
EXPECTED_SHA256 = "20d83d69f8be9e6d7754e07fe3c493e46a0df17d8734954ed7d827e0ec372db2"


def kaggle_command() -> list[str]:
    executable = shutil.which("kaggle")
    if executable:
        return [executable]
    return [sys.executable, "-c", "from kaggle.cli import main; main()"]


def checkpoint_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(account: str, checkpoint: Path) -> Path:
    actual = checkpoint_sha(checkpoint)
    if actual != EXPECTED_SHA256:
        raise ValueError(f"warm-start SHA mismatch: {actual}")
    target = REPO / "ops" / "_push" / account / "_crc_warmstart_dataset"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, target / "preprocessor.pth")
    metadata = {
        "id": f"{account}/{SLUG}",
        "title": "QPC V4 uniform seed 1 warm start",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (target / "dataset-metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return target


def dataset_exists(handle: str) -> bool:
    result = subprocess.run(
        kaggle_command() + ["datasets", "files", handle],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and "preprocessor.pth" in result.stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--account", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--write-only", action="store_true")
    args = parser.parse_args()

    target = prepare(args.account, args.checkpoint.resolve())
    handle = f"{args.account}/{SLUG}"
    print(f"[crc-warmstart] prepared {target} sha256={EXPECTED_SHA256}")
    if args.write_only:
        return
    if dataset_exists(handle):
        print(f"[crc-warmstart] exists: {handle}")
        return
    command = kaggle_command() + ["datasets", "create", "-p", str(target)]
    raise SystemExit(subprocess.run(command, text=True).returncode)


if __name__ == "__main__":
    main()

