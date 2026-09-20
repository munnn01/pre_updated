#!/usr/bin/env python
"""Download one exact output file from a historical Kaggle notebook version."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import (
    ApiListKernelSessionOutputRequest,
)


def select_file(files, suffix: str):
    matches = [item for item in files if str(item.file_name).endswith(suffix)]
    if len(matches) != 1:
        names = [str(item.file_name) for item in files]
        raise ValueError(f"expected one '*{suffix}', found {len(matches)} in {names}")
    return matches[0]


def download(account: str, slug: str, version: int, suffix: str, out_dir: Path) -> dict:
    api = KaggleApi()
    api.authenticate()
    request = ApiListKernelSessionOutputRequest()
    request.user_name = account
    request.kernel_slug = slug
    # Backend labels historical versions as v1, v2, ...; the bare integer
    # accepted by the CLI parser is not forwarded and returns the latest run.
    request.version_label = f"v{version}"
    request.page_size = 200
    with api.build_kaggle_client() as client:
        response = client.kernels.kernels_api_client.list_kernel_session_output(request)
    item = select_file(response.files or [], suffix)
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / Path(str(item.file_name)).name
    remote = requests.get(item.url, timeout=120)
    remote.raise_for_status()
    destination.write_bytes(remote.content)
    digest = hashlib.sha256(remote.content).hexdigest()
    return {
        "account": account,
        "slug": slug,
        "version": version,
        "source_file": str(item.file_name),
        "destination": str(destination),
        "bytes": len(remote.content),
        "sha256": digest,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--account", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--suffix", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = download(
        args.account, args.slug, args.version, args.suffix, args.out.resolve()
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

