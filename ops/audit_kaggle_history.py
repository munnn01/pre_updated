"""Build a local, credential-free manifest of Kaggle notebook history.

The token pool is read locally and never written to the audit directory.  The
manifest stores notebook metadata, every retrievable source version, and the
latest output-file inventory so experiment lineage can be reviewed offline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
from requests import HTTPError

PROJECT_MARKERS = (
    "pre_updated",
    "preprocessing_upgrade",
    "pre-processing",
    "munnn01",
    "vcm_preprocess",
    "probe_action",
    "probe_detection",
    "probe_background",
    "kineticscleaned",
    "coco-2017-dataset",
    "bd_rate",
    "bd-rate",
)
PROJECT_SLUG = re.compile(
    r"^(?:preupd|pre-updated|u(?:6|7|8|9|10)(?:-|$)|bl-|final1$|"
    r"h264-eval-single$|tuan3?-|retrain-3gb$|ratelamda015$|version6$)",
    re.IGNORECASE,
)


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if hasattr(value, "to_dict"):
        return _plain(value.to_dict())
    if hasattr(value, "__dict__"):
        return {
            key.removeprefix("_"): _plain(item)
            for key, item in vars(value).items()
            if not key.startswith("__")
        }
    return str(value)


def _slug_path(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def _get_kernel(client: Any, account: str, slug: str, version: int | None = None) -> Any:
    request = ApiGetKernelRequest()
    request.user_name = account
    request.kernel_slug = f"{slug}/{version}" if version is not None else slug
    return client.kernels.kernels_api_client.get_kernel(request)


def _retry_rate_limit(call: Any, attempts: int = 6) -> Any:
    for attempt in range(attempts):
        try:
            return call()
        except HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status != 429 or attempt == attempts - 1:
                raise
            delay = min(60, 5 * (2**attempt))
            print(f"[rate-limit] sleeping {delay}s", flush=True)
            time.sleep(delay)
    raise AssertionError("unreachable")


def _metadata_value(metadata: Any, name: str, default: Any = None) -> Any:
    return getattr(metadata, name, default)


def scan_account(account: str, token: str, out_dir: Path) -> dict[str, Any]:
    os.environ["KAGGLE_API_TOKEN"] = token
    api = KaggleApi()
    api.authenticate()
    kernels = api.kernels_list(page_size=200, mine=True, sort_by="dateRun") or []
    account_dir = out_dir / "sources" / account
    account_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    with api.build_kaggle_client() as client:
        visible = [item for item in kernels if item is not None and getattr(item, "ref", "")]
        for number, item in enumerate(visible, 1):
            ref = str(item.ref)
            owner, slug = ref.split("/", 1)
            record: dict[str, Any] = {
                "account": account,
                "ref": ref,
                "list_metadata": _plain(item),
                "versions": [],
                "latest_output_files": [],
                "errors": [],
            }
            try:
                latest = _retry_rate_limit(lambda owner=owner, slug=slug: _get_kernel(client, owner, slug))
                metadata = latest.metadata
                current_version = int(_metadata_value(metadata, "current_version_number", 1) or 1)
                record["current_version"] = current_version
                record["metadata"] = _plain(metadata)
                for version in range(1, current_version + 1):
                    try:
                        response = (
                            latest
                            if version == current_version
                            else _retry_rate_limit(
                                lambda owner=owner, slug=slug, version=version: _get_kernel(
                                    client, owner, slug, version
                                )
                            )
                        )
                        source = str(response.blob.source)
                        source_path = account_dir / _slug_path(slug) / f"v{version}.ipynb"
                        source_path.parent.mkdir(parents=True, exist_ok=True)
                        source_path.write_text(source, encoding="utf-8")
                        lowered = source.lower()
                        record["versions"].append(
                            {
                                "version": version,
                                "sha256": hashlib.sha256(source.encode()).hexdigest(),
                                "bytes": len(source.encode()),
                                "project_relevant": any(marker in lowered for marker in PROJECT_MARKERS),
                                "source_path": str(source_path.relative_to(out_dir)),
                            }
                        )
                    except Exception as exc:  # keep the rest of the audit useful
                        record["errors"].append(f"source v{version}: {type(exc).__name__}: {exc}")
                record["project_relevant"] = bool(PROJECT_SLUG.search(slug)) or any(
                    version["project_relevant"] for version in record["versions"]
                )
                if record["project_relevant"]:
                    try:
                        page_token = None
                        while True:
                            files = _retry_rate_limit(
                                lambda ref=ref, page_token=page_token: api.kernels_list_files(
                                    ref, page_token=page_token, page_size=200
                                )
                            )
                            record["latest_output_files"].extend(_plain(getattr(files, "files", []) or []))
                            page_token = getattr(files, "next_page_token", None)
                            if not page_token:
                                break
                    except Exception as exc:
                        record["errors"].append(f"files: {type(exc).__name__}: {exc}")
            except Exception as exc:
                record["errors"].append(f"latest: {type(exc).__name__}: {exc}")
            records.append(record)
            print(f"[{account}] {number}/{len(visible)} {ref}", flush=True)

    account_manifest = out_dir / "accounts" / f"{account}.json"
    account_manifest.parent.mkdir(parents=True, exist_ok=True)
    account_manifest.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "account": account,
        "listed": len(kernels),
        "visible": len(records),
        "versions": sum(len(record["versions"]) for record in records),
        "relevant_versions": sum(
            int(version["project_relevant"]) for record in records for version in record["versions"]
        ),
        "errors": sum(len(record["errors"]) for record in records),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--accounts", nargs="*")
    args = parser.parse_args()

    pool = json.loads(args.pool.read_text(encoding="utf-8-sig"))
    accounts = args.accounts or sorted(pool)
    missing = [account for account in accounts if account not in pool]
    if missing:
        raise SystemExit(f"missing tokens for: {', '.join(missing)}")
    args.out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for account in accounts:
        summaries.append(scan_account(account, str(pool[account]), args.out))
        (args.out / "summary.json").write_text(
            json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    print(json.dumps(summaries, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
