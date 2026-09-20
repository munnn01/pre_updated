"""Download compact result artifacts and logs for audited Kaggle notebooks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from audit_kaggle_history import _retry_rate_limit
from kaggle.api.kaggle_api_extended import KaggleApi

RESULT_PATTERN = (
    r"(?i)(?:^|/)(?:[^/]*(?:result|metric|summary|report|probe|record|history|"
    r"gate|split|accounting|capability)[^/]*\.(?:json|jsonl|csv|npz|txt)|[^/]*\.log)$"
)


def _load_existing(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--accounts", nargs="*")
    args = parser.parse_args()

    pool = json.loads(args.pool.read_text(encoding="utf-8-sig"))
    account_files = sorted((args.audit / "accounts").glob("*.json"))
    if args.accounts:
        wanted = set(args.accounts)
        account_files = [path for path in account_files if path.stem in wanted]

    result_root = args.audit / "results"
    result_root.mkdir(parents=True, exist_ok=True)
    for account_file in account_files:
        account = account_file.stem
        if account not in pool:
            print(f"[skip] no token for {account}", flush=True)
            continue
        os.environ["KAGGLE_API_TOKEN"] = str(pool[account])
        api = KaggleApi()
        api.authenticate()
        records = json.loads(account_file.read_text(encoding="utf-8"))
        manifest_path = result_root / account / "download_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = _load_existing(manifest_path)
        selected = [record for record in records if record.get("project_relevant")]
        for number, record in enumerate(selected, 1):
            ref = str(record["ref"])
            slug = ref.split("/", 1)[1]
            if manifest.get(ref, {}).get("downloaded"):
                print(f"[{account}] {number}/{len(selected)} cached {ref}", flush=True)
                continue
            target = result_root / account / slug
            entry: dict[str, Any] = {"ref": ref, "downloaded": False, "files": []}
            try:
                files, _ = _retry_rate_limit(
                    lambda api=api, ref=ref, target=target: api.kernels_output(
                        ref,
                        str(target),
                        file_pattern=RESULT_PATTERN,
                        quiet=True,
                        page_size=200,
                    )
                )
                entry["downloaded"] = True
                entry["files"] = [str(Path(path).relative_to(args.audit)) for path in files]
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
            manifest[ref] = entry
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
            print(
                f"[{account}] {number}/{len(selected)} {ref} files={len(entry['files'])}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
