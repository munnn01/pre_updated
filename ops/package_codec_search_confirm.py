#!/usr/bin/env python
"""Package the verified 1,000-video H.264/H.265 AR confirmation for GitHub.

Only explicitly allowlisted Kaggle artifacts enter the archive. Source videos,
model weights and Kaggle credentials are never bundled.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ops.merge_codec_search_confirm import aggregate, load_shards

CONFIRM_COMMIT = "0fab4b95ae65e2e93ff1985a6491e2e3bf38dec1"
FILES_PER_SHARD = ("shard_result.json", "shard_records.jsonl", "run.log", "frozen_policy.json")
NOTEBOOKS = {
    "h264": (
        "https://www.kaggle.com/code/trnhlng/codec-search-confirm-h264-s0",
        "https://www.kaggle.com/code/baoancut/codec-search-confirm-h264-s1",
        "https://www.kaggle.com/code/huolgggnuyen/codec-search-confirm-h264-s2",
        "https://www.kaggle.com/code/shungg05/codec-search-confirm-h264-s3",
    ),
    "h265": (
        "https://www.kaggle.com/code/trnhlng/codec-search-confirm-h265-s0",
        "https://www.kaggle.com/code/baoancut/codec-search-confirm-h265-s1",
        "https://www.kaggle.com/code/huolgggnuyen/codec-search-confirm-h265-s2",
        "https://www.kaggle.com/code/shungg05/codec-search-confirm-h265-s3",
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shard_directories(root: Path, codec: str) -> list[Path]:
    return [root / codec / f"shard_{i}" / "outputs" / "codec_search_confirm"
            / codec / f"shard_{i}" for i in range(4)]


def verify(root: Path, frozen: Path) -> tuple[dict, dict, dict]:
    config = json.loads(frozen.read_text(encoding="utf-8"))
    # Git stores this text file with LF; Windows checkout may use CRLF.
    frozen_digest = hashlib.sha256(frozen.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    summaries = {}
    provenance = {}
    for codec in ("h264", "h265"):
        source = root / codec / "merged_1000_result.json"
        summary = json.loads(source.read_text(encoding="utf-8"))
        dirs = shard_directories(root, codec)
        rows, metadata = load_shards(dirs, expected_count=1000)
        independently_merged = aggregate(rows, metadata, repeats=2000)
        if independently_merged != summary:
            raise ValueError(f"recomputed {codec} result differs from the saved summary")
        if summary["code_commit"] != CONFIRM_COMMIT:
            raise ValueError(f"unexpected {codec} confirmation code commit")
        if summary["pilot_code_commit"] != config["pilot_code_commit"]:
            raise ValueError(f"unexpected {codec} pilot code commit")
        if summary["manifest_sha256"] != frozen_digest:
            raise ValueError(f"unexpected {codec} frozen-policy digest")
        if summary["n"] != config["confirm_clips"] or summary["qps"] != config["qps"]:
            raise ValueError(f"unexpected {codec} sample size or QPs")
        expected_limits = {key: config["policies"][codec][key]
                           for key in ("active", "kl_slack", "feature_slack")}
        if summary["limits"] != expected_limits:
            raise ValueError(f"unexpected {codec} policy limits")
        for directory in dirs:
            for name in FILES_PER_SHARD:
                if not (directory / name).is_file():
                    raise FileNotFoundError(directory / name)
            if sha256(directory / "frozen_policy.json") != frozen_digest:
                raise ValueError(f"frozen policy mismatch in {directory}")
        summaries[codec] = summary
        provenance[codec] = {"source_result_sha256": sha256(source), "notebooks": list(NOTEBOOKS[codec])}
    if summaries["h264"]["test_fingerprint"] != summaries["h265"]["test_fingerprint"]:
        raise ValueError("H.264 and H.265 did not use the same 1,000 videos")
    return config, summaries, provenance


def write_archive(path: Path, directories: list[Path]) -> None:
    """Create a deterministic archive with a flat, merge-script-ready layout."""
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0, compresslevel=9) as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as archive:
                for shard, directory in enumerate(directories):
                    for name in FILES_PER_SHARD:
                        payload = (directory / name).read_bytes()
                        entry = tarfile.TarInfo(f"shard_{shard}/{name}")
                        entry.size = len(payload)
                        entry.mode = 0o644
                        entry.mtime = 0
                        entry.uid = entry.gid = 0
                        entry.uname = entry.gname = ""
                        archive.addfile(entry, io.BytesIO(payload))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path,
                        default=REPO / "results" / "codec_search_ar_confirm_1000")
    parser.add_argument("--refresh", action="store_true",
                        help="replace only the package files previously generated by this script")
    args = parser.parse_args()
    frozen = REPO / "configs" / "codec_search_ar_frozen_v1.json"
    config, summaries, provenance = verify(args.input_root, frozen)
    out = args.out_dir
    expected_names = {"README.md", "manifest.json", "h264_result.json", "h265_result.json",
                      "h264_shards.tar.gz", "h265_shards.tar.gz"}
    if out.exists():
        existing = {item.name for item in out.iterdir()}
        if (not args.refresh and existing != {"README.md"}) or (args.refresh and
                ("README.md" not in existing or existing - expected_names)):
            raise FileExistsError(f"release directory contains unexpected files: {out}")
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for codec in ("h264", "h265"):
        source = args.input_root / codec / "merged_1000_result.json"
        result_name = f"{codec}_result.json"
        archive_name = f"{codec}_shards.tar.gz"
        # GitHub serves LF-normalized text blobs. Preserve those exact bytes
        # (and checksum) even when the Kaggle download was written with CRLF.
        (out / result_name).write_bytes(source.read_bytes().replace(b"\r\n", b"\n"))
        write_archive(out / archive_name, shard_directories(args.input_root, codec))
        files[result_name] = {"sha256": sha256(out / result_name), "bytes": (out / result_name).stat().st_size}
        files[archive_name] = {"sha256": sha256(out / archive_name),
                               "bytes": (out / archive_name).stat().st_size}
    manifest = {
        "release": "codec_search_ar_confirm_1000_v1",
        "scope": "research artifact for frozen r2plus1d_18; not a universal AR codec or production encoder",
        "dataset": config["dataset"], "split": config["confirm_split"],
        "videos": config["confirm_clips"], "qps": config["qps"],
        "test_fingerprint": summaries["h264"]["test_fingerprint"],
        "pilot_code_commit": config["pilot_code_commit"],
        "confirm_code_commit": CONFIRM_COMMIT,
        "frozen_policy_sha256": summaries["h264"]["manifest_sha256"],
        "primary_model": config["primary_model"], "cross_model": config["cross_model"],
        "files": files,
        "results": {
            codec: {
                "primary_bd_rate_top1_pct": summary["analyzers"]["r2plus1d_18"]["metrics"]["policy"]["bd_rate_top1_pct"],
                "primary_bootstrap_ci95_pct": summary["analyzers"]["r2plus1d_18"]["bootstrap"]["ci95_pct"],
                "cross_bd_rate_top1_pct": summary["analyzers"]["r3d_18"]["metrics"]["policy"]["bd_rate_top1_pct"],
                "cross_bootstrap_ci95_pct": summary["analyzers"]["r3d_18"]["bootstrap"]["ci95_pct"],
                **provenance[codec],
            }
            for codec, summary in summaries.items()
        },
    }
    (out / "manifest.json").write_bytes((json.dumps(manifest, indent=2, ensure_ascii=False,
                                                  allow_nan=False) + "\n").encode("utf-8"))
    print(json.dumps({"out": str(out), "files": files,
                      "primary_bd_rate": {codec: value["primary_bd_rate_top1_pct"]
                                          for codec, value in manifest["results"].items()}},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
