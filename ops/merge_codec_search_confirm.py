#!/usr/bin/env python
"""Merge disjoint confirmation shards before calculating aggregate BD-rate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np

from ops.codec_search_ar import QPS, compare, selected, summarize
from src.metrics.bd_rate import bd_rate
from src.models.codec_search import CANDIDATES


def load_shards(directories: list[Path], expected_count: int) -> tuple[list[dict], dict]:
    reports = []
    rows = []
    for directory in directories:
        report = json.loads((directory / "shard_result.json").read_text(encoding="utf-8"))
        records = [json.loads(line) for line in (directory / "shard_records.jsonl").open(encoding="utf-8")]
        if len(records) != report["n"]:
            raise ValueError(f"record count mismatch in {directory}")
        if any(row.get("schema") != 2 or row.get("codec") != report["codec"] for row in records):
            raise ValueError(f"invalid confirmation schema or codec in {directory}")
        reports.append(report)
        rows.extend(records)
    first = reports[0]
    expected_shards = set(range(first["shards"]))
    if len(reports) != first["shards"] or {report["shard"] for report in reports} != expected_shards:
        raise ValueError("missing or repeated confirmation shard")
    for report in reports:
        for key in ("experiment", "codec", "split", "total_expected", "qps",
                    "test_fingerprint", "manifest_sha256", "code_commit", "pilot_code_commit", "limits"):
            if report[key] != first[key]:
                raise ValueError(f"inconsistent {key} across shards")
    if first["split"] != "test" or first["total_expected"] != expected_count or len(rows) != expected_count:
        raise ValueError("confirmation does not have the frozen 1,000 TEST clips")
    ids = [row["sequence_id"] for row in rows]
    if len(set(ids)) != expected_count:
        raise ValueError("duplicate or missing TEST video")
    got_fingerprint = hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()[:16]
    if got_fingerprint != first["test_fingerprint"]:
        raise ValueError("merged TEST fingerprint mismatch")
    for row in rows:
        if len(row["measurements"]) != len(QPS) or [m["qp"] for m in row["measurements"]] != list(QPS):
            raise ValueError("missing or disordered QP measurements")
        for measurement in row["measurements"]:
            candidates = measurement["candidates"]
            if [candidate["name"] for candidate in candidates] != list(CANDIDATES):
                raise ValueError("candidate list disagrees with frozen policy")
            chosen = candidates[selected(candidates, first["limits"])]
            if not all(key in candidate for candidate in (candidates[0], chosen)
                       for key in ("cross_correct", "cross_target_prob")):
                raise ValueError("missing cross-model outcome for anchor or chosen arm")
    return rows, first


def bootstrap(rows: list[dict], limits: dict, field: str, repeats: int, seed: int) -> dict:
    array = np.empty((len(rows), len(QPS), 2, 2), dtype=np.float64)
    for i, row in enumerate(rows):
        for j, measurement in enumerate(row["measurements"]):
            candidates = measurement["candidates"]
            for k, candidate in enumerate((candidates[0], candidates[selected(candidates, limits)])):
                array[i, j, k] = (candidate["bpp"], float(candidate[field]))
    rng = np.random.default_rng(seed)
    values = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(repeats):
            sample = array[rng.integers(0, len(rows), len(rows))].mean(axis=0)
            metric = bd_rate(sample[:, 0, 0], sample[:, 0, 1],
                             sample[:, 1, 0], sample[:, 1, 1])
            if math.isfinite(metric):
                values.append(metric)
    if len(values) < .95 * repeats:
        raise ValueError("too many undefined bootstrap BD-rates")
    return {"n_resamples": len(values), "resampling_unit": "video",
            "seed": seed, "ci95_pct": np.quantile(values, [.025, .975]).tolist(),
            "median_pct": float(np.median(values))}


def aggregate(rows: list[dict], metadata: dict, repeats: int) -> dict:
    limits = metadata["limits"]
    report = {key: metadata[key] for key in (
        "experiment", "codec", "split", "total_expected", "qps", "test_fingerprint",
        "manifest_sha256", "code_commit", "pilot_code_commit", "limits")}
    report["n"] = len(rows)
    report["analyzers"] = {}
    for model, field in (("r2plus1d_18", "correct"), ("r3d_18", "cross_correct")):
        anchor, _ = summarize(rows, lambda _: 0, field=field)
        policy, choices = summarize(rows, lambda candidates: selected(candidates, limits), field=field)
        report["analyzers"][model] = {
            "curves": {"identity128": anchor, "policy": policy},
            "choices": {"policy": choices},
            "metrics": {"policy": compare(anchor, policy)},
            "bootstrap": bootstrap(rows, limits, field, repeats, seed=20260923),
        }
    primary = report["analyzers"]["r2plus1d_18"]["metrics"]["policy"]["bd_rate_top1_pct"]
    report["primary_success_below_minus_15"] = primary is not None and primary < -15.0
    report["interpretation"] = "Frozen-policy, 1000-video TEST confirmation relative to 104-clip pilot."
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap < 100:
        raise ValueError("bootstrap must have at least 100 resamples")
    rows, metadata = load_shards(args.shard_dir, expected_count=1000)
    report = aggregate(rows, metadata, args.bootstrap)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"codec": report["codec"], "videos": report["n"],
                      "matched": report["analyzers"]["r2plus1d_18"]["metrics"]["policy"],
                      "matched_ci95": report["analyzers"]["r2plus1d_18"]["bootstrap"]["ci95_pct"],
                      "cross": report["analyzers"]["r3d_18"]["metrics"]["policy"],
                      "primary_success_below_minus_15": report["primary_success_below_minus_15"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
