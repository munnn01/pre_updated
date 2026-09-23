#!/usr/bin/env python
"""Run one shard of the frozen 1,000-video codec-search confirmation.

This deliberately imports the pilot's candidate generation, real-codec
measurement and label-free selector. No calibration is run on TEST.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import torch

from ops.codec_search_ar import QPS, case_for_clip, compare, selected, summarize
from ops.rcts_pilot import balanced_indices, clip_id, fingerprint
from src.codecs.standard import StandardCodec, ffmpeg_available
from src.data.video_dataset import VideoClipDataset
from src.models.codec_search import CANDIDATES
from src.tasks.action_recognition import ActionRecognitionAnalyzer


def load_frozen(path: Path, codec: str) -> tuple[dict, dict, str]:
    raw = path.read_bytes()
    config = json.loads(raw)
    if config["experiment"] != "codec_search_ar_confirm_v1":
        raise ValueError("unexpected frozen experiment")
    if tuple(config["qps"]) != QPS or tuple(config["candidates"]) != CANDIDATES:
        raise ValueError("frozen QPs or candidates disagree with code")
    policy = config["policies"][codec]
    limits = {key: policy[key] for key in ("active", "kl_slack", "feature_slack")}
    if not limits["active"]:
        raise ValueError("frozen confirmation requires an active policy")
    return config, limits, hashlib.sha256(raw).hexdigest()


def clip_keys(index: dict) -> dict[str, set[str]]:
    return {split: {clip_id(record) for record in records}
            for split, records in index.items() if isinstance(records, list)}


def collect_shard(dataset, indices, analyzer, cross, codec, limits, out, shard):
    directory = out / "clip_cache"
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for number, index in enumerate(indices, 1):
        path = directory / f"clip_{index:05d}.json"
        expected_id = clip_id(dataset.samples[index])
        if path.exists():
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("schema") != 2 or row.get("codec") != codec.codec or row.get("sequence_id") != expected_id:
                raise ValueError(f"stale or mismatched confirmation cache: {path}")
        else:
            row = case_for_clip(dataset, index, analyzer, codec, cross=cross, cross_limits=limits)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(row, allow_nan=False), encoding="utf-8")
            temporary.replace(path)
        if len(row["measurements"]) != len(QPS) or any(
            [candidate["name"] for candidate in measurement["candidates"]] != list(CANDIDATES)
            for measurement in row["measurements"]
        ):
            raise ValueError(f"incomplete confirmation measurements: {path}")
        rows.append(row)
        if number % 10 == 0 or number == len(indices):
            print(f"[test shard={shard}] {number}/{len(indices)} codec={codec.codec}", flush=True)
    return rows


def shard_report(rows, limits, metadata):
    selectors = {"identity128": lambda _: 0,
                 "policy": lambda candidates: selected(candidates, limits)}
    report = dict(metadata)
    report["analyzers"] = {}
    for model, field in (("r2plus1d_18", "correct"), ("r3d_18", "cross_correct")):
        base, base_choices = summarize(rows, selectors["identity128"], field=field)
        trial, trial_choices = summarize(rows, selectors["policy"], field=field)
        report["analyzers"][model] = {
            "curves": {"identity128": base, "policy": trial},
            "choices": {"identity128": base_choices, "policy": trial_choices},
            "metrics": {"policy": compare(base, trial)},
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--codec", choices=("h264", "h265"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, default=REPO / "configs" / "codec_search_ar_frozen_v1.json")
    parser.add_argument("--shard", type=int, required=True)
    args = parser.parse_args()
    if not ffmpeg_available():
        raise SystemExit("ffmpeg and ffprobe are required")
    config, limits, manifest_sha = load_frozen(args.frozen, args.codec)
    shards = int(config["shards"])
    if not 0 <= args.shard < shards:
        raise ValueError("shard outside frozen range")
    random.seed(53)
    np.random.seed(53)
    torch.manual_seed(53)
    index = json.loads(args.index.read_text(encoding="utf-8"))
    keys = clip_keys(index)
    if (keys["train"] & keys["test"]) or (keys["val"] & keys["test"]):
        raise ValueError("TEST shares source IDs with TRAIN or VAL")
    dataset = VideoClipDataset(args.index, split="test", num_frames=config["frames"],
                               frame_size=config["frame_size"],
                               temporal_stride=config["temporal_stride"],
                               train=False, return_metadata=True)
    count = int(config["confirm_clips"])
    if len(dataset) < count:
        raise ValueError(f"only {len(dataset)} TEST videos, need {count}")
    all_indices = balanced_indices(dataset.samples, count, config["confirm_salt"])
    if len(all_indices) != count or len({clip_id(dataset.samples[i]) for i in all_indices}) != count:
        raise ValueError("confirmation sample is incomplete or duplicates videos")
    indices = [index for position, index in enumerate(all_indices) if position % shards == args.shard]
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    print(f"[confirm] codec={args.codec} shard={args.shard}/{shards} "
          f"videos={len(indices)} test_fingerprint={fingerprint(dataset.samples, all_indices)} "
          f"manifest_sha256={manifest_sha}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    analyzer = ActionRecognitionAnalyzer(config["primary_model"], clip_size=112).freeze().to(device)
    cross = ActionRecognitionAnalyzer(config["cross_model"], clip_size=112).freeze().to(device)
    codec = StandardCodec(args.codec, preset=config["preset"])
    rows = collect_shard(dataset, indices, analyzer, cross, codec, limits, out, args.shard)
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    metadata = {
        "experiment": config["experiment"], "codec": args.codec,
        "split": "test", "shard": args.shard, "shards": shards,
        "n": len(rows), "total_expected": count, "qps": list(QPS),
        "test_fingerprint": fingerprint(dataset.samples, all_indices),
        "shard_fingerprint": fingerprint(dataset.samples, indices),
        "manifest_sha256": manifest_sha, "code_commit": git_sha,
        "pilot_code_commit": config["pilot_code_commit"], "limits": limits,
    }
    report = shard_report(rows, limits, metadata)
    with (out / "shard_records.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    (out / "shard_result.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"codec": args.codec, "shard": args.shard,
                      "matched": report["analyzers"][config["primary_model"]]["metrics"]["policy"],
                      "cross": report["analyzers"][config["cross_model"]]["metrics"]["policy"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
