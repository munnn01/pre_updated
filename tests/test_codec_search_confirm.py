import hashlib
import json
import tarfile
from pathlib import Path

import pytest
import torch

from ops import codec_search_ar
from ops.codec_search_confirm import load_frozen
from ops.merge_codec_search_confirm import load_shards
from ops.package_codec_search_confirm import FILES_PER_SHARD, write_archive
from ops.push_codec_search_confirm import render_cell
from src.models.codec_search import CANDIDATES


class _Dataset:
    def __getitem__(self, index):
        source = torch.zeros(3, 16, 128, 128)
        return source, 1, {"sequence_id": "class/clip.mp4"}


class _Codec:
    codec = "h264"

    def _encode_decode_clip(self, clip, qp):
        return clip, 0.5


def test_frozen_limits_and_manifest():
    path = Path(__file__).resolve().parents[1] / "configs" / "codec_search_ar_frozen_v1.json"
    h264, h264_limits, digest = load_frozen(path, "h264")
    _, h265_limits, other_digest = load_frozen(path, "h265")
    assert h264_limits == h265_limits == {"active": True, "kl_slack": .1, "feature_slack": .05}
    assert h264["confirm_clips"] == 1000 and h264["confirm_split"] == "test"
    assert len(digest) == 64 and digest == other_digest


def test_confirm_cross_model_scores_only_anchor_and_frozen_selection(monkeypatch):
    calls = {"primary": 0, "cross": 0}

    def predict(analyzer, video):
        calls[analyzer] += 1
        return torch.tensor([[0.0, 1.0]]), torch.tensor([[1.0, 0.0]])

    monkeypatch.setattr(codec_search_ar, "predict_and_feature", predict)
    limits = {"active": True, "kl_slack": .1, "feature_slack": .05}
    row = codec_search_ar.case_for_clip(_Dataset(), 0, "primary", _Codec(),
                                         cross="cross", cross_limits=limits)
    assert row["schema"] == 2 and len(row["measurements"]) == 5
    assert calls == {"primary": 37, "cross": 10}
    for measurement in row["measurements"]:
        candidates = measurement["candidates"]
        assert [candidate["name"] for candidate in candidates] == list(CANDIDATES)
        picked = codec_search_ar.selected(candidates, limits)
        assert candidates[picked]["name"] == "area96"
        assert [i for i, candidate in enumerate(candidates) if "cross_correct" in candidate] == [0, picked]


def test_kaggle_cell_is_commit_pinned():
    cell = render_cell("a" * 40, "h264", 2)
    assert 'REF="' + "a" * 40 + '"' in cell
    assert 'CODEC="h264"' in cell and 'SHARD="2"' in cell
    assert "__REF__" not in cell and "__SHARD__" not in cell


def test_merge_requires_unique_videos_and_complete_shards(tmp_path):
    ids = [f"class/video_{i:02d}.mp4" for i in range(8)]
    fp = hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()[:16]
    directories = []
    for shard in range(4):
        directory = tmp_path / str(shard)
        directory.mkdir()
        directories.append(directory)
        report = {"experiment": "codec_search_ar_confirm_v1", "codec": "h264", "split": "test",
                  "shard": shard, "shards": 4, "n": 2, "total_expected": 8,
                  "qps": [30, 35, 40, 45, 50], "test_fingerprint": fp,
                  "manifest_sha256": "m", "code_commit": "c", "pilot_code_commit": "p",
                  "limits": {"active": True, "kl_slack": .1, "feature_slack": .05}}
        (directory / "shard_result.json").write_text(json.dumps(report), encoding="utf-8")
        with (directory / "shard_records.jsonl").open("w", encoding="utf-8") as handle:
            for i in (shard, shard + 4):
                measurements = []
                for qp in (30, 35, 40, 45, 50):
                    candidates = []
                    for j, name in enumerate(CANDIDATES):
                        candidate = {"name": name, "bpp": 1 / (1 + j + qp / 10),
                                     "kl_source": 0, "feature_distance": 0,
                                     "source_confidence": .9, "source_top1_agrees": True,
                                     "correct": True, "target_prob": .7}
                        if j in (0, len(CANDIDATES) - 1):
                            candidate.update({"cross_correct": True, "cross_target_prob": .6})
                        candidates.append(candidate)
                    measurements.append({"qp": qp, "candidates": candidates})
                handle.write(json.dumps({"schema": 2, "codec": "h264", "sequence_id": ids[i],
                                         "measurements": measurements}) + "\n")
    rows, report = load_shards(directories, expected_count=8)
    assert len(rows) == 8 and report["test_fingerprint"] == fp
    with pytest.raises(ValueError, match="missing or repeated confirmation shard"):
        load_shards(directories[:3] + directories[:1], expected_count=8)


def test_release_archive_is_deterministic_and_allowlisted(tmp_path):
    directories = []
    for i in range(4):
        directory = tmp_path / f"input_{i}"
        directory.mkdir()
        directories.append(directory)
        for name in FILES_PER_SHARD:
            (directory / name).write_text(f"shard={i} file={name}\n", encoding="utf-8")
        (directory / "private_token.txt").write_text("do-not-package", encoding="utf-8")
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    write_archive(first, directories)
    write_archive(second, directories)
    assert first.read_bytes() == second.read_bytes()
    with tarfile.open(first, "r:gz") as archive:
        names = archive.getnames()
        assert len(names) == 4 * len(FILES_PER_SHARD)
        assert names == [f"shard_{i}/{name}" for i in range(4) for name in FILES_PER_SHARD]
        assert not any("private_token" in name for name in names)
