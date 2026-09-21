from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "ops" / "push_longtrain_fullval_v12.py"
    spec = importlib.util.spec_from_file_location("push_longtrain_fullval_v12", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_h264_metadata_mounts_account_local_private_dataset() -> None:
    module = _module()
    meta = module.metadata(
        "qktttttttttt",
        "longtrain-v12",
        "h264",
        "qktttttttttt/crc-v5-h264-v12-source-s282001",
    )
    assert meta["dataset_sources"] == [
        "qktttttttttt/kineticscleaned",
        "qktttttttttt/crc-v5-h264-v12-source-s282001",
    ]
    assert meta["kernel_sources"] == []


def test_h265_metadata_uses_public_frozen_source() -> None:
    module = _module()
    meta = module.metadata("thuha1205", "longtrain-v12", "h265", None)
    assert meta["dataset_sources"] == [
        "qktttttttttt/kineticscleaned",
        "baooo25r/crc-v5-h265-minus24-candidate-v1",
    ]
    assert meta["kernel_sources"] == []


def test_source_rules_reject_wrong_mount_type() -> None:
    module = _module()
    with pytest.raises(ValueError, match="account-local"):
        module.metadata("qktttttttttt", "longtrain-v12", "h264", None)
    with pytest.raises(ValueError, match="owned"):
        module.metadata(
            "qktttttttttt",
            "longtrain-v12",
            "h264",
            "someone-else/crc-v5-h264-v12-source-s282001",
        )
    with pytest.raises(ValueError, match="frozen public"):
        module.metadata("thuha1205", "longtrain-v12", "h265", "x/y")


def test_rendered_cell_resumes_three_epochs_and_full_evaluates() -> None:
    module = _module()
    cell = module.render_cell("a" * 40, "h264", 282001, "b" * 64, "c" * 64)
    for placeholder in (
        "__REF__",
        "__CODEC__",
        "__SEED__",
        "__EXPECTED_BEST_SHA__",
        "__EXPECTED_LAST_SHA__",
    ):
        assert placeholder not in cell
    assert "train.epochs=4" in cell
    assert "train.max_steps=null" in cell
    assert "train.resume=true" in cell
    assert "train.finetune=false" in cell
    assert 'FINAL="$TRAIN_OUT/checkpoints/preprocessor_last.pth"' in cell
    assert "eval.num_shards=1" in cell
    assert 'report["n_eval"] == 1010' in cell
    assert 'state["global_step"] >= 13000' in cell


def test_render_rejects_invalid_identity() -> None:
    module = _module()
    with pytest.raises(ValueError, match="40-character"):
        module.render_cell("main", "h264", 1, "b" * 64, "c" * 64)
    with pytest.raises(ValueError, match="codec"):
        module.render_cell("a" * 40, "vp9", 1, "b" * 64, "c" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        module.render_cell("a" * 40, "h265", 1, "short", "c" * 64)
