from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "ops" / "push_fullval_v11.py"
    spec = importlib.util.spec_from_file_location("push_fullval_v11", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_h264_metadata_uses_only_account_local_v10_kernel() -> None:
    module = _module()
    meta = module.metadata(
        "baooo25r",
        "fullval-v11",
        "h264",
        "baooo25r/preupd-dual-checkpoint-v10-r1",
    )
    assert meta["dataset_sources"] == ["qktttttttttt/kineticscleaned"]
    assert meta["kernel_sources"] == ["baooo25r/preupd-dual-checkpoint-v10-r1"]


def test_h265_metadata_uses_frozen_public_dataset_without_kernel_source() -> None:
    module = _module()
    meta = module.metadata("wagur124705", "fullval-v11", "h265", None)
    assert meta["dataset_sources"] == [
        "qktttttttttt/kineticscleaned",
        "baooo25r/crc-v5-h265-minus24-candidate-v1",
    ]
    assert meta["kernel_sources"] == []


def test_source_rules_reject_confounded_mounts() -> None:
    module = _module()
    with pytest.raises(ValueError, match="account-local"):
        module.metadata("baooo25r", "fullval-v11", "h264", None)
    with pytest.raises(ValueError, match="only the frozen"):
        module.metadata("wagur124705", "fullval-v11", "h265", "x/y")


def test_rendered_cell_is_full_validation_only() -> None:
    module = _module()
    ref = "a" * 40
    sha = "b" * 64
    cell = module.render_cell(ref, "h264", sha)
    assert "__REF__" not in cell
    assert "__CODEC__" not in cell
    assert "__EXPECTED_SHA__" not in cell
    assert 'eval.num_shards=1' in cell
    assert 'report["n_eval"] == 1010' in cell
    assert "train.py" not in cell
    assert "target_ratio\"]), 0.0" in cell
    assert "dual_lr\"]), 0.015" in cell


def test_render_rejects_invalid_identity() -> None:
    module = _module()
    with pytest.raises(ValueError, match="40-character"):
        module.render_cell("main", "h264", "b" * 64)
    with pytest.raises(ValueError, match="codec"):
        module.render_cell("a" * 40, "vp9", "b" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        module.render_cell("a" * 40, "h265", "short")
