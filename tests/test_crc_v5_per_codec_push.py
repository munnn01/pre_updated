import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "ops" / "push_crc_v5_per_codec.py"
    spec = importlib.util.spec_from_file_location("push_crc_v5_per_codec", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_rendered_job_is_paired_codec_specific_and_has_no_timeout():
    module = _module()
    ref = "a" * 40
    sha = "b" * 64
    cell = module.render_cell(ref, sha, 281001, "treatment-first")
    assert f'REF="{ref}"' in cell
    assert f'EXPECTED_SHA="{sha}"' in cell
    assert 'ARM_ORDER="treatment-first"' in cell
    assert "run_arm treatment 0.015" in cell
    assert "run_arm control 0.005" in cell
    assert "loss.rate_constraint.per_codec.h265.dual_lr=0.005" in cell
    assert "eval.shard_idx=1" in cell
    assert "crc-v5-pc1-val-v1" in cell
    assert "timeout " not in cell.lower()


def test_metadata_attaches_account_local_stage1_dataset():
    module = _module()
    meta = module.metadata("baooo25r", "preupd-crc-v5-pc1-r1", "stage1-v1")
    assert meta["id"] == "baooo25r/preupd-crc-v5-pc1-r1"
    assert meta["dataset_sources"] == [
        "qktttttttttt/kineticscleaned",
        "baooo25r/stage1-v1",
    ]
    assert meta["enable_gpu"] is True
    assert meta["enable_internet"] is True


def test_render_validation_rejects_ambiguous_inputs():
    module = _module()
    with pytest.raises(ValueError, match="commit SHA"):
        module.render_cell("main", "b" * 64, 1, "control-first")
    with pytest.raises(ValueError, match="SHA-256"):
        module.render_cell("a" * 40, "bad", 1, "control-first")
    with pytest.raises(ValueError, match="arm_order"):
        module.render_cell("a" * 40, "b" * 64, 1, "bad-order")
