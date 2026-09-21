import importlib.util
from pathlib import Path

import pytest


def _push_module():
    path = Path(__file__).resolve().parents[1] / "ops" / "push_dual_checkpoint_v10.py"
    spec = importlib.util.spec_from_file_location("push_dual_checkpoint_v10", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_v10_cell_freezes_h265_and_runs_full_h264_matrix():
    module = _push_module()
    order = "t0_lr5,t0_lr15,tm5_lr5,tm5_lr15,tm25_lr10"
    cell = module.render_cell("a" * 40, 282001, order)
    assert 'EXPECTED_SHA="a3580b32e1554071811888238ea249ce9ae26e7b32392a2d975e4bd6cd0a9c8e"' in cell
    assert "evaluate_h265 eval_original_val0of20 0 20 crc-v5-val-v1" in cell
    assert "evaluate_h265 eval_fresh_val2of20 2 20 dual-checkpoint-v10-val-v1" in cell
    assert "codec.ste_codec=h264 codec.ste_alternate=false" in cell
    assert 'test "$(sha256sum "$H265_CKPT"' in cell
    for arm in module.ARMS:
        assert f"{arm})" in cell
    assert "timeout " not in cell.lower()


def test_v10_metadata_mounts_public_candidate_and_kinetics():
    module = _push_module()
    meta = module.metadata("shungg05", "preupd-dual-checkpoint-v10-r1")
    assert meta["dataset_sources"] == [
        "qktttttttttt/kineticscleaned",
        "baooo25r/crc-v5-h265-minus24-candidate-v1",
    ]
    assert meta["enable_gpu"] is True
    assert meta["enable_internet"] is True


def test_v10_arm_order_requires_each_arm_once():
    module = _push_module()
    valid = "t0_lr5,t0_lr15,tm5_lr5,tm5_lr15,tm25_lr10"
    assert module.validate_arm_order(valid) == valid
    with pytest.raises(ValueError, match="exactly once"):
        module.validate_arm_order("t0_lr5,t0_lr5,tm5_lr5,tm5_lr15,tm25_lr10")
    with pytest.raises(ValueError, match="exactly once"):
        module.validate_arm_order("t0_lr5,t0_lr15")


def test_candidate_publisher_constants_match_registered_result():
    path = Path(__file__).resolve().parents[1] / "ops" / "publish_h265_candidate.py"
    spec = importlib.util.spec_from_file_location("publish_h265_candidate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.EXPECTED_SHA == "a3580b32e1554071811888238ea249ce9ae26e7b32392a2d975e4bd6cd0a9c8e"
    assert module.EXPECTED_H265_BD_RATE == pytest.approx(-24.2623162235382)
    assert module.EXPECTED_N_EVAL == 44
