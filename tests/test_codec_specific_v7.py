import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "ops" / "push_codec_specific_v7.py"
    spec = importlib.util.spec_from_file_location("push_codec_specific_v7", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_codec_specific_factorial_and_cell_protocol():
    module = _module()
    assert module.ARMS == {
        "t0_lr5": {"target": "0.0", "dual_lr": "0.005"},
        "t0_lr15": {"target": "0.0", "dual_lr": "0.015"},
        "tm15_lr5": {"target": "-0.15", "dual_lr": "0.005"},
        "tm15_lr15": {"target": "-0.15", "dual_lr": "0.015"},
    }
    cell = module.render_cell("a" * 40, "b" * 64, "h265", "tm15_lr15", 270921)
    assert 'CODEC="h265"' in cell
    assert 'TARGET="-0.15"' in cell
    assert 'DUAL_LR="0.015"' in cell
    assert 'codec.ste_codec="$CODEC" codec.ste_alternate=false' in cell
    assert 'eval.codecs="[$CODEC]"' in cell
    assert "run_arm control false 0.0 0.0 0.001" in cell
    assert 'run_arm "$ARM" true "$TARGET" "$DUAL_LR" 0.0' in cell
    assert "eval.split=val" in cell and "eval.split=test" not in cell
    assert "timeout " not in cell.lower()


def test_codec_specific_rejects_invalid_arm_and_attaches_public_checkpoint():
    module = _module()
    with pytest.raises(ValueError):
        module.render_cell("a" * 40, "b" * 64, "av1", "t0_lr5", 1)
    with pytest.raises(ValueError):
        module.render_cell("a" * 40, "b" * 64, "h264", "unknown", 1)
    meta = module.metadata("shungg05", "codec-v7")
    assert meta["dataset_sources"] == [
        "qktttttttttt/kineticscleaned",
        "dieulinhh/crc-v5-t0-lr1-stage1-v1",
    ]
    assert meta["enable_gpu"] is True
