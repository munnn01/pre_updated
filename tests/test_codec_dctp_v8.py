import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "ops" / "push_codec_dctp_v8.py"
    spec = importlib.util.spec_from_file_location("push_codec_dctp_v8", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_codec_specific_composition_settings_and_cell():
    module = _module()
    assert module.SETTINGS["h264"]["residuals"] == "0.0 0.125 0.25 0.5"
    assert module.SETTINGS["h265"]["residuals"] == "0.0 0.0625 0.125 0.25"
    cell = module.render_cell("a" * 40, "h265", "b" * 64)
    assert 'CODEC="h265"' in cell
    assert 'RATE_ARM="tm15_lr15"' in cell
    assert 'DCT="0.6"' in cell
    assert 'TEMPORAL="0.35"' in cell
    assert 'RESIDUALS="0.0 0.0625 0.125 0.25"' in cell
    assert "codec-dctp-v8-val-v1" in cell
    assert 'eval.codecs="[$CODEC]"' in cell
    assert "eval.split=test" not in cell
    assert "timeout " not in cell.lower()


def test_composition_source_is_exact_treatment_and_private_kernel_source():
    module = _module()
    cell = module.render_cell("a" * 40, "h264", "b" * 64)
    assert "codec_specific_v7_${CODEC}_${RATE_ARM}_${RATE_ARM}" in cell
    assert "sha256sum" in cell
    meta = module.metadata(
        "vtk269",
        "codec-dctp-v8-h264-r1",
        "vtk269/preupd-codec-v7-h264-t0-lr15-r1",
    )
    assert meta["kernel_sources"] == ["vtk269/preupd-codec-v7-h264-t0-lr15-r1"]
    assert meta["dataset_sources"] == ["qktttttttttt/kineticscleaned"]
    with pytest.raises(ValueError):
        module.render_cell("a" * 40, "av1", "b" * 64)
