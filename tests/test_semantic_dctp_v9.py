from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "ops" / "push_semantic_dctp_v9.py"
    spec = importlib.util.spec_from_file_location("push_semantic_dctp_v9", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_v9_cell_has_registered_response_surface_and_fresh_split() -> None:
    module = _module()
    cell = module.render_cell("a" * 40, "h264", "b" * 64)
    for arm in (
        "p25_q1:0.25:1.0",
        "p25_q2:0.25:2.0",
        "p25_q4:0.25:4.0",
        "p50_q1:0.50:1.0",
        "p50_q2:0.50:2.0",
        "p50_q4:0.50:4.0",
        "p375_q2:0.375:2.0",
    ):
        assert arm in cell
    assert "--residual-scale 0.0" in cell
    assert "--dct-band-start 2" in cell
    assert "--qp-slope 0.75" in cell
    assert "semantic-dctp-v9-val-v1" in cell
    assert "eval.split=test" not in cell
    assert "timeout " not in cell.lower()


def test_v9_is_codec_specific_and_attaches_private_source_kernel() -> None:
    module = _module()
    h264 = module.render_cell("a" * 40, "h264", "b" * 64)
    h265 = module.render_cell("a" * 40, "h265", "b" * 64)
    assert 'RATE_ARM="t0_lr15"' in h264
    assert 'TEMPORAL="0.0"' in h264
    assert 'RATE_ARM="tm15_lr15"' in h265
    assert 'TEMPORAL="0.35"' in h265
    assert "sha256sum" in h264
    meta = module.metadata(
        "vtk269",
        "semantic-dctp-v9-h264-r1",
        "vtk269/preupd-codec-v7-h264-t0-lr15-r1",
    )
    assert meta["is_private"] is True
    assert meta["kernel_sources"] == ["vtk269/preupd-codec-v7-h264-t0-lr15-r1"]


def test_evaluator_restores_semantic_area_from_each_checkpoint() -> None:
    engine = (
        Path(__file__).resolve().parents[1] / "src" / "engine.py"
    ).read_text(encoding="utf-8")
    assert '"semantic_protect_area"' in engine
