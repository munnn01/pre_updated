import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "ops" / "probe_joint_roi.py"
    spec = importlib.util.spec_from_file_location("probe_joint_roi", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_roi_arm_grid_is_frozen_and_has_matched_global_controls():
    module = _module()
    assert module.ARM_CONFIG == {
        "roi50_bg3": (0.50, 0, 3),
        "roi50_bg6": (0.50, 0, 6),
        "roi65_bg6": (0.65, 0, 6),
        "roi65_bg9": (0.65, 0, 9),
        "roi50m2_bg6": (0.50, -2, 6),
        "roi65m2_bg9": (0.65, -2, 9),
    }
    assert {config[2] for config in module.ARM_CONFIG.values()} == {3, 6, 9}


def test_crf_grid_parser_rejects_duplicates():
    module = _module()
    try:
        module._grid("30,30", "crfs")
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate CRFs were accepted")
