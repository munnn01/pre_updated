import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "ops" / "push_roi_probe.py"
    spec = importlib.util.spec_from_file_location("push_roi_probe", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_d1_cell_is_crf_based_and_has_no_timeout():
    module = _module()
    cell = module.render_cell("a" * 40, "d1", "h264", "mc3_18", 200, "24,30,36,42,48")
    assert "--phase d1" in cell
    assert '--crfs "$CRFS"' in cell
    assert "--qps" not in cell
    assert "timeout " not in cell.lower()
    assert 'EVAL_BACKBONE="mc3_18"' in cell


def test_f0_metadata_does_not_attach_large_dataset():
    module = _module()
    assert module.metadata("wagur124705", "roi-v3-f0", "f0")["dataset_sources"] == []
    assert module.metadata("wagur124705", "roi-v3-f0", "f0")["enable_gpu"] is False


def test_notebook_executes_rendered_script_as_bash():
    module = _module()
    payload = module.notebook("set -euo pipefail\necho ok\n")
    source = "".join(payload["cells"][0]["source"])
    assert source.startswith("%%bash\nset -euo pipefail\n")
    assert payload["cells"][0]["id"] == "roi-v3-probe"
