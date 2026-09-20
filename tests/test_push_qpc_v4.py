import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "ops" / "push_qpc_v4.py"
    spec = importlib.util.spec_from_file_location("push_qpc_v4", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_matrix_arms_are_matched_except_conditioning_and_sampling():
    module = _module()
    assert module.ARMS["control"] == {"arch": "additive", "weights": "null"}
    assert module.ARMS["uniform"] == {
        "arch": "additive_cond",
        "weights": "null",
    }
    assert module.ARMS["bdweighted"]["arch"] == "additive_cond"
    assert module.ARMS["bdweighted"]["weights"].startswith("[0.089,")


def test_rendered_train_cell_uses_current_dataset_protocol():
    module = _module()
    cell = module.render_cell("a" * 40, "bdweighted", 2, 16)
    assert 'ARCH="additive_cond"' in cell
    assert 'SEED="2"' in cell and 'EPOCHS="16"' in cell
    assert 'QP_WEIGHTS="[0.089,0.215,0.288,0.285,0.123]"' in cell
    assert "--assert-fingerprint 30f083f8520a" in cell
    assert "--split val" in cell
    assert "timeout " not in cell.lower()


def test_notebook_is_bash_and_metadata_attaches_cleaned_kinetics():
    module = _module()
    source = "".join(module.notebook("set -e\n")["cells"][0]["source"])
    assert source.startswith("%%bash\nset -e\n")
    meta = module.metadata("vtk269", "qpc-v4")
    assert meta["dataset_sources"] == ["qktttttttttt/kineticscleaned"]
    assert meta["enable_gpu"] is True


def test_commit_resolution_rejects_unknown_ref():
    module = _module()
    assert len(module.resolve_local_commit("HEAD")) == 40
    try:
        module.resolve_local_commit("0" * 40)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown commit was accepted")
