import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "ops" / "push_qpc_v4_eval.py"
    spec = importlib.util.spec_from_file_location("push_qpc_v4_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_eval_cell_is_validation_only_and_uses_fixed_subsplit():
    module = _module()
    cell = module.render_cell("a" * 40, "uniform", 2, 0, 5)
    assert 'ARM="uniform"' in cell
    assert 'ARCH="additive_cond"' in cell
    assert 'SEED="2"' in cell
    assert "eval.split=val" in cell
    assert 'SHARD_IDX="0"' in cell and 'NUM_SHARDS="5"' in cell
    assert "eval.shard_salt=qpc-v4-val-v1" in cell
    assert "eval.per_sequence=true" in cell
    assert "eval.held_out_backbone=r2plus1d_18" in cell
    assert "eval.split=test" not in cell
    assert "timeout " not in cell.lower()


def test_eval_metadata_attaches_exact_same_account_train_kernel():
    module = _module()
    meta = module.metadata(
        "wagur124705",
        "preupd-qpc-v4-eval-control-s0-v5",
        "preupd-qpc-v4-control-s0",
    )
    assert meta["dataset_sources"] == ["qktttttttttt/kineticscleaned"]
    assert meta["kernel_sources"] == ["wagur124705/preupd-qpc-v4-control-s0"]
    assert meta["enable_gpu"] is True


def test_control_checkpoint_path_and_arch_are_pinned():
    module = _module()
    cell = module.render_cell("b" * 40, "control", 0, 0, 5)
    assert 'ARCH="additive"' in cell
    assert "qpc_v4_${ARM}_s${SEED}/checkpoints/preprocessor.pth" in cell
    assert 'state.get("epoch", -1)) == 14' in cell
