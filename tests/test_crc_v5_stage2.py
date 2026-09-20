import importlib.util
from pathlib import Path

import pytest
import torch


def _module(name: str):
    path = Path(__file__).resolve().parents[1] / "ops" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_stage1_publisher_validates_lineage(tmp_path):
    module = _module("publish_crc_stage1")
    ckpt = tmp_path / "preprocessor.pth"
    torch.save(
        {
            "cfg": {
                "model": {"arch": "additive_cond", "cond_dim": 1},
                "seed": 260920,
                "out_dir": "/kaggle/working/outputs/crc_v5_tm5_lr1_stage1",
            },
            "epoch": 15,
            "global_step": 16185,
            "best_val": 1.45,
        },
        ckpt,
    )
    audit = module.validate_checkpoint(ckpt, "tm5_lr1", 260920)
    assert len(audit["sha256"]) == 64
    assert audit["global_step"] == 16185
    with pytest.raises(ValueError, match="does not match arm"):
        module.validate_checkpoint(ckpt, "t0_lr1", 260920)


def test_recovery_notebook_runs_stage2_stage3_eval_and_is_hash_pinned():
    module = _module("push_crc_v5_stage2")
    sha = "a" * 64
    cell = module.render_cell("b" * 40, "tm5_lr1", sha)
    assert f'EXPECTED_SHA="{sha}"' in cell
    assert "configs/qpc_v4_ar.yaml" not in cell
    assert "configs/crc_v5_ar.yaml" in cell
    assert "run_arm stage2 control false 0.0 0.0 0.001" in cell
    assert 'run_arm stage3 "$ARM"' in cell
    assert '[${stage}-eval] real-codec validation' in cell
    assert "eval.split=test" not in cell
    assert "timeout " not in cell.lower()


def test_stage2_metadata_keeps_dataset_in_same_account():
    module = _module("push_crc_v5_stage2")
    meta = module.metadata("huolgggnuyen", "stage2", "private-stage1")
    assert meta["dataset_sources"] == [
        "qktttttttttt/kineticscleaned",
        "huolgggnuyen/private-stage1",
    ]
