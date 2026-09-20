from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "ops" / "push_crc_v6_crossover.py"
    spec = importlib.util.spec_from_file_location("push_crc_v6_crossover", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_crossover_uses_new_validation_shard_and_exact_two_arms() -> None:
    module = _module()
    cell = module.render_cell("b" * 40, "a" * 64)
    assert "run_arm control false 0.0 0.001" in cell
    assert "run_arm t0_lr5 true 0.005 0.0" in cell
    assert "eval.shard_salt=crc-v6-crossover-v1" in cell
    assert "eval.num_shards=10" in cell
    assert "eval.split=test" not in cell
    assert "configs/qpc_v4_ar.yaml" not in cell
    assert "timeout " not in cell.lower()


def test_crossover_dataset_is_account_local_copy() -> None:
    module = _module()
    meta = module.metadata("shungg05", "cross", "canonical-dieulinh")
    assert meta["dataset_sources"] == [
        "qktttttttttt/kineticscleaned",
        "shungg05/canonical-dieulinh",
    ]
