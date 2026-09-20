from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from src.models.additive_cond import AdditiveCondPreprocessor


def _module(name: str):
    path = Path(__file__).resolve().parents[1] / "ops" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_checkpoint_conversion_is_hash_pinned_and_expands_condition(tmp_path) -> None:
    module = _module("make_dctp_v6_ckpt")
    source_model = AdditiveCondPreprocessor(cond_dim=1)
    source = tmp_path / "source.pth"
    target = tmp_path / "target.pth"
    torch.save(
        {
            "model": source_model.state_dict(),
            "cfg": {"model": {"arch": "additive_cond", "cond_dim": 1}},
            "epoch": 15,
            "global_step": 16185,
        },
        source,
    )
    sha = module.file_sha256(source)
    audit = module.convert(
        source,
        target,
        sha,
        residual_scale=0.25,
        dct_strength=1.0,
        dct_threshold=1.0,
        temporal_strength=0.35,
        seed=220921,
    )
    converted = torch.load(target, map_location="cpu", weights_only=False)
    assert audit["source_sha256"] == sha
    assert converted["cfg"]["model"]["arch"] == "additive_dct"
    assert converted["model"]["film.0.weight"].shape[1] == 3
    assert torch.equal(
        converted["model"]["film.0.weight"][:, :1],
        source_model.state_dict()["film.0.weight"],
    )
    assert torch.count_nonzero(converted["model"]["film.0.weight"][:, 1:]) == 0


def test_dctp_pusher_has_full_factorial_and_public_checkpoint() -> None:
    module = _module("push_dctp_v6")
    assert len(module.ARMS) == 8
    combos = {
        (v["residual"], v["dct"], v["temporal"]) for v in module.ARMS.values()
    }
    assert len(combos) == 8
    meta = module.metadata("shungg05", "dctp")
    assert meta["dataset_sources"][1] == "dieulinhh/crc-v5-t0-lr1-stage1-v1"


def test_dctp_cell_is_eval_only_on_fresh_validation_shard() -> None:
    module = _module("push_dctp_v6")
    cell = module.render_cell("b" * 40, "r25_d10_t35", "a" * 64)
    assert 'RESIDUAL="0.25"' in cell
    assert 'DCT="1.0"' in cell
    assert 'TEMPORAL="0.35"' in cell
    assert "dctp-v6-screen-v1" in cell
    assert "eval.num_shards=10" in cell
    assert "train.py" not in cell
    assert "eval.split=test" not in cell
    assert "timeout " not in cell.lower()
