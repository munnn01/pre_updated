from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from src.config import load_config
from src.engine import (
    _codec_task_regret_constraint,
    _sequence_reports,
    _task_regret_constraint_settings,
)
from src.losses import LossWeights, preprocessing_loss


ROOT = Path(__file__).resolve().parents[1]


def _push_module():
    path = ROOT / "ops" / "push_v13_task_regret_screen.py"
    spec = importlib.util.spec_from_file_location("push_v13_task_regret", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_task_regret_settings_are_codec_qp_conditioned() -> None:
    cfg = load_config(str(ROOT / "configs" / "v13_task_regret_ar.yaml"))
    settings = _task_regret_constraint_settings(
        cfg, cfg["train"]["qp_list"], ("h264",)
    )
    assert settings["enabled"] is True
    assert settings["cells"] == [
        "h264:30", "h264:35", "h264:40", "h264:45", "h264:50"
    ]
    assert _codec_task_regret_constraint(settings, "h264") == {
        "epsilon": 0.0,
        "dual_lr": 0.001,
        "rate_weight": 1.0,
    }
    assert cfg["loss"]["rate_constraint"]["enabled"] is False
    assert cfg["train"]["epochs"] == 3
    assert cfg["train"]["max_steps"] is None


def test_task_regret_settings_reject_invalid_values() -> None:
    cfg = {
        "loss": {
            "task_regret_constraint": {
                "enabled": True,
                "epsilon": -0.1,
            }
        }
    }
    with pytest.raises(ValueError, match="must be non-negative"):
        _task_regret_constraint_settings(cfg, [30], ("h264",))
    cfg["loss"]["task_regret_constraint"] = {
        "per_codec": {"h266": {"dual_lr": 0.1}}
    }
    with pytest.raises(ValueError, match="unsupported codec"):
        _task_regret_constraint_settings(cfg, [30], ("h264",))


def test_precomputed_task_regret_is_the_primal_objective() -> None:
    class Analyzer:
        def accuracy_loss(self, *_args):
            raise AssertionError("precomputed task loss must prevent a second forward")

        def features(self, _x):
            return []

    source = torch.zeros(1, 3, 2, 4, 4)
    reconstruction = source.clone()
    task_loss = torch.tensor(2.0, requires_grad=True)
    task_objective = task_loss - 1.5
    rate_objective = torch.tensor(-0.2, requires_grad=True)
    weights = LossWeights(
        lam_task=0.4,
        omega=0.0,
        beta=2.0,
        tau=0.0,
        delta=0.0,
        gamma=0.0,
        gamma_res=0.0,
        kappa=0.0,
        kappa_t=0.0,
        mu=0.0,
    )
    parts = preprocessing_loss(
        Analyzer(),
        source,
        reconstruction,
        torch.tensor(0.1),
        target=torch.tensor([0]),
        w=weights,
        rate_objective=rate_objective,
        task_loss=task_loss,
        task_objective=task_objective,
    )
    assert torch.allclose(parts["loss"], torch.tensor(-0.2))
    assert torch.allclose(parts["loss_task"], torch.tensor(2.0))
    assert torch.allclose(parts["loss_task_objective"], torch.tensor(0.5))
    parts["loss"].backward()
    assert task_loss.grad is not None and task_loss.grad.item() == pytest.approx(0.4)
    assert rate_objective.grad is not None
    assert rate_objective.grad.item() == pytest.approx(2.0)


def test_v13_notebook_is_locked_and_uses_only_requested_sources() -> None:
    module = _push_module()
    sha = "b" * 64
    cell = module.render_cell("a" * 40, "h264", 292001, sha)
    assert "configs/v13_task_regret_ar.yaml" in cell
    assert "train.epochs=3" in cell
    assert "train.max_steps=null" in cell
    assert "eval.num_shards=5" in cell
    assert "v13-task-regret-screen-v1" in cell
    assert 'assert report["n_eval"] == 208' in cell
    assert "timeout " not in cell.lower()

    h264 = module.metadata(
        "qktttttttttt",
        "v13-h264",
        "h264",
        "qktttttttttt/crc-v5-h264-v12-source-s282001",
    )
    assert h264["dataset_sources"] == [
        "qktttttttttt/kineticscleaned",
        "qktttttttttt/crc-v5-h264-v12-source-s282001",
    ]
    h265 = module.metadata("baoancut", "v13-h265", "h265", None)
    assert h265["dataset_sources"] == [
        "qktttttttttt/kineticscleaned",
        "baooo25r/crc-v5-h265-minus24-candidate-v1",
    ]
    with pytest.raises(ValueError, match="owned by target"):
        module.metadata(
            "trnhlng", "bad", "h264",
            "qktttttttttt/crc-v5-h264-v12-source-s282001",
        )


def test_sequence_report_counts_only_finite_bd(tmp_path: Path, monkeypatch) -> None:
    def point(bpp: float, prob: float) -> dict:
        return {"bpp": bpp, "target_prob": prob, "top1": 0}

    store = {
        "clip": {
            "sequence_id": "clip",
            "path": "class/clip.mp4",
            "class": "class",
            "codecs": {
                "h264": {"30": point(0.2, 0.8), "35": point(0.1, 0.5)},
                "prep+h264": {
                    "30": point(0.19, 0.8), "35": point(0.09, 0.5)
                },
            },
        }
    }
    monkeypatch.setattr("src.engine.bd_rate", lambda *_args: float("nan"))
    monkeypatch.setattr("src.engine.bd_metric", lambda *_args: 0.0)
    _sequence_reports(tmp_path, store, [30, 35])
    report = json.loads(
        (tmp_path / "sequence_bd_rate.json").read_text(encoding="utf-8")
    )
    assert report["n_sequences"] == 1
    assert report["sequences_with_bd"] == 0
    assert report["sequences_with_finite_bd_by_codec"] == {
        "h264": 0,
        "h265": 0,
    }
