import importlib.util
from pathlib import Path

import pytest
import torch
from src.config import load_config
from src.engine import (
    _codec_rate_constraint,
    _evaluation_codecs,
    _load_state_compat,
    _rate_cond,
    _rate_constraint_settings,
)
from src.models.additive_cond import AdditiveCondPreprocessor


def _push_module():
    path = Path(__file__).resolve().parents[1] / "ops" / "push_crc_v5.py"
    spec = importlib.util.spec_from_file_location("push_crc_v5", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_codec_rate_condition_is_explicit_and_legacy_stays_compatible():
    legacy = _rate_cond(0.5, 2, "cpu", torch.float32)
    assert legacy.tolist() == [[0.5], [0.5]]
    h264 = _rate_cond(0.5, 1, "cpu", torch.float32, codec="h264", cond_dim=3)
    h265 = _rate_cond(0.5, 1, "cpu", torch.float32, codec="h265", cond_dim=3)
    assert h264.tolist() == [[0.5, 1.0, 0.0]]
    assert h265.tolist() == [[0.5, 0.0, 1.0]]
    with pytest.raises(ValueError):
        _rate_cond(0.5, 1, "cpu", torch.float32, cond_dim=3)


def test_qpc_warmstart_expands_film_with_zero_codec_columns():
    source = AdditiveCondPreprocessor(cond_dim=1)
    target = AdditiveCondPreprocessor(cond_dim=3)
    with torch.no_grad():
        source.film[0].weight.fill_(0.25)
        source.film[2].weight.normal_(0, 0.1)
        source.to_rgb.weight.normal_(0, 0.01)
    missing = _load_state_compat(target, source.state_dict())
    assert not missing
    assert torch.equal(target.film[0].weight[:, :1], source.film[0].weight)
    assert torch.count_nonzero(target.film[0].weight[:, 1:]) == 0
    x = torch.rand(1, 3, 4, 16, 16)
    with torch.no_grad():
        old = source(x, torch.tensor([[0.5]]))
        new264 = target(x, torch.tensor([[0.5, 1.0, 0.0]]))
        new265 = target(x, torch.tensor([[0.5, 0.0, 1.0]]))
    assert torch.equal(old, new264)
    assert torch.equal(old, new265)


def test_rate_constraint_validation_and_cells():
    cfg = {
        "loss": {
            "rate_constraint": {
                "enabled": True,
                "target_ratio": -0.05,
                "dual_lr": 0.005,
                "lambda_init": 0.01,
                "lambda_max": 1.0,
            }
        }
    }
    settings = _rate_constraint_settings(cfg, [30, 35])
    assert settings["target_ratio"] == -0.05
    assert settings["per_codec"] == {
        "h264": {"target_ratio": -0.05, "dual_lr": 0.005},
        "h265": {"target_ratio": -0.05, "dual_lr": 0.005},
    }
    assert settings["cells"] == ["h264:30", "h264:35", "h265:30", "h265:35"]


def test_rate_constraint_resolves_asymmetric_codec_overrides():
    cfg = {
        "loss": {
            "rate_constraint": {
                "enabled": True,
                "target_ratio": -0.05,
                "dual_lr": 0.001,
                "lambda_init": 0.01,
                "lambda_max": 1.0,
                "per_codec": {
                    "h264": {"target_ratio": 0.0, "dual_lr": 0.015},
                    "h265": {"target_ratio": 0.0, "dual_lr": 0.005},
                },
            }
        }
    }
    settings = _rate_constraint_settings(cfg, [30, 35])
    assert _codec_rate_constraint(settings, "h264") == {
        "target_ratio": 0.0,
        "dual_lr": 0.015,
    }
    assert _codec_rate_constraint(settings, "h265") == {
        "target_ratio": 0.0,
        "dual_lr": 0.005,
    }


def test_rate_constraint_rejects_bad_codec_overrides():
    base = {
        "loss": {
            "rate_constraint": {
                "per_codec": {"h266": {"dual_lr": 0.1}},
            }
        }
    }
    with pytest.raises(ValueError, match="unsupported codec"):
        _rate_constraint_settings(base, [30])

    base["loss"]["rate_constraint"]["per_codec"] = {
        "h264": {"dual_lr": -0.1}
    }
    with pytest.raises(ValueError, match="must be non-negative"):
        _rate_constraint_settings(base, [30])


def test_crc_v5_per_codec_config_is_the_registered_asymmetric_treatment():
    path = Path(__file__).resolve().parents[1] / "configs" / "crc_v5_per_codec_ar.yaml"
    cfg = load_config(str(path))
    settings = _rate_constraint_settings(cfg, cfg["train"]["qp_list"])
    assert cfg["codec"]["ste_alternate"] is True
    assert cfg["train"]["balanced_codec_qp"] is True
    assert _codec_rate_constraint(settings, "h264")["dual_lr"] == 0.015
    assert _codec_rate_constraint(settings, "h265")["dual_lr"] == 0.005
    assert _codec_rate_constraint(settings, "h264")["target_ratio"] == 0.0
    assert _codec_rate_constraint(settings, "h265")["target_ratio"] == 0.0


def test_rate_constraint_can_be_scoped_to_one_codec():
    cfg = {
        "loss": {
            "rate_constraint": {
                "enabled": True,
                "target_ratio": -0.15,
                "dual_lr": 0.015,
                "lambda_init": 0.01,
                "lambda_max": 1.0,
            }
        }
    }
    settings = _rate_constraint_settings(cfg, [30, 40], ("h265",))
    assert settings["codecs"] == ["h265"]
    assert settings["cells"] == ["h265:30", "h265:40"]


def test_evaluation_codec_subset_validation():
    assert _evaluation_codecs({}) == ("h264", "h265")
    assert _evaluation_codecs({"eval": {"codecs": "h264"}}) == ("h264",)
    assert _evaluation_codecs({"eval": {"codecs": ["h265"]}}) == ("h265",)
    with pytest.raises(ValueError):
        _evaluation_codecs({"eval": {"codecs": ["h264", "h264"]}})
    with pytest.raises(ValueError):
        _rate_constraint_settings({}, [30], ("av1",))


def test_crc_matrix_and_notebook_protocol():
    module = _push_module()
    assert set(module.ARMS) == {"t0_lr1", "tm5_lr1", "t0_lr5", "tm5_lr5"}
    cell = module.render_cell("a" * 40, "tm5_lr5")
    assert 'ENABLED="true"' in cell
    assert 'TARGET="-0.05"' in cell
    assert 'DUAL_LR="0.005"' in cell
    assert "configs/qpc_v4_ar.yaml" in cell
    assert 'mkdir -p "$STAGE1/checkpoints"' in cell
    assert "PREV_STAGE1=" in cell
    assert "run_arm control false 0.0 0.0 0.001" in cell
    assert 'run_arm "$ARM"' in cell
    assert "eval.split=val" in cell and "eval.num_shards=20" in cell
    assert "eval.split=test" not in cell
    assert "timeout " not in cell.lower()
    meta = module.metadata("shungg05", "crc-v5-control")
    assert meta["dataset_sources"] == ["qktttttttttt/kineticscleaned"]
    assert meta["kernel_sources"] == []
    resumed = module.metadata("huolgggnuyen", "crc-v5", attach_previous=True)
    assert resumed["kernel_sources"] == ["huolgggnuyen/crc-v5"]
