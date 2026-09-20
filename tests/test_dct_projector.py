from __future__ import annotations

import torch
from src.models.additive_cond import AdditiveCondPreprocessor
from src.models.dct_projector import DCTProjectedAdditivePreprocessor


def _cond(qp: float, codec: str = "h264") -> torch.Tensor:
    return torch.tensor([[qp, float(codec == "h264"), float(codec == "h265")]])


def test_projection_keeps_additive_checkpoint_layout() -> None:
    source = AdditiveCondPreprocessor(cond_dim=3)
    projected = DCTProjectedAdditivePreprocessor(cond_dim=3)
    assert source.state_dict().keys() == projected.state_dict().keys()
    assert all(
        source.state_dict()[key].shape == projected.state_dict()[key].shape
        for key in source.state_dict()
    )


def test_zero_projection_and_zero_residual_is_identity() -> None:
    torch.manual_seed(1)
    model = DCTProjectedAdditivePreprocessor(
        cond_dim=3, residual_scale=0.0, dct_strength=0.0, temporal_strength=0.0
    )
    x = torch.rand(1, 3, 2, 16, 16)
    assert torch.equal(model(x, _cond(0.5)), x)


def test_spatial_projection_reduces_weak_high_frequency_energy() -> None:
    torch.manual_seed(2)
    model = DCTProjectedAdditivePreprocessor(
        cond_dim=3,
        residual_scale=0.0,
        dct_strength=1.0,
        dct_threshold=2.0,
        qp_slope=0.0,
    )
    base = torch.full((1, 3, 2, 16, 16), 0.5)
    noise = 0.03 * torch.randn_like(base)
    x = (base + noise).clamp(0.0, 1.0)
    out = model(x, _cond(0.0))
    assert out.shape == x.shape
    assert float((out - base).pow(2).mean()) < float((x - base).pow(2).mean())
    assert 0.0 <= float(out.min()) <= float(out.max()) <= 1.0


def test_qp_and_codec_condition_control_projection_strength() -> None:
    model = DCTProjectedAdditivePreprocessor(
        cond_dim=3,
        dct_strength=1.0,
        qp_slope=0.75,
        h264_scale=1.0,
        h265_scale=0.5,
    )
    low_qp_h264 = float(model._conditioned_strength(_cond(0.2, "h264"), 1.0))
    high_qp_h264 = float(model._conditioned_strength(_cond(0.9, "h264"), 1.0))
    low_qp_h265 = float(model._conditioned_strength(_cond(0.2, "h265"), 1.0))
    assert low_qp_h264 > high_qp_h264
    assert low_qp_h264 > low_qp_h265


def test_temporal_projection_leaves_motion_less_affected_than_static_pixels() -> None:
    model = DCTProjectedAdditivePreprocessor(
        cond_dim=3,
        residual_scale=0.0,
        dct_strength=0.0,
        temporal_strength=1.0,
        motion_tau=0.05,
        qp_slope=0.0,
    )
    source = torch.zeros(1, 3, 2, 4, 4)
    source[:, :, 1, :, 2:] = 1.0
    spatial = source.clone()
    spatial[:, :, 1, :, :2] = 0.2  # artificial flicker in a source-static region
    out = model._temporal_project(source, spatial, torch.ones(1))
    assert torch.allclose(out[:, :, 1, :, :2], out[:, :, 0, :, :2], atol=1e-5)
    assert torch.allclose(out[:, :, 1, :, 2:], spatial[:, :, 1, :, 2:], atol=1e-5)
