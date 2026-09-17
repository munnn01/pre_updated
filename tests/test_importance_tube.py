import torch

from src.models.importance_tube import (
    ImportanceTubeSuppress,
    expand_importance_tube,
    feather_protection,
    masks_from_frame_detections,
)
from src.models.mask_suppress import suppress


def test_single_frame_is_od_r0_when_temporal_features_are_off():
    x = torch.rand(1, 3, 1, 32, 32)
    mask = torch.zeros(1, 1, 1, 32, 32)
    mask[..., 8:20, 9:21] = 1
    module = ImportanceTubeSuppress(
        sigma=4, temporal_radius=0, feather=0, temporal_strength=0
    )
    assert torch.allclose(module(x, mask), suppress(x, mask, 4))


def test_protected_core_is_exact_for_every_frame():
    x = torch.rand(1, 3, 4, 32, 32)
    mask = torch.zeros(1, 1, 4, 32, 32)
    mask[..., 10:18, 11:19] = 1
    out = ImportanceTubeSuppress(sigma=4, temporal_strength=0.8)(x, mask)
    assert torch.equal(out[..., 10:18, 11:19], x[..., 10:18, 11:19])


def test_temporal_branch_reduces_static_background_residual():
    base = torch.rand(1, 3, 1, 32, 32)
    noise = 0.01 * torch.randn(1, 3, 5, 32, 32)
    x = (base.expand(-1, -1, 5, -1, -1) + noise).clamp(0, 1)
    mask = torch.zeros(1, 1, 5, 32, 32)
    spatial = ImportanceTubeSuppress(
        sigma=2, temporal_radius=0, feather=0, temporal_strength=0
    )(x, mask)
    temporal = ImportanceTubeSuppress(
        sigma=2, temporal_radius=0, feather=0, temporal_strength=0.9,
        motion_tau=0.1,
    )(x, mask)
    tv_spatial = (spatial[:, :, 1:] - spatial[:, :, :-1]).abs().mean()
    tv_temporal = (temporal[:, :, 1:] - temporal[:, :, :-1]).abs().mean()
    assert tv_temporal < tv_spatial


def test_tube_expansion_feather_and_detection_stack():
    detections = [
        {"boxes": torch.tensor([[4.0, 5.0, 10.0, 12.0]]),
         "scores": torch.tensor([0.9]), "labels": torch.tensor([1])},
        {"boxes": torch.tensor([[6.0, 5.0, 12.0, 12.0]]),
         "scores": torch.tensor([0.9]), "labels": torch.tensor([1])},
    ]
    mask = masks_from_frame_detections(detections, 24, dilate=0)
    assert mask.shape == (1, 1, 2, 24, 24)
    expanded = expand_importance_tube(mask, temporal_radius=1)
    assert expanded.sum() >= mask.sum()
    soft = feather_protection(mask, radius=2)
    assert torch.equal(soft[mask == 1], torch.ones_like(soft[mask == 1]))
    assert ((soft > 0) & (soft < 1)).any()
