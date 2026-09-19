import torch
from src.codecs.roi import ROIRect
from src.models.task_roi import (
    action_importance_score,
    importance_window,
    motion_score,
    spatial_roi_regions,
    top_fraction_mask,
)


def test_top_fraction_mask_has_exact_budget():
    score = torch.arange(64.0).reshape(1, 1, 1, 8, 8)
    mask = top_fraction_mask(score, 0.25)
    assert int(mask.sum()) == 16


def test_importance_window_is_block_aligned_and_capped():
    score = torch.zeros(1, 1, 1, 128, 128)
    score[..., 32:80, 48:96] = 1
    rect = importance_window(score, 0.50, block=16)

    assert all(value % 16 == 0 for value in (rect.x, rect.y, rect.width, rect.height))
    assert rect.width * rect.height <= 0.50 * 128 * 128
    assert rect.x <= 48 < rect.x + rect.width
    assert rect.y <= 32 < rect.y + rect.height


def test_spatial_regions_put_protected_rectangle_first():
    protected = ROIRect(16, 16, 32, 32, -2)
    regions = spatial_roi_regions(protected, 64, 64, background_delta_qp=6)
    assert regions[0] == protected
    assert regions[1] == ROIRect(0, 0, 64, 64, 6)


def test_motion_and_saliency_fusion_is_label_free_and_deterministic():
    video = torch.zeros(1, 3, 3, 16, 16)
    video[:, :, 1:, 4:8, 4:8] = 1
    saliency = torch.zeros(1, 1, 3, 16, 16)
    saliency[..., 10:14, 10:14] = 1

    motion = motion_score(video)
    first = action_importance_score(saliency, video)
    second = action_importance_score(saliency, video)

    assert motion.shape == (1, 1, 1, 16, 16)
    assert first.shape == (1, 1, 1, 16, 16)
    assert torch.equal(first, second)
    assert not first.requires_grad
