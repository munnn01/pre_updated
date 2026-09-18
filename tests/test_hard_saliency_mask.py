import pytest
import torch
from src.models.task_mask import hard_saliency_mask


def test_hard_saliency_mask_enforces_clip_budget():
    saliency = torch.arange(32, dtype=torch.float32).reshape(1, 1, 2, 4, 4)
    mask = hard_saliency_mask(saliency, 0.25, "clip")

    assert mask.shape == saliency.shape
    assert mask.sum() == 8
    assert torch.equal(mask.unique(), torch.tensor([0.0, 1.0]))


def test_tube_mask_is_constant_in_time_and_spatially_budgeted():
    saliency = torch.rand(2, 1, 4, 8, 8)
    mask = hard_saliency_mask(saliency, 0.25, "tube")

    assert mask.shape == saliency.shape
    assert torch.equal(mask[:, :, :1].expand_as(mask), mask)
    assert torch.equal(mask[:, :, 0].sum(dim=(1, 2, 3)), torch.tensor([16.0, 16.0]))


@pytest.mark.parametrize(
    ("fraction", "mode"),
    [(0.0, "clip"), (1.0, "clip"), (0.25, "frame")],
)
def test_hard_saliency_mask_rejects_invalid_configuration(fraction, mode):
    with pytest.raises(ValueError):
        hard_saliency_mask(torch.rand(1, 1, 2, 4, 4), fraction, mode)
