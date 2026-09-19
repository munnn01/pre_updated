import pytest
import torch

from ops.probe_action_guarded import (
    _arm,
    _blend_ladder,
    _codec_grid,
    _float_grid,
    guard_choices,
    motion_tube_mask,
)


def test_guarded_arm_and_grids_are_stable():
    assert (
        _arm(0.65, 0.5, 2.0, 0.4, 0.97, 0.1)
        == "guard_p65_m50_s2_a0.4_r0.97_t0.1"
    )
    assert _float_grid("0.65,0.8", "protect", open_unit=True) == [0.65, 0.8]
    assert _codec_grid("h264,h265") == ["h264", "h265"]
    assert _blend_ladder(0.4, 4) == pytest.approx([0.4, 0.3, 0.2, 0.1])


@pytest.mark.parametrize("raw", ["", "0", "1", "0.5,0.5", "bad"])
def test_guarded_probe_rejects_invalid_open_unit_grids(raw):
    with pytest.raises(ValueError):
        _float_grid(raw, "grid", open_unit=True)


@pytest.mark.parametrize("raw", ["", "av1", "h264,h264"])
def test_guarded_probe_rejects_invalid_codec_grids(raw):
    with pytest.raises(ValueError):
        _codec_grid(raw)


def test_motion_tube_mask_has_exact_budget_and_is_temporally_stable():
    video = torch.zeros(1, 3, 3, 4, 4)
    video[:, :, 1:, :2, :2] = 1.0

    mask = motion_tube_mask(video, 0.25)

    assert mask.shape == (1, 1, 3, 4, 4)
    assert mask[:, :, 0].sum().item() == 4
    assert torch.equal(mask[:, :, 0], mask[:, :, 1])
    assert torch.equal(mask[:, :, 1], mask[:, :, 2])


def test_guard_chooses_strongest_passing_candidate_without_true_labels():
    source = torch.tensor([[4.0, 0.0], [0.0, 4.0]])
    candidates = torch.tensor(
        [
            [[0.0, 4.0], [0.0, 3.8]],  # sample 0 flips; sample 1 passes
            [[3.9, 0.0], [2.0, 2.1]],  # sample 0 passes; sample 1 too weak
        ]
    )

    choices = guard_choices(source, candidates, retention=0.95)

    assert choices.tolist() == [1, 0]


def test_guard_returns_identity_when_no_candidate_passes():
    source = torch.tensor([[4.0, 0.0]])
    candidates = torch.tensor([[[0.0, 4.0]], [[0.1, 3.0]]])

    assert guard_choices(source, candidates, retention=0.95).tolist() == [-1]
