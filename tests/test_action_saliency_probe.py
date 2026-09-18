import pytest
from ops.probe_action_saliency import (
    _arm,
    _codec_grid,
    _mode_grid,
    _protect_grid,
)


def test_action_saliency_names_and_grids_are_stable():
    assert _arm(0.25, "tube", 8.0, 0.75) == "sal25_tube_s8_t0.75"
    assert _protect_grid("0.15,0.25,0.4") == [0.15, 0.25, 0.4]
    assert _mode_grid("clip,tube") == ["clip", "tube"]
    assert _codec_grid("h264,h265") == ["h264", "h265"]


@pytest.mark.parametrize("raw", ["", "0", "1", "0.25,0.25", "bad"])
def test_action_saliency_rejects_invalid_protect_grid(raw):
    with pytest.raises(ValueError):
        _protect_grid(raw)


@pytest.mark.parametrize("raw", ["", "frame", "clip,clip"])
def test_action_saliency_rejects_invalid_modes(raw):
    with pytest.raises(ValueError):
        _mode_grid(raw)


@pytest.mark.parametrize("raw", ["", "av1", "h264,h264"])
def test_action_saliency_rejects_invalid_codecs(raw):
    with pytest.raises(ValueError):
        _codec_grid(raw)
