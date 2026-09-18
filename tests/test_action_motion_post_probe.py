import pytest
from ops.probe_action_motion_post import (
    _codec_grid,
    _global_arm,
    _motion_arm,
    _quantile_grid,
)


def test_action_motion_post_names_and_grids_are_stable():
    assert _global_arm(1.0, 45) == "global1q45"
    assert _motion_arm(0.75, 1.0, 45) == "motionq75_s1q45"
    assert _quantile_grid("0.5,0.75,0.5") == [0.5, 0.75]
    assert _codec_grid("h264,h265") == ["h264", "h265"]


@pytest.mark.parametrize("raw", ["", "0", "1", "-0.1", "1.1"])
def test_action_motion_post_rejects_invalid_quantiles(raw):
    with pytest.raises(ValueError):
        _quantile_grid(raw)


@pytest.mark.parametrize("raw", ["", "av1", "h264,h264"])
def test_action_motion_post_rejects_invalid_codecs(raw):
    with pytest.raises(ValueError):
        _codec_grid(raw)
