import pytest

from ops.probe_action_post import _arm, _codec_grid, _float_grid


def test_action_post_arm_and_grids_are_stable():
    assert _arm(1.0, 45) == "post1q45"
    assert _float_grid("1,2,1") == [1.0, 2.0]
    assert _codec_grid("h264,h265") == ["h264", "h265"]


@pytest.mark.parametrize("raw", ["", "0", "-1"])
def test_action_post_rejects_non_positive_sigmas(raw):
    with pytest.raises(ValueError):
        _float_grid(raw)


@pytest.mark.parametrize("raw", ["", "av1", "h264,h264"])
def test_action_post_rejects_invalid_codecs(raw):
    with pytest.raises(ValueError):
        _codec_grid(raw)
