import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ops"))

from ops.probe_background_suppression import _codec_grid


def test_codec_grid_accepts_one_or_both_codecs():
    assert _codec_grid("h264") == ["h264"]
    assert _codec_grid("h265,h264") == ["h265", "h264"]


@pytest.mark.parametrize("raw", ["", "h266", "h264,h264"])
def test_codec_grid_rejects_empty_unknown_or_duplicate_values(raw):
    with pytest.raises(ValueError):
        _codec_grid(raw)
