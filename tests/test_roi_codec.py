from fractions import Fraction

from src.codecs.roi import (
    ROICodec,
    ROIRect,
    build_addroi_filter,
    delta_qp_to_qoffset,
)


def test_delta_qp_is_persisted_as_exact_rational():
    assert delta_qp_to_qoffset(-2) == Fraction(-2, 51)
    assert delta_qp_to_qoffset(6) == Fraction(2, 17)


def test_filter_preserves_first_region_priority():
    protected = ROIRect(16, 16, 32, 32, -2)
    background = ROIRect(0, 0, 64, 64, 6)
    chain = build_addroi_filter([protected, background], 64, 64)

    assert chain.split(",")[0].startswith("addroi=x=16:y=16")
    assert "qoffset=-2/51" in chain
    assert chain.split(",")[1].startswith("addroi=x=0:y=0:w=64:h=64")


def test_v3_command_uses_crf_aq_and_warning_logs(tmp_path):
    command = ROICodec("h264", crf=36, aq_mode=2).encoder_command(
        64,
        64,
        tmp_path / "clip.264",
        [ROIRect(0, 0, 64, 64, 3)],
    )

    assert "-crf" in command and command[command.index("-crf") + 1] == "36"
    assert "-qp" not in command
    assert command[command.index("-loglevel") + 1] == "warning"
    params = command[command.index("-x264-params") + 1]
    assert "aq-mode=2" in params
    assert "aq-strength=1" in params


def test_rectangle_must_fit_frame():
    try:
        build_addroi_filter([ROIRect(48, 0, 32, 16, 0)], 64, 64)
    except ValueError as exc:
        assert "exceeds" in str(exc)
    else:
        raise AssertionError("out-of-frame ROI was accepted")
