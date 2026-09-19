"""Codec-native region-of-interest encoding for the V3 experiments.

Unlike :mod:`src.codecs.standard`, this adapter deliberately uses CRF with
adaptive quantisation.  FFmpeg's x264 wrapper rejects ROI side data in CQP
mode, so the legacy constant-QP codec must not be used as the V3 anchor.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

_ENCODER = {"h264": "libx264", "h265": "libx265"}
_MUXER = {"h264": "h264", "h265": "hevc"}
_ROI_REJECTION = re.compile(
    r"(?:skipping|ignoring|discarding).{0,80}(?:roi|region)|"
    r"(?:roi|region).{0,80}(?:unsupported|not supported|disabled|requires? aq)",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class ROIRect:
    """A block-aligned ROI rectangle and its requested semantic QP offset."""

    x: int
    y: int
    width: int
    height: int
    delta_qp: int = 0

    def validate(self, frame_width: int, frame_height: int) -> None:
        if min(self.x, self.y, self.width, self.height) < 0:
            raise ValueError("ROI coordinates and dimensions must be non-negative")
        if self.width == 0 or self.height == 0:
            raise ValueError("ROI width and height must be positive")
        if self.x + self.width > frame_width or self.y + self.height > frame_height:
            raise ValueError("ROI rectangle exceeds the frame")
        if not -51 <= self.delta_qp <= 51:
            raise ValueError("delta_qp must be in [-51, 51]")


@dataclass
class ROIEncodeResult:
    reconstruction: np.ndarray
    bpp: float
    coded_bytes: int
    stderr: str
    command: list[str]
    regions: list[dict]


def delta_qp_to_qoffset(delta_qp: int, qp_range: int = 51) -> Fraction:
    """Translate a semantic QP delta to FFmpeg's rational qoffset.

    Both 8-bit x264 and x265 apply the side-data offset over a 51-step QP
    range.  The requested delta and this rational are both persisted because
    encoder AQ can make the realised block QP differ from the request.
    """
    if qp_range <= 0:
        raise ValueError("qp_range must be positive")
    if not -qp_range <= delta_qp <= qp_range:
        raise ValueError(f"delta_qp must be in [-{qp_range}, {qp_range}]")
    return Fraction(int(delta_qp), int(qp_range))


def build_addroi_filter(regions: Sequence[ROIRect], frame_width: int, frame_height: int) -> str:
    """Build an ordered ``addroi`` chain; first matching region wins."""
    filters: list[str] = []
    for region in regions:
        region.validate(frame_width, frame_height)
        offset = delta_qp_to_qoffset(region.delta_qp)
        filters.append(
            "addroi="
            f"x={region.x}:y={region.y}:w={region.width}:h={region.height}:"
            f"qoffset={offset.numerator}/{offset.denominator}"
        )
    return ",".join(filters)


def codec_capabilities() -> dict:
    """Inspect the active FFmpeg binary without assuming a Kaggle build."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return {
            "ffmpeg": None,
            "addroi": False,
            "libx264": False,
            "libx265": False,
        }
    filters = subprocess.run(
        [ffmpeg, "-hide_banner", "-filters"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    ).stdout
    encoders = subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    ).stdout
    return {
        "ffmpeg": ffmpeg,
        "addroi": bool(re.search(r"\baddroi\b", filters)),
        "libx264": bool(re.search(r"\blibx264\b", encoders)),
        "libx265": bool(re.search(r"\blibx265\b", encoders)),
    }


class ROICodec:
    """FFmpeg CRF+AQ codec with ordered ROI side-data and strict log checks."""

    def __init__(
        self,
        codec: str = "h264",
        crf: int = 35,
        preset: str = "medium",
        fps: int = 25,
        aq_mode: int = 2,
        aq_strength: float = 1.0,
        gop: int = 32,
    ) -> None:
        if codec not in _ENCODER:
            raise ValueError(f"codec must be one of {list(_ENCODER)}")
        if not 0 <= crf <= 51:
            raise ValueError("crf must be in [0, 51]")
        if aq_mode <= 0:
            raise ValueError("AQ must be enabled for ROI experiments")
        if aq_strength <= 0 or gop <= 0 or fps <= 0:
            raise ValueError("aq_strength, gop, and fps must be positive")
        self.codec = codec
        self.crf = int(crf)
        self.preset = preset
        self.fps = int(fps)
        self.aq_mode = int(aq_mode)
        self.aq_strength = float(aq_strength)
        self.gop = int(gop)

    def encoder_command(
        self,
        width: int,
        height: int,
        bitstream: str | Path,
        regions: Sequence[ROIRect] = (),
        *,
        crf: int | None = None,
    ) -> list[str]:
        value = self.crf if crf is None else int(crf)
        if not 0 <= value <= 51:
            raise ValueError("crf must be in [0, 51]")
        params = (
            f"aq-mode={self.aq_mode}:aq-strength={self.aq_strength:g}:"
            f"keyint={self.gop}:min-keyint={self.gop}:scenecut=0"
        )
        command = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(self.fps),
            "-i",
            "pipe:0",
        ]
        if regions:
            command += ["-vf", build_addroi_filter(regions, width, height)]
        command += [
            "-c:v",
            _ENCODER[self.codec],
            "-preset",
            self.preset,
            "-crf",
            str(value),
            "-x264-params" if self.codec == "h264" else "-x265-params",
            params,
            "-pix_fmt",
            "yuv420p",
            "-g",
            str(self.gop),
            "-f",
            _MUXER[self.codec],
            str(bitstream),
        ]
        return command

    def _encode_decode_clip(
        self,
        clip: np.ndarray,
        regions: Sequence[ROIRect] = (),
        *,
        crf: int | None = None,
    ) -> ROIEncodeResult:
        if clip.ndim != 4 or clip.shape[-1] != 3 or clip.dtype != np.uint8:
            raise ValueError("clip must be uint8 [T,H,W,3] RGB")
        t, height, width, _ = clip.shape
        with tempfile.TemporaryDirectory() as temp_dir:
            suffix = "264" if self.codec == "h264" else "265"
            bitstream = Path(temp_dir) / f"clip.{suffix}"
            command = self.encoder_command(width, height, bitstream, regions, crf=crf)
            encoded = subprocess.run(
                command,
                input=clip.tobytes(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=False,
            )
            stderr = encoded.stderr.decode("utf-8", errors="replace")
            if encoded.returncode:
                raise RuntimeError(f"FFmpeg ROI encode failed (rc={encoded.returncode}):\n{stderr}")
            if regions and _ROI_REJECTION.search(stderr):
                raise RuntimeError(f"FFmpeg rejected ROI side data:\n{stderr}")
            coded_bytes = bitstream.stat().st_size
            decoded = subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(bitstream),
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "pipe:1",
                ],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                check=False,
            )
            if decoded.returncode:
                message = decoded.stderr.decode("utf-8", errors="replace")
                raise RuntimeError(f"FFmpeg ROI decode failed:\n{message}")
            raw = np.frombuffer(decoded.stdout, dtype=np.uint8)

        frame_values = height * width * 3
        n_frames = raw.size // frame_values
        if n_frames == 0:
            raise RuntimeError("FFmpeg decoded zero frames")
        reconstruction = raw[: n_frames * frame_values].reshape(n_frames, height, width, 3)
        if n_frames < t:
            reconstruction = np.concatenate(
                [
                    reconstruction,
                    np.repeat(reconstruction[-1:], t - n_frames, axis=0),
                ],
                axis=0,
            )
        reconstruction = reconstruction[:t]
        return ROIEncodeResult(
            reconstruction=reconstruction,
            bpp=8.0 * coded_bytes / (t * height * width),
            coded_bytes=coded_bytes,
            stderr=stderr,
            command=[str(item) for item in command],
            regions=[
                {
                    **asdict(region),
                    "qoffset": str(delta_qp_to_qoffset(region.delta_qp)),
                }
                for region in regions
            ],
        )
