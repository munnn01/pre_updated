"""Standard-codec anchors: bare H.264 (x264) and H.265 (x265) via ffmpeg.

These are the *baselines* the preprocessor+CompressAI pipeline is compared
against. A clip is piped to ffmpeg as raw RGB, encoded at a constant QP, and
decoded straight back to raw RGB. We measure the real coded size, so the
reported bpp is honest and directly comparable to CompressAI's entropy-coded
bpp.

    bpp = 8 * encoded_file_bytes / (T * H * W)

No neural network is involved -- this is exactly "what you get from H.264/H.265
alone", which is the anchor curve for the BD-Rate comparison.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

_ENCODER = {"h264": "libx264", "h265": "libx265"}
_MUXER = {"h264": "h264", "h265": "hevc"}  # raw-bitstream muxer (NOT "264"/"265")
_PICTURE_TYPE_ID = {"I": 0, "P": 1, "B": 2}


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


class StandardCodec:
    """ffmpeg-backed constant-QP H.264 / H.265 codec.

    Args:
        codec: "h264" or "h265".
        qp: constant quantisation parameter (higher = lower rate).
        preset: x264/x265 speed preset.
        fps: container frame-rate (only affects timing metadata, not quality).
    """

    def __init__(
        self,
        codec: str = "h264",
        qp: int = 35,
        preset: str = "medium",
        fps: int = 25,
    ):
        if codec not in _ENCODER:
            raise ValueError(f"codec must be one of {list(_ENCODER)}")
        self.codec = codec
        self.qp = qp
        self.preset = preset
        self.fps = fps

    # -- single clip -------------------------------------------------------
    def _encode_decode_clip(
        self, clip: np.ndarray, qp: int | None = None
    ) -> Tuple[np.ndarray, float]:
        """clip: [T,H,W,C] uint8 RGB. Returns (recon [T,H,W,C] uint8, bpp)."""
        recon, bpp, _ = self._encode_decode_clip_with_metadata(clip, qp=qp)
        return recon, bpp

    def _encode_decode_clip_with_metadata(
        self, clip: np.ndarray, qp: int | None = None
    ) -> Tuple[np.ndarray, float, List[int]]:
        """Encode/decode one clip and expose the realised I/P/B picture types.

        Picture types come from the actual elementary stream through ffprobe;
        this is deliberately different from inferring a GOP pattern from frame
        indices.  The returned ids are I=0, P=1, B=2, other/unknown=3.
        """
        t, h, w, c = clip.shape
        assert c == 3
        qp = self.qp if qp is None else qp
        with tempfile.TemporaryDirectory() as td:
            bitstream = Path(td) / f"clip.{'264' if self.codec == 'h264' else '265'}"

            # RGB -> encoded elementary stream.
            enc = subprocess.run(
                [
                    "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
                    "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{w}x{h}", "-r", str(self.fps), "-i", "pipe:0",
                    "-c:v", _ENCODER[self.codec],
                    "-preset", self.preset, "-qp", str(qp),
                    "-pix_fmt", "yuv420p",
                    "-f", _MUXER[self.codec],
                    str(bitstream),
                ],
                input=clip.tobytes(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True,
            )
            _ = enc
            coded_bytes = bitstream.stat().st_size

            probe = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "frame=pict_type", "-of", "csv=p=0",
                    str(bitstream),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            picture_types = [
                _PICTURE_TYPE_ID.get(line.strip().split(",")[0], 3)
                for line in probe.stdout.splitlines()
                if line.strip()
            ]

            # Encoded stream -> raw RGB back.
            dec = subprocess.run(
                [
                    "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
                    "-i", str(bitstream),
                    "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            raw = np.frombuffer(dec.stdout, dtype=np.uint8)

        n = raw.size // (h * w * c)
        if n == 0:
            recon = clip.copy()
        else:
            recon = raw[: n * h * w * c].reshape(n, h, w, c)
            if n < t:  # pad short decode by repeating last frame
                pad = np.repeat(recon[-1:], t - n, axis=0)
                recon = np.concatenate([recon, pad], axis=0)
            recon = recon[:t]
        if not picture_types:
            picture_types = [3] * t
        elif len(picture_types) < t:
            picture_types.extend([picture_types[-1]] * (t - len(picture_types)))
        picture_types = picture_types[:t]
        bpp = 8.0 * coded_bytes / (t * h * w)
        return recon, bpp, picture_types

    # -- batch of clips ----------------------------------------------------
    @torch.no_grad()
    def compress_decompress_items(
        self, x: torch.Tensor, qp: int | None = None
    ) -> Tuple[torch.Tensor, List[float]]:
        """Encode a batch and return the real bpp of every input sequence."""
        b, c, t, h, w = x.shape
        arr = (x.clamp(0, 1) * 255).round().byte().cpu().numpy()  # [B,C,T,H,W]
        recons = []
        bpps = []
        for i in range(b):
            clip = np.transpose(arr[i], (1, 2, 3, 0))  # [T,H,W,C]
            rec, bpp = self._encode_decode_clip(clip, qp=qp)
            recons.append(np.transpose(rec, (3, 0, 1, 2)))  # [C,T,H,W]
            bpps.append(bpp)
        out = torch.from_numpy(np.stack(recons)).float().div_(255.0).to(x.device)
        return out, [float(v) for v in bpps]

    @torch.no_grad()
    def compress_decompress_items_with_metadata(
        self, x: torch.Tensor, qp: int | None = None
    ) -> Tuple[torch.Tensor, List[float], torch.Tensor]:
        """Real codec batch plus per-frame picture-type ids ``[B,T]``."""
        b, c, t, h, w = x.shape
        arr = (x.clamp(0, 1) * 255).round().byte().cpu().numpy()
        recons, bpps, picture_types = [], [], []
        for i in range(b):
            clip = np.transpose(arr[i], (1, 2, 3, 0))
            rec, bpp, types = self._encode_decode_clip_with_metadata(clip, qp=qp)
            recons.append(np.transpose(rec, (3, 0, 1, 2)))
            bpps.append(float(bpp))
            picture_types.append(types)
        out = torch.from_numpy(np.stack(recons)).float().div_(255.0).to(x.device)
        types = torch.tensor(picture_types, dtype=torch.long, device=x.device)
        return out, bpps, types

    @torch.no_grad()
    def compress_decompress(
        self, x: torch.Tensor, qp: int | None = None
    ) -> Tuple[torch.Tensor, float]:
        """Encode a batch and return its mean bpp (backward-compatible API)."""
        out, bpps = self.compress_decompress_items(x, qp=qp)
        return out, float(np.mean(bpps))
