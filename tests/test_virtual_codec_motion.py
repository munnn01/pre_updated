"""Checks for the motion-compensated virtual-codec proxy (needs torch).

The default (motion=False) path is the legacy zero-motion frame-difference
proxy and must be unchanged. With motion=True the proxy does block motion
estimation + compensation, so on TRANSLATING content the coded residual (hence
bpp) drops toward what a real inter-coder achieves, while on STATIC content the
chosen motion vectors are ~zero and the two paths agree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.virtual_codec import VirtualCodec


def _translating_clip(shift: int = 3, T: int = 5, size: int = 64) -> torch.Tensor:
    """A clip whose frames are a fixed texture translated by ``shift`` px/frame.
    Zero-motion prediction sees a large frame difference; block ME can track it."""
    torch.manual_seed(0)
    base = torch.rand(1, 3, size + T * shift + 8, size + T * shift + 8)
    # smooth a little so the content is codec-plausible (not per-pixel noise)
    base = F.avg_pool2d(base, kernel_size=3, stride=1, padding=1)
    frames = []
    for t in range(T):
        off = t * shift
        frames.append(base[:, :, off:off + size, off:off + size])
    return torch.stack(frames, dim=2)               # [1,3,T,size,size]


def _static_clip(T: int = 5, size: int = 64) -> torch.Tensor:
    torch.manual_seed(1)
    frame = F.avg_pool2d(torch.rand(1, 3, size, size), 3, 1, 1)
    return frame.unsqueeze(2).expand(1, 3, T, size, size).contiguous()


def test_motion_off_is_default() -> None:
    """Default construction leaves motion OFF (legacy behaviour preserved)."""
    assert VirtualCodec().motion is False


def test_motion_lowers_bpp_on_translation() -> None:
    """On a translating clip, motion compensation must cost fewer bits than the
    zero-motion frame-difference proxy at the same quality."""
    clip = _translating_clip()
    q = 5
    cod_off = VirtualCodec(motion=False, colorspace="yuv420")
    cod_on = VirtualCodec(motion=True, motion_range=8, motion_block=16,
                          colorspace="yuv420")
    _, bpp_off = cod_off.compress_decompress(clip, q)
    _, bpp_on = cod_on.compress_decompress(clip, q)
    assert bpp_on < bpp_off, f"motion bpp {bpp_on} should beat zero-motion {bpp_off}"


def test_motion_matches_on_static_content() -> None:
    """On a static clip the best motion vector is ~zero, so motion on/off give
    essentially the same bpp (chosen MV=0 reproduces the frame-difference path)."""
    clip = _static_clip()
    q = 5
    _, bpp_off = VirtualCodec(motion=False).compress_decompress(clip, q)
    _, bpp_on = VirtualCodec(motion=True, motion_range=8,
                             motion_block=16).compress_decompress(clip, q)
    assert abs(bpp_on - bpp_off) <= 1e-4 + 0.02 * abs(bpp_off), \
        f"static bpp should match: off={bpp_off} on={bpp_on}"


def test_motion_reconstruction_valid_and_differentiable() -> None:
    """Forward stays finite/differentiable with motion on; gradient reaches x."""
    clip = _translating_clip().clone().requires_grad_(True)
    cod = VirtualCodec(motion=True, motion_range=6, motion_block=16)
    x_hat, bpp = cod(clip, 3)                        # training forward
    assert x_hat.shape == clip.shape
    assert torch.isfinite(bpp)
    (x_hat.mean() + bpp).backward()
    assert clip.grad is not None and torch.isfinite(clip.grad).all()


def test_motion_rgb_path() -> None:
    """The legacy rgb colourspace path also honours motion compensation."""
    clip = _translating_clip()
    _, bpp_off = VirtualCodec(motion=False, colorspace="rgb").compress_decompress(clip, 5)
    _, bpp_on = VirtualCodec(motion=True, motion_range=8, motion_block=16,
                             colorspace="rgb").compress_decompress(clip, 5)
    assert bpp_on < bpp_off, f"rgb motion bpp {bpp_on} should beat {bpp_off}"


if __name__ == "__main__":
    test_motion_off_is_default()
    test_motion_lowers_bpp_on_translation()
    test_motion_matches_on_static_content()
    test_motion_reconstruction_valid_and_differentiable()
    test_motion_rgb_path()
    print("virtual_codec motion self-check passed")
