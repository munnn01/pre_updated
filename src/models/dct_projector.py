"""Codec/QP-conditioned weak-frequency projection for action recognition.

CRC-V5 proved that a free additive residual can improve recognition while still
adding 21--33% real-codec bitrate.  This module keeps the warm-started semantic
editor, but projects its output into a rate-reducing family:

* only weak high-frequency 8x8 DCT coefficients are attenuated;
* strong edges and every low-frequency coefficient are preserved;
* an optional causal temporal projection only blends pixels that are already
  static in the source clip;
* projection strength decreases toward high QP, where recognition is fragile,
  and can differ between H.264 and H.265.

The projection has no learned parameters.  Its state dict is therefore exactly
load-compatible with :class:`AdditiveCondPreprocessor`; experiment arms differ
only through checkpointed configuration, not hidden weights.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .additive_cond import AdditiveCondPreprocessor
from .virtual_codec import _dct_basis


class DCTProjectedAdditivePreprocessor(AdditiveCondPreprocessor):
    """Warm-started semantic residual followed by bounded rate projection."""

    def __init__(
        self,
        temporal_frames: int = 8,
        strength: float = 1.0,
        cond_dim: int = 3,
        residual_scale: float = 0.0,
        dct_strength: float = 1.0,
        dct_threshold: float = 1.0,
        dct_softness: float = 0.25,
        dct_block: int = 8,
        dct_band_start: int = 4,
        temporal_strength: float = 0.0,
        motion_tau: float = 0.05,
        qp_slope: float = 0.65,
        h264_scale: float = 1.0,
        h265_scale: float = 1.0,
    ):
        super().__init__(temporal_frames=temporal_frames, strength=strength, cond_dim=cond_dim)
        if cond_dim != 3:
            raise ValueError("DCT projection requires cond_dim=3 ([QP,h264,h265])")
        bounded = {
            "residual_scale": residual_scale,
            "dct_strength": dct_strength,
            "temporal_strength": temporal_strength,
            "qp_slope": qp_slope,
            "h264_scale": h264_scale,
            "h265_scale": h265_scale,
        }
        if any(not 0.0 <= float(value) <= 1.0 for value in bounded.values()):
            raise ValueError(f"projection strengths/scales must be in [0,1]: {bounded}")
        if dct_threshold <= 0.0 or dct_softness <= 0.0 or motion_tau <= 0.0:
            raise ValueError("dct_threshold, dct_softness, and motion_tau must be positive")
        if dct_block < 2 or not 1 <= dct_band_start <= 2 * (dct_block - 1):
            raise ValueError("invalid DCT block or band start")
        self.residual_scale = float(residual_scale)
        self.dct_strength = float(dct_strength)
        self.dct_threshold = float(dct_threshold)
        self.dct_softness = float(dct_softness)
        self.dct_block = int(dct_block)
        self.dct_band_start = int(dct_band_start)
        self.temporal_strength = float(temporal_strength)
        self.motion_tau = float(motion_tau)
        self.qp_slope = float(qp_slope)
        self.h264_scale = float(h264_scale)
        self.h265_scale = float(h265_scale)

    def _conditioned_strength(self, cond: torch.Tensor, base: float) -> torch.Tensor:
        """Return one bounded strength per clip from [QP,h264,h265]."""
        qp = cond[:, 0].clamp(0.0, 1.0)
        codec_scale = (
            cond[:, 1].clamp(0.0, 1.0) * self.h264_scale
            + cond[:, 2].clamp(0.0, 1.0) * self.h265_scale
        )
        qp_scale = (1.0 - self.qp_slope * qp).clamp(0.0, 1.0)
        return (float(base) * codec_scale * qp_scale).clamp(0.0, 1.0)

    def _spatial_project(self, frames: torch.Tensor, strength: torch.Tensor) -> torch.Tensor:
        if self.dct_strength == 0.0:
            return frames
        n, c, h, w = frames.shape
        block = self.dct_block
        pad_h = (-h) % block
        pad_w = (-w) % block
        padded = F.pad(frames, (0, pad_w, 0, pad_h), mode="replicate")
        hp, wp = padded.shape[-2:]
        blocks = padded.unfold(-2, block, block).unfold(-2, block, block)
        nh, nw = blocks.shape[2], blocks.shape[3]
        flat = blocks.contiguous().reshape(-1, block, block)
        basis = _dct_basis(block).to(device=frames.device, dtype=frames.dtype)
        coeff = basis @ flat @ basis.transpose(0, 1)
        coeff = coeff.reshape(n, c, nh, nw, block, block)

        idx = torch.arange(block, device=frames.device)
        high = (idx[:, None] + idx[None, :]) >= self.dct_band_start
        high_f = high.to(frames.dtype).view(1, 1, 1, 1, block, block)
        magnitude = coeff.abs()
        selected = magnitude * high_f
        mean = selected.sum(dim=(-2, -1), keepdim=True) / high.sum().clamp_min(1)
        cutoff = self.dct_threshold * mean
        softness = self.dct_softness * mean + 1e-6
        weak = torch.sigmoid((cutoff - magnitude) / softness) * high_f
        attenuate = strength.view(n, 1, 1, 1, 1, 1) * weak
        coeff = coeff * (1.0 - attenuate)

        flat = coeff.reshape(-1, block, block)
        restored = basis.transpose(0, 1) @ flat @ basis
        restored = restored.reshape(n, c, nh, nw, block, block)
        restored = restored.permute(0, 1, 2, 4, 3, 5).reshape(n, c, hp, wp)
        return restored[..., :h, :w].clamp(0.0, 1.0)

    def _temporal_project(
        self,
        source: torch.Tensor,
        spatial: torch.Tensor,
        strength: torch.Tensor,
    ) -> torch.Tensor:
        if self.temporal_strength == 0.0 or source.shape[2] < 2:
            return spatial
        outputs = [spatial[:, :, 0]]
        for index in range(1, source.shape[2]):
            motion = (source[:, :, index] - source[:, :, index - 1]).abs().mean(
                dim=1, keepdim=True
            )
            static = torch.exp(-motion / self.motion_tau)
            alpha = strength[:, None, None, None] * static
            current = spatial[:, :, index]
            outputs.append(current + alpha * (outputs[-1] - current))
        return torch.stack(outputs, dim=2).clamp(0.0, 1.0)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if cond is None:
            cond = x.new_zeros(x.shape[0], self.cond_dim)
            cond[:, 1] = 1.0
        edited = super().forward(x, cond, mask=mask)
        candidate = x + self.residual_scale * (edited - x)
        b, c, t, h, w = candidate.shape
        frames = candidate.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        spatial_strength = self._conditioned_strength(cond, self.dct_strength)
        spatial_strength = spatial_strength.repeat_interleave(t)
        spatial = self._spatial_project(frames, spatial_strength)
        spatial = spatial.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        temporal_strength = self._conditioned_strength(cond, self.temporal_strength)
        return self._temporal_project(x, spatial, temporal_strength)


__all__ = ["DCTProjectedAdditivePreprocessor"]
