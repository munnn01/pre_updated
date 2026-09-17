"""Shared spatial/temporal suppression primitive for OD and action recognition.

At ``T=1`` this is the detector-mask R0 transform.  For video, detections are
expanded into short object tubes and only static, unprotected background is
pulled toward the previous processed frame.  The protected core remains an
exact identity, so neither the spatial nor temporal branch can rewrite objects.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mask_suppress import protect_mask, suppress


def _as_video_mask(mask: torch.Tensor, x: torch.Tensor | None = None) -> torch.Tensor:
    if mask.ndim == 4:
        mask = mask.unsqueeze(2)
    if mask.ndim != 5 or mask.shape[1] != 1:
        raise ValueError(f"expected mask [B,1,T,H,W] or [B,1,H,W], got {tuple(mask.shape)}")
    if x is not None:
        if x.ndim != 5 or x.shape[1] != 3:
            raise ValueError(f"expected video [B,3,T,H,W], got {tuple(x.shape)}")
        if mask.shape[0] not in (1, x.shape[0]) or mask.shape[2:] != x.shape[2:]:
            raise ValueError(f"mask {tuple(mask.shape)} is incompatible with video {tuple(x.shape)}")
    return mask.clamp(0.0, 1.0)


def expand_importance_tube(mask: torch.Tensor, temporal_radius: int = 1) -> torch.Tensor:
    """Protect detections in neighbouring frames to bridge one-frame misses."""
    mask = _as_video_mask(mask)
    radius = int(temporal_radius)
    if radius < 0:
        raise ValueError("temporal_radius must be non-negative")
    if radius == 0 or mask.shape[2] == 1:
        return mask
    return F.max_pool3d(mask, kernel_size=(2 * radius + 1, 1, 1),
                        stride=1, padding=(radius, 0, 0))


def feather_protection(mask: torch.Tensor, radius: int = 4) -> torch.Tensor:
    """Add a soft protection band while keeping the original core exactly one."""
    mask = _as_video_mask(mask)
    radius = int(radius)
    if radius < 0:
        raise ValueError("feather radius must be non-negative")
    if radius == 0:
        return mask
    kernel = 2 * radius + 1
    expanded = F.max_pool3d(mask, kernel_size=(1, kernel, kernel), stride=1,
                            padding=(0, radius, radius))
    soft = F.avg_pool3d(expanded, kernel_size=(1, kernel, kernel), stride=1,
                        padding=(0, radius, radius))
    return torch.where(mask >= 1.0, torch.ones_like(mask), soft).clamp(0.0, 1.0)


def masks_from_frame_detections(
    detections: Sequence[dict],
    size: int,
    score_thresh: float = 0.5,
    dilate: float = 0.15,
) -> torch.Tensor:
    """Stack torchvision detections into one ``[1,1,T,H,W]`` protection tube."""
    if not detections:
        raise ValueError("detections must contain at least one frame")
    masks = [
        protect_mask(d["boxes"], d["scores"], d["labels"], size,
                     score_thresh=score_thresh, dilate=dilate).unsqueeze(2)
        for d in detections
    ]
    return torch.cat(masks, dim=2)


class ImportanceTubeSuppress(nn.Module):
    """Parameter-free spatial suppression plus motion-gated background stability.

    ``mask=1`` is an exact identity.  Temporal stabilization is strongest only
    where consecutive source frames are already similar, limiting ghost trails
    around moving people and manipulated objects.
    """

    def __init__(self, sigma: float = 8.0, temporal_radius: int = 1,
                 feather: int = 4, temporal_strength: float = 0.5,
                 motion_tau: float = 0.05):
        super().__init__()
        if not 0.0 <= temporal_strength <= 1.0:
            raise ValueError("temporal_strength must be in [0,1]")
        if motion_tau <= 0:
            raise ValueError("motion_tau must be positive")
        self.sigma = float(sigma)
        self.temporal_radius = int(temporal_radius)
        self.feather = int(feather)
        self.temporal_strength = float(temporal_strength)
        self.motion_tau = float(motion_tau)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = _as_video_mask(mask, x)
        if mask.shape[0] == 1 and x.shape[0] > 1:
            mask = mask.expand(x.shape[0], -1, -1, -1, -1)
        core = mask
        protection = feather_protection(
            expand_importance_tube(mask, self.temporal_radius), self.feather
        )
        spatial = suppress(x, protection, self.sigma)
        if x.shape[2] == 1 or self.temporal_strength == 0.0:
            return x * core + spatial * (1.0 - core)

        frames = [x[:, :, 0] * core[:, :, 0]
                  + spatial[:, :, 0] * (1.0 - core[:, :, 0])]
        for t in range(1, x.shape[2]):
            motion = (x[:, :, t] - x[:, :, t - 1]).abs().mean(dim=1, keepdim=True)
            static = torch.exp(-motion / self.motion_tau)
            alpha = self.temporal_strength * static * (1.0 - protection[:, :, t])
            candidate = spatial[:, :, t] + alpha * (frames[-1] - spatial[:, :, t])
            out = x[:, :, t] * core[:, :, t] + candidate * (1.0 - core[:, :, t])
            frames.append(out)
        return torch.stack(frames, dim=2).clamp(0.0, 1.0)


__all__ = [
    "ImportanceTubeSuppress",
    "expand_importance_tube",
    "feather_protection",
    "masks_from_frame_detections",
]
