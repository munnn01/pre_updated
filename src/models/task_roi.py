"""Label-free task-importance maps converted to codec-aligned rectangles."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from src.codecs.roi import ROIRect


def top_fraction_mask(score: torch.Tensor, fraction: float) -> torch.Tensor:
    """Return an exact-budget binary mask for ``[B,1,1,H,W]`` scores."""
    if score.ndim != 5 or score.shape[1:3] != (1, 1):
        raise ValueError("score must have shape [B,1,1,H,W]")
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0,1)")
    flat = score.reshape(score.shape[0], -1)
    count = max(1, int(round(flat.shape[1] * fraction)))
    indices = flat.topk(count, dim=1, largest=True, sorted=False).indices
    return torch.zeros_like(flat).scatter_(1, indices, 1.0).reshape_as(score)


def motion_score(video: torch.Tensor) -> torch.Tensor:
    """Stable temporal motion score with shape ``[B,1,1,H,W]``."""
    if video.ndim != 5 or video.shape[1] != 3:
        raise ValueError("video must have shape [B,3,T,H,W]")
    if video.shape[2] < 2:
        return torch.zeros(
            video.shape[0],
            1,
            1,
            video.shape[3],
            video.shape[4],
            device=video.device,
            dtype=video.dtype,
        )
    delta = (video[:, :, 1:] - video[:, :, :-1]).abs().mean(dim=1, keepdim=True)
    return delta.amax(dim=2, keepdim=True)


def _normalise(score: torch.Tensor) -> torch.Tensor:
    flat = score.flatten(1)
    low = flat.amin(dim=1).view(-1, 1, 1, 1, 1)
    high = flat.amax(dim=1).view(-1, 1, 1, 1, 1)
    return (score - low) / (high - low).clamp_min(1e-8)


def action_importance_score(
    saliency: torch.Tensor,
    video: torch.Tensor,
    *,
    saliency_weight: float = 0.65,
) -> torch.Tensor:
    """Fuse label-free teacher saliency with a stable motion tube score."""
    if saliency.ndim != 5 or saliency.shape[1] != 1:
        raise ValueError("saliency must have shape [B,1,T,H,W]")
    if not 0.0 <= saliency_weight <= 1.0:
        raise ValueError("saliency_weight must be in [0,1]")
    saliency_tube = saliency.amax(dim=2, keepdim=True)
    motion = motion_score(video)
    if saliency_tube.shape[-2:] != motion.shape[-2:]:
        saliency_tube = F.interpolate(
            saliency_tube.squeeze(2),
            motion.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).unsqueeze(2)
    return (
        saliency_weight * _normalise(saliency_tube) + (1.0 - saliency_weight) * _normalise(motion)
    ).detach()


def _window_shape(rows: int, cols: int, budget: float) -> tuple[int, int]:
    limit = max(1, int(math.floor(rows * cols * budget + 1e-9)))
    candidates = [
        (height, width)
        for height in range(1, rows + 1)
        for width in range(1, cols + 1)
        if height * width <= limit
    ]
    frame_aspect = cols / rows
    # Use as much of the preregistered area budget as possible, then prefer a
    # window whose shape resembles the frame instead of a thin strip.
    return max(
        candidates,
        key=lambda shape: (
            shape[0] * shape[1],
            -abs(math.log((shape[1] / shape[0]) / frame_aspect)),
        ),
    )


def importance_window(
    score: torch.Tensor,
    budget: float,
    *,
    block: int = 16,
    delta_qp: int = 0,
) -> ROIRect:
    """Find the highest-scoring block-aligned rectangle within an area cap.

    ``score`` may be ``[H,W]`` or ``[1,1,1,H,W]``.  A single rectangle is used
    for every frame, so no time-varying mask is sent to the decoder.
    """
    if score.ndim == 5:
        if score.shape[:3] != (1, 1, 1):
            raise ValueError("batched score must have shape [1,1,1,H,W]")
        score = score[0, 0, 0]
    if score.ndim != 2:
        raise ValueError("score must be [H,W]")
    if not 0.0 < budget <= 0.65:
        raise ValueError("budget must be in (0,0.65]")
    if block <= 0:
        raise ValueError("block must be positive")
    height, width = (int(value) for value in score.shape)
    if height % block or width % block:
        raise ValueError("frame dimensions must be divisible by block")
    grid = F.avg_pool2d(score[None, None].float(), kernel_size=block, stride=block)[0, 0]
    rows, cols = (int(value) for value in grid.shape)
    win_rows, win_cols = _window_shape(rows, cols, budget)
    sums = F.conv2d(
        grid[None, None],
        torch.ones(1, 1, win_rows, win_cols, device=grid.device),
    )[0, 0]
    flat_index = int(sums.argmax().item())
    out_cols = int(sums.shape[1])
    row, col = divmod(flat_index, out_cols)
    return ROIRect(
        x=col * block,
        y=row * block,
        width=win_cols * block,
        height=win_rows * block,
        delta_qp=delta_qp,
    )


def spatial_roi_regions(
    protected: ROIRect,
    frame_width: int,
    frame_height: int,
    *,
    background_delta_qp: int,
) -> list[ROIRect]:
    """Protected region first, then full-frame background (first match wins)."""
    protected.validate(frame_width, frame_height)
    return [
        protected,
        ROIRect(0, 0, frame_width, frame_height, background_delta_qp),
    ]
