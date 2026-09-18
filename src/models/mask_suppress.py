"""Detector-mask protection + background suppression (the R0 preprocessor).

Philosophy, and why it is not another learned editor: every learned preprocessor
this project tried on images lost. The AR checkpoint costs ~12% more bits at
equal detection mAP, and the detection-trained one destroys 25% of the
detector's mAP *before the codec is involved* (mAP(pre)/mAP(x) = 0.75). Both
failures share a mechanism — a learned module may ADD or RESHAPE structure, and
detection mAP does not reward that. The detection literature's large numbers come
from the opposite move: remove background, keep objects (ROI-Packing −44%,
dual-region JPEG −26%). That needs no parameters at all.

So: the frozen detector runs on the SOURCE image (encoder-side analysis — the
encoder has the image and may analyse it freely; the decoder needs no side
information), its boxes are dilated into a protection mask, and everything
outside the mask is heavily blurred.  The default keeps object regions exact;
the optional dual-region arm applies only a declared, mild Gaussian there.

Both functions are deliberately stateless and parameter-free: there is nothing to
overfit the training proxy with, and nothing to select.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def protect_mask(boxes, scores, labels, size: int, score_thresh: float = 0.5,
                 dilate: float = 0.15, min_margin_px: float = 0.0,
                 grid: int = 1) -> torch.Tensor:
    """[1,1,S,S] mask, 1 = protect. Boxes are xyxy in the (already resized) frame.

    ``dilate`` is a fraction of each box's own size.  ``min_margin_px`` adds a
    lower bound to that context halo, which is important for small objects.
    ``grid`` optionally expands the final box to codec-friendly block boundaries.
    Both default to the historical behaviour (no fixed halo, one-pixel grid).
    Boxes below ``score_thresh`` are NOT protected: a detection the detector is
    unsure about should not buy bit budget.

    The mask is allocated on the BOXES' device — the detector returns CUDA
    tensors at eval time while the image may be on either device, and a CPU mask
    against a CUDA frame is a RuntimeError that a CPU-only local test cannot see
    (it cost one Kaggle cycle to learn that here).
    """
    if min_margin_px < 0:
        raise ValueError("min_margin_px must be non-negative")
    grid = int(grid)
    if grid <= 0:
        raise ValueError("grid must be positive")

    boxes_t = torch.as_tensor(boxes)
    scores_t = torch.as_tensor(scores).reshape(-1)
    m = torch.zeros(1, 1, size, size, device=boxes_t.device)
    keep = scores_t >= score_thresh
    for b in boxes_t.reshape(-1, 4)[keep]:
        x1, y1, x2, y2 = [float(v) for v in b]
        w, h = x2 - x1, y2 - y1
        pad_x = max(float(min_margin_px), dilate * w)
        pad_y = max(float(min_margin_px), dilate * h)
        x1, x2 = x1 - pad_x, x2 + pad_x
        y1, y2 = y1 - pad_y, y2 + pad_y
        x1 = max(0, int(math.floor(x1 / grid) * grid))
        y1 = max(0, int(math.floor(y1 / grid) * grid))
        x2 = min(size, int(math.ceil(x2 / grid) * grid))
        y2 = min(size, int(math.ceil(y2 / grid) * grid))
        if x2 > x1 and y2 > y1:
            m[:, :, y1:y2, x1:x2] = 1.0
    return m


def gaussian_filter(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Depthwise separable Gaussian filter for ``[B,C,T,H,W]`` tensors.

    The filter is depthwise (``groups=C``) so RGB is blurred channel-wise.
    A 1-channel kernel against a 3-channel input raises a RuntimeError, which is
    the shape of the bug that cost one debugging cycle in the probe — hence the
    explicit groups argument and the test that pins it.
    """
    if sigma <= 0:
        return x
    if x.ndim != 5:
        raise ValueError(f"expected [B,C,T,H,W], got {tuple(x.shape)}")
    k = int(2 * round(2 * sigma) + 1)
    c = x.shape[1]
    ax = torch.arange(k, dtype=x.dtype, device=x.device) - (k - 1) / 2
    g = torch.exp(-ax.pow(2) / (2 * sigma * sigma))
    g = g / g.sum()
    gx = g.view(1, 1, k, 1).expand(c, 1, k, 1).contiguous()
    gy = g.view(1, 1, 1, k).expand(c, 1, 1, k).contiguous()
    b, _, t, h, w = x.shape
    flat = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    pad = k // 2
    blur = F.conv2d(F.pad(flat, (pad, pad, 0, 0), mode="reflect"), gx, groups=c)
    blur = F.conv2d(F.pad(blur, (0, 0, pad, pad), mode="reflect"), gy, groups=c)
    return blur.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)


def motion_preserving_gaussian(
    x: torch.Tensor,
    sigma: float,
    motion_quantile: float = 0.75,
    dilation: int = 2,
    feather: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Blur temporally static pixels while preserving motion-bearing regions.

    This is a decoder-only, zero-bit AR post-filter.  Motion is estimated from
    adjacent decoded frames, so no detector, optical-flow model, or side
    information is required.  The highest-motion pixels are protected exactly;
    spatial dilation retains action context and a soft outer band avoids a hard
    compositing edge.

    Returns ``(filtered, protection_mask)`` where the mask has shape
    ``[B,1,T,H,W]`` and values in ``[0,1]``.
    """
    if x.ndim != 5:
        raise ValueError(f"expected [B,C,T,H,W], got {tuple(x.shape)}")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if not 0.0 < motion_quantile < 1.0:
        raise ValueError("motion_quantile must be between zero and one")
    if int(dilation) != dilation or dilation < 0:
        raise ValueError("dilation must be a non-negative integer")
    if int(feather) != feather or feather < 0:
        raise ValueError("feather must be a non-negative integer")

    dilation = int(dilation)
    feather = int(feather)
    b, _, t, _, _ = x.shape
    if t < 2:
        mask = torch.zeros(
            (b, 1, t, x.shape[-2], x.shape[-1]), dtype=x.dtype, device=x.device
        )
    else:
        delta = (x[:, :, 1:] - x[:, :, :-1]).abs().mean(dim=1, keepdim=True)
        motion = torch.zeros(
            (b, 1, t, x.shape[-2], x.shape[-1]), dtype=x.dtype, device=x.device
        )
        motion[:, :, :-1] = delta
        motion[:, :, 1:] = torch.maximum(motion[:, :, 1:], delta)
        thresholds = torch.quantile(
            motion.float().flatten(1), motion_quantile, dim=1, keepdim=True
        ).to(dtype=x.dtype).view(b, 1, 1, 1, 1)
        # Strict comparison makes a truly static clip select no motion pixels.
        mask = (motion > thresholds).to(dtype=x.dtype)

    if dilation:
        radius = dilation
        mask = F.max_pool3d(
            mask,
            kernel_size=(3, 2 * radius + 1, 2 * radius + 1),
            stride=1,
            padding=(1, radius, radius),
        )
    if feather:
        radius = feather
        soft = F.avg_pool3d(
            mask,
            kernel_size=(1, 2 * radius + 1, 2 * radius + 1),
            stride=1,
            padding=(0, radius, radius),
        )
        mask = torch.maximum(mask, soft).clamp_(0.0, 1.0)

    blurred = gaussian_filter(x, sigma)
    return x * mask + blurred * (1.0 - mask), mask


def dual_region_suppress(
    x: torch.Tensor,
    mask: torch.Tensor,
    background_sigma: float,
    roi_sigma: float = 0.0,
) -> torch.Tensor:
    """Filter ROI and background independently, then blend with ``mask``.

    ``roi_sigma=0`` is exactly the original identity-ROI R0 transform.  A small
    positive ROI sigma tests the dual-region hypothesis without changing the
    bitstream or transmitting the mask.  Soft feather masks remain valid.
    """
    roi = gaussian_filter(x, roi_sigma)
    background = gaussian_filter(x, background_sigma)
    m = mask.expand_as(x)
    return roi * m + background * (1 - m)


def suppress(x: torch.Tensor, mask: torch.Tensor, sigma: float) -> torch.Tensor:
    """Heavy blur outside the mask; protected regions are byte-identical."""
    return dual_region_suppress(x, mask, background_sigma=sigma, roi_sigma=0.0)


def mask_from_detections(det_out: dict, size: int, score_thresh: float = 0.5,
                         dilate: float = 0.15, min_margin_px: float = 0.0,
                         grid: int = 1) -> torch.Tensor:
    """Convenience: build the mask from one entry of the detector's output."""
    return protect_mask(det_out["boxes"], det_out["scores"], det_out["labels"],
                        size, score_thresh, dilate, min_margin_px, grid)


__all__ = [
    "dual_region_suppress",
    "gaussian_filter",
    "mask_from_detections",
    "motion_preserving_gaussian",
    "protect_mask",
    "suppress",
]
