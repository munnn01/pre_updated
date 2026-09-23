"""Bounded encoder-side operators for a real-codec task-sensitivity pilot.

The operators are deliberately discrete.  Their supervision comes from measured
bytes and action-model regret after a full H.264/H.265 encode, rather than from a
surrogate rate gradient.  No action label is needed to construct a candidate.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Action:
    name: str
    spatial: float = 0.0
    temporal: float = 0.0
    protect: bool = True


ACTIONS = (
    Action("identity"),
    Action("uniform_s20", spatial=0.20, protect=False),
    Action("uniform_s40", spatial=0.40, protect=False),
    Action("spatial_s25", spatial=0.25),
    Action("spatial_s45", spatial=0.45),
    Action("temporal_t20", temporal=0.20),
    Action("joint_s25_t15", spatial=0.25, temporal=0.15),
    Action("joint_s45_t25", spatial=0.45, temporal=0.25),
)


def _protection(saliency: np.ndarray, block: int = 16) -> np.ndarray:
    """Protect the top 25% of coarse task-saliency blocks, plus neighbours in time."""
    t, h, w = saliency.shape
    gh, gw = max(1, (h + block - 1) // block), max(1, (w + block - 1) // block)
    coarse = np.stack(
        [cv2.resize(s, (gw, gh), interpolation=cv2.INTER_AREA) for s in saliency]
    )
    threshold = np.quantile(coarse, 0.75)
    mask = coarse >= threshold
    stable = mask.copy()
    if t > 1:
        stable[1:] |= mask[:-1]
        stable[:-1] |= mask[1:]
    return np.stack(
        [cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
         for m in stable]
    ).astype(np.float32)[..., None]


def _motion_compensation(clip: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Warp previous source frame to current coordinates; suppress unreliable warps."""
    t, h, w, _ = clip.shape
    warped = clip.copy()
    confidence = np.zeros((t, h, w, 1), dtype=np.float32)
    magnitudes: list[float] = []
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    for k in range(1, t):
        current = cv2.cvtColor(clip[k], cv2.COLOR_RGB2GRAY)
        previous = cv2.cvtColor(clip[k - 1], cv2.COLOR_RGB2GRAY)
        # Flow from current coordinates into the previous source image.
        flow = cv2.calcOpticalFlowFarneback(
            current, previous, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mapped = cv2.remap(
            clip[k - 1], xx + flow[..., 0], yy + flow[..., 1],
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
        )
        error = np.abs(mapped.astype(np.float32) - clip[k].astype(np.float32)).mean(2)
        warped[k] = mapped
        confidence[k, :, :, 0] = np.exp(-error / 18.0)
        magnitudes.append(float(np.linalg.norm(flow, axis=2).mean()))
    return warped, confidence, float(np.mean(magnitudes)) if magnitudes else 0.0


def make_candidates(
    clip: np.ndarray, saliency: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Return uint8 RGB [T,H,W,3] candidates and label-free source features."""
    if clip.ndim != 4 or clip.shape[-1] != 3 or clip.dtype != np.uint8:
        raise ValueError("clip must be uint8 RGB [T,H,W,3]")
    if saliency.shape != clip.shape[:3]:
        raise ValueError("saliency must match [T,H,W]")
    source = clip.astype(np.float32)
    blurred = np.stack([cv2.GaussianBlur(f, (0, 0), 1.5) for f in source])
    warped, confidence, flow_mean = _motion_compensation(clip)
    protect = _protection(saliency)
    available = 1.0 - protect
    variants: dict[str, np.ndarray] = {}
    for action in ACTIONS:
        if action.name == "identity":
            variants[action.name] = clip.copy()
            continue
        gate = available if action.protect else 1.0
        edited = source + action.spatial * gate * (blurred - source)
        if action.temporal:
            edited += action.temporal * available * confidence * (
                warped.astype(np.float32) - edited
            )
        variants[action.name] = np.clip(np.rint(edited), 0, 255).astype(np.uint8)
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in clip])
    info = {
        "luma_mean": float(gray.mean() / 255.0),
        "luma_std": float(gray.std() / 255.0),
        "edge": float(np.abs(np.diff(gray.astype(np.float32), axis=2)).mean() / 255.0),
        "frame_diff": float(np.abs(np.diff(gray.astype(np.float32), axis=0)).mean() / 255.0)
        if len(gray) > 1 else 0.0,
        "flow_mean": flow_mean / max(clip.shape[1:3]),
        "warp_confidence": float(confidence[1:].mean()) if len(gray) > 1 else 0.0,
        "protected_fraction": float(protect.mean()),
        "saliency_mean": float(saliency.mean()),
        "saliency_std": float(saliency.std()),
    }
    return variants, info


def select_teacher_action(
    candidates: list[dict], *, max_ce_regret: float = 0.02,
) -> int:
    """Choose the cheapest safe action on TRAIN only; identity is always safe."""
    if not candidates or candidates[0]["name"] != "identity":
        raise ValueError("first candidate must be identity")
    anchor = candidates[0]
    best, best_rate = 0, float(anchor["bpp"])
    for index, candidate in enumerate(candidates[1:], 1):
        if candidate["ce"] - anchor["ce"] > max_ce_regret:
            continue
        if anchor["correct"] and not candidate["correct"]:
            continue
        rate = float(candidate["bpp"])
        if rate < best_rate:
            best, best_rate = index, rate
    return best


__all__ = ["ACTIONS", "Action", "make_candidates", "select_teacher_action"]
