"""Label-free, measured-byte selection of task-preserving codec candidates."""

from __future__ import annotations

import cv2
import numpy as np


CANDIDATES = (
    "identity128", "area112", "area96", "area112_up128",
    "blur020_128", "blur040_128",
)


def make_candidates(clip: np.ndarray) -> dict[str, np.ndarray]:
    """Generate fixed, codec-independent RGB candidates from a 128-pixel clip."""
    if clip.ndim != 4 or clip.shape[1:] != (128, 128, 3) or clip.dtype != np.uint8:
        raise ValueError("expected uint8 RGB [T,128,128,3]")
    area112 = np.stack([cv2.resize(f, (112, 112), interpolation=cv2.INTER_AREA) for f in clip])
    area96 = np.stack([cv2.resize(f, (96, 96), interpolation=cv2.INTER_AREA) for f in clip])
    up = np.stack([cv2.resize(f, (128, 128), interpolation=cv2.INTER_LINEAR) for f in area112])
    blurred = np.stack([cv2.GaussianBlur(f, (0, 0), 1.5) for f in clip]).astype(np.float32)
    source = clip.astype(np.float32)
    return {
        "identity128": clip.copy(), "area112": area112, "area96": area96,
        "area112_up128": up,
        "blur020_128": np.clip(np.rint(source + .2 * (blurred - source)), 0, 255).astype(np.uint8),
        "blur040_128": np.clip(np.rint(source + .4 * (blurred - source)), 0, 255).astype(np.uint8),
    }


def normalized_bpp(native_bpp: float, height: int, width: int, reference: int = 128) -> float:
    """Report every stream with the same original-pixel denominator."""
    return float(native_bpp) * height * width / (reference * reference)


def choose_action(measurements: list[dict], kl_slack: float, feature_slack: float) -> int:
    """Choose by observed bytes and label-free task distance; identity is feasible."""
    if not measurements or measurements[0]["name"] != "identity128":
        raise ValueError("identity128 must be the first candidate")
    anchor = measurements[0]
    choices = [0]
    for i, candidate in enumerate(measurements[1:], 1):
        if candidate["bpp"] >= anchor["bpp"]:
            continue
        if candidate["kl_source"] > anchor["kl_source"] + kl_slack:
            continue
        if candidate["feature_distance"] > anchor["feature_distance"] + feature_slack:
            continue
        if candidate["source_confidence"] >= .6 and not candidate["source_top1_agrees"]:
            continue
        choices.append(i)
    return min(choices, key=lambda i: measurements[i]["bpp"])

