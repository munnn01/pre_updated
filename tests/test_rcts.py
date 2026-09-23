"""Checks for the RCTS action contract and train-only teacher decisions."""

import numpy as np

from src.models.rcts import ACTIONS, make_candidates, select_teacher_action
from ops.rcts_pilot import balanced_indices


def test_candidates_are_bounded_and_temporal_first_frame_is_identity():
    rng = np.random.default_rng(7)
    clip = rng.integers(0, 256, (4, 32, 32, 3), dtype=np.uint8)
    saliency = rng.random((4, 32, 32)).astype(np.float32)
    variants, info = make_candidates(clip, saliency)
    assert tuple(variants) == tuple(action.name for action in ACTIONS)
    assert np.array_equal(variants["identity"], clip)
    assert np.array_equal(variants["temporal_t20"][0], clip[0])
    assert all(v.shape == clip.shape and v.dtype == np.uint8 for v in variants.values())
    assert 0 < info["protected_fraction"] < 1


def test_teacher_uses_measured_rate_subject_to_regret_and_flip():
    rows = [
        {"name": "identity", "bpp": 1.0, "ce": 0.2, "correct": True},
        {"name": "cheap_bad_ce", "bpp": 0.5, "ce": 0.3, "correct": True},
        {"name": "cheap_flip", "bpp": 0.6, "ce": 0.2, "correct": False},
        {"name": "safe", "bpp": 0.8, "ce": 0.21, "correct": True},
    ]
    assert select_teacher_action(rows, max_ce_regret=0.02) == 3


def test_balanced_subset_spans_full_class_range():
    records = [{"path": f"class_{i}/clip.mp4", "label": i} for i in range(400)]
    picked = balanced_indices(records, 80, "rcts-train-v1")
    assert len(picked) == 80
    assert len(set(picked)) == 80
    assert max(picked) >= 300
    assert min(picked) < 100
    assert picked == balanced_indices(records, 80, "rcts-train-v1")
