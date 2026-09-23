"""Codec-search contracts that can invalidate the BD-rate comparison."""

import numpy as np

from src.models.codec_search import CANDIDATES, choose_action, make_candidates, normalized_bpp


def test_candidate_sizes_and_original_pixel_rate_denominator():
    clip = np.random.default_rng(3).integers(0, 256, (2, 128, 128, 3), dtype=np.uint8)
    variants = make_candidates(clip)
    assert tuple(variants) == CANDIDATES
    assert variants["area112"].shape == (2, 112, 112, 3)
    assert variants["area96"].shape == (2, 96, 96, 3)
    assert variants["area112_up128"].shape == clip.shape
    assert np.array_equal(variants["identity128"], clip)
    # Same exact byte count must have the same normalized bpp at every size.
    for size in (128, 112, 96):
        native_bpp = 8 * 1000 / (2 * size * size)
        assert abs(normalized_bpp(native_bpp, size, size) - 8 * 1000 / (2 * 128 * 128)) < 1e-12


def test_search_rejects_semantic_harm_and_never_uses_label_fields():
    rows = [
        {"name": "identity128", "bpp": 1.0, "kl_source": .1,
         "feature_distance": .1, "source_confidence": .8, "source_top1_agrees": True, "correct": False},
        {"name": "area112", "bpp": .5, "kl_source": .2,
         "feature_distance": .1, "source_confidence": .8, "source_top1_agrees": True, "correct": True},
        {"name": "area96", "bpp": .7, "kl_source": .11,
         "feature_distance": .11, "source_confidence": .8, "source_top1_agrees": False, "correct": True},
        {"name": "blur020_128", "bpp": .8, "kl_source": .11,
         "feature_distance": .11, "source_confidence": .8, "source_top1_agrees": True, "correct": False},
    ]
    assert choose_action(rows, .02, .02) == 3
    rows[3]["source_top1_agrees"] = False
    assert choose_action(rows, .02, .02) == 0

