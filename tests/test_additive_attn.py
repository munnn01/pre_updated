"""Checks for the attentive dual-branch additive preprocessor (needs torch).

pytest-collectable AND runnable standalone -- same pattern as
tests/test_additive_cond.py. Pins the identity-at-init discipline, the
QP-conditioned attention wiring, resolution-agnosticism, and gradient flow.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.additive_attn import AttentiveAdditivePreprocessor


def _trained_like(scale: float = 0.02) -> AttentiveAdditivePreprocessor:
    """A model with a LIVE residual: perturb the zero-init to_rgb so the edit
    is non-zero but small enough to stay off the [0,1] clamp."""
    pre = AttentiveAdditivePreprocessor()
    with torch.no_grad():
        pre.to_rgb.weight.normal_(0.0, scale)
        pre.to_rgb.bias.normal_(0.0, scale)
    return pre


def test_identity_at_init_for_any_cond() -> None:
    """zero-init to_rgb: an untrained model is EXACT identity for any cond."""
    x = torch.rand(1, 3, 16, 64, 64)
    with torch.no_grad():
        y_none = AttentiveAdditivePreprocessor()(x, None)
        y_one = AttentiveAdditivePreprocessor()(x, torch.ones(1, 1))
    assert torch.equal(y_none, x), "cond=None must be exact identity at init"
    assert torch.equal(y_one, x), "cond=1 must be exact identity at init"


def test_unconditioned_equals_light_cond() -> None:
    """cond=None and cond=0 must agree exactly (same default semantics)."""
    pre = _trained_like()
    x = torch.rand(1, 3, 8, 48, 48)
    with torch.no_grad():
        y0 = pre(x, None)
        yz = pre(x, torch.zeros(1, 1))
    assert torch.equal(y0, yz), "cond=None must default to zeros"


def test_cond_changes_live_residual() -> None:
    """Conditionability: with a live FiLM head the attention fusion (hence the
    residual) must differ across conditions. Pins cond -> FiLM -> attn -> out."""
    pre = _trained_like()
    with torch.no_grad():
        pre.film[2].weight.normal_(0.0, 0.2)
        pre.film[2].bias.normal_(0.0, 0.2)
    x = torch.rand(2, 3, 8, 48, 48)
    with torch.no_grad():
        y_lo = pre(x, torch.zeros(2, 1))
        y_hi = pre(x, torch.ones(2, 1))
    assert not torch.equal(y_lo, y_hi), "cond must reach the output when FiLM is live"


def test_mask_is_ignored() -> None:
    """This model is ungated: a mask must not change the output."""
    pre = _trained_like()
    x = torch.rand(1, 3, 8, 48, 48)
    with torch.no_grad():
        y0 = pre(x)
        y1 = pre(x, None, torch.ones(1, 1, 8, 48, 48))
    assert torch.equal(y0, y1), "mask must be ignored"


def test_shape_range_resolution_agnostic() -> None:
    """Shape preserved, output clamped, any square resolution / clip length."""
    pre = _trained_like()
    for t, hw in ((16, 128), (16, 224), (4, 96)):
        x = torch.rand(2, 3, t, hw, hw)
        y = pre(x, torch.full((2, 1), 0.5))
        assert y.shape == x.shape, (y.shape, x.shape)
        assert y.min() >= 0.0 and y.max() <= 1.0, "output must be clamped to [0,1]"


def test_attention_weights_sum_to_one() -> None:
    """The fusion is a softmax over the two branches, so per-pixel per-channel
    weights sum to 1 -- a genuine attention, not the lineage's scalar gate."""
    pre = AttentiveAdditivePreprocessor()
    # Recompute the branch logits + softmax the same way forward() does, on one
    # frame, and check the branch weights sum to 1 everywhere.
    x = torch.rand(1, 3, 4, 32, 32)
    b, c, t, h, w = x.shape
    frames = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    with torch.no_grad():
        import torch.nn.functional as F
        sp = F.relu(pre.spatial_stem(frames))
        body = pre.spatial_residual.body
        sp = sp + body[2](F.relu(body[0](sp)))
        pad = x[:, :, :1].expand(b, c, pre.tframes - 1, h, w)
        padded = torch.cat([pad, x], dim=2)
        win = padded.unfold(2, pre.tframes, 1)
        stack = win.permute(0, 2, 5, 1, 3, 4).reshape(b * t, pre.tframes * c, h, w)
        tp = F.relu(pre.temporal_stem(stack))
        l_sp, l_tp = pre.attn.spatial(sp), pre.attn.temporal(tp)
        weights = torch.softmax(torch.stack([l_sp, l_tp], dim=0), dim=0)
    assert torch.allclose(weights.sum(dim=0), torch.ones_like(l_sp), atol=1e-5)


def test_gradients_flow_to_every_param() -> None:
    """Gradient reaches every parameter, attention + FiLM included."""
    pre = AttentiveAdditivePreprocessor()
    x = torch.rand(2, 3, 16, 64, 64)
    pre(x, torch.full((2, 1), 0.5)).mean().backward()
    for name, p in pre.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"no/NaN grad: {name}"
    # gradient flows INTO cond when the FiLM head is live
    pre2 = _trained_like()
    with torch.no_grad():
        pre2.film[2].weight.normal_(0.0, 0.2)
    cond = torch.zeros(1, 1, requires_grad=True)
    pre2(torch.rand(1, 3, 8, 48, 48), cond).mean().backward()
    assert cond.grad is not None and float(cond.grad.abs()) > 0, \
        "gradient must flow INTO cond when film is live"


def test_strength_is_the_operating_point() -> None:
    """0 -> exact identity; s -> x + s*(y1 - x) at a FIXED cond."""
    pre = _trained_like()
    x = torch.rand(1, 3, 16, 64, 64) * 0.4 + 0.3
    cond = torch.full((1, 1), 0.5)
    with torch.no_grad():
        y1 = pre(x, cond)
        pre.strength = 0.5
        yh = pre(x, cond)
        pre.strength = 0.0
        y0 = pre(x, cond)
    assert not torch.allclose(y1, x, atol=1e-6), "fixture residual must be live"
    assert torch.equal(y0, x), "strength=0 must be exact identity"
    assert torch.allclose(yh, x + 0.5 * (y1 - x), atol=1e-6), \
        "strength=s must scale the residual linearly"


if __name__ == "__main__":
    test_identity_at_init_for_any_cond()
    test_unconditioned_equals_light_cond()
    test_cond_changes_live_residual()
    test_mask_is_ignored()
    test_shape_range_resolution_agnostic()
    test_attention_weights_sum_to_one()
    test_gradients_flow_to_every_param()
    test_strength_is_the_operating_point()
    print("additive_attn self-check passed")
