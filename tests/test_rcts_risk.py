"""Safety and source-disjointness checks for risk-aware RCTS."""

import numpy as np
import torch

from ops.rcts_risk import split_examples
from src.models.rcts_risk import RiskPredictor, choose_action


def test_predictor_shape_and_gradients():
    model = RiskPredictor(8)
    result = model(torch.randn(2, 130, 4, 8, 8), torch.tensor([0.0, 1.0]))
    assert result.shape == (2, 8, 3)
    result.square().mean().backward()
    assert model.head[-1].weight.grad is not None


def test_selector_identity_fallback_and_safe_rate_choice():
    pred = np.zeros((3, 3))
    pred[1] = [-0.3, 0.1, -10]  # fast, but too much predicted CE regret
    pred[2] = [-0.2, 0.01, -10]
    assert choose_action(pred, 0.02, 0.1) == 2
    pred[2, 2] = 10  # harmful-flip risk
    assert choose_action(pred, 0.02, 0.1) == 0


def test_three_splits_are_source_disjoint():
    rows = [{"sequence_id": f"class/x{i:03d}"} for i in range(100)]
    fit, stop, cal = split_examples(rows)
    groups = [{r["sequence_id"] for r in part} for part in (fit, stop, cal)]
    assert list(map(len, groups)) == [70, 10, 20]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])

