import torch
from ops.gates_qpc import _build_audit_model, _overall_pass
from src.models.additive import AdditivePreprocessor
from src.models.additive_cond import AdditiveCondPreprocessor


def test_audit_rebuilds_matched_unconditioned_control():
    model = _build_audit_model({"cfg": {"model": {"arch": "additive", "temporal_frames": 8}}})
    assert isinstance(model, AdditivePreprocessor)


def test_audit_rebuilds_qp_conditioned_arm():
    model = _build_audit_model(
        {
            "cfg": {
                "model": {
                    "arch": "additive_cond",
                    "temporal_frames": 8,
                    "cond_dim": 1,
                }
            }
        }
    )
    assert isinstance(model, AdditiveCondPreprocessor)
    assert hasattr(model, "film")


def test_control_is_condition_invariant_at_the_model_boundary():
    model = AdditivePreprocessor()
    with torch.no_grad():
        model.to_rgb.weight.normal_(0, 0.01)
    clip = torch.rand(1, 3, 4, 32, 32)
    with torch.no_grad():
        low = model(clip, torch.tensor([[0.303]]))
        high = model(clip, torch.tensor([[0.645]]))
    assert torch.equal(low, high)


def test_qpc_overall_gate_rejects_unused_conditioning():
    assert not _overall_pass("additive_cond", regime=True, utilized=False, stable=True)
    assert _overall_pass("additive_cond", regime=True, utilized=True, stable=True)
    assert _overall_pass("additive", regime=True, utilized=False, stable=True)
