import torch

from src.models.cast_ar import CASTTemporalPost


def _clip(batch=2, frames=5, size=24):
    return torch.rand(batch, 3, frames, size, size)


def test_cast_post_is_exact_identity_at_initialisation():
    torch.manual_seed(0)
    model = CASTTemporalPost(width=8, bases=3)
    x = _clip()
    picture_types = torch.tensor([[0, 2, 2, 1, 1], [0, 2, 2, 1, 1]])
    out = model(x, 40, picture_types)
    assert torch.equal(out, x)


def test_cast_identity_has_live_strength_gradient():
    torch.manual_seed(0)
    model = CASTTemporalPost(width=8, bases=3)
    x = _clip()
    target = (x * 0.9).detach()
    out = model(x, 45)
    (out - target).square().mean().backward()
    assert model.post_strength.grad is not None
    assert model.post_strength.grad.abs() > 0


def test_cast_uses_qp_and_real_picture_types_once_open():
    torch.manual_seed(0)
    model = CASTTemporalPost(width=8, bases=3)
    with torch.no_grad():
        model.post_strength.fill_(1.0)
        model.qp_film[-1].weight.normal_(0, 0.05)
        model.picture_embedding.weight.normal_(0, 0.05)
    x = _clip(batch=1)
    ip = torch.tensor([[0, 1, 1, 1, 1]])
    ib = torch.tensor([[0, 2, 2, 1, 1]])
    low = model(x, 30, ip)
    high = model(x, 50, ip)
    bframes = model(x, 30, ib)
    assert low.shape == x.shape
    assert not torch.allclose(low, high)
    assert not torch.allclose(low, bframes)


def test_cast_rejects_bad_metadata_shape():
    model = CASTTemporalPost(width=8)
    x = _clip(batch=1, frames=4)
    try:
        model(x, 40, torch.zeros(1, 3, dtype=torch.long))
    except ValueError as exc:
        assert "picture_types" in str(exc)
    else:
        raise AssertionError("bad picture-type shape was accepted")
