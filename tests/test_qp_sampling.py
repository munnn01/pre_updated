from types import SimpleNamespace

import pytest
from src.engine import _qp_sampling_weights, _training_codec_setup

QPS = [30, 35, 40, 45, 50]


def test_uniform_sampling_is_the_backward_compatible_default():
    assert _qp_sampling_weights({}, QPS) is None


def test_bd_weights_are_normalised_and_aligned():
    raw = [0.089, 0.215, 0.288, 0.285, 0.123]
    weights = _qp_sampling_weights({"qp_sampling_weights": raw}, QPS)
    assert weights is not None
    assert sum(weights) == pytest.approx(1.0)
    assert weights == pytest.approx(raw)
    assert weights[2] > weights[0]


@pytest.mark.parametrize(
    "raw",
    ([1, 2], [0, 0, 0, 0, 0], [1, -1, 1, 1, 1], [1, float("nan"), 1, 1, 1]),
)
def test_invalid_qp_sampling_weights_fail_before_training(raw):
    with pytest.raises(ValueError):
        _qp_sampling_weights({"qp_sampling_weights": raw}, QPS)


def test_sampling_grid_and_proxy_mapping_remain_independent():
    train = {
        "qp_list": QPS,
        "qp_to_quality": {30: 8, 35: 5, 40: 3, 45: 2, 50: 1},
        "qp_sampling_weights": [0.089, 0.215, 0.288, 0.285, 0.123],
    }
    qps, mapping = _training_codec_setup(train, SimpleNamespace(qualities=[1, 2, 3, 5, 8]))
    assert qps == QPS
    assert mapping == train["qp_to_quality"]
