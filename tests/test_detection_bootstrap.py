"""Regression tests for the OD measurement path."""

from __future__ import annotations

from src.engine import _loss_weights
from src.metrics.detection import (
    _remap_bootstrap_sample,
    paired_bootstrap_detection_bd,
)


def _metric(preds, gt_by_id, image_ids, ann_meta):
    del gt_by_id, image_ids, ann_meta
    return sum(p["score"] for p in preds) / len(preds), 0.0


def test_bootstrap_preserves_duplicate_image_multiplicity():
    gt = {
        7: [{"id": 70, "image_id": 7, "category_id": 1, "bbox": [0, 0, 2, 2]}],
        8: [{"id": 80, "image_id": 8, "category_id": 1, "bbox": [0, 0, 2, 2]}],
    }
    remapped, ids, mapping = _remap_bootstrap_sample([7, 7, 8], gt)
    assert ids == [1, 2, 3]
    assert mapping == [(7, 1), (7, 2), (8, 3)]
    assert [remapped[i][0]["image_id"] for i in ids] == ids
    assert len({remapped[i][0]["id"] for i in ids}) == 3


def test_each_arm_gets_its_own_paired_bootstrap_ci():
    ids = [1, 2, 3, 4]
    qps = [30, 35, 40, 45, 50]
    records = {arm: {} for arm in ("anchor", "prep", "sandwich")}
    for qi, qp in enumerate(qps):
        metric = 0.5 + 0.05 * qi
        for arm, rate_scale in (("anchor", 1.0), ("prep", 0.8), ("sandwich", 0.6)):
            records[arm][("h264", qp)] = {
                image_id: (
                    rate_scale * (0.1 + 0.1 * qi),
                    [{"image_id": image_id, "category_id": 1,
                      "bbox": [0, 0, 1, 1], "score": metric}],
                )
                for image_id in ids
            }
    gt = {
        i: [{"id": i, "image_id": i, "category_id": 1, "bbox": [0, 0, 1, 1]}]
        for i in ids
    }
    ci = paired_bootstrap_detection_bd(
        records,
        codec="h264",
        qps=qps,
        gt_by_id=gt,
        image_ids=ids,
        ann_meta={"categories": [{"id": 1, "name": "thing"}]},
        arms=("prep", "sandwich"),
        n_boot=30,
        seed=4,
        metric_fn=_metric,
    )
    assert set(ci) == {"prep", "sandwich"}
    assert ci["prep"]["n_draws"] == 30
    assert ci["sandwich"]["n_draws"] == 30
    assert -21.0 < ci["prep"]["lo"] < -19.0
    assert -41.0 < ci["sandwich"]["lo"] < -39.0


def test_rho_is_wired_from_config():
    weights = _loss_weights({"loss": {"rho": 1.25, "lam_task": 2.0}})
    assert weights.rho == 1.25
    assert weights.lam_task == 2.0
