#!/usr/bin/env python
"""Probe: detector-mask-driven background suppression, no training.

Why this exists. Every *learned* preprocessor the project tried on images failed:
the AR checkpoint costs ~12% more bits at equal detection mAP, and the
detection-trained one destroys 25% of the detector's mAP before the codec is even
involved (mAP(pre)/mAP(x) = 0.75). The detection literature's large numbers come
from the opposite philosophy — REMOVE background, keep objects (ROI-Packing
-44%, dual-region JPEG -26%, Rozek VCIP 2023) — which needs no learned editor and
therefore cannot overfit the proxy codec or add structure the detector dislikes.

Design: run the frozen detector on the SOURCE image (the encoder has the image
and may analyse it freely; the decoder needs no side information), dilate its
boxes into a protection mask, and heavily blur everything OUTSIDE the mask.  The
dual-region extension can mildly denoise the ROI as well.  A second, zero-bit
Gaussian filter can be enabled after decoding only at high QP.  The historical
identity-ROI transform remains the default and every extension is an explicit
ablation arm.

Arms: anchor (codec(x)) vs masked (codec(suppress(x))) on the mAP axis, with
several blur strengths, both codecs, the project's QP grid. Per-image records are
saved so the CI is recomputed offline (CPU) rather than in the kernel.

Usage:
  python ops/probe_background_suppression.py --images <val2017> --ann <instances_val2017.json> \
      --n-images 500 --size 320 --sigmas 4,8,16 --out outputs/probe_bgsuppress
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.codecs.standard import StandardCodec, ffmpeg_available  # noqa: E402
from src.metrics.bd_rate import bd_rate  # noqa: E402
from src.metrics.detection import paired_bootstrap_detection_bd  # noqa: E402
from src.models.importance_tube import feather_protection  # noqa: E402
from src.models.mask_suppress import (  # noqa: E402
    dual_region_suppress,
    gaussian_filter,
    protect_mask,
)
from probe_detection import (  # noqa: E402  (ops/ is on sys.path when run from repo root)
    Detector,
    _coco_box,
    coco_map,
    load_coco,
    scaled_gt,
)


def _float_grid(raw: str, name: str, *, positive: bool = False) -> list[float]:
    values = [float(x) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError(f"{name} must not be empty")
    if positive and any(x <= 0 for x in values):
        raise ValueError(f"{name} values must be positive")
    if not positive and any(x < 0 for x in values):
        raise ValueError(f"{name} values must be non-negative")
    return list(dict.fromkeys(values))


def _codec_grid(raw: str) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("codecs must be a non-empty unique list")
    invalid = set(values) - {"h264", "h265"}
    if invalid:
        raise ValueError(f"unsupported codecs: {sorted(invalid)}")
    return values


def _base_arm(background_sigma: float, roi_sigma: float) -> str:
    name = f"blur{background_sigma:g}"
    if roi_sigma > 0:
        name += f"_roi{roi_sigma:g}"
    return name


def _post_arm(base: str, post_sigma: float, post_min_qp: int) -> str:
    return f"{base}_post{post_sigma:g}q{post_min_qp}"


def _predictions(det_out: dict, detector: Detector, image_id: int) -> list[dict]:
    keep = det_out["scores"] >= detector.score_thresh
    return [
        {"image_id": image_id, "category_id": int(label), "bbox": _coco_box(box),
         "score": float(score)}
        for box, score, label in zip(
            det_out["boxes"][keep], det_out["scores"][keep], det_out["labels"][keep]
        )
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--n-images", type=int, default=500)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--qps", default="30,35,40,45,50")
    ap.add_argument("--codecs", default="h264,h265",
                    help="comma-separated subset of h264,h265")
    ap.add_argument("--sigmas", default="4,8,16")
    ap.add_argument("--roi-sigmas", default="0",
                    help="comma-separated Gaussian sigmas inside protected ROI")
    ap.add_argument("--post-sigmas", default="0",
                    help="comma-separated zero-bit Gaussian sigmas after decode")
    ap.add_argument("--post-min-qp", type=int, default=45,
                    help="post-filter is identity below this QP")
    ap.add_argument("--score", type=float, default=0.5)
    ap.add_argument("--eval-score", type=float, default=0.05)
    ap.add_argument("--dilate", type=float, default=0.15)
    ap.add_argument("--min-margin-px", type=float, default=0.0,
                    help="minimum context halo per box side before block alignment")
    ap.add_argument("--mask-grid", type=int, default=1,
                    help="expand protected boxes to this pixel grid (1 disables)")
    ap.add_argument("--feather", type=int, default=0,
                    help="soft protection band in pixels around the exact-identity boxes")
    ap.add_argument("--mask-backbone", default="fasterrcnn_mobilenet_v3_large_fpn",
                    choices=["fasterrcnn_resnet50_fpn",
                             "fasterrcnn_mobilenet_v3_large_fpn",
                             "retinanet_resnet50_fpn_v2",
                             "fcos_resnet50_fpn"])
    ap.add_argument("--eval-backbone", default="fasterrcnn_resnet50_fpn",
                    choices=["fasterrcnn_resnet50_fpn",
                             "fasterrcnn_mobilenet_v3_large_fpn",
                             "retinanet_resnet50_fpn_v2",
                             "fcos_resnet50_fpn"])
    ap.add_argument("--allow-same-detector", action="store_true",
                    help="allow an on-teacher diagnostic; never use it for a claim")
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="paired image-bootstrap draws (0 skips inline CI)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/probe_bgsuppress")
    a = ap.parse_args()

    if not ffmpeg_available():
        print("[bg] ffmpeg missing -> abort")
        raise SystemExit(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sigmas = _float_grid(a.sigmas, "sigmas", positive=True)
    roi_sigmas = _float_grid(a.roi_sigmas, "roi_sigmas")
    post_sigmas = _float_grid(a.post_sigmas, "post_sigmas")
    positive_post_sigmas = [s for s in post_sigmas if s > 0]
    qps = [int(q) for q in a.qps.split(",")]
    codecs = _codec_grid(a.codecs)
    if a.min_margin_px < 0:
        raise ValueError("min_margin_px must be non-negative")
    if a.mask_grid <= 0:
        raise ValueError("mask_grid must be positive")

    if a.mask_backbone == a.eval_backbone and not a.allow_same_detector:
        raise ValueError(
            "mask and eval detectors must differ for held-out evaluation; "
            "pass --allow-same-detector only for an explicitly on-teacher diagnostic"
        )

    ann_meta, items = load_coco(Path(a.images), Path(a.ann), a.n_images, a.size, a.seed)
    items = [(i, t.to(device), hw, an) for i, t, hw, an in items]
    gt_by_id = {i: scaled_gt(an, a.size, hw) for i, _, hw, an in items}
    ids = [i for i, _, _, _ in items]
    print(f"[bg] {len(items)} images at {a.size}px, background={sigmas}, "
          f"roi={roi_sigmas}, post={post_sigmas}@qp>={a.post_min_qp}")

    mask_det = Detector(device, score_thresh=a.score, backbone=a.mask_backbone)
    print(f"[bg] mask detector={a.mask_backbone}; held-out evaluator={a.eval_backbone}")

    def mAP(pred_list):
        return coco_map(pred_list, gt_by_id, ids, ann_meta)[0]

    from torchvision.ops import nms  # noqa: E402  (only needed for the mask tally)
    masks = {}
    core_cover = []
    for i, t, hw, _ in items:
        d = mask_det.predict(t)[0]
        core = protect_mask(d["boxes"], d["scores"], d["labels"], a.size,
                            a.score, a.dilate, a.min_margin_px, a.mask_grid)
        core_cover.append(float(core.mean()))
        masks[i] = (feather_protection(core, a.feather).squeeze(2)
                    if a.feather else core)
    # The two networks must differ scientifically, but they need not occupy GPU
    # memory simultaneously: masks are now materialised and detached.
    del mask_det
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    eval_det = Detector(device, score_thresh=a.eval_score, backbone=a.eval_backbone)
    cover = float(np.mean([m.mean().item() for m in masks.values()]))
    print(f"[bg] mean core/effective protected fraction = "
          f"{np.mean(core_cover):.3f}/{cover:.3f} "
          f"(1 - this is how much of the image may be destroyed)")

    pre_specs = [("anchor", None, None)] + [
        (_base_arm(background_sigma, roi_sigma), background_sigma, roi_sigma)
        for background_sigma in sigmas for roi_sigma in roi_sigmas
    ]
    if len({name for name, _, _ in pre_specs}) != len(pre_specs):
        raise ValueError("sigma grids produce duplicate arm names")
    arms = []
    for base, _, _ in pre_specs:
        arms.append(base)
        arms.extend(_post_arm(base, s, a.post_min_qp) for s in positive_post_sigmas)
    rec = {arm: {} for arm in arms}
    for codec_name in codecs:
        for qp in qps:
            sc = StandardCodec(codec=codec_name, qp=qp, preset="medium")
            for arm in arms:
                rec[arm].setdefault((codec_name, qp), {})
            for i, t, hw, _ in items:
                encoded = []
                for base, background_sigma, roi_sigma in pre_specs:
                    xv = t if base == "anchor" else dual_region_suppress(
                        t, masks[i], background_sigma=background_sigma,
                        roi_sigma=roi_sigma,
                    )
                    decoded, bpp = sc.compress_decompress_items(xv)
                    encoded.append((base, decoded, float(bpp[0])))

                base_predictions = eval_det.predict(torch.cat(
                    [decoded for _, decoded, _ in encoded], dim=0
                ))
                for (base, _, bpp), det_out in zip(encoded, base_predictions):
                    preds = _predictions(det_out, eval_det, i)
                    rec[base][(codec_name, qp)][i] = (bpp, preds)

                for post_sigma in positive_post_sigmas:
                    if qp < a.post_min_qp:
                        for base, _, bpp in encoded:
                            post_name = _post_arm(base, post_sigma, a.post_min_qp)
                            rec[post_name][(codec_name, qp)][i] = (
                                bpp, rec[base][(codec_name, qp)][i][1]
                            )
                        continue
                    post_batch = torch.cat([
                        gaussian_filter(decoded, post_sigma)
                        for _, decoded, _ in encoded
                    ], dim=0)
                    post_predictions = eval_det.predict(post_batch)
                    for (base, _, bpp), det_out in zip(encoded, post_predictions):
                        post_name = _post_arm(base, post_sigma, a.post_min_qp)
                        rec[post_name][(codec_name, qp)][i] = (
                            bpp, _predictions(det_out, eval_det, i)
                        )
            print(f"[bg] {codec_name} qp{qp} done", flush=True)

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"n_images": len(items), "size": a.size, "cover": cover,
              "core_cover": float(np.mean(core_cover)), "feather": a.feather,
              "sigmas": sigmas, "roi_sigmas": roi_sigmas,
              "post_sigmas": post_sigmas, "post_min_qp": a.post_min_qp,
              "score": a.score, "eval_score": a.eval_score,
              "dilate": a.dilate, "min_margin_px": a.min_margin_px,
              "mask_grid": a.mask_grid, "mask_backbone": a.mask_backbone,
              "eval_backbone": a.eval_backbone, "codecs": codecs, "curves": {}}
    for codec_name in codecs:
        curves = {}
        for arm in arms:
            rates, aps = [], []
            for qp in qps:
                slot = rec[arm][(codec_name, qp)]
                rates.append(float(np.mean([v[0] for v in slot.values()])))
                aps.append(mAP([p for v in slot.values() for p in v[1]]))
            curves[arm] = {"rate": rates, "mAP": aps}
            print(f"[bg] {codec_name} {arm:24s} bpp={['%.4f' % r for r in rates]} "
                  f"mAP={['%.4f' % m for m in aps]}")
        for arm in arms[1:]:
            curves[arm]["bd_vs_anchor"] = bd_rate(
                curves["anchor"]["rate"], curves["anchor"]["mAP"],
                curves[arm]["rate"], curves[arm]["mAP"])
            print(f"[bg] {codec_name} {arm}: BD = "
                  f"{curves[arm]['bd_vs_anchor']:+.2f}%")
        result["curves"][codec_name] = curves

    # records for the offline bootstrap CI
    flat = {}
    for arm in arms:
        for (codec_name, qp), slot in rec[arm].items():
            img = sorted(slot)
            boxes, offs, labs = [], [0], []
            for i in img:
                for p in slot[i][1]:
                    boxes.append(p["bbox"] + [p["score"]])
                    labs.append(p["category_id"])
                offs.append(len(boxes))
            tag = f"{arm}_{codec_name}_{qp}"
            flat[f"{tag}_img"] = np.asarray(img, dtype=np.int64)
            flat[f"{tag}_bpp"] = np.asarray([slot[i][0] for i in img], dtype=np.float32)
            flat[f"{tag}_boxes"] = np.asarray(boxes, dtype=np.float32).reshape(-1, 5)
            flat[f"{tag}_labels"] = np.asarray(labs, dtype=np.int32)
            flat[f"{tag}_offsets"] = np.asarray(offs, dtype=np.int64)
    np.savez_compressed(out_dir / "per_image_records.npz", **flat)
    print(f"[bg] wrote {out_dir / 'per_image_records.npz'}")

    def finite(o):
        if isinstance(o, dict):
            return {k: finite(v) for k, v in o.items()}
        if isinstance(o, list):
            return [finite(v) for v in o]
        if isinstance(o, float) and not np.isfinite(o):
            return None
        return o

    result_path = out_dir / "probe_bgsuppress.json"
    result_path.write_text(json.dumps(finite(result), indent=2))
    print(f"[bg] wrote preliminary {result_path}")

    # Persist detector records and point estimates before the CPU-heavy CI.
    # A Kaggle timeout during bootstrap must not throw away the expensive GPU
    # sweep.  Each completed codec is checkpointed independently as well.
    for codec_name in codecs:
        if not a.bootstrap:
            break
        print(f"[bg] bootstrap {codec_name}: {a.bootstrap} paired draws", flush=True)
        result.setdefault("bootstrap_ci", {})[codec_name] = (
            paired_bootstrap_detection_bd(
                rec,
                codec=codec_name,
                qps=qps,
                gt_by_id=gt_by_id,
                image_ids=ids,
                ann_meta=ann_meta,
                arms=arms[1:],
                n_boot=a.bootstrap,
                seed=a.seed,
            )
        )
        result_path.write_text(json.dumps(finite(result), indent=2))
        print(f"[bg] checkpointed bootstrap {codec_name}", flush=True)

    print(f"[bg] wrote {result_path}")


if __name__ == "__main__":
    main()
