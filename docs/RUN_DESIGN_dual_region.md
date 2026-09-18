# Frozen screening design — OD dual-region PRE + Gaussian POST

## Question

Starting from the frozen R0 winner (`blur4`, score `0.2`, dilation `0.30`,
feather `8`), does either mild ROI denoising or zero-bit post-decoder Gaussian
filtering improve the rate–mAP curve under real x264 and x265?

## Screening run

- Data: deterministic 100-image COCO val subset, seed 0, 320 px.
- Mask analyzer: Faster R-CNN MobileNetV3-Large-FPN.
- Held-out evaluator: Faster R-CNN ResNet50-FPN.
- Codecs: x264 and x265, All-Intra, QP `30,35,40,45,50`.
- Frozen mask geometry: score `0.2`, dilation `0.30`, feather `8`, no fixed halo,
  no block alignment.  Keeping this fixed isolates the two new mechanisms.
- PRE arms: background sigma `4` crossed with ROI sigma `0` or `1`.
- POST arms: identity or global Gaussian sigma `1`; POST is identity below QP 40.
- Anchor POST is included to measure artifact removal without PRE.
- No inline bootstrap.  Per-image records are persisted for offline paired CI.

This is a mechanism screen, not a confirmatory result.  A candidate advances
only when it improves both codecs without a per-QP mAP gap below `-0.05`.  The
winning configuration is then frozen and rerun on at least 500 images with
paired bootstrap CI.  Fixed 5–10 px context halos and 8/16-pixel mask alignment
are implemented but intentionally deferred to a separate ablation.
