# Held-out confirmation: halo-8 background suppression + QP-gated POST

Status: frozen before launching the 500-image runs on 2026-09-18.

## Motivation

The `preprocessing_upgrade_10_OD` study found that an ungated full-frame
Gaussian POST filter helps at high QP but damages clean, low-QP frames.  Its
pre-registered follow-up therefore proposed codec-specific gates: QP 40 for
H.264 and QP 45 for H.265.  Those measurements used an exploratory
non-held-out analyzer, so they are treated as mechanism evidence only.

The held-out 100-image screen in this repository rejected 16-pixel mask-grid
alignment and retained an 8-pixel context halo.  On the same screen,
`halo8 + POST(sigma=1, QP>=45)` reached -11.53% H.264 and -4.54% H.265 BD-rate,
with the worst per-QP mAP gap at -0.0198.

## Frozen runs

Both runs use the same deterministic COCO shuffle seed (`20260918`) so the two
gate thresholds can be compared point-for-point.

- Data: 500 annotated COCO val2017 images, letterboxed to 320 x 320.
- QPs: 30, 35, 40, 45, 50; real x264/x265, medium preset, intra-only.
- Mask analyzer: Faster R-CNN MobileNet-V3-Large-FPN, score 0.20.
- Held-out evaluator: Faster R-CNN ResNet50-FPN, score 0.05.
- PRE: background Gaussian sigma 4, dilation 0.30, minimum box halo 8 px,
  feather 8 px, no mask-grid alignment, identity ROI.
- POST: full-frame Gaussian sigma 1 after decode, zero extra bits.
- `wagur124705`: POST gate QP >= 40.
- `htran123456`: POST gate QP >= 45.
- Inline bootstrap is disabled so the GPU run reliably preserves all detector
  records.  The saved paired per-image records are the input to 1,000 offline
  bootstrap draws after both kernels finish.

## Reading rule

1. Anchors and PRE-only curves must match across the two runs; otherwise the
   threshold comparison is invalid.
2. A candidate must have negative BD-rate on both codecs and a per-QP mAP gap
   no worse than -0.05.
3. For each codec, select QP40 or QP45 only from these two frozen candidates;
   do not tune another threshold on this sample.
4. The selected candidate becomes the headline only if the paired 95% CI
   excludes zero for at least one codec.  Otherwise retain it as exploratory.

