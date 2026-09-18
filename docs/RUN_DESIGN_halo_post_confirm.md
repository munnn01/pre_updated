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

Execution is sequential by task at the user's request: phase A evaluates H.264
only and compares QP40 against QP45.  H.265 is not launched until the H.264
result has been read and the next configuration has been fixed.

## Phase A result: H.264

Both 500-image runs produced byte-for-byte identical anchor and PRE-only
curves, so the gate comparison is paired on the same sample.

| gate | halo8 + POST BD-rate | worst per-QP mAP gap |
|---|---:|---:|
| QP >= 40 | -12.097% | -0.00457 |
| QP >= 45 | **-13.145%** | **-0.00414** |

QP45 wins by 1.048 percentage points.  The entire difference is at QP40:
enabling POST there lowers mAP by 0.00501 relative to PRE-only.  H.264 is
therefore frozen at **QP >= 45**; QP40 is closed and must not be retuned.

## Phase B: H.265-only comparison

The completed QP40 run already contains the H.265 QP40 curve on the same seed.
The only additional run is H.265-only QP45 with all other fields unchanged.
QP40 and QP45 are compared only after confirming exact anchor and PRE-only
curve equality.  No additional threshold is opened on this sample.

## Phase B result and final gate

The H.265 QP45 run matched the QP40 anchor and PRE-only curves exactly.

| gate | halo8 + POST BD-rate | worst per-QP mAP gap |
|---|---:|---:|
| QP >= 40 | -6.734% | -0.01007 |
| QP >= 45 | **-8.017%** | **-0.00651** |

QP45 wins by 1.283 percentage points.  As on H.264, applying POST at QP40
damages the otherwise preserved QP40 point.  The final shared decoder rule is
therefore **Gaussian POST sigma 1 only when QP >= 45 for both codecs**.

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
