# R0 confirmatory design — held-out OD background suppression

Status: frozen before the 500-image run.  The two exploratory reads are fully
disclosed below and must not be pooled with the confirmatory sample.

## Frozen configuration

- Data: COCO 2017 validation, deterministic seed 0, first 500 items after the
  probe's seeded shuffle, letterboxed to 320 x 320.
- Mask analyzer: pretrained Faster R-CNN MobileNet-V3-Large-FPN, score threshold
  0.20, box dilation 0.30.
- Held-out evaluator: pretrained Faster R-CNN ResNet50-FPN, score threshold 0.05.
- Candidate: background Gaussian `sigma=4`; protected boxes remain exact and an
  8-pixel soft protection band removes the hard boundary.
- Anchor: unmodified decoded image.
- Codecs: real x264 and x265, medium preset, QP 30, 35, 40, 45, 50.  A one-frame
  item is necessarily intra-only.
- Metric: COCO mAP@[.50:.95], BD-rate at equal mAP.
- Uncertainty: 1,000 paired image-bootstrap draws with replacement, identical
  sampled occurrences for anchor and candidate, separate CI per codec.

No sigma, mask threshold, dilation, feather, QP, detector, image count, or seed
may be changed after this run starts.

## Registered reading

The candidate passes R0 only if:

1. the point-estimate BD-rate is below zero for both codecs;
2. the 95% bootstrap CI excludes zero for at least one codec; and
3. candidate mAP is never more than 0.05 below its same-codec anchor at any QP.

If either codec has non-negative BD-rate, the shared suppression claim is
rejected.  R1 may open only after this R0 gate passes; R1 must subsequently beat
this exact `sigma=4` arm rather than a retuned R0.

## Exploratory evidence that selected the frozen arm

| Run | Images | Candidate | H.264 BD | H.265 BD | Reading |
|---|---:|---|---:|---:|---|
| v1 | 50 | hard-mask sigma 8 | +11.26% | +3.60% | rejected |
| v3 | 50 | feathered sigma 2 | -1.46% | +4.31% | rejected |
| v3 | 50 | feathered sigma 4 | -4.85% | +3.89% | mixed |
| v4 | 200 | feathered sigma 4 | -9.97% | -4.92% | advances to confirmatory |

The OD-derived action-recognition tube is a separate rejected exploratory arm:
H.264/H.265 BD-rate was +9.80%/+8.57% on 20 Kinetics clips.  It is not rerun or
silently combined with R0.  The existing full-sample AR checkpoint remains the
AR branch while this experiment answers only the missing OD question.
