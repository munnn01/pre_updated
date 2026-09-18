# OD generalization matrix and AR transfer decision

Status: point-estimate matrix complete. Date: 2026-09-19.

The OD method frozen before this matrix is background Gaussian sigma 4, an
8-pixel object-context halo, and decoder Gaussian sigma 1 only at QP >= 45.
Negative BD-rate means fewer bits at equal task quality.

| Screen | Evaluator / change | H.264 BD-rate | H.265 BD-rate | Worst OD mAP gap |
|---|---|---:|---:|---:|
| Original held-out, n=500 | Faster R-CNN ResNet50 | -13.145% | -8.017% | -0.00651 |
| New sample, n=500 | Faster R-CNN ResNet50 | -15.412% | -7.959% | -0.00613 |
| Cross-evaluator, n=200 | FCOS ResNet50 | -7.119% | -7.298% | -0.02055 |
| Cross-evaluator, n=200 | RetinaNet ResNet50 v2 | -7.291% | -5.122% | -0.01451 |
| Reversed analyzer, n=200 | MobileNet evaluator, ResNet50 mask | -8.240% | -3.148% | -0.01404 |

Every pre-registered OD screen has a negative point-estimate BD-rate for both
available codecs and stays above the -0.05 mAP-gap guardrail.  The development
decision is therefore to freeze the OD transform and stop parameter search.
This is not yet a publication-level uncertainty claim: per-image records are
retained for paired confidence intervals and a final full-scale evaluation.

The direct AR transfer failed.  Whole-frame decoder Gaussian sigma 1 produced
target-probability BD-rates of +6.102% (H.264) and +2.454% (H.265), with top-1
BD-rates of +5.241% and +2.882%.  At QP45 it lost 8.0 and 8.5 top-1 percentage
points respectively.  The result rejects whole-frame Gaussian POST for the
frozen r3d_18 analyzer; it does not reject AR-specific, motion-preserving
post-processing.

## Decision log

- **Decision:** freeze OD architecture and direct new development compute to AR.
- **Supported claim:** the OD mechanism generalizes at the point-estimate level
  across a new sample, three held-out evaluators, both H.264 and H.265, and a
  reversed mask/evaluator pair.
- **Not yet supported:** exact population effect size or statistical
  significance; those require paired uncertainty and the final evaluation.
- **Rejected alternative:** further OD sigma/halo/QP search on these samples,
  because it would add multiplicity without resolving the remaining uncertainty.
- **AR next step:** test a decoder-only motion mask that preserves temporal
  change regions exactly and denoises only static regions.
