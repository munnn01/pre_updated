# QPC-V4 real-codec validation results (2026-09-20)

## Decision

**Close scalar QP conditioning and do not run the 1,159-clip test.** QPC-uniform
improves the median relative to matched retraining controls, but every absolute
top-1 BD-rate remains positive, one H.264 seed violates the preregistered
no-regression rule, and paired bootstrap intervals cross zero on both codecs.

The next defensible AR direction is codec-native rate-constrained training,
where the objective directly penalizes added real-codec bits per QP/codec. More
epochs, another QP sampling distribution, or another saliency-mask grid would
not address the observed failure mechanism.

## Validity checks

- Results are from Kaggle notebook **version 2** only. Version 1 is an excluded
  infrastructure failure with `n_eval=0` caused by an unsalted hash collision.
- Every version-2 job evaluated the same 202 validation clips from salted hash
  shard `0/5`; all six anchor curves have identical SHA-256 prefix
  `f12839fd9f71`.
- Analyzer: held-out R2Plus1D-18. Codecs: real x264 and x265, preset medium.
  QPs: `[30,35,40,45,50]`.
- All six jobs used their seed-matched epoch-14 best checkpoint.
- The untouched 1,159-clip test split was not read.

## Top-1 BD-rate results

Negative is better. Positive means preprocessing requires more bitrate than the
unprocessed anchor at equal top-1 accuracy.

| Seed | Control H.264 | QPC H.264 | QPC-control | Control H.265 | QPC H.265 | QPC-control |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | +2.22% | +4.52% | **+2.31 pp** | +4.88% | +0.74% | -4.14 pp |
| 1 | +8.00% | +0.77% | -7.24 pp | +5.12% | +0.65% | -4.47 pp |
| 2 | +3.65% | +1.40% | -2.25 pp | +0.55% | +0.73% | +0.18 pp |
| Median | +3.65% | +1.40% | **-2.25 pp** | +4.88% | +0.73% | **-4.14 pp** |

The family median improves by more than one percentage point on both codecs,
but H.264 seed 0 regresses by +2.31 points, exceeding the registered maximum
regression of +2 points. More importantly, no QPC run has negative top-1
BD-rate, let alone the desired `<= -15%`.

## Accuracy and bitrate decomposition

QPC does improve accuracy at every QP. Across the three QPC seeds, the smallest
per-QP gain over the anchor is +1.98 top-1 percentage points on H.264 and +3.96
points on H.265. The failure is therefore not task preservation; it is rate
cost.

Median QPC bitrate overhead relative to the anchor:

| Codec | QP50 | QP45 | QP40 | QP35 | QP30 |
|---|---:|---:|---:|---:|---:|
| H.264 | +24.37% | +34.23% | +42.67% | +47.01% | +46.44% |
| H.265 | +7.75% | +17.22% | +29.08% | +37.01% | +40.17% |

Conditioning behaves in the intended direction: compared with the control it
reduces QP30 overhead (H.264 `58.48% -> 46.44%`; H.265
`50.87% -> 40.17%`) while allowing more intervention at QP50. The remaining
overhead is still far too large for the accuracy gain, so scalar conditioning
fixes allocation but not compressibility.

## Target-probability sensitivity check

Target probability is smoother than 0/1 top-1 on 202 clips. Its family-median
QPC-control differences are only `-0.74 pp` on H.264 and `-0.36 pp` on H.265,
both below the one-point continuation threshold.

An exploratory paired sequence bootstrap (500 resamples, seed 260920) gives:

| Metric | Codec | Median difference | 95% interval | P(improvement) | P(improvement >=1 pp) |
|---|---|---:|---:|---:|---:|
| top-1 BD-rate | H.264 | -2.25 pp | [-7.40, +3.76] | 0.764 | 0.652 |
| top-1 BD-rate | H.265 | -4.14 pp | [-7.24, +0.90] | 0.940 | 0.858 |
| target-prob BD-rate | H.264 | -0.74 pp | [-3.35, +2.34] | 0.638 | 0.360 |
| target-prob BD-rate | H.265 | -0.36 pp | [-2.98, +1.25] | 0.774 | 0.426 |

All intervals include zero. These intervals are exploratory because the
continuation rule was defined on point estimates and seed behavior, but they
reinforce the decision not to spend the untouched test set.

## Preregistered gate evaluation

| Criterion | Result |
|---|---|
| Median QPC improvement >=1 pp on both codecs | Pass for top-1 |
| No seed regresses by more than 2 pp | **Fail: H.264 seed 0 = +2.31 pp** |
| Per-QP accuracy guardrail | Pass |
| Absolute goal `BD-rate <= -15%` on both codecs | **Fail: all values positive** |
| Evidence robust under paired resampling | **Fail: both top-1 intervals cross zero** |

Overall continuation decision: **FAIL — no confirmatory test**.

## Next method

The evidence supports one narrow conclusion: QP conditioning is learned and
changes allocation correctly, but the virtual-codec objective rewards edits
whose real H.264/H.265 rate cost is excessive. A next experiment should change
the rate model, not the dataset split or conditioning sampler:

1. retain QP and codec conditioning;
2. alternate H.264/H.265-aware STE or a calibrated differentiable rate proxy;
3. optimize added bitrate relative to the raw anchor, with a separate Lagrange
   multiplier for each codec/QP;
4. keep the same three-seed additive control;
5. require negative validation BD-rate before reading test.

This is a new mechanism and must receive a new preregistered protocol. It should
not be presented as an extension of the failed scalar-QPC result.
