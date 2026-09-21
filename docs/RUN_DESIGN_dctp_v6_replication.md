# DCTP-V6 replicated structural screen after codec-specific V7

## Decision

Codec-specific V7 confirmed that stronger scalar rate penalties reduce BD-rate,
but no H.264 arm was negative in both replicas and the replicated H.265 winner
(`tm15_lr15`) reached only about -3.9% on average. Added bitrate remained about
29--34% for H.264 and 21--24% for H.265. This closes further scalar-dual tuning
on the inspected validation screen.

The next experiment tests a structural mechanism that cannot freely add
high-frequency energy: weak high-frequency 8x8 DCT attenuation, optionally
combined with a bounded semantic residual and static-region temporal smoothing.

## Fixed source and response

- Source checkpoint: public `dieulinhh/crc-v5-t0-lr1-stage1-v1`.
- Required SHA-256:
  `a2ad614b874d756db4b5db5fb2a0cefbae300353ed3044a3c27652d0f1b0e5f2`.
- No training is performed; every arm is a deterministic projection of the same
  source bytes.
- Real x264/x265, preset medium, QP `[30,35,40,45,50]`.
- Held-out R2Plus1D-18 on validation shard `0/10`, salt
  `dctp-v6-screen-v1`. The test split is not read.
- Primary responses: H.264 and H.265 Top-1 BD-rate versus their same-codec raw
  anchors, plus worst-QP Top-1 gap.
- Continuation gate: both codec BD-rates negative, at least one `<= -15%`, and
  every per-QP Top-1 gap `>= -0.05`.

Using the canonical checkpoint rather than the best V7 treatment avoids adaptive
checkpoint-selection bias. A V7 winner can be composed with the structural
winner only after this screen freezes the projection settings.

## Replicated full factorial

The full `2^3` design crosses:

| Factor | Low | High |
|---|---:|---:|
| retained semantic residual | 0.00 | 0.25 |
| weak-HF DCT attenuation | 0.60 | 1.00 |
| static temporal stabilization | 0.00 | 0.35 |

The original randomized eight-account assignment is retained as replicate 1.
The remaining eight accounts form replicate 2; their arm order was shuffled
with `random.Random(220922)`.

| Account | Arm | Replicate |
|---|---|---:|
| `shungg05` | `r25_d10_t0` | 1 |
| `dieulinhh` | `r25_d06_t0` | 1 |
| `huolgggnuyen` | `r0_d06_t35` | 1 |
| `baooo25r` | `r0_d10_t0` | 1 |
| `trmnguyn111` | `r0_d06_t0` | 1 |
| `wagur124705` | `r25_d06_t35` | 1 |
| `htran123456` | `r0_d10_t35` | 1 |
| `vtk269` | `r25_d10_t35` | 1 |
| `dngbolm` | `r25_d06_t0` | 2 |
| `hieusunday0412` | `r25_d06_t35` | 2 |
| `hoangminhhuy123` | `r25_d10_t35` | 2 |
| `linhowi05` | `r25_d10_t0` | 2 |
| `nguyenhoanglan1232` | `r0_d06_t0` | 2 |
| `ngynanhthuw` | `r0_d10_t35` | 2 |
| `thuha1205` | `r0_d10_t0` | 2 |
| `tranthihongdieu` | `r0_d06_t35` | 2 |

Account is the independent run-level replicate. Clips within a run are repeated
measurements and must not be counted as independent treatment replicates.

## Selection rule

Reject an arm unless both replicas satisfy the sign and Top-1 guardrails. Among
survivors, rank by the larger (worse) of the two codec BD-rates, then by the
worst-QP Top-1 loss. If no arm survives, close this fixed DCTP region rather than
tuning on the same shard; the next mechanism must be trained subtractive
prediction or codec-native ROI/rate allocation.
