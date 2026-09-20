# QPC-V4 training results and real-codec validation decision (2026-09-20)

> Final status: the corrected real-codec validation completed and QPC did not
> pass its continuation gate. See `QPC_V4_REAL_CODEC_RESULTS_2026-09-20.md`.

## Outcome

All nine matched training jobs completed successfully. The mechanism gate is
positive: QP-conditioned models change their behavior with QP, remain inside
the preregistered edit regime, and do not blow up. Uniform-QP QPC is the selected
family for the real-codec validation screen. No test clips have been used.

This is **not yet evidence of BD-rate improvement**. The observations below are
proxy validation loss and label-free edit diagnostics. The next six jobs compare
three uniform-QPC seeds against their three matched unconditioned controls on
the same deterministic validation subset using real H.264 and H.265.

## Checkpoint integrity

- Every job produced best and last checkpoints, a 16-epoch log, audit JSON and
  audit return code 0.
- All checkpoint tensors are finite.
- Parameter counts match the intended models: additive control 9,795; QPC
  10,371.
- All best checkpoints are epoch 14 (15,106 optimizer steps); last checkpoints
  are epoch 16. Epochs 15--16 degraded validation loss, so the saved best model
  selection behaved correctly.
- Sampling logs confirm uniform sampling for control/uniform arms and
  `[0.089,0.215,0.288,0.285,0.123]` for all BD-weighted arms.

## Matched training results

| Seed | Control best loss | QPC uniform | QPC BD-weighted | Uniform - control | Weighted - control |
|---:|---:|---:|---:|---:|---:|
| 0 | 1.5004 | 1.4370 | 1.4761 | -0.0634 | -0.0243 |
| 1 | 1.5244 | 1.4921 | 1.4791 | -0.0323 | -0.0453 |
| 2 | 1.4379 | 1.4206 | 1.4270 | -0.0173 | -0.0109 |
| Median family value | 1.5004 | 1.4370 | 1.4761 |  |  |

Both conditioned families beat their seed-matched control on proxy validation
loss in all three seeds. Uniform QPC has the best family median and is therefore
selected without inspecting real-codec results.

## Conditioning audit

| Family / seed | HF spread QP30--QP50 | RMS spread | Mean RMS | Overall gate |
|---|---:|---:|---:|---|
| control 0 | 0.00% | 0.00% | 0.06044 | pass |
| control 1 | 0.00% | 0.00% | 0.06068 | pass |
| control 2 | 0.00% | 0.00% | 0.06125 | pass |
| uniform 0 | 28.82% | 15.38% | 0.05457 | pass |
| uniform 1 | 25.39% | 11.94% | 0.05536 | pass |
| uniform 2 | 35.27% | 20.64% | 0.05351 | pass |
| BD-weighted 0 | 22.23% | 10.20% | 0.05751 | pass |
| BD-weighted 1 | 18.55% | 10.00% | 0.05716 | pass |
| BD-weighted 2 | 24.25% | 12.16% | 0.05850 | pass |

Controls are exactly condition-invariant, as required. Every QPC job exceeds
the 3% twin-noise utilization threshold by a wide margin. Uniform conditioning
is behaviorally stronger and yields slightly smaller edits than BD-weighted.

## Evidence assessment

Strengths:

- three matched seeds per family;
- identical data, teacher, loss, optimizer and epoch budget;
- condition-invariant negative controls behave exactly as predicted;
- family choice was made before real-codec validation.

Important limitations:

- the proxy validation loss covers only the configured eight validation
  batches per QP, not all 1,010 validation clips;
- the label-free conditioning audit uses 32 validation clips;
- lower virtual-codec loss does not guarantee lower real H.264/H.265 BD-rate;
- the desired -15% result remains far outside the historical AR range and is
  not supported by these training diagnostics alone.

## Frozen real-codec screen

Evaluate control and uniform QPC, seeds 0/1/2, on `eval.split=val`, deterministic
salted hash shard `0/5` (about 200 clips), held-out R2Plus1D-18, real x264/x265
preset medium, QP `[30,35,40,45,50]`, with per-sequence records. The salt
`qpc-v4-val-v1` is mandatory because the canonical val split already uses
`md5(key) % 10 == 1`; reusing the unsalted hash modulo 5 makes shard 0 empty.
The first dispatch exposed exactly this zero-sample condition and is excluded
as an infrastructure failure. The six corrected jobs use the exact best
checkpoints above. BD-weighted is not advanced because uniform was selected by
the preregistered proxy criterion.

Continue to the untouched 1,159-clip test only if uniform QPC improves the
three-seed median over retrained control by at least one BD-rate percentage
point on both codecs, no seed regresses by more than two points, and accuracy
guardrails pass. Otherwise close scalar QP conditioning.
