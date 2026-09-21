# Semantic-DCTP V9 replicated response screen

## Question

Can the learned codec/QP residual identify task-important block tubes without
paying the residual's 13--31% bitrate overhead?  V9 runs the V7 editor only to
derive an absolute-intervention map, protects the top blocks consistently over
time, and applies stronger subtractive DCT projection everywhere else.  The
residual itself is multiplied by zero and is never transmitted.

## Experimental unit and blocks

The checkpoint/account is the independent unit; clip rows within a job are
paired repeated measurements.  The two registered V7 checkpoints per codec
remain the blocks:

| Account | Codec | Source arm | Checkpoint SHA-256 |
|---|---|---|---|
| `vtk269` | H.264 | `t0_lr15` rep 1 | `1e1cdeca65c5bd6c101db8c82b8da8716ba778e7a72f5ead4d9aa31f08caa4a1` |
| `dngbolm` | H.264 | `t0_lr15` rep 2 | `ca0beebd811652e7200f50c9ce03a76baf0f36bdb41ae10dbed1926596fb6e58` |
| `huolgggnuyen` | H.265 | `tm15_lr15` rep 1 | `40e3d4123951ef30b3424a76aa8b3bf1fdbb7f7ba1655e37629ab2d74649e280` |
| `thuha1205` | H.265 | `tm15_lr15` rep 2 | `7b1803348901bda5a0b7b6fbb45f106309d3a560d07675bd594ec077e84764fa` |

Completed private V7 kernel output is attached only to a new private kernel on
the same account.  Every job verifies the exact SHA before evaluation.

## Registered response surface

A `2 x 3` grid crosses protected block area `{0.25, 0.50}` with weak-coefficient
threshold `{1, 2, 4}`.  The center point `(0.375, 2)` checks curvature.  Fixed
settings are residual scale `0`, DCT strength `1`, band start `2`, QP slope
`0.75`; temporal strength is `0` for H.264 and `0.35` for H.265 based on the
replicated V6 codec interaction.

Evaluation uses real x264/x265, QP `[30,35,40,45,50]`, held-out R2Plus1D-18,
per-sequence records and a fresh validation salt `semantic-dctp-v9-val-v1`
(shard `0/10`).  The test split is not read.

## Gate

An exact arm advances only if both checkpoint blocks for its codec:

1. have negative same-codec Top-1 BD-rate;
2. keep every Top-1 QP gap at least `-0.05`;
3. improve over the paired V7 baseline.

The codec-routed candidate meets the project target only if both codec winners
are negative and at least one codec reaches `<= -15%`.  If no exact arm passes,
close fixed DCT projection; the next mechanism must learn the subtractive
frequency mask through real-codec rate feedback rather than widen this grid.
