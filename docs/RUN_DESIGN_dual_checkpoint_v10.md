# Dual-checkpoint V10: frozen H.265 candidate and H.264 factorial

## Question and immutable positive control

Can H.264 be optimized independently while preserving, byte-for-byte, the
CRC-V5 H.265 candidate that measured `-24.2623%` BD-rate and `+0.03183`
BD-accuracy on 44 clips from validation shard `0/20`?

The H.265 checkpoint SHA-256 is fixed to
`a3580b32e1554071811888238ea249ce9ae26e7b32392a2d975e4bd6cd0a9c8e`.
It is mounted read-only, never passed to an optimizer, and checked again after
every H.264 arm. The historical score is a positive-control reproduction, not a
full-validation claim. A new locked shard is also evaluated to measure transfer.

## H.264 treatments

Every account runs every treatment from a fresh copy of the immutable H.265
checkpoint. `codec.ste_codec=h264` and `codec.ste_alternate=false`, so no H.265
step can update the H.264 copy and no H.264 step can update the frozen H.265 file.

| Arm | Target added-rate | Dual LR | Role |
|---|---:|---:|---|
| `t0_lr5` | 0% | 0.005 | Historical coefficient |
| `t0_lr15` | 0% | 0.015 | High dual step |
| `tm5_lr5` | -5% | 0.005 | Low target, low step |
| `tm5_lr15` | -5% | 0.015 | Low target, high step |
| `tm25_lr10` | -2.5% | 0.010 | Center point / curvature check |

The four corners form a full 2x2 factorial over target and dual LR. The center
point checks whether interpolation is plausible before any response-surface
follow-up.

## Blocking, replication, and order

The five Kaggle accounts are run blocks. Every block contains all five arms and
uses one calibration seed, so treatment is not confounded with account. Arm order
is a cyclic Latin-square schedule: every arm occupies each run position once.
The independent unit is a complete 500-step H.264 calibration run; QPs and clips
are repeated measurements, not independent replicates.

| Account | Seed | Arm order |
|---|---:|---|
| `baooo25r` | 282001 | t0_lr5, t0_lr15, tm5_lr5, tm5_lr15, tm25_lr10 |
| `dieulinhh` | 282002 | t0_lr15, tm5_lr5, tm5_lr15, tm25_lr10, t0_lr5 |
| `shungg05` | 282003 | tm5_lr5, tm5_lr15, tm25_lr10, t0_lr5, t0_lr15 |
| `huolgggnuyen` | 282004 | tm5_lr15, tm25_lr10, t0_lr5, t0_lr15, tm5_lr5 |
| `trmnguyn111` | 282005 | tm25_lr10, t0_lr5, t0_lr15, tm5_lr5, tm5_lr15 |

## Locked evaluation and gates

- H.265 reproduction: shard `0/20`, salt `crc-v5-val-v1`; requires `n=44`,
  BD-rate within 0.25 percentage points of `-24.2623`, and BD-accuracy within
  0.01 of `+0.03183`.
- Fresh H.265/H.264 screen: shard `2/20`, salt
  `dual-checkpoint-v10-val-v1`. It is not inspected before launch.
- Infrastructure: every source SHA matches, every H.264 run has 500 steps and
  exactly five finite H.264 duals, and the H.265 SHA remains unchanged.
- H.264 continuation: median BD-rate `<= -15%`, at least three of five block
  results `<= -15%`, and no QP loses more than 0.05 Top-1 versus its anchor.
- H.265 generalization is reported separately on the fresh shard. Failure there
  invalidates a general H.265 claim but cannot alter the frozen historical model.

Only a selected H.264 arm that passes these gates proceeds to full validation.
The test split remains untouched.
