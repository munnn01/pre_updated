# Full-validation V11: locked H.264 replications and frozen H.265

V11 is confirmatory evaluation only. It performs no training, checkpoint
selection, coefficient search, or early stopping. All curves are measured on the
complete locked Kinetics validation split of 1,010 clips using five real-codec QPs
and the held-out `r2plus1d_18` evaluator.

## Experimental units

- H.264 has five independent training-seed checkpoints. Each account evaluates
  its own immutable V10 `t0_lr15` checkpoint on the same complete validation set.
  Training seed is the replication unit; clips are repeated measurements within a
  seed and are not counted as five additional model replications.
- H.265 has one frozen checkpoint with SHA-256
  `a3580b32e1554071811888238ea249ce9ae26e7b32392a2d975e4bd6cd0a9c8e`.
  It is evaluated once on the complete set. Repeating it on five accounts would
  be deterministic pseudoreplication, so one separate account runs it.

## Locked checkpoint allocation

| Account | Codec | Checkpoint SHA-256 | Source |
|---|---|---|---|
| `baooo25r` | H.264 | `53a16b33d625c8212e3bbd8284e91cae6af5501f8f6af7a95191cf49e6b89f45` | account-local V10 output |
| `dieulinhh` | H.264 | `72da23443c3c0d4fe3a26c8f5d740d46dad63a7ab3bd6810257d4096459b1379` | account-local V10 output |
| `shungg05` | H.264 | `d696adb926235b11d3059d65e3d6d57ea2a44e0c6d47b36c5a5cfc3a0096a51c` | account-local V10 output |
| `huolgggnuyen` | H.264 | `8e00abef306aa602db6ddfa627c8ad0518a062738e1836855fa21ad6a9e76478` | account-local V10 output |
| `trmnguyn111` | H.264 | `8994e4eb8ae24116545d3f3b20f8645d6b6ebc9fecc447a9ce65ef1339f8b433` | account-local V10 output |
| `wagur124705` | H.265 | `a3580b32e1554071811888238ea249ce9ae26e7b32392a2d975e4bd6cd0a9c8e` | frozen public dataset |

Every kernel verifies its checkpoint SHA before evaluation and again after it.
The H.264 kernels additionally verify `target_ratio=0`, `dual_lr=0.015`,
`global_step=500`, `cond_dim=3`, and H.264-only calibration.

## Pre-registered responses and gates

Primary response is Top-1 BD-rate for `prep+codec` versus the unprocessed codec;
lower is better. Secondary response is BD-accuracy; higher is better.

The development target passes only if:

1. full-validation BD-rate is negative for both codecs;
2. at least one codec reaches BD-rate at or below -15%;
3. mean H.264 BD-accuracy across five seeds is non-negative;
4. H.265 BD-accuracy is non-negative; and
5. the five-seed H.264 mean has a 95% confidence interval whose upper bound is
   below zero.

All five H.264 seed results are reported, including failures. No result from the
earlier 44- or 46-clip screens is used in the full-validation estimate.
