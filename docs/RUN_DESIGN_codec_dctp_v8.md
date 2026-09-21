# Codec-DCTP V8 composition response screen

## Question

Codec-specific V7 and fixed DCTP each produced modest, reproducible gains but
neither reached the requested `-15%` BD-rate target. Their reported BD-rates
cannot be added. V8 directly evaluates whether a bounded fraction of the
codec-specific learned residual can be retained before the best structural DCT
projection for that codec.

This is an adaptive validation experiment. It uses a new deterministic
validation screen (`codec-dctp-v8-val-v1`, shard `0/10`) and does not read the
test split.

## Blocks and source checkpoints

Two independently calibrated V7 checkpoints are retained for each codec. Each
checkpoint is a block and runs its baseline plus every registered composition
level on the same clips and anchors.

| Account | Codec | V7 arm | Checkpoint SHA-256 |
|---|---|---|---|
| `vtk269` | H.264 | `t0_lr15` rep 1 | `1e1cdeca65c5bd6c101db8c82b8da8716ba778e7a72f5ead4d9aa31f08caa4a1` |
| `dngbolm` | H.264 | `t0_lr15` rep 2 | `ca0beebd811652e7200f50c9ce03a76baf0f36bdb41ae10dbed1926596fb6e58` |
| `huolgggnuyen` | H.265 | `tm15_lr15` rep 1 | `40e3d4123951ef30b3424a76aa8b3bf1fdbb7f7ba1655e37629ab2d74649e280` |
| `thuha1205` | H.265 | `tm15_lr15` rep 2 | `7b1803348901bda5a0b7b6fbb45f106309d3a560d07675bd594ec077e84764fa` |

The completed private V7 kernel is attached only to a new kernel owned by the
same account. No additional checkpoint is made public.

## Registered response levels

- H.264 structural setting: DCT `1.0`, temporal `0.0`; residual scale
  `{0, 0.125, 0.25, 0.5}`.
- H.265 structural setting: DCT `0.6`, temporal `0.35`; residual scale
  `{0, 0.0625, 0.125, 0.25}`.
- Every block also evaluates the unprojected V7 checkpoint as a paired baseline.
- Real selected codec only, QP `[30,35,40,45,50]`, preset medium, held-out
  R2Plus1D-18, per-sequence rows enabled.

The zero-residual level measures the structural projection alone. The interior
levels test curvature and whether a small semantic residual provides a better
rate-accuracy compromise than either endpoint.

## Gate and selection

For a residual level to continue, both checkpoint blocks for that codec must:

1. produce negative same-codec Top-1 BD-rate versus the raw-codec anchor;
2. improve over their paired unprojected V7 baseline;
3. keep every Top-1 QP gap versus raw anchor at least `-0.05`.

The two codec winners are then hard-routed by deployment codec. The combined
candidate meets the project target only when both selected codec BD-rates are
negative and at least one is `<= -15%`. If V8 fails, do not tune further on this
screen; move to trained subtractive prediction or codec-native ROI/rate control.
