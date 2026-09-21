# CRC-V5-PC1: asymmetric per-codec dual follow-up

## Scientific status

This is a new, pre-registered follow-up. It does not alter or retroactively
replace CRC-V5. The `baooo25r/t0_lr5` H.265 result motivated the hypothesis and
therefore remains exploratory evidence, not a confirmatory replicate.

## Question and primary contrast

Does increasing only the H.264 dual update rate improve H.264 BD-rate without
removing the H.265 benefit of the symmetric `t0_lr5` recipe?

- Concurrent control: H.264 `dual_lr=0.005`, H.265 `dual_lr=0.005`.
- Treatment: H.264 `dual_lr=0.015`, H.265 `dual_lr=0.005`.
- Both arms use `target_ratio=0.0`, the same Stage-1 checkpoint bytes, training
  clips, 500-step randomized complete codec/QP blocks, evaluator, and seed.
- The account/checkpoint seed is the independent replicate. QPs and clips are
  repeated measurements, not independent replicates.

The only intended difference between paired control and treatment is H.264's
dual learning rate. This isolates that change from the H.265 setting.

## Execution and evaluation

Use `configs/crc_v5_ar.yaml` with scalar `t0_lr5` for the concurrent control and
`configs/crc_v5_per_codec_ar.yaml` for treatment. Randomize control/treatment run
order within each account block. Use at least three independently trained
Stage-1 checkpoints and persist each checkpoint SHA-256.

The locked screening set is validation shard `1/20` with salt
`crc-v5-pc1-val-v1`; it is distinct from CRC-V5's shard `0/20`. Do not inspect
the test split. If screening passes, repeat the exact selected recipe on the
complete validation split before any test evaluation.

## Pre-registered continuation gates

1. Infrastructure: paired arms share Stage-1 SHA, data IDs, anchors, preset,
   evaluator, and exactly 500 calibration steps.
2. Mechanism: all ten codec/QP duals are finite and at least one H.264 dual
   moves from `lambda_init`.
3. H.264 primary: treatment improves paired H.264 BD-rate by at least 2
   percentage points in at least two of three independent blocks.
4. H.265 non-inferiority: treatment loses no more than 2 BD-rate percentage
   points versus paired control, and no codec/QP point loses more than 0.05
   Top-1.
5. Absolute continuation: both codec BD-rates are negative. The desired
   `<= -15%` goal remains a later confirmation target and is not relaxed.

Changing thresholds or selecting accounts after seeing results makes the run
exploratory and requires another fresh validation shard for confirmation.
