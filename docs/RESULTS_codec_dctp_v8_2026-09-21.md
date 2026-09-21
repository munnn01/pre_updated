# Codec-DCTP V8 validation results (2026-09-21)

## Decision

The direct V7-residual plus DCT composition is valid and produces small
replicated gains, but it does not reach the requested `-15%` Top-1 BD-rate.
Close residual-scale tuning.  Continue with Semantic-DCTP V9, which uses the
learned residual only as a block-protection signal and never transmits the
rate-adding residual.

All four jobs completed with `rc=0`.  Each evaluated 102 clips from the same
fresh validation screen (`codec-dctp-v8-val-v1`, shard `0/10`).  H.264 anchors
have hash prefix `be68df22b035`; H.265 anchors have `ca1a0c5f9c32`.  Source
checkpoint SHA-256 values match the registered V7 checkpoints.

## Results

Negative Top-1 BD-rate is better.  `Min gap` is the worst preprocessed-minus-
anchor Top-1 difference over the five QPs.

| Codec | Arm | Replica 1 BD-rate | Replica 2 BD-rate | Min gap range | Mean added-rate range |
|---|---|---:|---:|---:|---:|
| H.264 | V7 baseline | +0.09% | +2.98% | +0.05 to +0.06 | +31.20% to +31.54% |
| H.264 | residual 0.000 | -0.47% | -0.47% | -0.09 | -0.82% |
| H.264 | residual 0.125 | -3.02% | -0.93% | -0.05 to -0.03 | +2.29% to +2.33% |
| H.264 | residual 0.250 | +0.40% | -1.41% | -0.03 to -0.02 | +5.69% to +5.74% |
| H.264 | residual 0.500 | **-6.88%** | **-1.23%** | +0.02 to +0.03 | +13.09% to +13.28% |
| H.265 | V7 baseline | +7.70% | +7.80% | -0.02 to -0.01 | +21.61% to +21.71% |
| H.265 | residual 0.000 | +1.43% | +1.43% | -0.04 | -1.07% |
| H.265 | residual 0.0625 | **-1.15%** | **-0.29%** | -0.03 to -0.02 | -0.01% |
| H.265 | residual 0.125 | +0.95% | -3.24% | -0.03 | +1.05% to +1.09% |
| H.265 | residual 0.250 | -0.85% | +0.30% | -0.02 to -0.01 | +3.27% to +3.34% |

H.264 residual `0.5` and H.265 residual `0.0625` are the only useful replicated
settings under the accuracy guard.  Neither codec has a replicate at or below
`-15%`, and the checkpoint spread is too large to justify selecting the best
single run.  The rate decomposition identifies the mechanism: the semantic
residual still adds up to 13% bitrate even after scaling, while zero-residual
DCT alone is not aggressive enough.
