# V12: longer codec-specific calibration and full 1,010-clip evaluation

V11 established that the 500-step checkpoints do not generalize: H.264 averaged
`+1.92%` BD-rate and the frozen H.265 checkpoint produced `+1.05%` on the full
validation set. V12 tests the specific hypothesis that 500 real-codec calibration
steps were insufficient. Training duration is the only intended treatment change.

## Locked training treatment

- Resume optimizer, model, per-QP dual variables, epoch and step state from the
  audited 500-step checkpoint; do not restart fine-tuning.
- The checkpoint records `epoch=1, global_step=500`. Set the terminal epoch to 4,
  which runs three additional complete epochs. With 8,632 training clips and batch
  size 2, this adds approximately 12,948 steps and ends near step 13,448.
- Train one codec per checkpoint (`ste_alternate=false`).
- Preserve all architecture, losses, uniform randomized-complete-block QP schedule,
  learning rate and rate targets.
- H.264 preserves `target_ratio=0`, `dual_lr=0.015`.
- H.265 preserves `target_ratio=0`, `dual_lr=0.005`.
- Early stopping and cosine decay remain disabled. The final checkpoint
  (`preprocessor_last.pth`) is evaluated, so epoch selection cannot leak validation
  performance into the result.

## Replication and allocation

Five independent H.264 lineages continue their own V10 checkpoints. Each pair is
copied byte-for-byte into a private dataset owned by the target account; no new
H.264 checkpoint is published publicly:

| Target account | Origin lineage | Seed | Private dataset |
|---|---|---:|---|
| `qktttttttttt` | `baooo25r` | 282001 | `qktttttttttt/crc-v5-h264-v12-source-s282001` |
| `ngynanhthuw` | `dieulinhh` | 282002 | `ngynanhthuw/crc-v5-h264-v12-source-s282002` |
| `tranthihongdieu` | `shungg05` | 282003 | `tranthihongdieu/crc-v5-h264-v12-source-s282003` |
| `linhowi05` | `huolgggnuyen` | 282004 | `linhowi05/crc-v5-h264-v12-source-s282004` |
| `hoangminhhuy123` | `trmnguyn111` | 282005 | `hoangminhhuy123/crc-v5-h264-v12-source-s282005` |

`thuha1205` runs one H.265 continuation with seed 283001 from the immutable public
500-step checkpoint. It is one trained lineage, not five replications.

## Evaluation and pre-registered gates

Every final checkpoint is evaluated on all 1,010 locked validation clips with real
H.264 or H.265 at QP 30/35/40/45/50 and held-out `r2plus1d_18`. No screening shard
is used for selection.

For H.264, report all five seeds, mean, SD and 95% confidence interval. H.265 is
reported as one fixed continuation without a between-seed confidence interval. V12
passes only if:

1. mean H.264 BD-rate and the H.265 BD-rate are negative;
2. the upper 95% confidence bound for H.264 is below zero;
3. at least one codec has BD-rate at or below -15%; and
4. mean H.264 BD-accuracy and H.265 BD-accuracy are non-negative.

If V12 fails, training duration alone is rejected as the explanation; the next
experiment must change the QP-dependent edit policy or rate objective rather than
continuing the same recipe for still more epochs.
