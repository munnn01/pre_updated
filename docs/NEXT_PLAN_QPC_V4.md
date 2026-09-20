# QPC-V4: matched retraining control and QP-conditioned development

**Status:** protocol frozen after a full Kaggle-history audit; no QPC-V4
checkpoint/result was found in any retrievable notebook source or output.

## Focal question

Does learned QP conditioning improve real-codec AR BD-rate beyond ordinary
retraining variance, and does BD-weighted QP sampling improve utilization of
that condition?

## Observation and assumptions

- **Located evidence:** ROI-V3 has no training stage. Its 200 clips per job are
  evaluation samples, so “not enough training” cannot explain its failure.
- **Located evidence:** the current Kinetics hash split has 8,632 train, 1,010
  validation and 1,159 untouched test clips over 400 classes.
- **Located evidence:** the valid additive-kappa10 lineage used 16 epochs and
  approximately 17,264 optimizer steps; a six-epoch QPC screen would be an
  undertrained comparison.
- **Prediction:** useful QP conditioning changes edit RMS/HF between QP30 and
  QP50 and improves validation real-codec BD-rate relative to the matched
  unconditioned family, not merely relative to an old checkpoint.
- **Disconfirmation:** FiLM behavior remains within 3% twin noise, or its
  two-seed median does not beat matched unconditioned retraining.

## Candidates considered

1. Train ROI-V3 longer — rejected: it contains no learned parameters.
2. Increase only dataset size — deferred: the cleaned proxy split is small
   (about 21.6 training clips/class), but 16 matched epochs first isolate the
   effect of conditioning without changing two factors at once.
3. One QPC run — rejected: cannot separate conditioning from seed variance.
4. **Decision:** matched 3-family x 3-seed experiment:
   unconditioned additive, uniform-QP FiLM, and BD-weighted-QP FiLM.
5. Codec+QP conditioning through real-codec STE — deferred until QPC-V4 shows
   FiLM utilization; otherwise it adds a second mechanism and much higher cost.

## Frozen training protocol

- Dataset: `qktttttttttt/kineticscleaned`, canonical hash split.
- Train/val only; test remains unread during training and gate selection.
- 16 frames, size 128, stride 2, batch 8, 16 epochs.
- Teachers: sampled R3D-18 and MC3-18; later evaluation is R2Plus1D-18.
- QPs: `30,35,40,45,50`; validation evaluates all five equally.
- Uniform weights: no sampling override.
- BD weights aligned to the QP points:
  `[0.089, 0.215, 0.288, 0.285, 0.123]`.
- The only family deltas are architecture and QP sampling. Each family uses
  seeds 0, 1 and 2.

| Account | Family | Seed |
|---|---|---:|
| `wagur124705` | additive control | 0 |
| `htran123456` | additive control | 1 |
| `hoangminhhuy123` | QPC uniform | 0 |
| `shungg05` | QPC uniform | 1 |
| `vtk269` | QPC BD-weighted | 0 |
| `nguyenhoanglan1232` | QPC BD-weighted | 1 |
| `dngbolm` | additive control | 2 |
| `hieusunday0412` | QPC uniform | 2 |
| `linhowi05` | QPC BD-weighted | 2 |

## Gates and continuation rule

Each job runs a label-free audit on validation clips only:

1. no-blow-up RMS `< 0.14`;
2. QPC behavior differs between QP30 and QP50 by more than 3% in RMS or HF;
3. checkpoints and validation history are complete with no NaN/OOM;
4. unconditioned controls must be invariant to QP at the model boundary.

After all nine complete, compare family medians and seed ranges without reading
test. Continue only the best valid family to a real-codec validation screen.
Only after that family and strength are frozen may the 1,159-clip test be used.

Continue QPC only if its three-seed median beats the matched control by at least
one percentage point on both H.264 and H.265 validation BD-rate, with no seed
regressing by more than two points, and the FiLM utilization gate passes. Then
freeze family/strength and run the untouched 1,159-clip test. If this rule fails,
close scalar QP conditioning; a poor result alone is not a reason to add epochs
or tune on test.
