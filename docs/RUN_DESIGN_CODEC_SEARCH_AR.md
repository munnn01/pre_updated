# Real-codec search and spatial-resampling screen for AR

This protocol is frozen before the two Kaggle jobs are started. The primary
goal remains negative Top-1 BD-rate on both H.264 and H.265, with at least one
codec at or below -15%. This 104-clip screen is exploratory development work.

## Mechanism

Encode six fixed versions of each 16-frame, 128x128 source clip at each QP:
identity128, area112, area96, area112_up128, blur020_128, blur040_128.
The 112-up-128 control separates antialiasing from fewer coded pixels. The
decoder-side task analyzer is frozen r2plus1d_18 with 112-pixel input.
The independent r3d_18 is only a cross-backbone evaluator.

For every candidate, measure the actual H.264/H.265 elementary-stream bytes.
All reported bpp use the original denominator 16*128*128, including 112/96
streams. This is essential to a fair rate comparison. The same decoded clip
is scored by the frozen analyzer for KL divergence from source logits,
layer-2 feature cosine distance, Top-1 and target probability.

At inference, select the lowest-byte candidate whose source KL and feature
distance do not exceed identity's values plus train-calibrated tolerances.
For high-confidence source predictions (>=0.6), candidate Top-1 must match
the source Top-1. If no candidate qualifies, transmit identity. This selection
uses source pixels, measured bytes and analyzer outputs; no ground-truth label.

## Calibration and evaluation

- Dataset: public qktttttttttt/kineticscleaned, canonical hash split with
  test fingerprint 30f083f8520a. Train 400 class-balanced clips, validation
  104 fixed class-balanced clips; no test-set measurement in this screen.
- QPs: 30,35,40,45,50; FFmpeg libx264/libx265, preset medium, yuv420p.
- Calibrate only on the 400 training clips. Grid:
  KL slack [0,.005,.02,.05,.10], feature slack [0,.005,.02,.05]. Select the
  most negative train Top-1 BD-rate that also has nonnegative BD-accuracy,
  worst per-QP Top-1 drop at most one percentage point, and mean same-QP
  rate ratio below one. If none qualify, use identity.
- Validation reports every fixed arm and policy on matched r2plus1d_18 and
  cross-backbone r3d_18. Per-clip/arm records, calibration grid, clean
  accuracy, rate curves and actual dimensions are saved. No validation
  labels influence thresholds or candidate selection.
- A positive screen is still exploratory. Before any >=15% claim, use a
  larger development set, independent source-video holdout, paired
  bootstrap intervals, per-QP accuracy checks and multiple codec runs.

Failure interpretation: if 112/96 only reduce pixels while destroying Top-1,
or the measured-byte search returns identity, do not train a selector to
imitate it. The clean-resize control determines whether the failure already
occurs before encoding.
