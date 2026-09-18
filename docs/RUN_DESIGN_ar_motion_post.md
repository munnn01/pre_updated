# Preregistered development screen: motion-preserving AR decoder POST

Status: frozen before launch. Date: 2026-09-19.

## Hypothesis and mechanism

**Observation:** whole-frame Gaussian POST hurts r3d_18 action recognition at
QP45/50 even though the same denoising mechanism helps object detection.

**Hypothesis:** spatial smoothing removes motion-bearing edges required by the
video analyzer.  A zero-bit decoder filter can retain those cues by deriving a
motion mask from adjacent decoded frames, copying high-motion pixels exactly,
and applying Gaussian sigma 1 only to temporally static pixels.

**Prediction:** one of the two frozen motion masks improves target-class
probability BD-rate on both codecs while avoiding the top-1 loss of the
whole-frame Gaussian negative control.

**Disconfirming result:** both motion arms have non-negative target-probability
BD-rate, or materially reduce top-1 at either enabled QP.  That closes this
hand-designed decoder-filter family rather than triggering another grid on the
same split.

## Frozen development matrix

- Split: Kinetics hash `val`; the previous `test` clips are not used for tuning.
- Clips: first 200 deterministic clips, 16 frames, stride 2, 128 px.
- Codecs/QPs: H.264 and H.265, QP 30/35/40/45/50, medium preset.
- Gate: identity below QP45.
- Arms: anchor; whole-frame Gaussian sigma 1 negative control; motion masks at
  quantiles 0.50 and 0.75 with spatial dilation 2 and feather 2.
- Frozen analyzers: r3d_18 (primary), mc3_18 and r2plus1d_18 (mechanism screens).
- Primary metric: target-class probability. Secondary metric: top-1.
- Every arm shares the exact anchor bitstream and persists per-clip records.

## Reading and selection rules

1. An arm passes one analyzer only if target-probability BD-rate is negative on
   both codecs and its top-1 change is no worse than -0.01 at QP45 and QP50.
2. Select at most one quantile: among passing arms, choose the lowest mean
   target-probability BD-rate across codecs. The whole-frame control is never a
   selectable arm.
3. Advance only if the same quantile passes at least two of three analyzers.
4. If advanced, freeze it and evaluate once on the untouched `test` split with
   paired clip bootstrap. Do not reopen quantile, sigma, dilation, feather, or
   QP after observing the test result.
5. A null result is informative: retain the frozen OD result and move AR work
   to training-aware robustness or an explicitly temporal learned method with a
   separate train/validation/test protocol.
