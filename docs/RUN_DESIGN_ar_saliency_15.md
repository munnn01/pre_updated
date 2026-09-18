# Development screen: action-saliency suppression for -15% AR BD-rate

Status: frozen before launch. Date: 2026-09-19.

## Focal question

Can an encoder-side, zero-side-information action-saliency mask reduce real
H.264 and H.265 bitrate by at least 15% at equal Top-1 accuracy?

The decoder-only Gaussian and motion-preserving filters are closed: every
target-probability BD-rate was non-negative.  A -15% target needs an encoder
operation that changes the coded signal and therefore the actual number of
bits, rather than another zero-bit decoder filter.

## Candidate mechanisms considered

1. **Action-gradient saliency suppression (selected pilot).** Protect the
   teacher's decision-critical pixels exactly; blur and temporally stabilise
   the rest before coding. It is immediately testable and has a falsifiable
   bitrate/accuracy prediction.
2. **Learned temporal preprocessor.** Higher capacity, but requires a new
   training and checkpoint-selection cycle. It remains the next rung if the
   parameter-free screen establishes that saliency-localised suppression has a
   useful operating region.
3. **Spatial QP map / ROI coding.** Potentially stronger, but it changes the
   codec interface and needs signalled side information. It is outside this
   zero-side-information screen.

The selected method uses the source prediction as a pseudo-label, never the
ground-truth action label. A frozen r3d_18 teacher computes input-gradient
saliency. A hard mask protects a fixed 15%, 25%, or 40% budget. `clip` ranks
spatiotemporal pixels jointly; `tube` takes a temporal maximum and repeats a
stable spatial mask. Outside the mask, Gaussian sigma 8 plus motion-gated
temporal stabilisation strength 0.75 removes high-frequency and inter-frame
residual cost.

## Frozen screen

- Kinetics hash split: `val`, first 200 deterministic clips.
- Geometry: 16 frames, stride 2, 128 px.
- Codecs: x264 and x265 medium, QP 30/35/40/45/50.
- Teacher: r3d_18 for every mask.
- Evaluators: r3d_18 ceiling screen, then held-out mc3_18 and r2plus1d_18.
- Arms: anchor plus the Cartesian product of protect fractions
  `{0.15, 0.25, 0.40}` and modes `{clip, tube}`.
- Primary metric: BD-rate on Top-1. Target probability is a continuous
  diagnostic. Per-clip records and clean pre-codec metrics are mandatory.

## Pre-registered decision rule

An arm reaches the requested target only when all conditions hold:

1. Top-1 BD-rate is at most **-15% on both H.264 and H.265**.
2. `processed - anchor` Top-1 is at least -0.05 at every QP for both codecs.
3. The same arm passes r3d_18 and at least one held-out evaluator. A same-model
   result alone is treated as teacher-specific selection bias, not transfer.

If multiple arms pass, choose the least aggressive one: largest protected
fraction, then `tube` over `clip`. Freeze it and evaluate once on the untouched
`test` split with paired clip bootstrap. If none passes, do not relax -15% after
seeing the result; move to a learned temporal preprocessor or a codec-native ROI
design with an explicit side-information accounting protocol.

## Adversarial review

- **Teacher leakage:** same-model saliency may manufacture a win. Held-out
  evaluators are a mandatory gate.
- **Mask flicker:** a changing hard mask can increase residual bits. The stable
  `tube` arm is the competing explanation/control.
- **Saliency incompleteness:** input gradients can be sparse or locally noisy.
  A 5x5 saliency blur and three fixed budgets test robustness without using the
  evaluation outcome to redraw the mask.
- **False BD win:** a large bitrate reduction could hide one collapsed QP.
  The -0.05 per-QP Top-1 guardrail vetoes such an arm.
