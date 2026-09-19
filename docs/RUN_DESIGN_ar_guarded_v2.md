# Development screen V2: guarded saliency-motion AR prefilter

Status: frozen before launch. Date: 2026-09-19.

## Evidence that motivated the redesign

The V1 saliency screen completed on the same deterministic 200-clip validation
set.  Its best arm (`sal40_tube_s8_t0.75`) had positive Top-1 BD-rate on every
evaluator: R3D-18 `+35.47/+23.08%`, MC3-18 `+31.29/+21.24%`, and R2Plus1D-18
`+25.10/+18.50%` for H.264/H.265.  Source Top-1 fell by 19--23 percentage
points before coding.  V1 is therefore closed; V2 does not reinterpret it as a
partial win.

## Focal question

Can a label-free, per-clip source guard preserve action evidence while a mild
saliency-and-motion-localised transform reduces real H.264/H.265 Top-1 BD-rate?

## Candidate directions and decision

- **Idea A -- context-safe guarded suppression.** Protect most saliency and
  motion context, then permit only a mild edit. This has the lowest expected
  source damage but may not save 15% rate.
- **Idea B -- rate-seeking guarded suppression.** Protect less context and allow
  a stronger edit, but retain the same automatic fallback. This has more rate
  headroom and higher teacher-selection risk.
- **Rejected for this screen -- another hard mask.** V1 directly falsified the
  assumption that a small protected region contains enough action evidence.
- **Deferred -- learned preprocessor and codec-native ROI/QP maps.** They require
  training or a changed codec interface and should not be mixed into this
  parameter-free screen.

Decision: run A and B as separate, predeclared families. Do not merge their
grids after observing results.

## Mechanism

1. A frozen R3D-18 teacher produces a source pseudo-label and input-gradient
   saliency; no ground-truth label enters preprocessing.
2. A stable motion tube protects the spatial locations with the largest
   temporal difference. The protected core is the union of the saliency and
   motion tubes.
3. Gaussian/temporal background simplification is blended with the source at a
   bounded strength rather than applied fully.
4. Candidate blends are tried strongest-to-weakest. The first candidate that
   preserves the teacher pseudo-label and the configured fraction of source
   confidence is selected; otherwise the arm is exact identity for that clip.
5. The selected pixels alone enter the standard codec. No mask, pseudo-label,
   selection index, or side information is sent.

## Frozen development matrix

Common settings:

- Kinetics hash `val`, first 200 clips; 16 frames, stride 2, 128 px.
- x264/x265 medium; QP 30/35/40/45/50.
- R3D-18 guard teacher for every run.
- Evaluators: R3D-18, MC3-18, R2Plus1D-18.
- Four linear fallback levels per maximum blend; feather 1.
- Per-clip records, source metrics, acceptance rate, chosen blend, protected
  fraction, curves, BD metrics, and automated gate decisions are mandatory.

Family A (`context-safe`):

- saliency protect `{0.65, 0.80}`;
- motion protect `0.50`;
- Gaussian sigma `2`, temporal strength `0.10`;
- maximum blend `{0.25, 0.40}`;
- teacher-confidence retention `0.97`.

Family B (`rate-seeking`):

- saliency protect `{0.50, 0.65}`;
- motion protect `0.35`;
- Gaussian sigma `3`, temporal strength `0.15`;
- maximum blend `{0.35, 0.55}`;
- teacher-confidence retention `0.95`.

Each family is evaluated independently by all three backbones, yielding six
Kaggle jobs. Account identity is only a compute block; it is not a treatment.

## Predeclared screening gates

An arm is a local pass only when all are true:

1. Source Top-1 gap is at least `-0.02`.
2. The minimum per-QP Top-1 gap is at least `-0.05` for both codecs.
3. Top-1 BD-rate is at most `-15%` for both codecs.

A candidate advances only when the exact same arm locally passes R3D-18 and at
least one held-out evaluator. Target probability is diagnostic, not a substitute
for Top-1. If no arm passes, the requested `-15%` claim is rejected for this
parameter-free family; thresholds are not relaxed post hoc.

This is a development screen because its design responds to V1 validation
results. Any passing arm must be frozen and evaluated once on the untouched
test split with paired clip bootstrap before a confirmatory claim.

## Adversarial review

- **Teacher-specific guard:** R3D may preserve its own decision while harming
  another model. Held-out evaluators are mandatory and never participate in
  candidate selection.
- **Identity dilution:** frequent fallback can make an arm safe but useless.
  Acceptance and selected-blend distributions are reported with BD-rate.
- **Hidden label leakage:** only the source teacher prediction is allowed in the
  guard. Ground truth is consumed after selection for metrics.
- **Multiple comparisons:** the two families and their grids are fixed here;
  this remains screening rather than confirmatory inference.
- **False average win:** source and every-QP guardrails veto a favorable BD
  average that hides a collapsed operating point.
