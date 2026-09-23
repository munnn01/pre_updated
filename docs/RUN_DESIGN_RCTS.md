# RCTS: real-codec task-sensitivity pilot for action recognition

## Question and scope

Can a small policy, trained to select bounded PRE operations from actual
H.264/H.265 bit counts and task regret, improve the rate/Top-1 curve on a
different action recogniser? This pilot tests feasibility; 104 reused validation
clips cannot establish a final >15% result.

## Fixed protocol

- Canonical hash split: 8,632 train / 1,010 val / 1,159 historical test,
  expected test fingerprint `30f083f8520a`. Historical test is not used.
- Same deterministic source clip selection across all accounts/seeds: 80 train
  and 104 val, approximately class balanced and sorted by stable source IDs.
- 16 RGB frames, stride 2, 128 px; analyser input 112 px; x264/x265 preset
  medium; QP 30, 35, 40, 45, 50; same GOP and frame count for all arms.
- Train teacher: frozen `r3d_18`. Evaluation: `r3d_18` and held-out
  `r2plus1d_18`. Source-model pseudo labels make saliency maps, with no ground
  truth in the candidate generator or deployed policy.
- Eight actions: identity, two uniform spatial strengths, two task-protected
  spatial strengths, one motion-compensated temporal strength, and two joint
  strengths. The protection mask covers high-gradient task blocks and their
  adjacent frames. Temporal edits are gated by motion-warp agreement.
- The training oracle picks the lowest measured bpp candidate among those
  with cross-entropy regret <= 0.02 and no correct-to-wrong flip on `r3d_18`.
  It uses TRAIN labels only and is never reported as a deployed result.
- A 2-layer, 32-channel MLP predicts the oracle action from source-only
  features plus QP. A source-video-disjoint 20% internal holdout controls
  early stopping. Policy prediction confidence below 0.35 falls back to
  identity. Seed changes policy initialization; train/val source IDs stay fixed.
- Validation controls: identity, uniform spatial, task-protected spatial,
  task-protected temporal, joint, and learned policy. The policy chooses its
  action before any candidate is encoded.

## Readout

Primary readout is held-out `r2plus1d_18` Top-1 BD-rate against identity,
calculated from actual coded bytes at five QPs. Also report `r3d_18`, raw
rate/Top-1 curves, same-QP rate ratios, worst Top-1 gap, chosen actions,
source-video IDs and per-clip per-QP rows. Undefined BD-rate stays null.

Engineering progression gate: both codecs should show negative held-out
Top-1 BD-rate and at least one should show a credible path toward -15% before
scaling data. The scientific target is H.264 < 0%, H.265 < 0%, and at least
one < -15% on a fresh source-video-disjoint holdout, with paired uncertainty.
The 104-clip pilot is exploratory and has no confirmatory claim.

## Resource and reproducibility notes

Each account runs one codec and one policy seed. Approximately 80*5*8 = 3,200
training and 104*5*(5 or 6) = 2,600-3,120 validation real-codec encodes are
expected. Encoder and analyser weights, code commit, split fingerprints,
policy checkpoint and per-clip measurements are saved in the notebook output.
No artificial wall-time limit is specified by this notebook; Kaggle platform
limits remain in force.
