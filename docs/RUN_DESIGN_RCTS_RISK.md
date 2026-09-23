# RCTS risk-aware PRE screen (preregistered before Kaggle execution)

This is an exploratory follow-up to RCTS v2, where the H.264 policy had
held-out r2plus1d_18 Top-1 BD-rate +6.43% and the H.265 policy selected the
identity everywhere. The train-only oracle was not a deployable result.

## Hypothesis and frozen choices

The global 12-feature categorical selector failed to predict harmful flips.
A layout-preserving frozen r3d layer-2 feature map plus saliency and motion,
conditioned on QP, may predict each action's real-codec rate, task regret and
correct-to-incorrect risk. A safe candidate can then be selected without
encoding all candidates at inference. Identity is always the fallback.

- Dataset: public `qktttttttttt/kineticscleaned`, original hash split with
  test fingerprint `30f083f8520a`; no test labels used here.
- Train: 400 class-balanced clips; 16 frames, 128 pixels, stride 2; 5 QPs
  (30,35,40,45,50); 8 fixed RCTS actions; x264/x265 preset medium.
- Teacher targets: frozen pretrained r3d_18 and mc3_18 on identical decoded
  candidates; log(bpp/action divided by bpp/identity), maximum CE regret
  across teachers, and harmful flip on either teacher.
- Fit/early-stop/calibration: deterministic disjoint source-video split,
  70%/10%/20%, shared across all QPs and actions. Epoch selection uses only
  the 10% early-stop split. Risk/CE thresholds use only the 20% calibration
  split with <=1% aggregate harmful flips, <=2.5% per-QP harmful
  flips and <=0.02 mean CE regret. Identity if no threshold saves >=1% mean
  log-rate on calibration clips.
- Development evaluation: 104 fixed class-balanced validation clips, frozen
  r3d_18 and independent r2plus1d_18, true x264/x265 encodes, paired QP
  curves and Top-1 BD-rate; all raw per-clip outcomes retained.

Decision criterion for advancing: both held-out codec BD-rates negative on
development validation with one <= -15%, no material Top-1 drop at any QP,
and replication with a second seed. These exploratory results are NOT a final
claim. Final assessment requires a new source-video-disjoint holdout (ideally
full 1010 validation clips or a never-used test) with confidence intervals.

Controls remain identity, uniform_s20, spatial_s25, temporal_t20, and
joint_s25_t15. No candidate additions, model tuning on validation outcomes,
or codec-specific ex-post threshold changes are permitted in this screen.
