# CAST-AR F1: real-codec temporal POST mechanism screen

Status: preregistered before the three Kaggle jobs are launched.

## Question

Can a codec-specific, QP/GOP-conditioned temporal decoder module recover action
recognition information from the *same* H.264/H.265 bitstream, including on an
analyzer not used for training?

This is intentionally POST-only.  PRE and rate optimisation are not introduced
until the zero-bit recovery mechanism passes, so a negative result cannot be
hidden by a bitrate change or by a second learned module.

## Treatment

`CASTTemporalPost` consumes the decoded RGB clip, the real QP and per-frame
I/P/B types reported by ffprobe.  A depthwise 3-D temporal trunk emits four
content-dependent RGB residual bases.  A controller mixes the bases and predicts
a clip safety gate.  H.264 and H.265 never share trainable weights.  The global
residual strength is identity-initialised with a live first-step gradient.

Training loss contains label CE, task-loss regret relative to the decoded anchor,
target-probability regret, layer-2 feature-regret relative to the clean clip,
clean-logit distillation and a small edit magnitude penalty.  The frozen training
analyzer is `r3d_18`; `r2plus1d_18` is held out until the final validation pass.

## Frozen screen

- Dataset split: 1,200 deterministic Kinetics-cleaned train clips per seed.
- Training: 3 epochs, batch 1, AdamW `3e-4`, balanced cycle over QP
  `30/35/40/45/50`, real ffmpeg codec, preset `medium`.
- Validation: one final pass over 208 deterministic validation clips selected by
  salt `cast-ar-f1-val-v1`; no epoch selection on validation.
- Clip format: 16 frames, temporal stride 2, 128 px; analyzer input 112 px.
- Treatment and anchor consume the exact same bitstream and therefore have
  identical bpp at every point.

| Account | Codec | Seed | Purpose |
|---|---|---:|---|
| `baoancut` | H.264 | 313001 | primary H.264 screen |
| `trnhlng` | H.265 | 313002 | primary H.265 screen |
| `huolgggnuyen` | H.264 | 313003 | independent H.264 replication |

The account spelling above follows the credential-pool key.  It corresponds to
the user-supplied `huolgnguyen` account name.

## Decision rule

A job passes only on held-out `r2plus1d_18` when all three conditions hold:

1. target-probability BD-rate is at most `-5%`;
2. Top-1 BD-accuracy is non-negative; and
3. no QP loses more than `0.5` Top-1 percentage point.

Advance H.264 only if both H.264 seeds pass.  Advance H.265 only if its primary
job passes; replicate it before any full-validation claim.  A pass authorises
F2 (freeze POST, add codec/GOP-matched PRE with real-forward/proxy-backward).
A failure closes this POST architecture; it does not trigger tuning against the
same 208 clips.  The 1,159-clip test split remains untouched.

