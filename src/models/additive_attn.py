"""Attentive dual-branch additive preprocessor (Zhao arXiv:2512.15331, faithful
"conditional attention" reading).

The lineage additive models fuse the spatial and temporal branches with a single
sigmoid GATE and a convex blend ``gate*sp + (1-gate)*tp`` (see ``additive.py`` /
``additive_cond.py``). That is a degenerate 2-way attention with scalar-per-pixel
weights and no rate conditioning of the fusion itself. Zhao et al. describe the
branch merge as *conditional attention*; their reported Kinetics-400 result
(≈ −17% BD-rate on BOTH codecs) is attributed by their own Table-3 ablation to
the preprocessor design, not to loss tuning — and the repo's reimplementation
plateaus near −5.9% with the gate fusion. This module supplies the missing
piece: a per-channel, per-pixel SOFTMAX attention between the two branches whose
logits are modulated by FiLM(QP), so the model can allocate the additive edit
across the spatial vs temporal evidence differently at each rate operating point.

    sp = spatial_branch(frame)                 # [B*T,16,H,W]
    tp = temporal_branch(causal 8-frame window)# [B*T,16,H,W]
    l_sp, l_tp = attn_sp(sp), attn_tp(tp)      # per-channel logits
    l_sp, l_tp = FiLM(QP)(l_sp), FiLM(QP)(l_tp)# rate-conditioned
    w = softmax([l_sp, l_tp], dim=branch)      # [2,B*T,16,H,W], sums to 1
    fused = w0*sp + w1*tp
    x_pre = x + strength * to_rgb(fused)       # to_rgb zero-init -> identity@init

Identity at init: ``to_rgb`` is zero-init, so the untrained model is EXACTLY the
identity (out == x) regardless of the attention state — the lineage's
identity-start discipline. The FiLM output layer is also zero-init so the
conditioned fusion equals the unconditioned fusion until training moves it.

NOT load-compatible with the additive ``best.pt`` (the fusion module tree
differs): this is a new architecture (``arch: additive_attn``), trained from
scratch. Fully convolutional and resolution-agnostic (train 128, eval any size).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentiveAdditivePreprocessor(nn.Module):
    """Two-branch additive editor with QP-conditioned softmax attention fusion.

    ~10k parameters. ``cond`` is the normalised-QP rate condition ([B, cond_dim],
    1 = heavy compression); ``mask`` is accepted for training-loop compatibility
    and ignored (this model is ungated by design, like the additive lineage)."""

    def __init__(self, temporal_frames: int = 8, strength: float = 1.0,
                 cond_dim: int = 1):
        super().__init__()
        if temporal_frames < 1:
            raise ValueError(f"temporal_frames must be >= 1, got {temporal_frames}")
        self.tframes = int(temporal_frames)
        self.strength = float(strength)
        self.cond_dim = int(cond_dim)

        # Spatial branch: stem + residual body (same tree as the additive lineage).
        self.spatial_stem = nn.Conv2d(3, 16, 3, padding=1)
        self.spatial_residual = nn.Module()
        self.spatial_residual.body = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1))
        # Temporal branch: causal window of tframes*3 channels -> 16.
        self.temporal_stem = nn.Conv2d(self.tframes * 3, 16, 3, padding=1)

        # Conditional attention: per-branch 1x1 logit projections, fused by a
        # softmax over the branch axis. FiLM(QP) modulates the logits so the
        # spatial/temporal balance is rate-dependent.
        self.attn = nn.Module()
        self.attn.spatial = nn.Conv2d(16, 16, 1)
        self.attn.temporal = nn.Conv2d(16, 16, 1)
        # FiLM emits (gamma, beta) over the 16 logit channels; zero-init output
        # so at init the conditioning is a no-op (gamma=beta=0).
        self.film = nn.Sequential(
            nn.Linear(self.cond_dim, 16), nn.LeakyReLU(0.1), nn.Linear(16, 32))
        nn.init.zeros_(self.film[2].weight)
        nn.init.zeros_(self.film[2].bias)

        # Output projection: zero-init -> identity at init (out = x + 0).
        self.to_rgb = nn.Conv2d(16, 3, 3, padding=1)
        nn.init.zeros_(self.to_rgb.weight)
        nn.init.zeros_(self.to_rgb.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: [B,3,T,H,W] in [0,1] -> edited clip, same shape, in [0,1]."""
        if x.ndim != 5 or x.shape[1] != 3:
            raise ValueError(f"expected [B,3,T,H,W], got {tuple(x.shape)}")
        b, c, t, h, w = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)   # [B*T,3,H,W]

        # Spatial branch.
        sp = F.relu(self.spatial_stem(frames))
        body = self.spatial_residual.body
        sp = sp + body[2](F.relu(body[0](sp)))                     # [B*T,16,H,W]

        # Temporal branch: causal tframes-window ending at each frame.
        pad = x[:, :, :1].expand(b, c, self.tframes - 1, h, w)
        padded = torch.cat([pad, x], dim=2)                        # [B,C,T+tf-1,H,W]
        win = padded.unfold(2, self.tframes, 1)                    # [B,C,T,H,W,tf]
        stack = win.permute(0, 2, 5, 1, 3, 4).reshape(b * t, self.tframes * c, h, w)
        tp = F.relu(self.temporal_stem(stack))                     # [B*T,16,H,W]

        # QP-conditioned per-channel softmax attention between the two branches.
        l_sp = self.attn.spatial(sp)                               # [B*T,16,H,W]
        l_tp = self.attn.temporal(tp)
        if cond is None:
            cond = frames.new_zeros(b, self.cond_dim)
        cond_f = cond.repeat_interleave(t, dim=0).to(sp.dtype)     # [B*T, cond_dim]
        gamma, beta = self.film(cond_f).chunk(2, dim=1)            # [B*T,16] each
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        l_sp = l_sp * (1.0 + gamma) + beta
        l_tp = l_tp * (1.0 + gamma) + beta
        # softmax over the branch axis (stack -> [2,B*T,16,H,W]). Named
        # ``attn_w`` so it does not shadow the ``w`` (width) unpacked above.
        attn_w = torch.softmax(torch.stack([l_sp, l_tp], dim=0), dim=0)
        fused = attn_w[0] * sp + attn_w[1] * tp                    # [B*T,16,H,W]

        out = frames + self.strength * self.to_rgb(fused)
        return out.clamp(0.0, 1.0).reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
