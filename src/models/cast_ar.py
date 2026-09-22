"""CAST-AR decoder-side temporal task enhancement.

This is the F1 component of Codec-Adaptive Safe Temporal Sandwich for action
recognition.  It operates *after* a real standard codec, so it changes no
bitstream bytes.  The module is intentionally small and identity-initialised:

    decoded clip -> temporal feature trunk -> K residual bases
                                      \-> content/QP mixture + safety gate

H.264 and H.265 use separate checkpoints.  QP is allowed to share structure
inside one codec, while the decoded picture type (I/P/B/other) exposes the
actual GOP realised by ffmpeg instead of assuming a P-only proxy.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _TemporalBlock(nn.Module):
    """Depthwise temporal-spatial filtering followed by channel mixing."""

    def __init__(self, channels: int):
        super().__init__()
        self.depthwise = nn.Conv3d(
            channels, channels, kernel_size=3, padding=1, groups=channels
        )
        self.pointwise = nn.Conv3d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(4, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.depthwise(x)
        y = self.pointwise(F.silu(y))
        return F.silu(self.norm(x + y))


class CASTTemporalPost(nn.Module):
    """Low-rank, codec-specific temporal POST for decoded RGB clips.

    Args:
        width: feature width of the temporal trunk.
        bases: number of content-conditioned residual bases.
        max_delta: maximum RGB edit amplitude before the learned global
            strength.  The output remains in ``[0, 1]``.
        qp_min/qp_max: normalisation range of the codec QP condition.

    ``picture_types`` is an integer tensor ``[B,T]`` with I=0, P=1, B=2 and
    other/unknown=3.  When unavailable it safely defaults to unknown.
    """

    def __init__(
        self,
        width: int = 24,
        bases: int = 4,
        max_delta: float = 0.15,
        qp_min: float = 20.0,
        qp_max: float = 51.0,
    ):
        super().__init__()
        if width % 4:
            raise ValueError("width must be divisible by four for GroupNorm")
        if bases < 1:
            raise ValueError("bases must be positive")
        self.width = int(width)
        self.bases = int(bases)
        self.max_delta = float(max_delta)
        self.qp_min = float(qp_min)
        self.qp_max = float(qp_max)

        # RGB plus first-order temporal residual.  Motion cues are therefore
        # explicit instead of being expected to emerge from frame-wise filters.
        self.input = nn.Conv3d(6, width, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(_TemporalBlock(width), _TemporalBlock(width))

        self.picture_embedding = nn.Embedding(4, width)
        self.qp_film = nn.Sequential(
            nn.Linear(1, width), nn.SiLU(), nn.Linear(width, 2 * width)
        )
        # A content-dependent residual codebook: every clip emits K candidate
        # corrections and a small controller mixes them.
        self.basis_head = nn.Conv3d(width, 3 * bases, kernel_size=3, padding=1)
        self.controller = nn.Sequential(
            nn.Linear(width + 1, width), nn.SiLU(), nn.Linear(width, bases + 1)
        )
        # Identity at initialisation without the historical zero*zero saddle:
        # the basis is alive, so the scalar strength receives a gradient on the
        # first step; after it moves, the full trunk receives gradients too.
        nn.init.normal_(self.basis_head.weight, std=1e-3)
        nn.init.zeros_(self.basis_head.bias)
        nn.init.zeros_(self.controller[-1].weight)
        nn.init.zeros_(self.controller[-1].bias)
        self.post_strength = nn.Parameter(torch.zeros(()))

    def qp_condition(self, qp: torch.Tensor, batch: int, like: torch.Tensor) -> torch.Tensor:
        if qp.ndim == 0:
            qp = qp.expand(batch)
        if qp.ndim == 2 and qp.shape[1] == 1:
            qp = qp[:, 0]
        if qp.ndim != 1 or qp.shape[0] != batch:
            raise ValueError(f"qp must be scalar or [B], got {tuple(qp.shape)}")
        denom = max(self.qp_max - self.qp_min, 1.0)
        return ((qp.to(device=like.device, dtype=like.dtype) - self.qp_min) / denom).clamp(0, 1)

    def forward(
        self,
        decoded: torch.Tensor,
        qp: torch.Tensor | float | int,
        picture_types: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if decoded.ndim != 5 or decoded.shape[1] != 3:
            raise ValueError(f"expected [B,3,T,H,W], got {tuple(decoded.shape)}")
        b, _, t, _, _ = decoded.shape
        if not torch.is_tensor(qp):
            qp = decoded.new_tensor(float(qp))
        q = self.qp_condition(qp, b, decoded)

        if picture_types is None:
            picture_types = torch.full(
                (b, t), 3, device=decoded.device, dtype=torch.long
            )
        if picture_types.shape != (b, t):
            raise ValueError(
                f"picture_types must have shape {(b, t)}, got {tuple(picture_types.shape)}"
            )
        picture_types = picture_types.to(decoded.device, dtype=torch.long).clamp(0, 3)

        prev = torch.cat([decoded[:, :, :1], decoded[:, :, :-1]], dim=2)
        motion = decoded - prev
        feat = self.input(torch.cat([decoded, motion], dim=1))
        # [B,T,W] -> [B,W,T,1,1]
        ptype = self.picture_embedding(picture_types).permute(0, 2, 1)
        feat = feat + ptype[:, :, :, None, None].to(feat.dtype)
        feat = self.blocks(F.silu(feat))

        gamma, beta = self.qp_film(q[:, None]).chunk(2, dim=1)
        feat = feat * (1.0 + gamma[:, :, None, None, None])
        feat = feat + beta[:, :, None, None, None]

        pooled = feat.mean(dim=(2, 3, 4))
        control = self.controller(torch.cat([pooled, q[:, None]], dim=1))
        mix = torch.softmax(control[:, : self.bases], dim=1)
        safety_gate = torch.sigmoid(control[:, self.bases :])

        basis = torch.tanh(self.basis_head(feat))
        basis = basis.reshape(b, self.bases, 3, t, decoded.shape[-2], decoded.shape[-1])
        delta = (basis * mix[:, :, None, None, None, None]).sum(dim=1)
        amplitude = self.max_delta * torch.tanh(self.post_strength)
        out = decoded + amplitude * safety_gate[:, :, None, None, None] * delta
        return out.clamp(0.0, 1.0)

    @torch.no_grad()
    def diagnostics(self, decoded: torch.Tensor, qp, picture_types=None) -> dict[str, float]:
        restored = self(decoded, qp, picture_types)
        edit = restored - decoded
        return {
            "post_strength": float(torch.tanh(self.post_strength)),
            "edit_l1": float(edit.abs().mean()),
            "edit_rms": float(edit.square().mean().sqrt()),
        }

