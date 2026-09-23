"""Risk-aware, codec-conditioned selector for real-codec RCTS measurements."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def source_feature(analyzer, video: torch.Tensor, saliency: np.ndarray) -> torch.Tensor:
    """Keep spatial/temporal layout; no target label or encoded candidate is used."""
    device = next(analyzer.parameters()).device
    x = analyzer._prep(video.to(device))
    net = analyzer.net
    h = net.layer2(net.layer1(net.stem(x)))
    semantic = F.adaptive_avg_pool3d(h, (4, 8, 8))
    # Channelwise normalization prevents large frozen features dominating maps.
    semantic = semantic / semantic.square().mean(dim=(2, 3, 4), keepdim=True).sqrt().clamp_min(1e-3)
    sal = torch.as_tensor(saliency, device=device, dtype=x.dtype)[None, None]
    sal = F.adaptive_avg_pool3d(sal, (4, 8, 8))
    sal = sal / sal.amax().clamp_min(1e-6)
    motion = (video[:, :, 1:] - video[:, :, :-1]).abs().mean(1, keepdim=True).to(device)
    motion = F.adaptive_avg_pool3d(motion, (4, 8, 8))
    return torch.cat((semantic, sal, motion), dim=1)[0].cpu().to(torch.float16)


class RiskPredictor(nn.Module):
    """Outputs per-action log-rate, CE regret, and harmful-flip logits."""

    def __init__(self, actions: int, channels: int = 130) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(channels, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(32, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.AdaptiveAvgPool3d((2, 4, 4)), nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(32 * 2 * 4 * 4 + 1, 96), nn.GELU(),
            nn.Linear(96, actions * 3),
        )
        self.actions = actions

    def forward(self, features: torch.Tensor, qp: torch.Tensor) -> torch.Tensor:
        z = self.encoder(features.float())
        return self.head(torch.cat((z, qp.float().reshape(-1, 1)), 1)).reshape(-1, self.actions, 3)


def choose_action(pred: np.ndarray, ce_limit: float, risk_limit: float) -> int:
    """Pick rate-saving safe action, with exact identity fallback."""
    if pred.ndim != 2 or pred.shape[1] != 3:
        raise ValueError("expected [actions, 3] prediction")
    valid = [i for i in range(1, len(pred)) if
             pred[i, 0] < 0 and pred[i, 1] <= ce_limit and
             1 / (1 + np.exp(-np.clip(pred[i, 2], -30, 30))) <= risk_limit]
    return min(valid, key=lambda i: pred[i, 0]) if valid else 0

