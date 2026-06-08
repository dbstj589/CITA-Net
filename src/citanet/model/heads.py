"""Prediction heads.

PairHead     : cross-KG observation-pair same/diff classifier. Input is
               [h_i, h_j, |h_i-h_j|, h_i*h_j, CTA features, dt/dist/text].
DanglingHead : per-observation abstain classifier. Input is the observation
               embedding plus a summary of its best cross-KG candidate evidence
               (a weakly-matched observation is likely dangling).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PairHead(nn.Module):
    # extra scalar features appended after the embedding interactions + CTA(6)
    N_EXTRA = 3   # dt_norm, dist_norm, text_cos

    def __init__(self, d_model: int, cta_dim: int = 6, hidden: int = 128):
        super().__init__()
        in_dim = 4 * d_model + cta_dim + self.N_EXTRA
        self.mlp = MLP(in_dim, hidden, 1)

    def forward(self, hi, hj, cta_feats, extra) -> torch.Tensor:
        inter = torch.cat([hi, hj, (hi - hj).abs(), hi * hj], dim=-1)
        x = torch.cat([inter, cta_feats, extra], dim=-1)
        return self.mlp(x).squeeze(-1)


class DanglingHead(nn.Module):
    N_EVID = 4   # best_logit, mean_logit, n_candidates_norm, max_sim

    def __init__(self, d_model: int, hidden: int = 64):
        super().__init__()
        self.mlp = MLP(d_model + self.N_EVID, hidden, 1)

    def forward(self, h, evidence) -> torch.Tensor:
        return self.mlp(torch.cat([h, evidence], dim=-1)).squeeze(-1)
