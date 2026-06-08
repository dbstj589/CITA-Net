"""Constraint-aware transition attention (§4.4).

For a time-ordered candidate pair (o_i -> o_j):

    score = w0 * sim_sem + b_time + b_motion + b_state + b_rel + b_src

    sim_sem  = cosine(h_i, h_j)
    b_time   = -BIG if t_j < t_i - eps else 0                 (hard, non-learn)
    b_motion = -alpha * softplus((req_speed - feas_speed)/beta)
    b_state  = gamma * log(C[s_i, s_j] + eps)
    b_rel    = delta * jaccard(neighbourhood_i, neighbourhood_j)   (M2+)
    b_src    = learnable bias indexed by (source_i, source_j)

All terms are soft except b_time. w0, alpha, beta, gamma, delta and the source
bias table are learnable. ``enabled_terms`` (config) selects which terms apply,
so ablations (e.g. drop b_motion, drop b_rel) are a config change.

``p_transition = sigmoid(score)``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CTAOutput:
    score: torch.Tensor          # (P,)
    p_transition: torch.Tensor   # (P,)
    sim_sem: torch.Tensor
    b_time: torch.Tensor
    b_motion: torch.Tensor
    b_state: torch.Tensor
    b_rel: torch.Tensor
    b_src: torch.Tensor

    def feature_matrix(self) -> torch.Tensor:
        """(P, 6) per-pair CTA contributions for the pair head / rationale."""
        return torch.stack([self.sim_sem, self.b_time, self.b_motion,
                            self.b_state, self.b_rel, self.b_src], dim=-1)


class CTA(nn.Module):
    def __init__(self, n_sources: int, cfg, state_eps: float = 0.01):
        super().__init__()
        self.enabled = set(cfg.cta.enabled_terms)
        self.time_eps = cfg.cta.time_eps_s
        self.big = cfg.cta.big_penalty
        self.state_eps = state_eps

        self.w0 = nn.Parameter(torch.tensor(1.0))
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.log_beta = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(cfg.cta.motion_beta)))))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.delta = nn.Parameter(torch.tensor(1.0))
        self.src_bias = nn.Parameter(torch.zeros(n_sources, n_sources))

    def forward(self, h: torch.Tensor, feats) -> CTAOutput:
        i, j = feats.pair_i, feats.pair_j
        hi, hj = h[i], h[j]
        sim = F.cosine_similarity(hi, hj, dim=-1)             # (P,)

        # b_time: hard backward penalty (candidates are forward, so ~0)
        backward = (-feats.p_dt) > self.time_eps              # t_j < t_i - eps
        b_time = torch.where(backward, torch.full_like(sim, -self.big),
                             torch.zeros_like(sim))

        # b_motion
        beta = torch.exp(self.log_beta)
        viol = (feats.p_req_speed - feats.p_feas_speed) / beta
        b_motion = -self.alpha * F.softplus(viol)

        # b_state
        b_state = self.gamma * torch.log(feats.p_state_compat + self.state_eps)

        # b_rel
        b_rel = self.delta * feats.p_rel_jaccard

        # b_src
        b_src = self.src_bias[feats.p_src_i, feats.p_src_j]

        # Zero any disabled term so BOTH the score and the exposed feature matrix
        # (used by the pair head + rationale) reflect exactly what is active.
        # This keeps the M2 ablation clean: with "rel" off, no relational signal
        # leaks into the pair head via b_rel.
        zero = torch.zeros_like(sim)
        if "time" not in self.enabled:
            b_time = zero
        if "motion" not in self.enabled:
            b_motion = zero
        if "state" not in self.enabled:
            b_state = zero
        if "rel" not in self.enabled:
            b_rel = zero
        if "src" not in self.enabled:
            b_src = zero

        score = self.w0 * sim + b_time + b_motion + b_state + b_rel + b_src
        return CTAOutput(score=score, p_transition=torch.sigmoid(score),
                         sim_sem=sim, b_time=b_time, b_motion=b_motion,
                         b_state=b_state, b_rel=b_rel, b_src=b_src)
