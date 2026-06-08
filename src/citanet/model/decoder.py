"""Latent identity-trajectory decoder (§4.5).

K learnable identity slots + 1 null (∅) slot. Observations are softly assigned
to slots by Sinkhorn-normalised cross-attention, with the CTA transition
affinity injected so that observations linked by a strong feasible transition
tend to share a slot. Row sums are 1 (each observation is fully distributed);
the ∅ column freely absorbs dangling observations.

Decoding (argmax over slots) yields, per slot: the grouped observations
(observation->identity is many-to-one), the time-sorted trajectory, and the
consecutive transitions. Grouping observations by ``kg_id`` inside a slot gives
the one-to-one identity<->identity map.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinkhorn_log(log_alpha: torch.Tensor, n_iters: int, tau: float = 1.0
                 ) -> torch.Tensor:
    """Sinkhorn-Knopp normalisation in log space (alternating row/column).

    For a SQUARE input and enough iterations this converges to a doubly
    stochastic matrix (both row and column sums -> 1). The final step is a row
    normalisation, so row sums are exactly 1 for any shape.
    """
    log = log_alpha / tau
    for _ in range(n_iters):
        log = log - torch.logsumexp(log, dim=1, keepdim=True)   # rows
        log = log - torch.logsumexp(log, dim=0, keepdim=True)   # cols
    log = log - torch.logsumexp(log, dim=1, keepdim=True)       # rows (exact)
    return torch.exp(log)


def sinkhorn_assign(log_alpha: torch.Tensor, n_iters: int, tau: float = 1.0,
                    col_strength: float = 0.5) -> torch.Tensor:
    """Row-stochastic Sinkhorn for slot assignment (§4.5).

    Each iteration row-normalises (so every observation is fully distributed,
    rows sum to 1) then applies a *gentle, mean-centred* column adjustment that
    balances the real slots RELATIVE to one another -- discouraging the
    degenerate all-mass-in-one-slot (or all-mass-in-∅) solution -- without the
    hard per-column marginal that would starve real slots and force identities
    of different sizes to fragment. The ∅ (last) column is left out of the
    balancing so it can freely absorb dangling observations. Rows sum to 1.
    """
    log = log_alpha / tau
    for _ in range(n_iters):
        log = log - torch.logsumexp(log, dim=1, keepdim=True)     # rows sum to 1
        real = log[:, :-1]
        col = torch.logsumexp(real, dim=0, keepdim=True)          # (1, K)
        real = real - col_strength * (col - col.mean())           # relative balance
        log = torch.cat([real, log[:, -1:]], dim=1)
    log = log - torch.logsumexp(log, dim=1, keepdim=True)         # rows (exact)
    return torch.exp(log)


class IdentityDecoder(nn.Module):
    def __init__(self, d_model: int, num_slots: int, sinkhorn_iters: int = 30,
                 tau: float = 0.5, col_strength: float = 0.1):
        super().__init__()
        self.d = d_model
        self.K = num_slots
        self.iters = sinkhorn_iters
        self.tau = tau
        self.col_strength = col_strength
        self.slots = nn.Parameter(torch.randn(num_slots, d_model) * 0.1)
        self.null = nn.Parameter(torch.randn(1, d_model) * 0.1)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.cta_gate = nn.Parameter(torch.tensor(1.0))

    def _cta_context(self, h: torch.Tensor, feats, p_transition: torch.Tensor
                     ) -> torch.Tensor:
        """Aggregate neighbour embeddings weighted by CTA transition prob, so a
        slot sees transition-linked observations together."""
        m = h.shape[0]
        i, j = feats.pair_i, feats.pair_j
        if i.numel() == 0:
            return h
        w = p_transition.detach().clamp(0, 1)
        agg = torch.zeros_like(h)
        wsum = torch.zeros(m)
        agg.index_add_(0, i, w.unsqueeze(-1) * h[j])
        agg.index_add_(0, j, w.unsqueeze(-1) * h[i])
        wsum.index_add_(0, i, w)
        wsum.index_add_(0, j, w)
        denom = wsum.clamp(min=1.0).unsqueeze(-1)
        return h + torch.sigmoid(self.cta_gate) * agg / denom

    def forward(self, h: torch.Tensor, feats, p_transition: torch.Tensor):
        h_ctx = self._cta_context(h, feats, p_transition)
        slots = torch.cat([self.slots, self.null], dim=0)       # (K+1, d)
        Z = (self.q(h_ctx) @ self.k(slots).t()) / math.sqrt(self.d)   # (M, K+1)
        A = sinkhorn_assign(Z, self.iters, self.tau, self.col_strength)  # (M, K+1)
        return Z, A
