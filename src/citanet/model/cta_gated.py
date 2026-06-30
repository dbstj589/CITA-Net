"""Learned per-term gating for CTA (extends, does not modify, cta.py).

Hypothesis (from the ablation re-score): the fixed per-term coefficients make the
constraint terms fire even on easy pairs where they are not needed, acting as
noise. Here each term is scaled by a learned gate g_k(context) in (0,1):

    score = w0 * g_sem*sim + g_time*b_time + g_motion*b_motion
            + g_state*b_state + g_rel*b_rel + g_src*b_src

The base terms (sim, b_time, ...) are computed EXACTLY as in CTA (we call
super().forward), so the only change is the learned scalar gate per term per pair.
With all gates == 1 this reduces bit-exactly to the fixed-weight CTA -- used as a
fidelity check (set ``force_gate_one=True``).

The gate is a tiny MLP over cheap, already-computed pair-context features
(semantic sim, normalised Δt/dist, speed-violation, state compat, rel jaccard,
text cos, per-endpoint type confidence, same-type flag, cross-KG flag).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .cta import CTA, CTAOutput

# order MUST match CTAOutput.feature_matrix(): sem, time, motion, state, rel, src
_TERMS = ("sem", "time", "motion", "state", "rel", "src")
CONTEXT_DIM = 11


class GatedCTA(CTA):
    def __init__(self, n_sources: int, cfg, state_eps: float = 0.01):
        super().__init__(n_sources, cfg, state_eps)
        hidden = int(getattr(cfg.cta, "gate_hidden", 32))
        self.gate = nn.Sequential(
            nn.Linear(CONTEXT_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, len(_TERMS)),
        )
        # near-identity init: bias high so initial gates ~sigmoid(3)=0.95, weights 0
        # => training starts close to the fixed-weight CTA, then learns to attenuate.
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, 3.0)
        self.force_gate_one = False          # fidelity-check switch
        self.last_gate = None                # (P,6) detached, for introspection
        self.last_raw_sim = None             # (P,)  detached raw cosine
        self.last_cross = None               # (P,)  bool

    def _context(self, base: CTAOutput, feats) -> torch.Tensor:
        i, j = feats.pair_i, feats.pair_j
        speed_viol = (feats.p_req_speed - feats.p_feas_speed) / 10.0
        same_type = (feats.type_id[i] == feats.type_id[j]).float()
        ctx = torch.stack([
            base.sim_sem,                              # semantic similarity
            torch.tanh(feats.p_dt / 180.0),            # normalised Δt
            torch.tanh(feats.p_dist / 1000.0),         # normalised dist
            torch.tanh(speed_viol),                    # speed-violation proxy
            feats.p_state_compat,                      # state compatibility
            feats.p_rel_jaccard,                       # relational jaccard
            feats.p_text_cos,                          # text cosine
            feats.type_conf[i],                        # endpoint type confidence
            feats.type_conf[j],
            same_type,                                 # same predicted type
            feats.pair_cross.float(),                  # cross-KG candidate?
        ], dim=-1)
        return ctx

    def forward(self, h: torch.Tensor, feats) -> CTAOutput:
        base = super().forward(h, feats)               # raw terms (disabled-zeroed)
        terms = base.feature_matrix()                  # (P,6): sem,time,motion,state,rel,src
        if self.force_gate_one:
            g = torch.ones_like(terms)
        else:
            g = torch.sigmoid(self.gate(self._context(base, feats)))   # (P,6) in (0,1)
        gated = g * terms                              # scale each term
        # score: w0 multiplies the (gated) sem term, exactly as CTA does for sim.
        score = (self.w0 * gated[:, 0] + gated[:, 1] + gated[:, 2]
                 + gated[:, 3] + gated[:, 4] + gated[:, 5])
        self.last_gate = g.detach()
        self.last_raw_sim = base.sim_sem.detach()
        self.last_cross = feats.pair_cross.detach()
        return CTAOutput(
            score=score, p_transition=torch.sigmoid(score),
            sim_sem=gated[:, 0], b_time=gated[:, 1], b_motion=gated[:, 2],
            b_state=gated[:, 3], b_rel=gated[:, 4], b_src=gated[:, 5])
