"""Top-level CITA-Net model.

Assembles: source-aware encoder -> (optional relation-context GNN, M2+) -> CTA
-> pair head + dangling head. The latent identity-trajectory decoder (M3) is
attached separately in :mod:`citanet.decode` / the full model wrapper so the
lite/g stages stay lightweight.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..data.ontology import Ontology
from .cta import CTA, CTAOutput
from .encoder import ObservationEncoder
from .featurize import FeatureSpace, ScenarioFeatures
from .heads import DanglingHead, PairHead


@dataclass
class ModelOutput:
    h: torch.Tensor               # (N, d) contextual observation embeddings
    cta: CTAOutput
    pair_logits: torch.Tensor     # (P,)
    dangling_logits: torch.Tensor # (N,)
    assign_logits: torch.Tensor | None = None   # (N, K+1) slot logits (M3)
    assign: torch.Tensor | None = None          # (N, K+1) Sinkhorn assignment


def _scatter_reduce(n: int, idx: torch.Tensor, val: torch.Tensor,
                    reduce: str, fill: float) -> torch.Tensor:
    out = torch.full((n,), fill, dtype=val.dtype)
    if idx.numel() > 0:
        out.scatter_reduce_(0, idx, val, reduce=reduce, include_self=False)
    return out


class CITANet(nn.Module):
    def __init__(self, fs: FeatureSpace, ontology: Ontology, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = ObservationEncoder(fs, ontology, cfg)
        d = cfg.encoder.d_model

        self.graph = None
        if cfg.graph.enabled:
            from .graph_encoder import RelationContextGNN
            from .featurize import REL_NAMES
            # n_relations must match the edge-type buckets featurize emits
            self.graph = RelationContextGNN(d, n_relations=len(REL_NAMES), cfg=cfg)

        self.cta = CTA(n_sources=len(fs.sources), cfg=cfg, state_eps=ontology.state_eps)
        self.pair_head = PairHead(d)
        self.dangling_head = DanglingHead(d)

        self.decoder = None
        if cfg.decoder.enabled:
            from .decoder import IdentityDecoder
            self.decoder = IdentityDecoder(
                d, num_slots=cfg.decoder.num_slots,
                sinkhorn_iters=cfg.decoder.sinkhorn_iters,
                tau=cfg.decoder.sinkhorn_temp,
                col_strength=cfg.decoder.sinkhorn_col_strength)

    def forward(self, feats: ScenarioFeatures) -> ModelOutput:
        h = self.encoder(feats)
        if self.graph is not None:
            h = self.graph(h, feats)

        cta = self.cta(h, feats)

        # --- pair head ---
        hi, hj = h[feats.pair_i], h[feats.pair_j]
        dt_norm = (feats.p_dt / 180.0).unsqueeze(-1)
        dist_norm = (feats.p_dist / 1000.0).unsqueeze(-1)
        extra = torch.cat([dt_norm, dist_norm, feats.p_text_cos.unsqueeze(-1)], dim=-1)
        pair_logits = self.pair_head(hi, hj, cta.feature_matrix(), extra)

        # --- dangling evidence per observation from cross-KG candidate pairs ---
        n = h.shape[0]
        cross = feats.pair_cross
        if cross.any():
            ci, cj = feats.pair_i[cross], feats.pair_j[cross]
            clog = pair_logits[cross]
            csim = cta.sim_sem[cross]
            endpoints = torch.cat([ci, cj])
            logvals = torch.cat([clog, clog])
            simvals = torch.cat([csim, csim])
            best = _scatter_reduce(n, endpoints, logvals, "amax", -20.0)
            mean = _scatter_reduce(n, endpoints, logvals, "mean", 0.0)
            maxsim = _scatter_reduce(n, endpoints, simvals, "amax", -1.0)
            count = torch.bincount(endpoints, minlength=n).float()
        else:
            best = torch.full((n,), -20.0)
            mean = torch.zeros(n)
            maxsim = torch.full((n,), -1.0)
            count = torch.zeros(n)
        evidence = torch.stack([best, mean, count / 5.0, maxsim], dim=-1)  # (N,4)
        dangling_logits = self.dangling_head(h, evidence)

        assign_logits = assign = None
        if self.decoder is not None:
            assign_logits, assign = self.decoder(h, feats, cta.p_transition)

        return ModelOutput(h=h, cta=cta, pair_logits=pair_logits,
                           dangling_logits=dangling_logits,
                           assign_logits=assign_logits, assign=assign)
