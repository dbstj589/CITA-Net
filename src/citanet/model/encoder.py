"""Source-aware multimodal observation encoder (§4.2).

Each observation is embedded across five channels -- text, spatio-temporal,
state, type, source -- then fused with a learnable **source-reliability gate**
g(src) in (0,1)^5 whose prior is initialised from ``sources.yaml`` reliability:

    h(o) = LayerNorm( sum_c  g_c(src) * f_c(o) )
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..data.ontology import Ontology
from .featurize import FeatureSpace
from .text import LearnedTextEncoder

N_CHANNELS = 5   # text, spatiotemporal, state, type, source


def _logit(p: float, eps: float = 1e-4) -> float:
    p = min(1 - eps, max(eps, p))
    return math.log(p / (1 - p))


class FourierFeatures(nn.Module):
    """Deterministic multi-band sin/cos features for (x, y, t)."""

    def __init__(self, n_bands: int = 8):
        super().__init__()
        bands = torch.tensor([2.0 ** i for i in range(n_bands)], dtype=torch.float32)
        self.register_buffer("bands", bands)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: (N, C) -> (N, C * n_bands * 2)
        scaled = coords.unsqueeze(-1) * self.bands  # (N, C, B)
        feats = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
        return feats.reshape(coords.shape[0], -1)


class ObservationEncoder(nn.Module):
    def __init__(self, fs: FeatureSpace, ontology: Ontology, cfg):
        super().__init__()
        d = cfg.encoder.d_model
        self.d_model = d
        self.use_gate = cfg.encoder.use_source_gate
        self.fs = fs

        self.text_encoder = LearnedTextEncoder(fs.vocab, d)

        n_bands = cfg.encoder.spatiotemporal_freqs
        self.fourier = FourierFeatures(n_bands)
        st_in = 3 * n_bands * 2          # (x, y, t) x bands x {sin,cos}
        self.st_mlp = nn.Sequential(nn.Linear(st_in, d), nn.ReLU(), nn.Linear(d, d))

        self.state_embed = nn.Embedding(len(fs.states), d)
        self.type_embed = nn.Embedding(len(fs.types), d)
        self.type_proj = nn.Linear(d + 1, d)     # type emb + type_confidence
        self.source_embed = nn.Embedding(len(fs.sources), d)

        # --- source-reliability gate ---
        self.gate_embed = nn.Embedding(len(fs.sources), d)
        self.gate_lin = nn.Linear(d, N_CHANNELS)
        nn.init.zeros_(self.gate_lin.weight)
        nn.init.zeros_(self.gate_lin.bias)
        prior = torch.tensor([[_logit(ontology.source_reliability(s))] * N_CHANNELS
                              for s in fs.sources], dtype=torch.float32)
        self.register_buffer("gate_prior", prior)   # (n_sources, 5)

        self.norm = nn.LayerNorm(d)

    def channel_features(self, feats) -> torch.Tensor:
        """Return per-channel features stacked: (N, 5, d)."""
        f_text = self.text_encoder(feats.token_ids, feats.token_mask)
        st = torch.cat([feats.xy, feats.t.unsqueeze(-1)], dim=-1)   # (N, 3)
        f_st = self.st_mlp(self.fourier(st))
        f_state = self.state_embed(feats.state_id)
        f_type = self.type_proj(torch.cat(
            [self.type_embed(feats.type_id), feats.type_conf.unsqueeze(-1)], dim=-1))
        f_source = self.source_embed(feats.source_id)
        return torch.stack([f_text, f_st, f_state, f_type, f_source], dim=1)

    def gate(self, source_id: torch.Tensor) -> torch.Tensor:
        logits = self.gate_lin(self.gate_embed(source_id)) + self.gate_prior[source_id]
        return torch.sigmoid(logits)                 # (N, 5)

    def forward(self, feats) -> torch.Tensor:
        ch = self.channel_features(feats)            # (N, 5, d)
        if self.use_gate:
            g = self.gate(feats.source_id).unsqueeze(-1)   # (N, 5, 1)
            fused = (g * ch).sum(dim=1)
        else:
            fused = ch.sum(dim=1)
        return self.norm(fused)                      # (N, d)
