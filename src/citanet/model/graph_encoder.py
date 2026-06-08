"""Relation-context graph encoder (§4.3) -- hand-rolled, no torch-geometric.

Observations form a heterogeneous graph: same-track temporal chains, entity
relation edges (follows / near / supports / ...), and co-event edges. A small
multi-relation GAT does 2 hops of message passing so an observation's embedding
absorbs its relational context. This is what lets the UNKNOWN-typed hard
positive (id_0003: "follows the lead T-72") be told apart from the lead tank
when their positions are ambiguous -- the discriminative signal is relational,
not spatial.

Edges are provided by :func:`citanet.model.featurize.featurize_scenario` as
``(src_idx, dst_idx, relation_id)`` triples. A self-loop relation is added so a
node always retains its own state.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _edge_softmax(scores: torch.Tensor, dst: torch.Tensor, n: int) -> torch.Tensor:
    """Softmax over incoming edges grouped by destination node, per head.

    scores: (E, H), dst: (E,) -> alpha: (E, H) summing to 1 within each dst.
    """
    E, H = scores.shape
    # numerical stability: subtract per-dst max
    max_per_dst = torch.full((n, H), float("-inf"))
    max_per_dst.scatter_reduce_(0, dst.unsqueeze(-1).expand(E, H), scores,
                                reduce="amax", include_self=False)
    max_gather = max_per_dst[dst]
    max_gather = torch.nan_to_num(max_gather, neginf=0.0)
    exp = torch.exp(scores - max_gather)
    denom = torch.zeros((n, H))
    denom.scatter_reduce_(0, dst.unsqueeze(-1).expand(E, H), exp,
                          reduce="sum", include_self=False)
    denom_gather = denom[dst].clamp(min=1e-12)
    return exp / denom_gather


class RGATLayer(nn.Module):
    def __init__(self, d: int, n_relations: int, n_heads: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        assert d % n_heads == 0, "d_model must be divisible by n_heads"
        self.d = d
        self.heads = n_heads
        self.dh = d // n_heads
        self.W = nn.Linear(d, d)
        self.rel_embed = nn.Embedding(n_relations + 1, d)   # +1 = self-loop
        self.self_rel = n_relations
        self.att_src = nn.Parameter(torch.randn(n_heads, self.dh) * 0.1)
        self.att_dst = nn.Parameter(torch.randn(n_heads, self.dh) * 0.1)
        self.leaky = nn.LeakyReLU(0.2)
        self.norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, src: torch.Tensor, dst: torch.Tensor,
                rel: torch.Tensor) -> torch.Tensor:
        n = h.shape[0]
        # add self-loops
        self_idx = torch.arange(n)
        src = torch.cat([src, self_idx])
        dst = torch.cat([dst, self_idx])
        rel = torch.cat([rel, torch.full((n,), self.self_rel, dtype=torch.long)])

        Wh = self.W(h)                                   # (N, d)
        msg = Wh[src] + self.rel_embed(rel)              # (E, d)
        E = msg.shape[0]
        msg_h = msg.view(E, self.heads, self.dh)
        dst_h = Wh[dst].view(E, self.heads, self.dh)
        e = self.leaky((msg_h * self.att_src).sum(-1) + (dst_h * self.att_dst).sum(-1))
        alpha = self.dropout(_edge_softmax(e, dst, n))   # (E, H)
        weighted = msg_h * alpha.unsqueeze(-1)           # (E, H, dh)
        agg = torch.zeros((n, self.heads, self.dh))
        agg.scatter_reduce_(0, dst.view(E, 1, 1).expand(E, self.heads, self.dh),
                            weighted, reduce="sum", include_self=False)
        agg = agg.reshape(n, self.d)
        return self.norm(h + F.elu(agg))                 # residual + norm


class RelationContextGNN(nn.Module):
    def __init__(self, d: int, n_relations: int, cfg):
        super().__init__()
        self.layers = nn.ModuleList([
            RGATLayer(d, n_relations, n_heads=cfg.graph.n_heads)
            for _ in range(cfg.graph.gnn_layers)
        ])

    def forward(self, h: torch.Tensor, feats) -> torch.Tensor:
        if not feats.edges:
            return h
        edges = torch.tensor(feats.edges, dtype=torch.long)   # (E, 3)
        src, dst, rel = edges[:, 0], edges[:, 1], edges[:, 2]
        for layer in self.layers:
            h = layer(h, src, dst, rel)
        return h
