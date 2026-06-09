"""Turn a parsed :class:`Scenario` into model-ready tensors.

Produces, per scenario:
  * per-observation channel tensors (text tokens, spatio-temporal, state, type,
    source) for the source-aware encoder;
  * a candidate-pair list (from blocking) with precomputed constraint inputs
    (dt, dist, required/feasible speed, state-compat, source ids, relation
    Jaccard) consumed by the CTA;
  * supervision: per-cross-KG-pair same/diff labels and per-observation dangling
    labels.

The :class:`FeatureSpace` (vocab + type/state/source indexers) is built once
from the training split so ids are stable across scenarios.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..data.blocking import CandidatePair, generate_candidates
from ..data.kinematics import feasibility
from ..data.loader import load_scenario
from ..data.ontology import Ontology, load_ontology
from ..data.schema import Observation, Scenario
from .text import Vocab

XY_SCALE = 1000.0     # metres -> normalised units
T_SCALE = 300.0       # seconds (scenario window)


@dataclass
class FeatureSpace:
    vocab: Vocab
    types: list[str]
    states: list[str]
    sources: list[str]
    max_tokens: int = 16

    def type_id(self, t: str) -> int:
        return self.types.index(t) if t in self.types else self.types.index("UNKNOWN")

    def state_id(self, s: str) -> int:
        return self.states.index(s) if s in self.states else 0

    def source_id(self, s: str) -> int:
        return self.sources.index(s) if s in self.sources else 0

    @classmethod
    def build(cls, data_root: str | Path, ontology: Ontology,
              train_ids: list[str], max_tokens: int = 16) -> "FeatureSpace":
        data_root = Path(data_root)
        labels: list[str] = []
        for sid in train_ids:
            jl = data_root / "scenarios" / sid / "observations.jsonl"
            for line in jl.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    labels.append(json.loads(line)["label_text"])
        vocab = Vocab.build(labels)
        return cls(vocab=vocab, types=ontology.type_names,
                   states=ontology.state_names, sources=ontology.source_names,
                   max_tokens=max_tokens)


@dataclass
class ScenarioFeatures:
    scenario_id: str
    obs: list[Observation]                 # time-sorted
    obs_ids: list[str]
    kg_ids: list[str]
    local_entity_ids: list[str]
    # channel tensors
    token_ids: torch.Tensor                # (N, L)
    token_mask: torch.Tensor               # (N, L)
    xy: torch.Tensor                       # (N, 2) normalised
    t: torch.Tensor                        # (N,) normalised
    state_id: torch.Tensor                 # (N,)
    type_id: torch.Tensor                  # (N,)
    type_conf: torch.Tensor                # (N,)
    source_id: torch.Tensor                # (N,)
    # candidate pairs + constraint inputs
    pairs: list[CandidatePair]
    pair_i: torch.Tensor                   # (P,)
    pair_j: torch.Tensor                   # (P,)
    pair_cross: torch.Tensor               # (P,) bool
    p_dt: torch.Tensor                     # (P,) seconds
    p_dist: torch.Tensor                   # (P,) metres
    p_req_speed: torch.Tensor              # (P,)
    p_feas_speed: torch.Tensor             # (P,)
    p_state_compat: torch.Tensor           # (P,) C[si,sj]
    p_src_i: torch.Tensor                  # (P,)
    p_src_j: torch.Tensor                  # (P,)
    p_rel_jaccard: torch.Tensor            # (P,)
    p_text_cos: torch.Tensor               # (P,)
    # labels
    pair_label: torch.Tensor               # (P,) in {0,1}, valid where cross
    dangling_label: torch.Tensor           # (N,) in {0,1}
    # adjacency for the GNN (M2): list of (src_idx, dst_idx, rel_id)
    edges: list[tuple[int, int, int]] = field(default_factory=list)
    n_relations: int = 0
    # bookkeeping
    entity_of: dict[str, str] = field(default_factory=dict)   # obs_id -> "kg:local"


def _rel_event_set(o: Observation) -> set[tuple[str, str]]:
    # Predicate-level (not target_ref): relation targets are KG-local entity ids
    # that never match across KGs, so comparing predicates + shared commons event
    # ids gives a relational signal that IS comparable cross-KG (e.g. a "follows"
    # observation vs a "near" one). This is what the CTA b_rel term consumes.
    s = {("rel", r.predicate) for r in o.relations}
    s |= {("event", e.event_id) for e in o.events}
    return s


REL_NAMES = ["follows", "near", "partOf", "firesAt", "engagedWith",
             "emplacedAt", "movesToward", "supports", "screens", "event", "sametrack"]


def _build_edges(obs: list[Observation], max_event_group: int | None) -> list[tuple[int, int, int]]:
    """Same-track temporal chains + relation-target edges + co-event edges.

    Co-event edges are built per (kg, event) group; with ``max_event_group``
    set, groups larger than the cap are skipped (a scenario-wide event like an
    advance otherwise yields O(M^2) edges). ``max_event_group=None`` reproduces
    the original (uncapped) behaviour exactly -- used by the small suite.
    """
    rel_id = {r: i for i, r in enumerate(REL_NAMES)}
    edges: list[tuple[int, int, int]] = []
    by_entity: dict[str, list[int]] = {}
    for k, o in enumerate(obs):
        by_entity.setdefault(f"{o.kg_id}:{o.local_entity_id}", []).append(k)
    for members in by_entity.values():
        members.sort(key=lambda k: obs[k].t)
        for a, b in zip(members, members[1:]):
            edges.append((a, b, rel_id["sametrack"]))
            edges.append((b, a, rel_id["sametrack"]))
    for k, o in enumerate(obs):
        for r in o.relations:
            for m in by_entity.get(f"{o.kg_id}:{r.target_ref}", []):
                edges.append((k, m, rel_id.get(r.predicate, rel_id["near"])))
    # co-event edges, grouped by (kg, event) to avoid the O(M^2) scan
    by_event: dict[tuple[str, str], list[int]] = {}
    for k, o in enumerate(obs):
        for ev in o.events:
            by_event.setdefault((o.kg_id, ev.event_id), []).append(k)
    for members in by_event.values():
        if max_event_group is not None and len(members) > max_event_group:
            continue
        for a in members:
            for b in members:
                if a != b:
                    edges.append((a, b, rel_id["event"]))
    return edges


def assemble_features(scenario_id: str, obs: list[Observation], pairs,
                      ontology: Ontology, fs: FeatureSpace,
                      pair_label: torch.Tensor, dangling_label: torch.Tensor,
                      max_event_group: int | None = None) -> ScenarioFeatures:
    """Build the tensor bundle from a time-sorted observation list + candidate
    pairs + precomputed labels. Shared by the small (JSON-LD) and large (JSONL)
    featurisation paths."""
    n = len(obs)
    L = fs.max_tokens
    token_ids = torch.zeros((n, L), dtype=torch.long)
    token_mask = torch.zeros((n, L), dtype=torch.long)
    for k, o in enumerate(obs):
        ids = fs.vocab.encode(o.label_text, L)
        token_ids[k, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        token_mask[k, : len(ids)] = 1

    ex = np.array([o.easting for o in obs], dtype=np.float64)
    ny = np.array([o.northing for o in obs], dtype=np.float64)
    tt = np.array([o.t for o in obs], dtype=np.float64)
    cx, cy = ex.mean(), ny.mean()
    xy = torch.tensor(np.stack([(ex - cx) / XY_SCALE, (ny - cy) / XY_SCALE], axis=1),
                      dtype=torch.float32)
    t_norm = torch.tensor((tt - tt.min()) / T_SCALE, dtype=torch.float32)
    state_id = torch.tensor([fs.state_id(o.state) for o in obs], dtype=torch.long)
    type_id = torch.tensor([fs.type_id(o.type) for o in obs], dtype=torch.long)
    type_conf = torch.tensor([o.type_confidence for o in obs], dtype=torch.float32)
    source_id = torch.tensor([fs.source_id(o.source) for o in obs], dtype=torch.long)

    P = len(pairs)
    pair_i = torch.tensor([p.i for p in pairs], dtype=torch.long)
    pair_j = torch.tensor([p.j for p in pairs], dtype=torch.long)
    pair_cross = torch.tensor([p.cross_kg for p in pairs], dtype=torch.bool)
    p_dt = torch.zeros(P); p_dist = torch.zeros(P)
    p_req = torch.zeros(P); p_feas = torch.zeros(P)
    p_sc = torch.zeros(P); p_relj = torch.zeros(P); p_tc = torch.zeros(P)
    p_si = torch.zeros(P, dtype=torch.long); p_sj = torch.zeros(P, dtype=torch.long)
    for q, p in enumerate(pairs):
        oi, oj = obs[p.i], obs[p.j]
        feas = feasibility(oi, oj, ontology)
        p_dt[q] = feas.dt_s
        p_dist[q] = feas.dist_m
        p_req[q] = feas.required_speed_mps
        p_feas[q] = feas.feasible_speed_mps
        p_sc[q] = ontology.state_compat(oi.state, oj.state, oi.type)
        p_si[q] = fs.source_id(oi.source)
        p_sj[q] = fs.source_id(oj.source)
        si, sj = _rel_event_set(oi), _rel_event_set(oj)
        union = si | sj
        p_relj[q] = (len(si & sj) / len(union)) if union else 0.0
        p_tc[q] = p.text_cos

    edges = _build_edges(obs, max_event_group)
    entity_of = {o.obs_id: f"{o.kg_id}:{o.local_entity_id}" for o in obs}

    return ScenarioFeatures(
        scenario_id=scenario_id, obs=obs,
        obs_ids=[o.obs_id for o in obs], kg_ids=[o.kg_id for o in obs],
        local_entity_ids=[o.local_entity_id for o in obs],
        token_ids=token_ids, token_mask=token_mask, xy=xy, t=t_norm,
        state_id=state_id, type_id=type_id, type_conf=type_conf, source_id=source_id,
        pairs=pairs, pair_i=pair_i, pair_j=pair_j, pair_cross=pair_cross,
        p_dt=p_dt, p_dist=p_dist, p_req_speed=p_req, p_feas_speed=p_feas,
        p_state_compat=p_sc, p_src_i=p_si, p_src_j=p_sj, p_rel_jaccard=p_relj,
        p_text_cos=p_tc, pair_label=pair_label, dangling_label=dangling_label,
        edges=edges, n_relations=len(REL_NAMES), entity_of=entity_of,
    )


def featurize_scenario(scenario_dir: str | Path, fs: FeatureSpace,
                       ontology: Ontology, cfg) -> ScenarioFeatures:
    scenario_dir = Path(scenario_dir)
    scn = load_scenario(scenario_dir, scenario_dir.parent.parent / "ontology")
    obs = sorted(scn.observations, key=lambda o: (o.t, o.obs_id))

    pairs = generate_candidates(obs, ontology,
                                dt_max_s=cfg.blocking.dt_max_s,
                                theta_text=cfg.blocking.theta_text,
                                r_err_floor_m=cfg.blocking.r_err_floor_m,
                                reach_vmax_mult=cfg.blocking.reach_vmax_mult,
                                reach_extra_m=cfg.blocking.reach_extra_m)

    obs_level = json.loads((scenario_dir / "labels" / "observation_level.json").read_text(encoding="utf-8"))
    same_pairs = {frozenset((p["obs_a"], p["obs_b"]))
                  for p in obs_level["cross_kg_pairs"] if p["label"] == "same"}
    pair_label = torch.zeros(len(pairs), dtype=torch.float32)
    for q, p in enumerate(pairs):
        if p.cross_kg and frozenset((obs[p.i].obs_id, obs[p.j].obs_id)) in same_pairs:
            pair_label[q] = 1.0

    dangling = json.loads((scenario_dir / "labels" / "dangling.json").read_text(encoding="utf-8"))
    dang_obs = set(dangling["dangling_observations"])
    dangling_label = torch.tensor([1.0 if o.obs_id in dang_obs else 0.0 for o in obs],
                                  dtype=torch.float32)

    return assemble_features(scn.scenario_id, obs, pairs, ontology, fs,
                             pair_label, dangling_label, max_event_group=None)
