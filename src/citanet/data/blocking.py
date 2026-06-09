"""Candidate generation / blocking (§4.1).

A candidate pair ``(o_i, o_j)`` with ``t_i <= t_j`` is kept iff ALL hold:
  (1) type-compatible        : equal types, or either side UNKNOWN
  (2) forward time window     : 0 <= t_j - t_i <= dt_max
  (3) reach radius            : dist <= v_max(type, state) * dt + r_err
                                with r_err = cep_i + cep_j + r_err_floor
  (4) semantic threshold      : text_cos(label_i, label_j) >= theta_text
                                -- BYPASSED when either side is UNKNOWN type
                                   (the classifier abstained, so its label is
                                   unreliable; spatiotemporal + relational cues
                                   must carry such hard positives, e.g.
                                   id_0003). Documented assumption (§8).

Blocking keeps the candidate set near-linear instead of O(M^2) and never uses
trained weights, so it is a stable, reproducible pre-filter.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .ontology import UNKNOWN_TYPE, Ontology
from .schema import Observation
from .text_utils import text_cosine


@dataclass
class CandidatePair:
    i: int                  # index into the observation list (earlier in time)
    j: int                  # index into the observation list (later in time)
    cross_kg: bool
    dist_m: float
    dt_s: float
    text_cos: float
    reach_m: float


def _dist(a: Observation, b: Observation) -> float:
    return math.hypot(a.easting - b.easting, a.northing - b.northing)


def generate_candidates(
    observations: list[Observation],
    ontology: Ontology,
    dt_max_s: float = 180.0,
    theta_text: float = 0.3,
    r_err_floor_m: float = 5.0,
    cross_kg_only: bool = False,
    reach_vmax_mult: float = 1.0,
    reach_extra_m: float = 0.0,
    use_text_gate: bool = True,
    type_by_category: bool = False,
) -> list[CandidatePair]:
    """Return surviving candidate pairs (ordered so observation i precedes j)."""
    obs = sorted(observations, key=lambda o: (o.t, o.obs_id))
    idx = {o.obs_id: k for k, o in enumerate(obs)}
    pairs: list[CandidatePair] = []
    n = len(obs)
    for a in range(n):
        oi = obs[a]
        for b in range(a + 1, n):
            oj = obs[b]
            dt = oj.t - oi.t
            if dt < 0:
                continue
            if dt > dt_max_s:
                # times are sorted; everything past here is further -> stop
                break
            if cross_kg_only and oi.kg_id == oj.kg_id:
                continue
            # (1) type compatibility
            if not ontology.types_compatible(oi.type, oj.type, type_by_category):
                continue
            # (3) reach radius (use the more permissive of the two endpoints'
            # kinematic ceilings; UNKNOWN/Moving is already permissive)
            vmax = max(ontology.v_max(oi.type, oi.state),
                       ontology.v_max(oj.type, oj.state))
            r_err = oi.cep_m + oj.cep_m + r_err_floor_m
            reach = reach_vmax_mult * vmax * dt + r_err + reach_extra_m
            dist = _dist(oi, oj)
            if dist > reach:
                continue
            # (4) semantic threshold, bypassed for UNKNOWN-typed endpoints
            tcos = text_cosine(oi.label_text, oj.label_text)
            unknown = (oi.type == UNKNOWN_TYPE or oj.type == UNKNOWN_TYPE)
            if use_text_gate and not unknown and tcos < theta_text:
                continue
            pairs.append(CandidatePair(
                i=idx[oi.obs_id], j=idx[oj.obs_id],
                cross_kg=(oi.kg_id != oj.kg_id),
                dist_m=dist, dt_s=dt, text_cos=tcos, reach_m=reach))
    return pairs


def blocking_recall(pairs: list[CandidatePair], observations: list[Observation],
                    gold_same_obs_pairs: set[frozenset]) -> float:
    """Fraction of gold same-identity cross-KG obs pairs retained by blocking.

    Note: pairs whose time gap exceeds ``dt_max`` are *intentionally* dropped,
    so this can be < 1.0 even on a perfect blocker. For an acceptance signal use
    :func:`identity_coverage`, which only requires each identity to stay
    connectable.
    """
    obs = sorted(observations, key=lambda o: (o.t, o.obs_id))
    kept = {frozenset((obs[p.i].obs_id, obs[p.j].obs_id)) for p in pairs}
    if not gold_same_obs_pairs:
        return 1.0
    hit = sum(1 for g in gold_same_obs_pairs if g in kept)
    return hit / len(gold_same_obs_pairs)


def identity_coverage(pairs: list[CandidatePair], observations: list[Observation],
                      identities: list[dict]) -> float:
    """Fraction of matched identities that remain *connectable* after blocking,
    i.e. at least one retained cross-KG candidate links an A-side and a B-side
    observation of that identity. This is the property blocking must preserve so
    no true match becomes unrecoverable downstream.
    """
    obs = sorted(observations, key=lambda o: (o.t, o.obs_id))
    kept_cross = [frozenset((obs[p.i].obs_id, obs[p.j].obs_id))
                  for p in pairs if p.cross_kg]
    if not identities:
        return 1.0
    covered = 0
    for ident in identities:
        members = set(ident["member_observations"])
        if any(pair <= members for pair in kept_cross):
            covered += 1
    return covered / len(identities)
