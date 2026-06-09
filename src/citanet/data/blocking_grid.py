"""Scalable blocking via a spatial hash grid + time buckets (§ scale-up).

The O(M^2) :func:`citanet.data.blocking.generate_candidates` compares every pair
within a sector. For large sectors that is wasteful, so this module indexes
observations by ``(cell_x, cell_y, time_bucket)`` and, for each observation,
only tests observations in nearby cells within the forward time window. The
spatial search radius is per-observation: it scales with that observation's own
kinematic reach (``v_max(type, state) * dt_max``), so slow types (infantry)
probe a tiny neighbourhood while fast types probe a larger one. Candidates that
survive must still pass the exact blocking criteria, so the result is identical
to the brute-force blocker (same criteria), only computed far cheaper.

Returns the same :class:`CandidatePair` objects and an index-compatible ordering.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .blocking import CandidatePair
from .ontology import UNKNOWN_TYPE, Ontology
from .schema import Observation
from .text_utils import text_cosine


@dataclass
class BlockingStats:
    n_obs: int
    n_candidates: int
    n_all_pairs: int                 # M*(M-1)/2
    reduction: float                 # n_all_pairs / n_candidates
    recall: float | None = None      # vs gold cross-KG same pairs (if provided)


def _cell(e: float, n: float, size: float) -> tuple[int, int]:
    return (int(math.floor(e / size)), int(math.floor(n / size)))


def generate_candidates_grid(
    observations: list[Observation],
    ontology: Ontology,
    dt_max_s: float = 180.0,
    theta_text: float = 0.3,
    r_err_floor_m: float = 5.0,
    cross_kg_only: bool = False,
    reach_vmax_mult: float = 1.0,
    reach_extra_m: float = 0.0,
    cell_size_m: float = 400.0,
    use_text_gate: bool = True,
    type_by_category: bool = False,
) -> list[CandidatePair]:
    obs = sorted(observations, key=lambda o: (o.t, o.obs_id))
    n = len(obs)
    idx = {o.obs_id: k for k, o in enumerate(obs)}
    # safe upper bound on a partner's CEP so the per-obs search radius never
    # under-covers a high-CEP (e.g. HUMINT) neighbour -> grid == brute force.
    max_cep = max((ontology.source_cep(s) for s in ontology.source_names), default=90.0)

    # spatial+time index: (cx, cy, tbucket) -> [k]
    tb = lambda t: int(math.floor(t / dt_max_s))
    grid: dict[tuple[int, int, int], list[int]] = {}
    for k, o in enumerate(obs):
        cx, cy = _cell(o.easting, o.northing, cell_size_m)
        grid.setdefault((cx, cy, tb(o.t)), []).append(k)

    pairs: list[CandidatePair] = []
    seen: set[tuple[int, int]] = set()
    for a in range(n):
        oi = obs[a]
        # per-obs reach radius: this obs's kinematic reach + its own CEP + the
        # worst-case partner CEP. A pair is found from whichever endpoint has the
        # larger reach, so per-obs v_max is sufficient; max_cep guards the CEP.
        vmax_i = ontology.v_max(oi.type, oi.state)
        reach_max = (reach_vmax_mult * vmax_i * dt_max_s
                     + oi.cep_m + max_cep + r_err_floor_m + reach_extra_m)
        rad = max(1, int(math.ceil(reach_max / cell_size_m)))
        cx, cy = _cell(oi.easting, oi.northing, cell_size_m)
        b = tb(oi.t)
        for tbk in (b, b + 1):                      # forward window spans <=2 buckets
            for dx in range(-rad, rad + 1):
                for dy in range(-rad, rad + 1):
                    bucket = grid.get((cx + dx, cy + dy, tbk))
                    if not bucket:
                        continue
                    for c in bucket:
                        if c == a:
                            continue
                        # order by time; ensure each unordered pair once
                        i, j = (a, c) if (obs[a].t, a) <= (obs[c].t, c) else (c, a)
                        key = (i, j)
                        if key in seen:
                            continue
                        seen.add(key)
                        cand = _check_pair(obs, i, j, ontology, dt_max_s, theta_text,
                                           r_err_floor_m, cross_kg_only,
                                           reach_vmax_mult, reach_extra_m, idx,
                                           use_text_gate, type_by_category)
                        if cand is not None:
                            pairs.append(cand)
    return pairs


def _check_pair(obs, i, j, ontology, dt_max_s, theta_text, r_err_floor_m,
                cross_kg_only, reach_vmax_mult, reach_extra_m, idx,
                use_text_gate=True, type_by_category=False):
    oi, oj = obs[i], obs[j]
    dt = oj.t - oi.t
    if dt < 0 or dt > dt_max_s:
        return None
    if cross_kg_only and oi.kg_id == oj.kg_id:
        return None
    if not ontology.types_compatible(oi.type, oj.type, type_by_category):
        return None
    vmax = max(ontology.v_max(oi.type, oi.state), ontology.v_max(oj.type, oj.state))
    r_err = oi.cep_m + oj.cep_m + r_err_floor_m
    reach = reach_vmax_mult * vmax * dt + r_err + reach_extra_m
    dist = math.hypot(oi.easting - oj.easting, oi.northing - oj.northing)
    if dist > reach:
        return None
    tcos = text_cosine(oi.label_text, oj.label_text)
    unknown = (oi.type == UNKNOWN_TYPE or oj.type == UNKNOWN_TYPE)
    if use_text_gate and not unknown and tcos < theta_text:
        return None
    return CandidatePair(i=idx[oi.obs_id], j=idx[oj.obs_id],
                         cross_kg=(oi.kg_id != oj.kg_id), dist_m=dist, dt_s=dt,
                         text_cos=tcos, reach_m=reach)


def blocking_stats(pairs: list[CandidatePair], observations: list[Observation],
                   gold_same_obs_pairs: set[frozenset] | None = None) -> BlockingStats:
    n = len(observations)
    all_pairs = n * (n - 1) // 2
    recall = None
    if gold_same_obs_pairs is not None:
        obs = sorted(observations, key=lambda o: (o.t, o.obs_id))
        kept = {frozenset((obs[p.i].obs_id, obs[p.j].obs_id)) for p in pairs}
        if gold_same_obs_pairs:
            recall = sum(1 for g in gold_same_obs_pairs if g in kept) / len(gold_same_obs_pairs)
        else:
            recall = 1.0
    return BlockingStats(n_obs=n, n_candidates=len(pairs), n_all_pairs=all_pairs,
                         reduction=(all_pairs / len(pairs)) if pairs else float("inf"),
                         recall=recall)


def gold_cross_kg_same_pairs(observations: list[Observation], gold: dict,
                             dt_max_s: float = 180.0) -> set[frozenset]:
    """Gold same-identity cross-KG observation pairs that are *within* the time
    window (the ones blocking is responsible for retaining)."""
    obs_to_gold = {a["obs_id"]: a["gold_identity_id"] for a in gold.get("assignment", [])}
    dang = set(gold.get("dangling_observations", []))
    by_obs = {o.obs_id: o for o in observations}
    # group obs by gold id, then enumerate cross-KG within-window pairs
    groups: dict[str, list[str]] = {}
    for oid, g in obs_to_gold.items():
        if oid in dang:
            continue
        groups.setdefault(g, []).append(oid)
    out: set[frozenset] = set()
    for g, oids in groups.items():
        for a in range(len(oids)):
            for b in range(a + 1, len(oids)):
                oa, ob = by_obs.get(oids[a]), by_obs.get(oids[b])
                if oa is None or ob is None or oa.kg_id == ob.kg_id:
                    continue
                if abs(oa.t - ob.t) <= dt_max_s:
                    out.add(frozenset((oa.obs_id, ob.obs_id)))
    return out
