"""Decoding observation/pair scores into entity-level identities (§2 invariants).

Two-level matching:
  * observation -> identity is many-to-one (an entity groups its observations);
  * identity <-> identity across KGs is one-to-one (greedy mutual matching).

Dangling entities (no good counterpart, or flagged by the dangling head) are
routed to abstain -- never force-matched. This module backs the M1/M2 evaluation
and is superseded by the Sinkhorn decoder in M3 for the full trajectory output.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .data.kinematics import feasibility
from .data.ontology import Ontology
from .model.citanet import ModelOutput
from .model.featurize import ScenarioFeatures


@dataclass
class PredEntity:
    key: str                     # "A:ent_A_001"
    kg_id: str
    local_entity_id: str
    obs_ids: list[str]
    dangling_prob: float


@dataclass
class PredIdentity:
    a_entity: str                # local id in KG A
    b_entity: str                # local id in KG B
    affinity: float
    member_obs: list[str]


@dataclass
class DecodeResult:
    identities: list[PredIdentity]
    dangling: list[PredEntity]
    entities: dict[str, PredEntity]
    affinity: dict[tuple[str, str], float]    # (A local, B local) -> score
    matched_keys: set[str] = field(default_factory=set)


def _entity_obs(feats: ScenarioFeatures) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for k, oid in enumerate(feats.obs_ids):
        key = f"{feats.kg_ids[k]}:{feats.local_entity_ids[k]}"
        groups.setdefault(key, []).append(k)
    return groups


def decode_entities(out: ModelOutput, feats: ScenarioFeatures,
                    match_threshold: float = 0.5,
                    dangling_threshold: float = 0.5,
                    agg: str = "mean") -> DecodeResult:
    # Decoding is inference-only; detach to avoid autograd bookkeeping/warnings.
    p_same = torch.sigmoid(out.pair_logits.detach())
    p_dang = torch.sigmoid(out.dangling_logits.detach())

    groups = _entity_obs(feats)
    # per-entity dangling probability = mean over its observations
    ent_dang: dict[str, float] = {}
    entities: dict[str, PredEntity] = {}
    for key, members in groups.items():
        kg, local = key.split(":", 1)
        d = float(p_dang[members].mean())
        ent_dang[key] = d
        entities[key] = PredEntity(key=key, kg_id=kg, local_entity_id=local,
                                   obs_ids=[feats.obs_ids[m] for m in members],
                                   dangling_prob=d)

    # --- aggregate cross-KG pair probs to entity-pair affinities ---
    bucket: dict[tuple[str, str], list[float]] = {}
    for q, p in enumerate(feats.pairs):
        if not feats.pair_cross[q]:
            continue
        ki = f"{feats.kg_ids[p.i]}:{feats.local_entity_ids[p.i]}"
        kj = f"{feats.kg_ids[p.j]}:{feats.local_entity_ids[p.j]}"
        a_key, b_key = (ki, kj) if feats.kg_ids[p.i] == "A" else (kj, ki)
        bucket.setdefault((a_key, b_key), []).append(float(p_same[q]))

    affinity: dict[tuple[str, str], float] = {}
    for (a_key, b_key), vals in bucket.items():
        t = torch.tensor(vals)
        score = float(t.mean()) if agg == "mean" else float(t.max())
        affinity[(a_key.split(":", 1)[1], b_key.split(":", 1)[1])] = score

    # --- greedy one-to-one matching ---
    ranked = sorted(bucket.items(),
                    key=lambda kv: (torch.tensor(kv[1]).mean() if agg == "mean"
                                    else torch.tensor(kv[1]).max()),
                    reverse=True)
    used: set[str] = set()
    identities: list[PredIdentity] = []
    matched_keys: set[str] = set()
    for (a_key, b_key), vals in ranked:
        if a_key in used or b_key in used:
            continue
        t = torch.tensor(vals)
        score = float(t.mean()) if agg == "mean" else float(t.max())
        if score < match_threshold:
            continue
        # abstain if either side is dominantly dangling
        if ent_dang[a_key] > dangling_threshold or ent_dang[b_key] > dangling_threshold:
            continue
        used.add(a_key); used.add(b_key)
        matched_keys.add(a_key); matched_keys.add(b_key)
        members = entities[a_key].obs_ids + entities[b_key].obs_ids
        identities.append(PredIdentity(
            a_entity=a_key.split(":", 1)[1], b_entity=b_key.split(":", 1)[1],
            affinity=score, member_obs=members))

    dangling = [e for key, e in entities.items() if key not in matched_keys]
    return DecodeResult(identities=identities, dangling=dangling,
                        entities=entities, affinity=affinity,
                        matched_keys=matched_keys)


# ===========================================================================
# M3 full decode: Sinkhorn slot assignment -> identities + trajectories
# ===========================================================================
@dataclass
class Transition:
    from_obs: str
    to_obs: str
    prob: float
    feasible_motion: bool
    required_speed_mps: float


@dataclass
class FullIdentity:
    slot: int
    type: str
    type_confidence: float
    match_confidence: float
    local_entity_ids: dict[str, str]          # {"A": .., "B": ..} one-to-one
    member_obs: list[tuple[str, float]]       # (obs_id, assign_prob)
    trajectory: list[dict]
    transitions: list[Transition]
    rationale: dict[str, float]
    fused_representative: dict = field(default_factory=dict)
    conflicts: list[dict] = field(default_factory=list)


@dataclass
class FullDecodeResult:
    identities: list[FullIdentity]
    dangling: list[dict]


def _pair_lookup(feats: ScenarioFeatures):
    d = {}
    for q, p in enumerate(feats.pairs):
        d[frozenset((feats.obs_ids[p.i], feats.obs_ids[p.j]))] = q
    return d


def decode_full(out: ModelOutput, feats: ScenarioFeatures, ontology: Ontology,
                null_threshold: float = 0.5) -> FullDecodeResult:
    """Decode the Sinkhorn assignment into identities with trajectories (§4.5).

    Entities are assigned to slots by the mean assignment of their observations;
    the ∅ slot (last column) absorbs danglings. Observations of the same slot
    and KG form that identity's local entity (one-to-one across KGs)."""
    assert out.assign is not None, "decode_full requires the M3 decoder (assign)"
    A = out.assign.detach()
    M, S = A.shape
    K = S - 1
    obs = feats.obs
    p_same = torch.sigmoid(out.pair_logits.detach())
    plut = _pair_lookup(feats)
    cta = out.cta

    groups = _entity_obs(feats)               # entity_key -> [obs idx]
    # entity -> (best_slot, null_mass, per-slot mean)
    ent_slot: dict[str, int] = {}
    ent_nullmass: dict[str, float] = {}
    ent_mass: dict[str, torch.Tensor] = {}
    for key, members in groups.items():
        mean_assign = A[members].mean(dim=0)
        ent_mass[key] = mean_assign
        ent_slot[key] = int(mean_assign.argmax())
        ent_nullmass[key] = float(mean_assign[K])

    # group entities by slot (excluding those routed to ∅)
    slot_entities: dict[int, list[str]] = {}
    dangling_keys: list[str] = []
    for key in groups:
        s = ent_slot[key]
        if s == K or ent_nullmass[key] > null_threshold:
            dangling_keys.append(key)
        else:
            slot_entities.setdefault(s, []).append(key)

    # Precompute (once, O(pairs)) the cross-KG candidate pairs grouped by the
    # (A-entity, B-entity) pair, and each entity's best cross candidate. This
    # avoids an O(identities * pairs) scan inside the loop, which is the decode
    # bottleneck at sector scale.
    ent_of = feats.entity_of
    cross_by_entpair: dict[tuple[str, str], list[int]] = {}
    ent_best_cross: dict[str, tuple[float, str]] = {}   # entity_key -> (prob, other_key)
    for q, p in enumerate(feats.pairs):
        if not feats.pair_cross[q]:
            continue
        ka, kb = ent_of[obs[p.i].obs_id], ent_of[obs[p.j].obs_id]
        a_key, b_key = (ka, kb) if ka.startswith("A:") else (kb, ka)
        cross_by_entpair.setdefault((a_key, b_key), []).append(q)
        pv = float(p_same[q])
        for me, other in ((ka, kb), (kb, ka)):
            if pv > ent_best_cross.get(me, (-1.0, ""))[0]:
                ent_best_cross[me] = (pv, other)

    identities: list[FullIdentity] = []
    for slot, keys in sorted(slot_entities.items()):
        a_keys = [k for k in keys if k.startswith("A:")]
        b_keys = [k for k in keys if k.startswith("B:")]
        # canonical one-to-one: highest-mass entity per KG; extras -> conflicts
        def best(keys_):
            return max(keys_, key=lambda k: float(ent_mass[k][slot])) if keys_ else None
        a_best, b_best = best(a_keys), best(b_keys)
        conflicts = []
        for k in a_keys + b_keys:
            if k not in (a_best, b_best):
                conflicts.append({"local_entity_id": k.split(":", 1)[1],
                                  "kg_id": k.split(":", 1)[0],
                                  "reason": "extra same-KG entity in slot (possible merge)"})
        local_ids = {}
        if a_best:
            local_ids["A"] = a_best.split(":", 1)[1]
        if b_best:
            local_ids["B"] = b_best.split(":", 1)[1]

        member_idx = [m for k in keys for m in groups[k]]
        member_idx.sort(key=lambda m: obs[m].t)
        members = [(obs[m].obs_id, float(A[m, slot])) for m in member_idx]

        # type / confidence: majority non-UNKNOWN type
        types = [obs[m].type for m in member_idx if obs[m].type != "UNKNOWN"]
        typ = max(set(types), key=types.count) if types else "UNKNOWN"
        tconf = float(sum(obs[m].type_confidence for m in member_idx) / len(member_idx))

        # trajectory
        traj = [{"obs_id": obs[m].obs_id, "time": obs[m].time_iso,
                 "easting": obs[m].easting, "northing": obs[m].northing,
                 "state": obs[m].state, "kg_id": obs[m].kg_id,
                 "source": obs[m].source} for m in member_idx]

        # transitions between consecutive observations
        transitions = []
        for m1, m2 in zip(member_idx, member_idx[1:]):
            o1, o2 = obs[m1], obs[m2]
            feas = feasibility(o1, o2, ontology)
            q = plut.get(frozenset((o1.obs_id, o2.obs_id)))
            prob = float(cta.p_transition.detach()[q]) if q is not None else \
                (1.0 if feas.feasible else 0.0)
            transitions.append(Transition(
                from_obs=o1.obs_id, to_obs=o2.obs_id, prob=prob,
                feasible_motion=bool(feas.feasible),
                required_speed_mps=round(feas.required_speed_mps, 3)))

        # rationale: mean CTA contributions over the canonical match's cross pairs
        cross_qs = cross_by_entpair.get((a_best, b_best), []) if (a_best and b_best) else []
        def cmean(t):
            return float(t.detach()[cross_qs].mean()) if cross_qs else 0.0
        rationale = {"sim_sem": cmean(cta.sim_sem), "b_time": cmean(cta.b_time),
                     "b_motion": cmean(cta.b_motion), "b_state": cmean(cta.b_state),
                     "b_rel": cmean(cta.b_rel), "b_src": cmean(cta.b_src)}
        match_conf = float(p_same[cross_qs].mean()) if cross_qs else float(A[member_idx, slot].mean())

        # fused representative: mean position + latest state
        if member_idx:
            fe = sum(obs[m].easting for m in member_idx) / len(member_idx)
            fn = sum(obs[m].northing for m in member_idx) / len(member_idx)
            rep = {"type": typ, "easting": round(fe, 2), "northing": round(fn, 2),
                   "last_state": obs[member_idx[-1]].state,
                   "last_time": obs[member_idx[-1]].time_iso}
        else:
            rep = {}

        identities.append(FullIdentity(
            slot=slot, type=typ, type_confidence=round(tconf, 3),
            match_confidence=round(match_conf, 3), local_entity_ids=local_ids,
            member_obs=members, trajectory=traj, transitions=transitions,
            rationale={k: round(v, 4) for k, v in rationale.items()},
            fused_representative=rep, conflicts=conflicts))

    # dangling entities -> abstain
    p_dang = torch.sigmoid(out.dangling_logits.detach())
    dangling = []
    for key in dangling_keys:
        kg, local = key.split(":", 1)
        members = groups[key]
        d_prob = max(ent_nullmass[key], float(p_dang[members].mean()))
        # nearest cross-KG candidate (highest pair prob), from the precomputed map
        nearest = None
        if key in ent_best_cross:
            bp, other_key = ent_best_cross[key]
            okg, olocal = other_key.split(":", 1)
            nearest = {"kg_id": okg, "local_entity_id": olocal, "pair_prob": round(bp, 3)}
        dangling.append({
            "kg_id": kg, "local_entity_id": local,
            "obs_ids": [obs[m].obs_id for m in members],
            "dangling_prob": round(d_prob, 3),
            "nearest_candidate": nearest,
            "decision": "abstain",
            "reason": ("routed to null slot" if ent_slot[key] == K
                       else "weak cross-KG support / high dangling probability"),
        })
    return FullDecodeResult(identities=identities, dangling=dangling)
