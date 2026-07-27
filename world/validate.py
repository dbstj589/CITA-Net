"""P4 — integrity validation. The GT must have ZERO structural violations by
construction; this asserts it and writes validation_report.json. Any failure
makes ``gen_world_gt`` exit non-zero.

Checks (spec §4):
  1. time monotonic          per-entity sample times strictly increase
  2. feasible motion         required speed between consecutive samples <= v_max
                             (and static states record speed 0)
  3. required attributes     no missing mandatory fields
  4. vocabulary              every type/state/relation/event is in the ontology
  5. triple count            unified_stkg.nt line count == manifest triple count
  6. cross-reference         no orphan references across the artefacts
"""
from __future__ import annotations

import math
from pathlib import Path

from .common import EVENT_KINDS, STATIC_STATES


def validate_sector(cfg, ontology, entities, units, landmarks, events, relations, traj_rows) -> dict:
    """In-memory per-sector validation (checks 1-4, 6; NOT triple count). Returns
    violation counts + a few examples per check, for streaming aggregation."""
    dt = float(cfg["time"]["dt_seconds"])
    ent_ids = {e.eid for e in entities}; unit_ids = set(units.keys())
    lm_ids = set(landmarks.keys()); ev_ids = {e.evid for e in events}
    type_of = {e.eid: e.true_type for e in entities}

    by_ent: dict[str, list[dict]] = {}
    for r in traj_rows:
        by_ent.setdefault(r["entity_id"], []).append(r)

    out = {"time_monotonic": [], "feasible_motion": [], "static_speed_zero": [],
           "required_attributes": [], "vocabulary": [], "cross_reference": []}
    tol = 1e-2
    states_ok = set(ontology.state_names); types_ok = set(ontology.type_names)
    rels_ok = set(ontology.relation_names)

    for eid, rows in by_ent.items():
        ts = [r["t"] for r in rows]
        if any(b <= a for a, b in zip(ts, ts[1:])):
            out["time_monotonic"].append(eid)
        typ = type_of.get(eid, "UNKNOWN")
        for a, b in zip(rows, rows[1:]):
            dist = math.hypot(b["easting"] - a["easting"], b["northing"] - a["northing"])
            if dist / dt > ontology.v_max(typ, b["state"]) + tol:
                out["feasible_motion"].append((eid, b["t"]))
        for r in rows:
            if r["state"] in STATIC_STATES and abs(r["speed"]) > tol:
                out["static_speed_zero"].append((eid, r["t"]))
            if r["state"] not in states_ok:
                out["vocabulary"].append(("state", r["state"]))
        if eid not in ent_ids:
            out["cross_reference"].append(("traj.entity", eid))

    for e in entities:
        for fld in ("eid", "true_type", "affiliation", "size"):
            if getattr(e, fld) in (None, ""):
                out["required_attributes"].append(("entity", e.eid, fld))
        if e.true_type not in types_ok:
            out["vocabulary"].append(("type", e.true_type))
        if e.parent_unit is not None and e.parent_unit not in unit_ids:
            out["cross_reference"].append(("entity.parent_unit", e.eid))
        for ev_id, _ in e.events:
            if ev_id not in ev_ids:
                out["cross_reference"].append(("entity.event", e.eid))
    for e in events:
        if e.kind not in EVENT_KINDS:
            out["vocabulary"].append(("event", e.kind))
        for p in e.participants:
            if p not in ent_ids:
                out["cross_reference"].append(("event.participant", e.evid))
    for r in relations:
        if r.predicate not in rels_ok:
            out["vocabulary"].append(("relation", r.predicate))
        if not (r.subject and r.predicate and r.obj) or r.start is None or r.end is None:
            out["required_attributes"].append(("relation", r.subject))
        if r.subject not in ent_ids:
            out["cross_reference"].append(("rel.subject", r.subject))
        pool = {"entity": ent_ids, "unit": unit_ids, "landmark": lm_ids}[r.obj_kind]
        if r.obj not in pool:
            out["cross_reference"].append(("rel.object", r.obj))
    return out


def run_validation(cfg, ontology, art, nt_path: Path, n_triples: int) -> tuple[dict, bool]:
    dt = float(cfg["time"]["dt_seconds"])
    entities = art["entities"]; units = art["units"]; landmarks = art["landmarks"]
    events = art["events"]; relations = art["relations"]; traj = art["traj_rows"]

    ent_ids = {e.eid for e in entities}
    unit_ids = set(units.keys())
    lm_ids = set(landmarks.keys())
    ev_ids = {e.evid for e in events}
    type_of = {e.eid: e.true_type for e in entities}

    report: dict = {}

    # group trajectory rows by entity (order preserved as written)
    by_ent: dict[str, list[dict]] = {}
    for r in traj:
        by_ent.setdefault(r["entity_id"], []).append(r)

    # 1. time monotonic
    viol = []
    for eid, rows in by_ent.items():
        ts = [r["t"] for r in rows]
        if any(b <= a for a, b in zip(ts, ts[1:])):
            viol.append(eid)
    report["time_monotonic"] = {"pass": not viol, "violations": len(viol),
                                "examples": viol[:5]}

    # 2. feasible motion (+ static speed 0)
    fviol, sviol = [], []
    tol = 1e-2                                   # 1 cm/s slack for rounding
    for eid, rows in by_ent.items():
        typ = type_of.get(eid, "UNKNOWN")
        for a, b in zip(rows, rows[1:]):
            dist = math.hypot(b["easting"] - a["easting"], b["northing"] - a["northing"])
            req = dist / dt
            vmax = ontology.v_max(typ, b["state"])
            if req > vmax + tol:
                fviol.append((eid, b["t"], round(req, 3), vmax))
        for r in rows:
            if r["state"] in STATIC_STATES and abs(r["speed"]) > tol:
                sviol.append((eid, r["t"], r["speed"]))
    report["feasible_motion"] = {"pass": not fviol, "violations": len(fviol),
                                 "examples": fviol[:5]}
    report["static_speed_zero"] = {"pass": not sviol, "violations": len(sviol),
                                   "examples": sviol[:5]}

    # 3. required attributes
    miss = []
    for e in entities:
        for fld in ("eid", "true_type", "affiliation", "size"):
            if getattr(e, fld) in (None, ""):
                miss.append(("entity", e.eid, fld))
    tcols = ("entity_id", "t", "easting", "northing", "elevation", "state", "speed", "heading")
    for r in traj:
        for c in tcols:
            if r.get(c) is None:
                miss.append(("traj", r.get("entity_id"), c)); break
    for e in events:
        if e.kind is None or e.start is None or e.end is None or e.participants is None:
            miss.append(("event", e.evid, "core"))
    for r in relations:
        if not (r.subject and r.predicate and r.obj) or r.start is None or r.end is None:
            miss.append(("relation", r.subject, "core"))
    report["required_attributes"] = {"pass": not miss, "violations": len(miss),
                                     "examples": miss[:5]}

    # 4. vocabulary
    vv = []
    types_ok = set(ontology.type_names); states_ok = set(ontology.state_names)
    rels_ok = set(ontology.relation_names)
    for e in entities:
        if e.true_type not in types_ok:
            vv.append(("type", e.true_type))
    for r in traj:
        if r["state"] not in states_ok:
            vv.append(("state", r["state"]))
    for r in relations:
        if r.predicate not in rels_ok:
            vv.append(("relation", r.predicate))
    for e in events:
        if e.kind not in EVENT_KINDS:
            vv.append(("event", e.kind))
    # dedup
    vv = list(dict.fromkeys(vv))
    report["vocabulary"] = {"pass": not vv, "violations": len(vv), "examples": vv[:10]}

    # 5. triple count
    n_lines = sum(1 for _ in open(nt_path, "r", encoding="utf-8"))
    report["triple_count"] = {"pass": n_lines == n_triples,
                              "nt_lines": n_lines, "manifest_triples": n_triples}

    # 6. cross-reference integrity
    xr = []
    for e in entities:
        if e.parent_unit is not None and e.parent_unit not in unit_ids:
            xr.append(("entity.parent_unit", e.eid, e.parent_unit))
        for ev_id, _ in e.events:
            if ev_id not in ev_ids:
                xr.append(("entity.event", e.eid, ev_id))
    for u in units.values():
        if u.parent is not None and u.parent not in unit_ids:
            xr.append(("unit.parent", u.uid, u.parent))
    for r in relations:
        if r.subject not in ent_ids:
            xr.append(("rel.subject", r.subject, r.predicate))
        pool = {"entity": ent_ids, "unit": unit_ids, "landmark": lm_ids}[r.obj_kind]
        if r.obj not in pool:
            xr.append(("rel.object", r.obj, r.obj_kind))
    for e in events:
        for p in e.participants:
            if p not in ent_ids:
                xr.append(("event.participant", e.evid, p))
        if e.landmark is not None and e.landmark not in lm_ids:
            xr.append(("event.landmark", e.evid, e.landmark))
    for eid in by_ent:
        if eid not in ent_ids:
            xr.append(("traj.entity", eid, ""))
    report["cross_reference"] = {"pass": not xr, "violations": len(xr), "examples": xr[:10]}

    ok = all(v["pass"] for v in report.values())
    report["_all_pass"] = ok
    return report, ok
