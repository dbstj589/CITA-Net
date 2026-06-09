#!/usr/bin/env python
"""Large-scale procedural battlefield-STKG generator (>= 1,000,000 triples).

Deterministic, sector-sharded, and isolated from the frozen small suite: writes
under ``data/battlefield_stkg_large/`` with its own (extended) ontology. Each
sector is a 5 km tile holding several formations; the canonical STKG is emitted
as **N-Triples** (one triple per line -> exact, streamable triple counting) plus
a flattened ``observations.jsonl`` cache and ``gold_identities`` labels (NO
exhaustive O(M^2) pair enumeration -- training pairs are derived later from the
gold assignment + blocking).

Complexity (no trivial duplication): a tank platoon of many same-type T-72s and
multiple infantry squads (mass same-type hard negatives); long tracks with some
crossing trajectories (instantaneous position is insufficient); unit hierarchy
(partOf/follows/supports/screens) for dense relations; per-robot FOV/occlusion
so ``dangling_ratio`` of objects are seen by one robot only; label paraphrase
pool, UNKNOWN/mis-ID, per-source type-confidence; Destroyed wrecks + Emplaced
towed guns. All matches are kinematically valid. train/dev/test sectors use
disjoint seed bases (leakage-safe).

Usage:
    python scripts/gen_dataset_large.py --build-suite                 # ~1M triples
    python scripts/gen_dataset_large.py --build-suite --target-triples 1000000
    python scripts/gen_dataset_large.py --split train --n-sectors 4 --seed 1000001
    python scripts/gen_dataset_large.py --split dev --target-triples 50000 --seed 2000001
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "battlefield_stkg_large"
ONTOLOGY_DIR = DATA_ROOT / "ontology"

EPOCH = datetime(2025, 6, 6, 0, 0, 0, tzinfo=timezone.utc)
CRS = "UTM52N"
BASE_E, BASE_N = 322000.0, 4158000.0
SECTOR_PITCH = 6000.0          # metres between sector tile origins

# IRI namespaces for N-Triples
NS = "https://example.org/stkg/"
GEO = "https://example.org/geo/"
PROV = "http://www.w3.org/ns/prov#"
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"

KG_SOURCES = {
    "A": ["EO_IR", "RADAR"],
    "B": ["RADAR", "SIGINT", "ACOUSTIC", "HUMINT"],
}
ROBOT_IDS = ["A", "B", "C", "D"]
KG_CLOCK_OFFSET = {"A": 0.0, "B": 2.0, "C": -1.5, "D": 1.0}

# ---- paraphrase / mis-id pools -------------------------------------------
LABELS = {
    "T-72": ["T-72 전차", "T-72 주력전차", "T72 tank", "전차 T-72", "T-72형 전차"],
    "T-62": ["T-62 전차", "T-62 tank", "T62 전차", "구형 전차 T-62"],
    "BMP-2": ["BMP-2 보병전투차", "BMP-2 IFV", "BMP2 장갑차", "보병전투차 BMP-2"],
    "BMP-1": ["BMP-1 보병전투차", "BMP-1 IFV", "BMP1 장갑차"],
    "BTR-80": ["BTR-80 장갑차", "BTR-80 APC", "BTR80 병력수송장갑차"],
    "2S1": ["2S1 자주포", "2S1 Gvozdika", "2S1 자주곡사포"],
    "ZSU-23-4": ["ZSU-23-4 자주대공포", "ZSU-23-4 Shilka", "ZSU23-4 대공차량"],
    "Soldier": ["보병", "소총수", "infantry", "dismount", "보병 1명", "도보 병력"],
    "ZPU-4": ["ZPU-4 대공포", "ZPU-4 견인대공포", "ZPU4 AA gun"],
    "MO-120-RT": ["MO-120-RT 박격포", "MO-120 견인박격포", "120mm 박격포"],
}
UNKNOWN_LABELS = ["미상 표적", "미상 차량", "정체불명 표적", "분류불가 기동표적", "미상 보병"]
# plausible same-category mis-identifications (lowers type_confidence)
MISID = {"T-72": "T-62", "T-62": "T-72", "BMP-2": "BMP-1", "BMP-1": "BMP-2"}

SOURCE_CEP = {"EO_IR": 15.0, "RADAR": 30.0, "SIGINT": 50.0, "ACOUSTIC": 60.0, "HUMINT": 90.0}
SOURCE_TYPEREL = {"EO_IR": 0.9, "RADAR": 0.4, "SIGINT": 0.6, "ACOUSTIC": 0.3, "HUMINT": 0.7}
SOURCE_REL = {"EO_IR": 0.9, "RADAR": 0.8, "SIGINT": 0.65, "ACOUSTIC": 0.55, "HUMINT": 0.55}


# ===========================================================================
# Roster data structures
# ===========================================================================
@dataclass
class Track:
    kg_id: str
    sample_times: list[float]
    sources: list[str]
    obj_type: str
    label_text: str
    type_conf_scale: float = 1.0


@dataclass
class Obj:
    gold_id: str
    true_type: str
    waypoints: list[tuple[float, float, float]]
    states: list[tuple[float, str]]
    unit: Optional[str] = None           # partOf target (unit id)
    relations: list[tuple[str, tuple]] = field(default_factory=list)
    events: list[tuple[str, str]] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    dangling: bool = False

    def pos(self, t: float) -> tuple[float, float]:
        wps = self.waypoints
        if t <= wps[0][0]:
            return wps[0][1], wps[0][2]
        if t >= wps[-1][0]:
            return wps[-1][1], wps[-1][2]
        for (t0, e0, n0), (t1, e1, n1) in zip(wps, wps[1:]):
            if t0 <= t <= t1:
                a = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                return e0 + a * (e1 - e0), n0 + a * (n1 - n0)
        return wps[-1][1], wps[-1][2]

    def vel(self, t: float) -> tuple[float, float]:
        e0, n0 = self.pos(max(self.waypoints[0][0], t - 1))
        e1, n1 = self.pos(min(self.waypoints[-1][0], t + 1))
        span = max(1e-6, min(self.waypoints[-1][0], t + 1) - max(self.waypoints[0][0], t - 1))
        de, dn = (e1 - e0) / span, (n1 - n0) / span
        return math.hypot(de, dn), math.degrees(math.atan2(de, dn)) % 360.0

    def state_at(self, t: float) -> str:
        st = self.states[0][1]
        for tf, s in self.states:
            if t >= tf:
                st = s
        return st


@dataclass
class Knobs:
    identities_per_sector: int = 40
    obs_per_track: int = 10
    dangling_ratio: float = 0.2
    n_robots: int = 2


# ===========================================================================
# Sector roster construction
# ===========================================================================
def _pick(rng, pool):
    return pool[int(rng.integers(0, len(pool)))]


def build_sector(rng: np.random.Generator, knobs: Knobs, sx: float, sy: float
                 ) -> tuple[list[Obj], dict[str, dict], dict[str, str], dict[str, str]]:
    """Return (objects, units, landmarks, events) for one sector. sx,sy = tile
    origin offset (metres) from BASE."""
    ox, oy = BASE_E + sx, BASE_N + sy
    n = knobs.identities_per_sector
    # recipe scaled to identities_per_sector (sums to ~n)
    n_tanks = max(6, round(n * 0.30))
    n_soldiers = max(6, round(n * 0.40))
    n_ifv = max(2, round(n * 0.12))
    n_spg = max(1, round(n * 0.05))
    n_towed = max(1, round(n * 0.05))
    n_wreck = max(1, round(n * 0.05))
    robots = ROBOT_IDS[: max(2, knobs.n_robots)]

    units: dict[str, dict] = {}
    landmarks = {f"lm_hill_{int(sx)}_{int(sy)}": (ox + 1200, oy + 800),
                 f"lm_cross_{int(sx)}_{int(sy)}": (ox + 2500, oy - 600)}
    events = {f"evt_adv_{int(sx)}_{int(sy)}": "maneuver",
              f"evt_fire_{int(sx)}_{int(sy)}": "fires"}
    ev_adv = next(iter(events))
    lm_hill, lm_cross = list(landmarks)[0], list(landmarks)[1]

    plt = f"unit_plt_{int(sx)}_{int(sy)}"
    units[plt] = {"type": "Unit", "label": "Tank Platoon", "partOf": None}
    sqds = []
    for i in range(max(1, n_soldiers // 9)):
        sid = f"unit_sqd{i}_{int(sx)}_{int(sy)}"
        units[sid] = {"type": "Unit", "label": f"Rifle Squad {i}", "partOf": plt}
        sqds.append(sid)

    objs: list[Obj] = []
    gid = [0]

    def new_gid():
        gid[0] += 1
        return f"id_{int(sx)}_{int(sy)}_{gid[0]:04d}"

    obs_n = knobs.obs_per_track
    T0, T1 = 0.0, 600.0          # 10-minute sector window -> long tracks

    def times(rng, k, lo=0, hi=590):
        pool = list(range(lo, hi, max(1, (hi - lo) // (k + 2))))
        return sorted(float(x) for x in rng.choice(pool, size=min(k, len(pool)), replace=False))

    def assign_robots(rng):
        if float(rng.random()) < knobs.dangling_ratio:
            return [robots[int(rng.integers(0, len(robots)))]], True   # dangling
        return robots, False

    def tracks_for(rng, obj_type, robots_seen, true_type):
        tr = []
        for r in robots_seen:
            src_pool = KG_SOURCES.get(r, KG_SOURCES["B"])
            srcs = [_pick(rng, src_pool) for _ in range(obs_n)]
            # uncertainty: per-track type / label realisation
            t_obs, lab, conf = obj_type, _pick(rng, LABELS.get(true_type, [true_type])), 1.0
            u = float(rng.random())
            if u < 0.12:                      # UNKNOWN (classifier abstains)
                t_obs, lab, conf = "UNKNOWN", _pick(rng, UNKNOWN_LABELS), 0.6
            elif u < 0.22 and true_type in MISID:   # plausible mis-ID
                t_obs = MISID[true_type]
                lab = _pick(rng, LABELS[t_obs]); conf = 0.7
            tr.append(Track(r, times(rng, obs_n), srcs, t_obs, lab, conf))
        return tr

    # --- tank platoon: many same-type T-72, column with follows-chain, two
    #     crossing pairs (lane swap) so instantaneous position can't separate ---
    prev = None
    lane_y = np.linspace(-400, 400, n_tanks)
    for i in range(n_tanks):
        e0 = ox + rng.uniform(-100, 100)
        y = oy + float(lane_y[i])
        de = rng.uniform(1500, 2500)
        # crossing: even/odd adjacent tanks swap their lateral offset midway
        y_end = y + (300 if i % 2 == 0 else -300)
        wp = [(T0, e0, y), (300.0, e0 + de / 2, (y + y_end) / 2), (T1, e0 + de, y_end)]
        robots_seen, dang = assign_robots(rng)
        rels = [("partOf", ("unit", plt))]
        if prev is not None:
            rels.append(("follows", ("obj", prev)))
        g = new_gid()
        objs.append(Obj(gold_id=g, true_type="T-72", waypoints=wp,
                        states=[(0.0, "Moving")], unit=plt, relations=rels,
                        events=[(ev_adv, "armor")], dangling=dang,
                        tracks=tracks_for(rng, "T-72", robots_seen, "T-72")))
        prev = g

    # --- infantry squads: many Soldiers, slow, part_of squad, near each other ---
    for s_i, sid in enumerate(sqds):
        base_e = ox + rng.uniform(400, 2000)
        base_n = oy + rng.uniform(-600, 600)
        squad_lead = None
        for k in range(min(9, n_soldiers - s_i * 9)):
            e0 = base_e + rng.uniform(-60, 60)
            n0 = base_n + rng.uniform(-60, 60)
            de, dn = rng.uniform(-200, 200), rng.uniform(-200, 200)
            wp = [(T0, e0, n0), (T1, e0 + de, n0 + dn)]
            robots_seen, dang = assign_robots(rng)
            rels = [("partOf", ("unit", sid))]
            if squad_lead is not None:
                rels.append(("near", ("obj", squad_lead)))
            g = new_gid()
            objs.append(Obj(gold_id=g, true_type="Soldier", waypoints=wp,
                            states=[(0.0, "Moving")], unit=sid, relations=rels,
                            events=[(ev_adv, "infantry")], dangling=dang,
                            tracks=tracks_for(rng, "Soldier", robots_seen, "Soldier")))
            if squad_lead is None:
                squad_lead = g

    # --- IFVs (support the platoon) ---
    for i in range(n_ifv):
        t = _pick(rng, ["BMP-2", "BMP-1"])
        e0 = ox + rng.uniform(-200, 400); n0 = oy + rng.uniform(-700, 700)
        wp = [(T0, e0, n0), (T1, e0 + rng.uniform(1000, 2000), n0 + rng.uniform(-300, 300))]
        robots_seen, dang = assign_robots(rng)
        objs.append(Obj(gold_id=new_gid(), true_type=t, waypoints=wp,
                        states=[(0.0, "Moving")], relations=[("supports", ("unit", plt))],
                        events=[(ev_adv, "ifv")], dangling=dang,
                        tracks=tracks_for(rng, t, robots_seen, t)))

    # --- SPG / SPAAG (halt then fire) ---
    for i in range(n_spg):
        t = _pick(rng, ["2S1", "ZSU-23-4"])
        e0 = ox + rng.uniform(1500, 2500); n0 = oy + rng.uniform(-900, -300)
        wp = [(T0, e0, n0), (200.0, e0 + 60, n0 + 20), (T1, e0 + 60, n0 + 20)]
        robots_seen, dang = assign_robots(rng)
        objs.append(Obj(gold_id=new_gid(), true_type=t,
                        waypoints=wp, states=[(0.0, "Moving"), (200.0, "Halted"), (300.0, "Firing")],
                        relations=[("firesAt", ("landmark", lm_cross))],
                        events=[(next(k for k in events if "fire" in k), "shooter")],
                        dangling=dang, tracks=tracks_for(rng, t, robots_seen, t)))

    # --- towed weapons (Emplaced, dangling-leaning: one robot only) ---
    for i in range(n_towed):
        t = _pick(rng, ["ZPU-4", "MO-120-RT"])
        e0 = ox + rng.uniform(800, 2600); n0 = oy + rng.uniform(-900, 900)
        wp = [(T0, e0, n0), (T1, e0, n0)]
        r = _pick(rng, robots)
        objs.append(Obj(gold_id=new_gid(), true_type=t, waypoints=wp,
                        states=[(0.0, "Emplaced"), (300.0, "Firing")],
                        relations=[("emplacedAt", ("landmark", lm_hill))],
                        events=[(next(k for k in events if "fire" in k), "shooter")],
                        dangling=True, tracks=tracks_for(rng, t, [r], t)))

    # --- Destroyed wrecks (stationary; b_state/b_motion bait), dangling ---
    for i in range(n_wreck):
        t = _pick(rng, ["T-72", "T-62"])
        e0 = ox + rng.uniform(-100, 2400); n0 = oy + rng.uniform(-500, 500)
        wp = [(T0, e0, n0), (T1, e0, n0)]
        r = _pick(rng, robots)
        objs.append(Obj(gold_id=new_gid(), true_type=t, waypoints=wp,
                        states=[(0.0, "Destroyed")], relations=[("near", ("unit", plt))],
                        events=[], dangling=True,
                        tracks=tracks_for(rng, t, [r], t)))

    return objs, units, landmarks, events


# ===========================================================================
# Realisation: roster -> observations + entities + gold
# ===========================================================================
def _iso(t: float) -> str:
    return (EPOCH + timedelta(seconds=t)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _mgrs(e, n):
    return f"52SCF{int(e) % 100000:05d}{int(n) % 100000:05d}"


def realise_sector(sector_id, objs, units, rng):
    # local entity ids per robot
    ent_counter: dict[str, int] = {}
    obj_local: dict[str, dict[str, str]] = {}
    for o in objs:
        obj_local[o.gold_id] = {}
        for tr in o.tracks:
            ent_counter[tr.kg_id] = ent_counter.get(tr.kg_id, 0) + 1
            obj_local[o.gold_id][tr.kg_id] = f"ent_{tr.kg_id}_{ent_counter[tr.kg_id]:04d}"

    def resolve(kg, tgt):
        kind, ref = tgt
        if kind == "landmark":
            return ("lm", ref)
        if kind == "unit":
            return ("unit", ref)            # units exist per-robot scope (shared id)
        if kind == "obj":
            lid = obj_local.get(ref, {}).get(kg)
            return ("ent", lid) if lid else None
        return None

    observations = []
    entities: dict[str, dict] = {}
    oc = [0]
    for o in objs:
        for tr in o.tracks:
            kg = tr.kg_id
            lid = obj_local[o.gold_id][kg]
            ekey = f"{kg}:{lid}"
            rels = []
            for pred, tgt in o.relations:
                r = resolve(kg, tgt)
                if r:
                    rels.append({"predicate": pred, "target_ref": r[1], "target_kind": r[0],
                                 "confidence": round(float(0.8 + 0.2 * rng.random()), 3)})
            evs = [{"event_id": e, "role": role} for e, role in o.events]
            entities[ekey] = {"local_entity_id": lid, "kg_id": kg, "object_type": tr.obj_type,
                              "relations": rels, "events": evs, "obs_ids": [],
                              "_gold": o.gold_id, "_dangling": o.dangling, "unit": o.unit}
            for i, tt in enumerate(tr.sample_times):
                oc[0] += 1
                oid = f"obs_{kg}_{oc[0]:05d}"
                src = tr.sources[i % len(tr.sources)]
                cep = SOURCE_CEP.get(src, 50.0)
                t_obs = tt + KG_CLOCK_OFFSET.get(kg, 0.0) + float(rng.normal(0, 1.0))
                e_t, n_t = o.pos(tt)
                sig = cep / 1.1774
                e_o = e_t + float(rng.normal(0, sig)); n_o = n_t + float(rng.normal(0, sig))
                st = o.state_at(tt)
                spd, hd = o.vel(tt)
                if st in ("Emplaced", "Halted", "Destroyed"):
                    spd = 0.0
                tconf = round(min(0.99, max(0.2, SOURCE_TYPEREL.get(src, 0.5) * tr.type_conf_scale
                                            + 0.05 * rng.random())), 3)
                sconf = round(min(0.99, max(0.2, SOURCE_REL.get(src, 0.5) + 0.05 * rng.random())), 3)
                observations.append({
                    "obs_id": oid, "kg_id": kg, "source": src, "local_entity_id": lid,
                    "label_text": tr.label_text, "type": tr.obj_type, "type_confidence": tconf,
                    "state": st, "state_confidence": sconf, "time": _iso(t_obs),
                    "location": {"crs": CRS, "easting": round(e_o, 2), "northing": round(n_o, 2),
                                 "elevation": 50.0, "mgrs": _mgrs(e_o, n_o)},
                    "cep_m": round(cep, 1),
                    "velocity": {"speed_mps": round(spd, 2), "heading_deg": round(hd, 1)},
                    "relations": [{"predicate": r["predicate"], "target_ref": r["target_ref"],
                                   "confidence": r["confidence"]} for r in rels],
                    "events": evs,
                    "provenance": {"wasAttributedTo": src, "generatedAtTime": _iso(t_obs)},
                })
                entities[ekey]["obs_ids"].append(oid)

    observations.sort(key=lambda o: (o["time"], o["obs_id"]))

    # gold (assignment only; no O(M^2) pair lists)
    obs_to_gold = {}
    identities, dang_obs, dang_ent = [], [], []
    by_gold: dict[str, list[str]] = {}
    for ek, e in entities.items():
        by_gold.setdefault(e["_gold"], []).extend(e["obs_ids"])
    for o in objs:
        members = sorted(by_gold.get(o.gold_id, []))
        for oid in members:
            obs_to_gold[oid] = o.gold_id
        if o.dangling:
            dang_obs.extend(members)
            for kg, lid in obj_local[o.gold_id].items():
                dang_ent.append({"kg_id": kg, "local_entity_id": lid, "true_type": o.true_type,
                                 "obs_ids": members})
        else:
            identities.append({"gold_identity_id": o.gold_id, "true_type": o.true_type,
                               "member_observations": members,
                               "member_local_entities": obj_local[o.gold_id]})
    gold = {"scenario_id": sector_id, "identities": identities,
            "dangling_observations": sorted(dang_obs), "dangling_local_entities": dang_ent,
            "assignment": [{"obs_id": k, "gold_identity_id": v} for k, v in sorted(obs_to_gold.items())]}
    return observations, entities, gold, obj_local


# ===========================================================================
# N-Triples emission (canonical STKG)  -> returns triple count
# ===========================================================================
def _iri(*parts):
    return "<" + NS + "large/" + "/".join(str(p) for p in parts) + ">"


def _lit(v, dtype=None):
    s = str(v).replace("\\", "\\\\").replace('"', '\\"')
    if dtype:
        return f'"{s}"^^<{dtype}>'
    return f'"{s}"'


def emit_ntriples(sector_dir: Path, sector_id, entities, observations, units,
                  landmarks, events) -> int:
    xsd = "http://www.w3.org/2001/XMLSchema#"
    lines: list[str] = []

    def T(s, p, o):
        lines.append(f"{s} {p} {o} .")

    P = lambda name: f"<{NS}{name}>"
    GP = lambda name: f"<{GEO}{name}>"

    # units
    for uid, u in units.items():
        s = _iri(sector_id, "unit", uid)
        T(s, f"<{RDF}type>", P("Unit"))
        T(s, f"<{RDFS}label>", _lit(u["label"]))
        if u.get("partOf"):
            T(s, P("partOf"), _iri(sector_id, "unit", u["partOf"]))
    # landmarks / events (commons within sector)
    for lid, (e, n) in landmarks.items():
        s = _iri(sector_id, "lm", lid)
        T(s, f"<{RDF}type>", P("Landmark"))
        T(s, GP("easting"), _lit(round(e, 2), xsd + "double"))
        T(s, GP("northing"), _lit(round(n, 2), xsd + "double"))
    for evid, kind in events.items():
        s = _iri(sector_id, "evt", evid)
        T(s, f"<{RDF}type>", P("Event"))
        T(s, P("eventKind"), _lit(kind))
    # entities
    for ek, e in entities.items():
        kg = e["kg_id"]
        s = _iri(sector_id, kg, e["local_entity_id"])
        T(s, f"<{RDF}type>", P("Entity"))
        T(s, P("objectType"), _lit(e["object_type"]))
        for r in e["relations"]:
            kind = r.get("target_kind", "ent")
            if kind == "ent":
                obj = _iri(sector_id, kg, r["target_ref"])
            elif kind == "unit":
                obj = _iri(sector_id, "unit", r["target_ref"])
            else:
                obj = _iri(sector_id, "lm", r["target_ref"])
            T(s, P(r["predicate"]), obj)
        for ev in e["events"]:
            T(s, P("participatesIn"), _iri(sector_id, "evt", ev["event_id"]))
    # observations
    for o in observations:
        kg = o["kg_id"]
        s = _iri(sector_id, kg, o["obs_id"])
        loc = o["location"]; v = o["velocity"]
        T(s, f"<{RDF}type>", P("Observation"))
        T(s, P("observationOf"), _iri(sector_id, kg, o["local_entity_id"]))
        T(s, P("time"), _lit(o["time"], xsd + "dateTime"))
        T(s, GP("easting"), _lit(loc["easting"], xsd + "double"))
        T(s, GP("northing"), _lit(loc["northing"], xsd + "double"))
        T(s, GP("elevation"), _lit(loc["elevation"], xsd + "double"))
        T(s, GP("crs"), _lit(loc["crs"]))
        T(s, GP("mgrs"), _lit(loc["mgrs"]))
        T(s, P("cepM"), _lit(o["cep_m"], xsd + "double"))
        T(s, P("objectType"), _lit(o["type"]))
        T(s, P("typeConfidence"), _lit(o["type_confidence"], xsd + "double"))
        T(s, P("state"), _lit(o["state"]))
        T(s, P("stateConfidence"), _lit(o["state_confidence"], xsd + "double"))
        T(s, P("labelText"), _lit(o["label_text"]))
        T(s, P("source"), _lit(o["source"]))
        T(s, P("speedMps"), _lit(v["speed_mps"], xsd + "double"))
        T(s, P("headingDeg"), _lit(v["heading_deg"], xsd + "double"))
        T(s, f"<{PROV}wasAttributedTo>", _lit(o["source"]))
        T(s, f"<{PROV}generatedAtTime>", _lit(o["provenance"]["generatedAtTime"], xsd + "dateTime"))

    # write per-robot shards (split lines by subject KG) + a units/commons shard
    robots = sorted({e["kg_id"] for e in entities.values()})
    # simplest: write all to one file per robot is overkill; write sector.nt once
    (sector_dir / "stkg.nt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


# ===========================================================================
# Per-sector driver + suite builder
# ===========================================================================
def write_sector(split: str, sector_id: str, seed: int, idx: int, knobs: Knobs) -> dict:
    rng = np.random.default_rng(seed)
    sx = (idx % 8) * SECTOR_PITCH
    sy = (idx // 8) * SECTOR_PITCH
    objs, units, landmarks, events = build_sector(rng, knobs, sx, sy)
    observations, entities, gold, obj_local = realise_sector(sector_id, objs, units, rng)

    sector_dir = DATA_ROOT / "sectors" / sector_id
    (sector_dir / "labels").mkdir(parents=True, exist_ok=True)
    n_triples = emit_ntriples(sector_dir, sector_id, entities, observations, units, landmarks, events)
    with open(sector_dir / "observations.jsonl", "w", encoding="utf-8") as fh:
        for o in observations:
            fh.write(json.dumps(o, ensure_ascii=False) + "\n")
    (sector_dir / "labels" / "gold_identities.json").write_text(
        json.dumps(gold, ensure_ascii=False), encoding="utf-8")
    (sector_dir / "labels" / "dangling.json").write_text(
        json.dumps({"scenario_id": sector_id,
                    "dangling_observations": gold["dangling_observations"],
                    "dangling_entities": gold["dangling_local_entities"]},
                   ensure_ascii=False), encoding="utf-8")

    n_a = sum(1 for e in entities.values() if e["kg_id"] == "A")
    n_b = sum(1 for e in entities.values() if e["kg_id"] == "B")
    manifest = {"scenario_id": sector_id, "split": split, "seed": seed, "crs": CRS,
                "epoch": EPOCH.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "n_observations": len(observations), "n_entities_A": n_a, "n_entities_B": n_b,
                "n_matched_identities": len(gold["identities"]),
                "n_dangling_entities": len(gold["dangling_local_entities"]),
                "n_triples": n_triples,
                "files": ["stkg.nt", "observations.jsonl", "manifest.json", "labels/"]}
    (sector_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    return manifest


SEED_BASE = {"train": 1_000_000, "dev": 2_000_000, "test": 3_000_000}


def build_split(split: str, knobs: Knobs, n_sectors: Optional[int],
                target_triples: Optional[int], start_idx: int = 0) -> tuple[list[str], int]:
    ids, total, idx = [], 0, 0
    base = SEED_BASE[split]
    while True:
        if n_sectors is not None and idx >= n_sectors:
            break
        sid = f"sec_{split}_{idx:04d}"
        m = write_sector(split, sid, base + idx, start_idx + idx, knobs)
        ids.append(sid); total += m["n_triples"]; idx += 1
        if target_triples is not None and total >= target_triples and n_sectors is None:
            break
        if n_sectors is None and target_triples is None:
            break
    return ids, total


def build_suite(target_triples: Optional[int], knobs: Knobs) -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    # dev/test fixed-size; train grows to hit the total target (default 1,000,000)
    target = target_triples or 1_000_000
    splits = {}
    totals = {}
    dev_ids, dev_t = build_split("dev", knobs, n_sectors=12, target_triples=None)
    test_ids, test_t = build_split("test", knobs, n_sectors=13, target_triples=None)
    splits["dev"], splits["test"] = dev_ids, test_ids
    totals["dev"], totals["test"] = dev_t, test_t

    # grow train until cumulative total >= target (and >= 50 sectors)
    train_ids, train_t, idx = [], 0, 0
    base = SEED_BASE["train"]
    while (train_t + dev_t + test_t) < target or len(train_ids) < 50:
        sid = f"sec_train_{idx:04d}"
        m = write_sector("train", sid, base + idx, idx, knobs)
        train_ids.append(sid); train_t += m["n_triples"]; idx += 1
        if len(train_ids) % 10 == 0:
            print(f"  train sectors={len(train_ids)} train_triples={train_t} "
                  f"total={train_t + dev_t + test_t}")
    splits["train"], totals["train"] = train_ids, train_t

    splits_dir = DATA_ROOT / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    for s, ids in splits.items():
        (splits_dir / f"{s}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")

    grand = sum(totals.values())
    global_manifest = {
        "data_root": str(DATA_ROOT.relative_to(REPO_ROOT)),
        "total_triples": grand,
        "target_triples": target,
        "sectors": {s: len(ids) for s, ids in splits.items()},
        "triples_per_split": totals,
        "knobs": vars(knobs),
        "seed_bases": SEED_BASE,
    }
    (DATA_ROOT / "manifest_global.json").write_text(
        json.dumps(global_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nTOTAL TRIPLES = {grand:,}  (target {target:,})  "
          f"sectors train/dev/test = {len(splits['train'])}/{len(splits['dev'])}/{len(splits['test'])}")
    print(f"global manifest -> {DATA_ROOT / 'manifest_global.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Large-scale CITA-Net STKG generator")
    ap.add_argument("--build-suite", action="store_true")
    ap.add_argument("--split", default="train", choices=["train", "dev", "test"])
    ap.add_argument("--n-sectors", type=int, default=None)
    ap.add_argument("--identities-per-sector", type=int, default=40)
    ap.add_argument("--obs-per-track", type=int, default=10)
    ap.add_argument("--dangling-ratio", type=float, default=0.2)
    ap.add_argument("--n-robots", type=int, default=2)
    ap.add_argument("--target-triples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    knobs = Knobs(identities_per_sector=args.identities_per_sector,
                  obs_per_track=args.obs_per_track,
                  dangling_ratio=args.dangling_ratio, n_robots=args.n_robots)

    if args.build_suite:
        build_suite(args.target_triples, knobs)
        return
    # single-split run (n-sectors or target-triples)
    if args.seed is not None:
        SEED_BASE[args.split] = args.seed
    ids, total = build_split(args.split, knobs, args.n_sectors, args.target_triples)
    print(f"[{args.split}] sectors={len(ids)} triples={total:,}")


if __name__ == "__main__":
    main()
