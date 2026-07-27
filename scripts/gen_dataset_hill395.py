#!/usr/bin/env python
"""Hill 395 (White Horse Mountain / 백마고지) battlefield-STKG generator.

A NEW battlefield domain modelled on the Battle of White Horse (1952-10-06..15),
emitted in the **same observation-centric format as the large suite** so it is a
drop-in for the CITA-Net cross-KG identity/track-association pipeline (grid
blocking, streaming loader, featurize_sector, train_large, eval). It writes under
``data/battlefield_hill395_large/`` with its own (Hill-395-grounded) ontology and
is fully isolated from the frozen suites.

Grounding: the object-type taxonomy, unit roster, terrain landmarks, Korean label
pool and the Oct-6..15 phase timeline are seeded from the Hill 395 source material
(entity dictionary / taxonomy / relation dictionary / name-graph) and then scaled
procedurally -- sub-units are decomposed (선두조/정찰조/증원대/잔존대/침투조) into many
same-type objects (mass hard negatives), units seen by one side only become
dangling, and labels are paraphrased / mis-identified per source (noise). All
matches stay kinematically valid; train/dev/test sectors use disjoint seed bases.

The realisation, gold/dangling derivation and N-Triples emission reuse the proven
large-suite machinery verbatim; only the DOMAIN (ontology, pools, sector roster)
is Hill-395-specific. The ontology YAML/JSON-LD files are emitted by
``write_ontology()`` so the suite is reproducible from this one script.

Usage:
    python scripts/gen_dataset_hill395.py --build-suite                 # ~1M triples
    python scripts/gen_dataset_hill395.py --build-suite --target-triples 1000000
    python scripts/gen_dataset_hill395.py --split dev --n-sectors 2 --emit-ontology
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "battlefield_hill395_large"
ONTOLOGY_DIR = DATA_ROOT / "ontology"
# Difficulty knob (robustness sweep): multiplies BOTH the injected position error
# AND the reported cep_m, keeping the data self-consistent so blocking reach grows
# with the noise. 1.0 = the frozen baseline. Overridden by main() via --noise-mult.
NOISE_MULT = 1.0

# Battle of White Horse opened 1952-10-06 19:00; use the day as the epoch.
EPOCH = datetime(1952, 10, 6, 0, 0, 0, tzinfo=timezone.utc)
CRS = "UTM52S"
# Hill 395 sits NW of Chorwon (~38.29N 127.10E); plausible UTM-52S origin.
BASE_E, BASE_N = 300000.0, 4238000.0
SECTOR_PITCH = 6000.0          # metres between sector tile origins

# IRI namespaces for N-Triples (mirror the large suite, distinct path segment)
NS = "https://example.org/stkg/"
GEO = "https://example.org/geo/"
PROV = "http://www.w3.org/ns/prov#"
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
PATH_SEG = "hill395"

# ---- intelligence sources (KG A = own direct sensors, KG B = standoff/indirect) -
KG_SOURCES = {
    "A": ["VISUAL_OBS", "RADAR", "ARTILLERY_OBS"],
    "B": ["AERIAL", "SIGINT", "ACOUSTIC", "HUMINT"],
}
ROBOT_IDS = ["A", "B", "C", "D"]
KG_CLOCK_OFFSET = {"A": 0.0, "B": 2.0, "C": -1.5, "D": 1.0}

SOURCE_CEP = {"VISUAL_OBS": 15.0, "RADAR": 30.0, "ARTILLERY_OBS": 35.0,
              "AERIAL": 25.0, "SIGINT": 50.0, "ACOUSTIC": 60.0, "HUMINT": 90.0}
SOURCE_TYPEREL = {"VISUAL_OBS": 0.90, "RADAR": 0.40, "ARTILLERY_OBS": 0.50,
                  "AERIAL": 0.70, "SIGINT": 0.60, "ACOUSTIC": 0.30, "HUMINT": 0.70}
SOURCE_REL = {"VISUAL_OBS": 0.90, "RADAR": 0.80, "ARTILLERY_OBS": 0.75,
              "AERIAL": 0.80, "SIGINT": 0.65, "ACOUSTIC": 0.55, "HUMINT": 0.55}
SOURCE_DESC = {
    "VISUAL_OBS": "Forward visual / EO observer (전방관측조)",
    "RADAR": "Ground-surveillance / counter-battery radar",
    "ARTILLERY_OBS": "Artillery forward observer (포병관측)",
    "AERIAL": "UN air reconnaissance / tac-air observer",
    "SIGINT": "Signals-intelligence emitter geolocation (무선감청)",
    "ACOUSTIC": "Acoustic gunfire / vehicle localisation (청음조)",
    "HUMINT": "Scout report / POW interrogation (정찰조/포로)",
}

# ===========================================================================
# realistic-v1 profile (OPT-IN via --profile realistic_v1). When None the
# generator behaves EXACTLY as the frozen baseline (all branches below are
# guarded by `if profile:` and consume no rng otherwise). This single constant
# is the authoritative record of the injected uncertainties (§ report table).
# ===========================================================================
REALISTIC_V1 = {
    "name": "realistic_v1",
    # ① per-SOURCE systematic clock offset (s); folded into the visible obs time.
    "clock_offset": {"RADAR": 0.0, "VISUAL_OBS": 5.0, "ARTILLERY_OBS": 3.0,
                     "AERIAL": 10.0, "ACOUSTIC": -20.0, "SIGINT": 40.0, "HUMINT": -60.0},
    # ① report/record delay added to the visible time (observation vs record time
    # split). Spec: (kind, *params). exp_clip=(mean,lo,hi); lognorm_clip=(median,sigma,lo,hi).
    "report_delay": {
        "RADAR": ("exp_clip", 5.0, 0.0, 10.0), "VISUAL_OBS": ("exp_clip", 5.0, 0.0, 10.0),
        "ARTILLERY_OBS": ("exp_clip", 5.0, 0.0, 10.0), "AERIAL": ("exp_clip", 5.0, 0.0, 10.0),
        "SIGINT": ("lognorm_clip", 120.0, 0.6, 60.0, 300.0),
        "ACOUSTIC": ("lognorm_clip", 120.0, 0.6, 60.0, 300.0),
        "HUMINT": ("lognorm_clip", 150.0, 0.6, 60.0, 300.0)},
    # ④ per-source position-error multiplier (scales injected error AND cep_m).
    "pos_mult": {"ACOUSTIC": 5.0, "RADAR": 0.5, "VISUAL_OBS": 1.0, "SIGINT": 3.0,
                 "HUMINT": 3.0, "AERIAL": 1.5, "ARTILLERY_OBS": 1.5},
    # ④ per-source UNKNOWN (classifier abstains) and plausible mis-ID rates,
    # decided PER OBSERVATION by that observation's source.
    "unknown_rate": {"RADAR": 0.80, "ACOUSTIC": 0.50, "SIGINT": 0.40, "HUMINT": 0.40,
                     "AERIAL": 0.25, "ARTILLERY_OBS": 0.20, "VISUAL_OBS": 0.10},
    "misid_rate": {"VISUAL_OBS": 0.15, "_default": 0.05},
    # ② fraction of (non-wreck) objects given a >=2-transition (fire-move-fire)
    # schedule; the rest keep >=1 transition but get per-object time jitter
    # (diversifies nearby same-type objects that previously shared one schedule).
    "multi_transition_frac": 0.30,
    # ③ target KG-A/B relation observation overlap (Jaccard); each relation is
    # seen by both KGs w.p. overlap, else by exactly one.
    "kg_relation_overlap": 0.60,
    # ③ relation enrichment: 1..3 extra unique predicates/object -> unique combos.
    "relation_enrich_max": 3,
}
_LM_PREDS = ["near", "screens", "movesToward", "occupies", "emplacedAt"]
_UNIT_PREDS = ["supports", "reinforces"]


def _draw_delay(spec, rng) -> float:
    """Report delay (s, >=0) for a source. None -> 0 (legacy)."""
    if spec is None:
        return 0.0
    kind = spec[0]
    if kind == "exp_clip":
        _, mean, lo, hi = spec
        return float(min(hi, max(lo, rng.exponential(mean))))
    if kind == "lognorm_clip":
        _, median, sigma, lo, hi = spec
        return float(min(hi, max(lo, median * math.exp(rng.normal(0.0, sigma)))))
    return 0.0


# ---- object-type taxonomy (faction-functional; faction baked into type so the
#      model never has to merge ROK with CCF). category drives by-category
#      blocking; same category across factions => hard negatives. ----------------
#   type: (category, mobility, base_speed_mps, default_state)
TYPES = {
    "UNKNOWN":       ("unknown",   "unknown", 25.0, "Unknown"),
    "Unit":          ("formation", "na",       0.0, "Unknown"),
    "ROK_Infantry":  ("infantry",  "foot",     2.0, "Holding"),
    "CCF_Infantry":  ("infantry",  "foot",     2.0, "Approaching"),
    "US_Tank":       ("tank",      "tracked", 12.0, "Moving"),
    "ROK_Tank":      ("tank",      "tracked", 11.0, "Moving"),
    "ROK_Artillery": ("artillery", "towed",    0.0, "Emplaced"),
    "US_Artillery":  ("artillery", "towed",    0.0, "Emplaced"),
    "CCF_Artillery": ("artillery", "towed",    0.0, "Emplaced"),
    "Mortar":        ("mortar",    "towed",    0.0, "Emplaced"),
    "CCF_AA":        ("aa",        "towed",    0.0, "Emplaced"),
    "QuadFifty":     ("aa",        "wheeled",  6.0, "Halted"),
    "Engineer":      ("engineer",  "foot",     2.0, "Moving"),
    "VehicleColumn": ("vehicle",   "wheeled", 14.0, "Moving"),
    "Searchlight":   ("support",   "towed",    0.0, "Emplaced"),
}

STATES = ["Moving", "Halted", "Approaching", "Engaging", "Occupying", "Holding",
          "Emplaced", "Firing", "Withdrawing", "Unknown", "Destroyed"]
# states at which a unit is essentially stationary (dug-in / firing / halted)
_STATIC_STATES = {"Occupying", "Holding", "Emplaced", "Firing", "Halted"}
# states at which a mobile unit is relocating at (near) full speed
_FAST_STATES = {"Moving", "Approaching", "Withdrawing", "Unknown", "Destroyed"}

# ---- paraphrase / mis-id pools (Korean, grounded in the entity dictionary) -----
LABELS = {
    "ROK_Infantry": ["국군 보병 대대", "국군 보병 중대", "국군 보병 소대", "국군 전초분대",
                     "국군 방어중대", "아군 보병", "국군 소총소대"],
    "CCF_Infantry": ["중공군 보병 대대", "중공군 돌격조", "중공군 침투조", "중공군 증원대",
                     "중공군 재공격대", "적 보병", "중공군 잔존대"],
    "US_Tank":       ["미군 M46 전차", "73d Tank Bn 전차", "미군 전차", "M46 tank"],
    "ROK_Tank":      ["국군 53전차중대 전차", "국군 전차", "ROK tank"],
    "ROK_Artillery": ["국군 포병 대대", "국군 105mm 곡사포", "아군 포병"],
    "US_Artillery":  ["미군 155mm 곡사포", "미군 8인치 곡사포", "미군 포병대대"],
    "CCF_Artillery": ["중공군 포병대", "중공군 122mm 곡사포", "적 포병"],
    "Mortar":        ["화학박격포 중대", "국군 박격포반", "중공군 박격포대", "120mm 박격포"],
    "CCF_AA":        ["중공군 대공화기대", "ZPU 견인대공포", "적 대공포"],
    "QuadFifty":     ["quad-50 대공포", "4연장 50구경 기관총", "quad-.50"],
    "Engineer":      ["국군 공병반", "적 공병조", "공병조"],
    "VehicleColumn": ["중공군 차량대", "UN 차량대", "보급 차량대", "적 보급차량"],
    "Searchlight":   ["조명탄 분대", "탐조등반", "조명진지"],
}
UNKNOWN_LABELS = ["미상 표적", "미상 보병", "정체불명 부대", "분류불가 기동표적", "미상 차량"]
# plausible same-category mis-identifications (lowers type_confidence)
MISID = {"US_Tank": "ROK_Tank", "ROK_Tank": "US_Tank",
         "ROK_Artillery": "US_Artillery", "US_Artillery": "ROK_Artillery",
         "CCF_AA": "QuadFifty", "QuadFifty": "CCF_AA"}


# ===========================================================================
# Roster data structures (identical contract to the large suite)
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
    unit: Optional[str] = None
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
    # OPT-IN uncertainty profile (e.g. REALISTIC_V1). None => frozen-identical.
    profile: Optional[dict] = None


# ===========================================================================
# Sector roster construction (Hill 395 order-of-battle + terrain + phase)
# ===========================================================================
def _pick(rng, pool):
    return pool[int(rng.integers(0, len(pool)))]


def _apply_state_dynamics(objs, rng, profile) -> None:
    """② profile-only. Give `multi_transition_frac` of the (non-wreck) objects a
    >=2-transition schedule (fire-move-fire / emplace-fire repeat); jitter the
    rest so nearby same-type objects no longer share one identical schedule.
    Mobility-aware so towed weapons never get a 'Moving' state."""
    frac = profile.get("multi_transition_frac", 0.3)
    for o in objs:
        if not o.states or o.states[0][1] == "Destroyed":
            continue                                   # wrecks stay destroyed
        mob = TYPES.get(o.true_type, (None, "unknown", 0.0, "Unknown"))[1]
        if float(rng.random()) < frac:
            seq = (["Moving", "Firing", "Moving", "Firing"] if mob in ("foot", "tracked", "wheeled")
                   else ["Emplaced", "Firing", "Emplaced", "Firing"])
            ts = sorted(float(x) for x in rng.uniform(40.0, 560.0, size=3))
            o.states = [(0.0, seq[0]), (ts[0], seq[1]), (ts[1], seq[2]), (ts[2], seq[3])]
        elif len(o.states) >= 2:
            new = [o.states[0]]
            for tf, s in o.states[1:]:
                new.append((max(1.0, min(599.0, tf + float(rng.uniform(-60.0, 60.0)))), s))
            o.states = sorted(new, key=lambda x: x[0])


def _enrich_relations(objs, rng, profile, units, landmarks) -> None:
    """③ profile-only. Append 1..N extra predicates (to landmarks/units already in
    this sector) per object so per-entity relation combos become unique and denser.
    Predicates are drawn from the existing ontology vocabulary only."""
    n_max = profile.get("relation_enrich_max", 3)
    lm_ids = list(landmarks)
    unit_ids = list(units)
    for o in objs:
        k = int(rng.integers(1, n_max + 1))
        for _ in range(k):
            if unit_ids and float(rng.random()) < 0.35:
                o.relations.append((_pick(rng, _UNIT_PREDS), ("unit", _pick(rng, unit_ids))))
            elif lm_ids:
                o.relations.append((_pick(rng, _LM_PREDS), ("landmark", _pick(rng, lm_ids))))


def build_sector(rng: np.random.Generator, knobs: Knobs, sx: float, sy: float
                 ) -> tuple[list[Obj], dict[str, dict], dict[str, tuple], dict[str, str]]:
    """One Hill 395 sub-engagement: ROK defenders holding the ridge while CCF
    sub-units assault it, with supporting armour / artillery / mortars / AA /
    engineers. sx,sy = tile origin offset (m) from BASE. Returns
    (objects, units, landmarks, events)."""
    ox, oy = BASE_E + sx, BASE_N + sy
    n = knobs.identities_per_sector
    # recipe scaled to identities_per_sector (sums to ~n)
    n_ccf_inf = max(6, round(n * 0.30))     # attacking sub-units (mass hard neg)
    n_rok_inf = max(6, round(n * 0.28))     # defending sub-units (mass hard neg)
    n_tank = max(2, round(n * 0.10))
    n_arty = max(1, round(n * 0.08))
    n_mortar = max(1, round(n * 0.05))
    n_aa = max(1, round(n * 0.05))
    n_engr = max(1, round(n * 0.05))
    n_veh = max(1, round(n * 0.04))
    n_wreck = max(1, round(n * 0.05))
    robots = ROBOT_IDS[: max(2, knobs.n_robots)]

    units: dict[str, dict] = {}
    landmarks = {
        f"lm_crest_{int(sx)}_{int(sy)}": (ox + 1500, oy + 600),
        f"lm_northtip_{int(sx)}_{int(sy)}": (ox + 1500, oy + 1700),
        f"lm_northslope_{int(sx)}_{int(sy)}": (ox + 1500, oy + 1100),
        f"lm_mlr_{int(sx)}_{int(sy)}": (ox + 1500, oy - 400),
        f"lm_objA_{int(sx)}_{int(sy)}": (ox + 1500, oy + 1000),
        f"lm_objB_{int(sx)}_{int(sy)}": (ox + 1500, oy + 1400),
        f"lm_valley_{int(sx)}_{int(sy)}": (ox + 200, oy + 200),
        f"lm_yokkok_{int(sx)}_{int(sy)}": (ox - 600, oy + 300),
    }
    lm_crest, lm_northtip, lm_northslope, lm_mlr, lm_objA, lm_objB, lm_valley, lm_yokkok = list(landmarks)

    events = {
        f"evt_attack_{int(sx)}_{int(sy)}": "Attack",
        f"evt_counter_{int(sx)}_{int(sy)}": "Counterattack",
        f"evt_fire_{int(sx)}_{int(sy)}": "FireSupport",
        f"evt_occupy_{int(sx)}_{int(sy)}": "Occupy",
        f"evt_illum_{int(sx)}_{int(sy)}": "Illuminate",
    }
    ev_attack, ev_counter, ev_fire, ev_occupy, ev_illum = list(events)

    # phase shifts dominant CCF assault states by sector (Oct 6..15 cycle)
    phase = int((sx / SECTOR_PITCH) + (sy / SECTOR_PITCH)) % 3

    # formations
    rok_rgt = f"unit_rok_rgt_{int(sx)}_{int(sy)}"
    ccf_rgt = f"unit_ccf_rgt_{int(sx)}_{int(sy)}"
    units[rok_rgt] = {"type": "Unit", "label": "ROK Regiment (defense)", "partOf": None}
    units[ccf_rgt] = {"type": "Unit", "label": "CCF Regiment (assault)", "partOf": None}
    rok_bns, ccf_bns = [], []
    for i in range(max(1, n_rok_inf // 9)):
        bid = f"unit_rok_bn{i}_{int(sx)}_{int(sy)}"
        units[bid] = {"type": "Unit", "label": f"ROK Infantry Battalion {i}", "partOf": rok_rgt}
        rok_bns.append(bid)
    for i in range(max(1, n_ccf_inf // 9)):
        bid = f"unit_ccf_bn{i}_{int(sx)}_{int(sy)}"
        units[bid] = {"type": "Unit", "label": f"CCF Infantry Battalion {i}", "partOf": ccf_rgt}
        ccf_bns.append(bid)

    objs: list[Obj] = []
    gid = [0]

    def new_gid():
        gid[0] += 1
        return f"id_{int(sx)}_{int(sy)}_{gid[0]:04d}"

    obs_n = knobs.obs_per_track
    T0, T1 = 0.0, 600.0          # 10-minute sub-engagement window -> long tracks

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
            t_obs, lab, conf = obj_type, _pick(rng, LABELS.get(true_type, [true_type])), 1.0
            u = float(rng.random())
            if u < 0.12:                      # UNKNOWN (classifier abstains)
                t_obs, lab, conf = "UNKNOWN", _pick(rng, UNKNOWN_LABELS), 0.6
            elif u < 0.22 and true_type in MISID:   # plausible mis-ID
                t_obs = MISID[true_type]
                lab = _pick(rng, LABELS[t_obs]); conf = 0.7
            tr.append(Track(r, times(rng, obs_n), srcs, t_obs, lab, conf))
        return tr

    # --- CCF infantry assault: many same-type sub-units (선두/정찰/증원/침투/잔존),
    #     column with follows-chain, crossing pairs on the slope (lane swap) so
    #     instantaneous position cannot separate them. movesToward the crest. ---
    ccf_states = {
        0: [(0.0, "Approaching"), (250.0, "Engaging")],                 # initial assault
        1: [(0.0, "Approaching"), (200.0, "Engaging"), (420.0, "Infiltrating_stub")],
        2: [(0.0, "Engaging"), (300.0, "Withdrawing")],                 # repulsed
    }[phase]
    # 'Infiltrating' isn't a kinematic state in the vocab; map to Occupying
    ccf_states = [(t, "Occupying" if s == "Infiltrating_stub" else s) for t, s in ccf_states]
    prev = None
    lane_y = np.linspace(1700, 600, n_ccf_inf)      # north tip -> crest
    for i in range(n_ccf_inf):
        e0 = ox + 1500 + rng.uniform(-250, 250)
        y = oy + float(lane_y[i])
        adv = rng.uniform(-350, -150)                # push south toward crest
        y_end = y + (200 if i % 2 == 0 else -200)    # adjacent swap -> crossing
        wp = [(T0, e0, y), (300.0, e0 + adv / 2, (y + y_end) / 2), (T1, e0 + adv, y_end)]
        robots_seen, dang = assign_robots(rng)
        bn = _pick(rng, ccf_bns)
        rels = [("partOf", ("unit", bn)), ("movesToward", ("landmark", lm_crest))]
        if prev is not None:
            rels.append(("follows", ("obj", prev)))
        g = new_gid()
        objs.append(Obj(gold_id=g, true_type="CCF_Infantry", waypoints=wp, states=ccf_states,
                        unit=bn, relations=rels, events=[(ev_attack, "assault")],
                        dangling=dang, tracks=tracks_for(rng, "CCF_Infantry", robots_seen, "CCF_Infantry")))
        prev = g

    # --- ROK infantry defense: many Soldiers/squads holding the line, near each
    #     other, partOf battalion; occasional counterattack toward the crest. ---
    for s_i in range(n_rok_inf):
        bn = _pick(rng, rok_bns)
        base_e = ox + 1500 + rng.uniform(-400, 400)
        base_n = oy + rng.uniform(-500, 300)             # MLR / crest band (south)
        e0 = base_e + rng.uniform(-60, 60); n0 = base_n + rng.uniform(-60, 60)
        if float(rng.random()) < 0.25:                   # counterattacking element
            wp = [(T0, e0, n0), (T1, e0 + rng.uniform(-80, 80), n0 + rng.uniform(120, 320))]
            st = [(0.0, "Holding"), (260.0, "Occupying")]
            rels = [("partOf", ("unit", bn)), ("occupies", ("landmark", lm_crest))]
            ev = (ev_counter, "counterattack")
        else:                                            # holding the line
            wp = [(T0, e0, n0), (T1, e0 + rng.uniform(-40, 40), n0 + rng.uniform(-40, 40))]
            st = [(0.0, "Holding"), (300.0, "Engaging")]
            rels = [("partOf", ("unit", bn)), ("emplacedAt", ("landmark", lm_mlr))]
            ev = (ev_occupy, "defense")
        robots_seen, dang = assign_robots(rng)
        objs.append(Obj(gold_id=new_gid(), true_type="ROK_Infantry", waypoints=wp, states=st,
                        unit=bn, relations=rels, events=[ev], dangling=dang,
                        tracks=tracks_for(rng, "ROK_Infantry", robots_seen, "ROK_Infantry")))

    # --- supporting armour (M46 / ROK tanks): screen the flanks, fire the valley ---
    for i in range(n_tank):
        t = _pick(rng, ["US_Tank", "ROK_Tank"])
        e0 = ox + rng.uniform(100, 500); n0 = oy + rng.uniform(-600, 200)
        wp = [(T0, e0, n0), (200.0, e0 + 300, n0 + 60), (T1, e0 + 300, n0 + 60)]
        robots_seen, dang = assign_robots(rng)
        objs.append(Obj(gold_id=new_gid(), true_type=t,
                        waypoints=wp, states=[(0.0, "Moving"), (200.0, "Halted"), (260.0, "Firing")],
                        relations=[("supports", ("unit", rok_rgt)), ("firesAt", ("landmark", lm_valley)),
                                   ("screens", ("landmark", lm_mlr))],
                        events=[(ev_fire, "shooter")], dangling=dang,
                        tracks=tracks_for(rng, t, robots_seen, t)))

    # --- artillery (ROK/US/CCF): Emplaced -> Firing, emplacedAt a position ---
    for i in range(n_arty):
        t = _pick(rng, ["ROK_Artillery", "US_Artillery", "CCF_Artillery"])
        far = t == "CCF_Artillery"
        e0 = ox + rng.uniform(800, 2400); n0 = oy + (rng.uniform(900, 1500) if far else rng.uniform(-1400, -700))
        wp = [(T0, e0, n0), (T1, e0, n0)]
        lm = lm_northslope if far else lm_mlr
        robots_seen, dang = assign_robots(rng)
        objs.append(Obj(gold_id=new_gid(), true_type=t, waypoints=wp,
                        states=[(0.0, "Emplaced"), (240.0, "Firing")],
                        relations=[("emplacedAt", ("landmark", lm)), ("supports", ("unit", ccf_rgt if far else rok_rgt))],
                        events=[(ev_fire, "shooter")], dangling=dang,
                        tracks=tracks_for(rng, t, robots_seen, t)))

    # --- mortars (Emplaced, dangling-leaning: one side only) ---
    for i in range(n_mortar):
        e0 = ox + rng.uniform(700, 2300); n0 = oy + rng.uniform(-1200, -500)
        wp = [(T0, e0, n0), (T1, e0, n0)]
        r = _pick(rng, robots)
        objs.append(Obj(gold_id=new_gid(), true_type="Mortar", waypoints=wp,
                        states=[(0.0, "Emplaced"), (260.0, "Firing")],
                        relations=[("emplacedAt", ("landmark", lm_mlr)), ("supports", ("unit", rok_rgt))],
                        events=[(ev_fire, "shooter")], dangling=True,
                        tracks=tracks_for(rng, "Mortar", [r], "Mortar")))

    # --- AA (CCF towed AA forward of the slope / mobile quad-50) ---
    for i in range(n_aa):
        t = _pick(rng, ["CCF_AA", "QuadFifty"])
        own = t == "QuadFifty"
        e0 = ox + rng.uniform(600, 2400)
        n0 = oy + (rng.uniform(-300, 200) if own else rng.uniform(1000, 1600))
        if own:
            wp = [(T0, e0, n0), (180.0, e0 + 40, n0), (T1, e0 + 40, n0)]
            st = [(0.0, "Halted"), (200.0, "Firing")]
        else:
            wp = [(T0, e0, n0), (T1, e0, n0)]
            st = [(0.0, "Emplaced"), (240.0, "Firing")]
        robots_seen, dang = assign_robots(rng)
        objs.append(Obj(gold_id=new_gid(), true_type=t, waypoints=wp, states=st,
                        relations=[("emplacedAt", ("landmark", lm_northslope if not own else lm_mlr))],
                        events=[(ev_fire, "shooter")], dangling=dang,
                        tracks=tracks_for(rng, t, robots_seen, t)))

    # --- engineers (move to the wire / valley; dangling-leaning) ---
    for i in range(n_engr):
        e0 = ox + rng.uniform(-300, 300); n0 = oy + rng.uniform(-200, 600)
        wp = [(T0, e0, n0), (T1, e0 + rng.uniform(-150, 150), n0 + rng.uniform(-150, 150))]
        r = _pick(rng, robots)
        objs.append(Obj(gold_id=new_gid(), true_type="Engineer", waypoints=wp,
                        states=[(0.0, "Moving")],
                        relations=[("near", ("landmark", lm_yokkok)), ("supports", ("unit", rok_rgt))],
                        events=[(ev_occupy, "engineer")], dangling=True,
                        tracks=tracks_for(rng, "Engineer", [r], "Engineer")))

    # --- vehicle columns (CCF supply / UN columns on the rear road) ---
    for i in range(n_veh):
        e0 = ox + rng.uniform(-600, -200); n0 = oy + rng.uniform(200, 1400)
        wp = [(T0, e0, n0), (T1, e0 + rng.uniform(1200, 2200), n0 + rng.uniform(-200, 200))]
        robots_seen, dang = assign_robots(rng)
        objs.append(Obj(gold_id=new_gid(), true_type="VehicleColumn", waypoints=wp,
                        states=[(0.0, "Moving")],
                        relations=[("movesToward", ("landmark", lm_northtip)), ("reinforces", ("unit", ccf_rgt))],
                        events=[(ev_attack, "logistics")], dangling=dang,
                        tracks=tracks_for(rng, "VehicleColumn", robots_seen, "VehicleColumn")))

    # --- searchlight / illumination position (Emplaced, one side) ---
    e0 = ox + rng.uniform(200, 1200); n0 = oy + rng.uniform(-1400, -900)
    objs.append(Obj(gold_id=new_gid(), true_type="Searchlight", waypoints=[(T0, e0, n0), (T1, e0, n0)],
                    states=[(0.0, "Emplaced"), (200.0, "Firing")],
                    relations=[("supports", ("unit", rok_rgt))],
                    events=[(ev_illum, "illuminate")], dangling=True,
                    tracks=tracks_for(rng, "Searchlight", [_pick(rng, robots)], "Searchlight")))

    # --- Destroyed wrecks (stationary; b_state/b_motion bait), dangling ---
    for i in range(n_wreck):
        t = _pick(rng, ["US_Tank", "CCF_Infantry", "VehicleColumn"])
        e0 = ox + rng.uniform(-100, 2400); n0 = oy + rng.uniform(-500, 1200)
        wp = [(T0, e0, n0), (T1, e0, n0)]
        objs.append(Obj(gold_id=new_gid(), true_type=t, waypoints=wp,
                        states=[(0.0, "Destroyed")], relations=[("near", ("landmark", lm_crest))],
                        events=[], dangling=True,
                        tracks=tracks_for(rng, t, [_pick(rng, robots)], t)))

    # ---- profile-only post-processing (② state dynamics, ③ relation density).
    #      Guarded so the frozen baseline is byte-identical. ----
    if getattr(knobs, "profile", None):
        _apply_state_dynamics(objs, rng, knobs.profile)
        _enrich_relations(objs, rng, knobs.profile, units, landmarks)

    return objs, units, landmarks, events


# ===========================================================================
# Realisation: roster -> observations + entities + gold (reused from large suite)
# ===========================================================================
def _iso(t: float) -> str:
    return (EPOCH + timedelta(seconds=t)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _mgrs(e, n):
    return f"52SDT{int(e) % 100000:05d}{int(n) % 100000:05d}"


def realise_sector(sector_id, objs, units, rng, knobs=None):
    profile = getattr(knobs, "profile", None) if knobs is not None else None
    ent_counter: dict[str, int] = {}
    obj_local: dict[str, dict[str, str]] = {}
    for o in objs:
        obj_local[o.gold_id] = {}
        for tr in o.tracks:
            ent_counter[tr.kg_id] = ent_counter.get(tr.kg_id, 0) + 1
            obj_local[o.gold_id][tr.kg_id] = f"ent_{tr.kg_id}_{ent_counter[tr.kg_id]:04d}"

    # ③ profile-only: per-object, per-relation KG-observation assignment. Each
    # relation is seen by both KGs w.p. `kg_relation_overlap`, else by exactly
    # one -> partial cross-KG relation overlap (legacy: every KG sees all).
    relkg: dict[str, dict[int, set]] = {}
    if profile:
        ov = profile.get("kg_relation_overlap", 1.0)
        for o in objs:
            o_kgs = sorted({tr.kg_id for tr in o.tracks})
            assign: dict[int, set] = {}
            for idx in range(len(o.relations)):
                if len(o_kgs) <= 1 or float(rng.random()) < ov:
                    assign[idx] = set(o_kgs)
                else:
                    assign[idx] = {o_kgs[int(rng.integers(0, len(o_kgs)))]}
            relkg[o.gold_id] = assign

    def resolve(kg, tgt):
        kind, ref = tgt
        if kind == "landmark":
            return ("lm", ref)
        if kind == "unit":
            return ("unit", ref)
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
            assign = relkg.get(o.gold_id) if profile else None
            rels = []
            for idx, (pred, tgt) in enumerate(o.relations):
                if assign is not None and kg not in assign.get(idx, set()):
                    continue                            # ③ this KG does not observe this relation
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
                # NOISE_MULT scales the actual position error AND the reported cep_m
                # together (robustness sweep); 1.0 = frozen baseline. The profile
                # further scales it per source (④) and shifts the visible time by a
                # per-source systematic offset + report delay (①).
                if profile:
                    cep = SOURCE_CEP.get(src, 50.0) * NOISE_MULT * profile["pos_mult"].get(src, 1.0)
                    off = profile["clock_offset"].get(src, KG_CLOCK_OFFSET.get(kg, 0.0))
                    delay = _draw_delay(profile["report_delay"].get(src), rng)
                    t_obs = tt + off + delay + float(rng.normal(0, 1.0))
                else:
                    cep = SOURCE_CEP.get(src, 50.0) * NOISE_MULT
                    t_obs = tt + KG_CLOCK_OFFSET.get(kg, 0.0) + float(rng.normal(0, 1.0))
                e_t, n_t = o.pos(tt)
                sig = cep / 1.1774
                e_o = e_t + float(rng.normal(0, sig)); n_o = n_t + float(rng.normal(0, sig))
                st = o.state_at(tt)
                spd, hd = o.vel(tt)
                if st in ("Emplaced", "Halted", "Destroyed", "Holding", "Occupying", "Firing"):
                    spd = 0.0
                # ④ per-source type reporting: UNKNOWN (classifier abstains) / plausible
                # mis-ID decided per-observation by the source's rates. Legacy: the
                # track-level type/label chosen once in tracks_for.
                if profile:
                    u = float(rng.random())
                    unk = profile["unknown_rate"].get(src, 0.10)
                    mis = profile["misid_rate"].get(src, profile["misid_rate"]["_default"])
                    if u < unk:
                        o_type, o_label, tcs = "UNKNOWN", _pick(rng, UNKNOWN_LABELS), 0.6
                    elif u < unk + mis and o.true_type in MISID:
                        o_type = MISID[o.true_type]
                        o_label, tcs = _pick(rng, LABELS[o_type]), 0.7
                    else:
                        o_type = o.true_type
                        o_label, tcs = _pick(rng, LABELS.get(o.true_type, [o.true_type])), 1.0
                else:
                    o_type, o_label, tcs = tr.obj_type, tr.label_text, tr.type_conf_scale
                tconf = round(min(0.99, max(0.2, SOURCE_TYPEREL.get(src, 0.5) * tcs
                                            + 0.05 * rng.random())), 3)
                sconf = round(min(0.99, max(0.2, SOURCE_REL.get(src, 0.5) + 0.05 * rng.random())), 3)
                rec = {
                    "obs_id": oid, "kg_id": kg, "source": src, "local_entity_id": lid,
                    "label_text": o_label, "type": o_type, "type_confidence": tconf,
                    "state": st, "state_confidence": sconf, "time": _iso(t_obs),
                    "location": {"crs": CRS, "easting": round(e_o, 2), "northing": round(n_o, 2),
                                 "elevation": 50.0, "mgrs": _mgrs(e_o, n_o)},
                    "cep_m": round(cep, 1),
                    "velocity": {"speed_mps": round(spd, 2), "heading_deg": round(hd, 1)},
                    "relations": [{"predicate": r["predicate"], "target_ref": r["target_ref"],
                                   "confidence": r["confidence"]} for r in rels],
                    "events": evs,
                    "provenance": {"wasAttributedTo": src, "generatedAtTime": _iso(t_obs)},
                }
                if profile:
                    # inspection-only: TRUE observation time (pipeline ignores extra keys)
                    rec["_true_time"] = _iso(tt)
                observations.append(rec)
                entities[ekey]["obs_ids"].append(oid)

    observations.sort(key=lambda o: (o["time"], o["obs_id"]))

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
    return "<" + NS + PATH_SEG + "/" + "/".join(str(p) for p in parts) + ">"


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

    for uid, u in units.items():
        s = _iri(sector_id, "unit", uid)
        T(s, f"<{RDF}type>", P("Unit"))
        T(s, f"<{RDFS}label>", _lit(u["label"]))
        if u.get("partOf"):
            T(s, P("partOf"), _iri(sector_id, "unit", u["partOf"]))
    for lid, (e, n) in landmarks.items():
        s = _iri(sector_id, "lm", lid)
        T(s, f"<{RDF}type>", P("Landmark"))
        T(s, GP("easting"), _lit(round(e, 2), xsd + "double"))
        T(s, GP("northing"), _lit(round(n, 2), xsd + "double"))
    for evid, kind in events.items():
        s = _iri(sector_id, "evt", evid)
        T(s, f"<{RDF}type>", P("Event"))
        T(s, P("eventKind"), _lit(kind))
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

    (sector_dir / "stkg.nt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


# ===========================================================================
# Ontology emission (Hill-395-grounded; reproducible)
# ===========================================================================
def _v_max_for(speed: float, mobility: str) -> dict:
    """Per-state max speed. Towed/structure self-propel at 0; mobile types move at
    `speed` in fast states, crawl (~0.5) when dug-in/firing, 2.0 while Engaging."""
    if mobility in ("towed", "na"):
        return {s: 0.0 for s in STATES}
    out = {}
    for s in STATES:
        if s in _FAST_STATES:
            out[s] = speed
        elif s == "Engaging":
            out[s] = min(speed, 2.0)
        else:                       # static / dug-in
            out[s] = 0.5
    return out


def _compat_matrix() -> dict:
    """Asymmetric same-object state-transition plausibility C[i][j] in (0,1].
    Rule-based so the 10x10 matrix stays consistent and editable."""
    low_after = {  # (earlier -> later): implausible for the SAME real object
        ("Withdrawing", "Occupying"): 0.2, ("Withdrawing", "Holding"): 0.25,
        ("Withdrawing", "Approaching"): 0.3,
        ("Emplaced", "Moving"): 0.3, ("Emplaced", "Approaching"): 0.3,
        ("Emplaced", "Withdrawing"): 0.4,
        ("Firing", "Moving"): 0.4, ("Firing", "Approaching"): 0.4,
        ("Occupying", "Approaching"): 0.4,
    }
    def cell(a: str, b: str) -> float:
        if a == b:
            return 1.0
        if a == "Destroyed":
            return 0.3 if b == "Unknown" else 0.01
        if b == "Destroyed":
            return 0.6                       # any active object can later be destroyed
        if a == "Unknown" or b == "Unknown":
            return 0.8
        return low_after.get((a, b), 0.85)

    return {a: {b: cell(a, b) for b in STATES} for a in STATES}


def write_ontology(ont_dir: Path, profile: Optional[dict] = None) -> None:
    ont_dir.mkdir(parents=True, exist_ok=True)

    def dump_yaml(path: Path, text: str):
        path.write_text(text, encoding="utf-8")

    # classes.yaml
    cl = ["# Hill 395 object-type taxonomy (faction-functional). Faction is baked",
          "# into the type so cross-faction merges are structurally avoided; same",
          "# category across factions yields hard negatives. v_max is state-",
          "# conditioned; towed weapons / org nodes self-propel at 0.\n"]
    for t, (cat, mob, spd, dstate) in TYPES.items():
        vm = _v_max_for(spd, mob)
        cl.append(f"{t}:")
        cl.append(f"  category: {cat}")
        cl.append(f"  mobility: {mob}")
        cl.append(f"  default_state: {dstate}")
        cl.append(f"  default_v_max_mps: {max(spd, 0.5) if mob not in ('towed','na') else 0.0}")
        inner = ", ".join(f"{s}: {vm[s]}" for s in STATES)
        cl.append(f"  v_max_mps: {{{inner}}}")
        cl.append("")
    dump_yaml(ont_dir / "classes.yaml", "\n".join(cl))

    # states.yaml
    C = _compat_matrix()
    st = ["# Hill 395 observation states + pairwise co-reference compatibility C",
          "# (b_state = gamma * log(C[s_i, s_j] + eps)). Asymmetric: e.g.",
          "# Withdrawing->Occupying is implausible for the same object; Destroyed->",
          "# {active} ~ 0. Per-type overrides push towed Emplaced->Moving very low.\n",
          "states:"]
    for s in STATES:
        st.append(f"  - {s}")
    st.append("\ncompatibility:")
    for a in STATES:
        inner = ", ".join(f"{b}: {C[a][b]}" for b in STATES)
        st.append(f"  {a}: {{{inner}}}")
    st.append("\noverrides:")
    for t, (cat, mob, spd, ds) in TYPES.items():
        if mob == "towed":
            st.append(f"  {t}:")
            st.append("    Emplaced: {Moving: 0.02, Approaching: 0.02}")
            st.append("    Firing: {Moving: 0.02, Approaching: 0.02}")
    st.append("\neps: 0.01")
    dump_yaml(ont_dir / "states.yaml", "\n".join(st))

    # sources.yaml
    sr = ["# Hill 395 intelligence sources with reliability priors. KG A uses own",
          "# direct sensors (VISUAL_OBS/RADAR/ARTILLERY_OBS); KG B uses standoff /",
          "# indirect sources (AERIAL/SIGINT/ACOUSTIC/HUMINT).\n"]
    for s in SOURCE_CEP:
        # profile inflates cep by the per-source position multiplier so the blocking
        # max_cep guard (max over ontology source_cep) still covers the injected error.
        cep_eff = SOURCE_CEP[s] * (profile["pos_mult"].get(s, 1.0) if profile else 1.0)
        sr.append(f"{s}:")
        sr.append(f"  description: {SOURCE_DESC[s]}")
        sr.append(f"  reliability: {SOURCE_REL[s]}")
        sr.append(f"  type_reliability: {SOURCE_TYPEREL[s]}")
        sr.append(f"  pos_reliability: {round(1.0 - min(cep_eff, 120.0) / 120.0, 2)}")
        sr.append(f"  cep_m: {round(cep_eff, 1)}")
        sr.append("")
    dump_yaml(ont_dir / "sources.yaml", "\n".join(sr))

    # relations.yaml (base 9 + participatesIn + 3 battle predicates)
    rels = {
        "partOf": (False, False, "Entity is a sub-unit / component of another entity"),
        "follows": (False, False, "Entity trails another entity (column order)"),
        "near": (True, False, "Entities are spatially proximate"),
        "firesAt": (False, False, "Entity engages another entity / location by fire"),
        "engagedWith": (True, False, "Two entities are in mutual engagement"),
        "emplacedAt": (False, False, "Entity is dug-in / positioned at a landmark"),
        "movesToward": (False, False, "Entity moves toward another entity / landmark"),
        "supports": (False, False, "Entity provides fire / logistic support to another"),
        "screens": (False, False, "Entity screens / covers a flank or advance"),
        "occupies": (False, False, "Entity takes / holds control of a position (점령)"),
        "withdrawsFrom": (False, False, "Entity pulls back from a position (철수)"),
        "reinforces": (False, False, "Entity reinforces / augments another unit (증원)"),
        "participatesIn": (False, True, "Entity takes part in an event (with a role)"),
    }
    rl = ["# Relation predicates for the Hill 395 STKG: the 9 base CITA-Net edges +",
          "# 3 battle-specific predicates (occupies/withdrawsFrom/reinforces) that the",
          "# 78 source Korean verbs map onto. `event` marks participation links.\n"]
    for p, (sym, ev, desc) in rels.items():
        rl.append(f"{p}:")
        rl.append(f"  symmetric: {'true' if sym else 'false'}")
        if ev:
            rl.append("  event: true")
        rl.append(f"  description: {desc}")
        rl.append("")
    dump_yaml(ont_dir / "relations.yaml", "\n".join(rl))

    # context.jsonld
    rel_ctx = "".join(
        f'    "{p}": {{"@id": "stkg:{p}", "@type": "@id"}},\n' for p in rels if p != "participatesIn")
    ctx = (
        '{\n  "@context": {\n'
        '    "stkg": "https://example.org/stkg/",\n'
        '    "geo": "https://example.org/geo/",\n'
        '    "prov": "http://www.w3.org/ns/prov#",\n'
        '    "xsd": "http://www.w3.org/2001/XMLSchema#",\n'
        '    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",\n'
        '    "Entity": "stkg:Entity",\n'
        '    "Observation": "stkg:Observation",\n'
        '    "Event": "stkg:Event",\n'
        '    "Landmark": "stkg:Landmark",\n'
        '    "time": {"@id": "stkg:time", "@type": "xsd:dateTime"},\n'
        '    "objectType": "stkg:objectType",\n'
        '    "typeConfidence": {"@id": "stkg:typeConfidence", "@type": "xsd:double"},\n'
        '    "state": "stkg:state",\n'
        '    "stateConfidence": {"@id": "stkg:stateConfidence", "@type": "xsd:double"},\n'
        '    "labelText": "stkg:labelText",\n'
        '    "source": "stkg:source",\n'
        '    "cepM": {"@id": "stkg:cepM", "@type": "xsd:double"},\n'
        '    "speedMps": {"@id": "stkg:speedMps", "@type": "xsd:double"},\n'
        '    "headingDeg": {"@id": "stkg:headingDeg", "@type": "xsd:double"},\n'
        '    "observationOf": {"@id": "stkg:observationOf", "@type": "@id"},\n'
        '    "participatesIn": {"@id": "stkg:participatesIn", "@type": "@id"},\n'
        '    "role": "stkg:role",\n'
        '    "easting": {"@id": "geo:easting", "@type": "xsd:double"},\n'
        '    "northing": {"@id": "geo:northing", "@type": "xsd:double"},\n'
        '    "elevation": {"@id": "geo:elevation", "@type": "xsd:double"},\n'
        '    "crs": "geo:crs",\n'
        '    "mgrs": "geo:mgrs",\n'
        f'{rel_ctx}'
        '    "wasAttributedTo": {"@id": "prov:wasAttributedTo", "@type": "@id"},\n'
        '    "generatedAtTime": {"@id": "prov:generatedAtTime", "@type": "xsd:dateTime"}\n'
        '  }\n}\n'
    )
    (ont_dir / "context.jsonld").write_text(ctx, encoding="utf-8")


# ===========================================================================
# Per-sector driver + suite builder (reused from large suite)
# ===========================================================================
def write_sector(split: str, sector_id: str, seed: int, idx: int, knobs: Knobs) -> dict:
    rng = np.random.default_rng(seed)
    sx = (idx % 8) * SECTOR_PITCH
    sy = (idx // 8) * SECTOR_PITCH
    objs, units, landmarks, events = build_sector(rng, knobs, sx, sy)
    observations, entities, gold, obj_local = realise_sector(sector_id, objs, units, rng, knobs)

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
    write_ontology(ONTOLOGY_DIR, knobs.profile)
    target = target_triples or 1_000_000
    splits, totals = {}, {}
    dev_ids, dev_t = build_split("dev", knobs, n_sectors=12, target_triples=None)
    test_ids, test_t = build_split("test", knobs, n_sectors=13, target_triples=None)
    splits["dev"], splits["test"] = dev_ids, test_ids
    totals["dev"], totals["test"] = dev_t, test_t

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
        "domain": "hill395_white_horse_1952",
        "total_triples": grand,
        "target_triples": target,
        "sectors": {s: len(ids) for s, ids in splits.items()},
        "triples_per_split": totals,
        "knobs": vars(knobs),
        "noise_mult": NOISE_MULT,
        "seed_bases": SEED_BASE,
    }
    (DATA_ROOT / "manifest_global.json").write_text(
        json.dumps(global_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nTOTAL TRIPLES = {grand:,}  (target {target:,})  "
          f"sectors train/dev/test = {len(splits['train'])}/{len(splits['dev'])}/{len(splits['test'])}")
    print(f"global manifest -> {DATA_ROOT / 'manifest_global.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Hill 395 (White Horse) CITA-Net STKG generator")
    ap.add_argument("--build-suite", action="store_true")
    ap.add_argument("--emit-ontology", action="store_true", help="(re)write ontology files only")
    ap.add_argument("--split", default="train", choices=["train", "dev", "test"])
    ap.add_argument("--n-sectors", type=int, default=None)
    ap.add_argument("--identities-per-sector", type=int, default=40)
    ap.add_argument("--obs-per-track", type=int, default=10)
    ap.add_argument("--dangling-ratio", type=float, default=0.2)
    ap.add_argument("--n-robots", type=int, default=2)
    ap.add_argument("--target-triples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--noise-mult", type=float, default=1.0,
                    help="scale injected position error AND reported cep_m (1.0=baseline)")
    ap.add_argument("--profile", default=None, choices=["realistic_v1"],
                    help="OPT-IN uncertainty profile. Omit => frozen-identical behaviour.")
    ap.add_argument("--out-root", default=None,
                    help="output data root (default data/battlefield_hill395_large). "
                         "REQUIRED to build a non-frozen suite elsewhere.")
    ap.add_argument("--force", action="store_true",
                    help="allow --build-suite to write into an existing non-empty dir")
    args = ap.parse_args()

    global DATA_ROOT, ONTOLOGY_DIR, NOISE_MULT
    NOISE_MULT = args.noise_mult
    if args.out_root is not None:
        # resolve to absolute: build_suite does DATA_ROOT.relative_to(REPO_ROOT)
        DATA_ROOT = Path(args.out_root).resolve()
        ONTOLOGY_DIR = DATA_ROOT / "ontology"

    # Safety guard: never silently overwrite an existing (e.g. the frozen) suite.
    if args.build_suite and DATA_ROOT.exists() and any(DATA_ROOT.iterdir()) and not args.force:
        raise SystemExit(
            f"refusing to --build-suite into existing non-empty dir {DATA_ROOT} "
            f"without --force (protects the frozen suite; pass --out-root for a new dir)")

    profile = REALISTIC_V1 if args.profile == "realistic_v1" else None
    knobs = Knobs(identities_per_sector=args.identities_per_sector,
                  obs_per_track=args.obs_per_track,
                  dangling_ratio=args.dangling_ratio, n_robots=args.n_robots,
                  profile=profile)

    if args.emit_ontology:
        write_ontology(ONTOLOGY_DIR, profile)
        print(f"ontology -> {ONTOLOGY_DIR}")
        return
    if args.build_suite:
        build_suite(args.target_triples, knobs)
        return
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    if not (ONTOLOGY_DIR / "classes.yaml").exists():
        write_ontology(ONTOLOGY_DIR, profile)
    if args.seed is not None:
        SEED_BASE[args.split] = args.seed
    ids, total = build_split(args.split, knobs, args.n_sectors, args.target_triples)
    print(f"[{args.split}] sectors={len(ids)} triples={total:,}")


if __name__ == "__main__":
    main()
