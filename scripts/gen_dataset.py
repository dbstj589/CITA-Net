#!/usr/bin/env python
"""Deterministic battlefield-STKG dataset generator for CITA-Net.

Produces, per scenario, the canonical STKG triple (``kg_A.jsonld`` +
``kg_B.jsonld`` + ``commons.jsonld``), a flattened ``observations.jsonl``
cache, a ``manifest.json``, and the ``labels/`` gold files described in the
dataset spec (§3). Everything is a deterministic function of ``--seed`` and
``--scenario-id`` so runs are reproducible and offline.

Design (§5): a scenario is a roster of *true objects*. Each true object has a
type, a ground-truth path (waypoints), a state schedule, relation/event
context, and a set of KGs that observe it. Matched objects are observed in
both KG_A and KG_B; *dangling* objects are observed in exactly one KG and have
no cross-KG counterpart -- CITA-Net must abstain on them.

``scn_0001`` is a fixed dev probe with a hand-authored roster yielding exactly:
29 observations, 6 entities in A / 6 in B, 5 matched identities (incl. two
T-72s id_0001 & id_0003, the latter a hard positive whose KG_B side is typed
UNKNOWN), and 2 dangling entities (ZPU-4 in A, MO-120-RT in B).

Usage:
    python scripts/gen_dataset.py --scenario-id scn_0001
    python scripts/gen_dataset.py --scenario-id scn_t001 --seed 101 --n-objects 8 --split train
    python scripts/gen_dataset.py --build-suite     # regenerate the whole default suite
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
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "battlefield_stkg_dataset"
ONTOLOGY_DIR = DATA_ROOT / "ontology"

EPOCH = datetime(2025, 6, 6, 0, 0, 0, tzinfo=timezone.utc)
CRS = "UTM52N"
BASE_E, BASE_N = 322000.0, 4158000.0   # UTM52N anchor (Korea zone)

# Per-KG sensor pools (robot A vs robot B carry different suites, §4.2).
KG_SOURCES = {
    "A": ["EO_IR", "RADAR"],
    "B": ["RADAR", "SIGINT", "ACOUSTIC", "HUMINT"],
}
KG_CLOCK_OFFSET = {"A": 0.0, "B": 2.0}   # B clock skew, within time_eps (5s)


# ---------------------------------------------------------------------------
# Roster data structures
# ---------------------------------------------------------------------------
@dataclass
class Track:
    """One KG's observation plan for a true object."""

    kg_id: str
    sample_times: list[float]
    sources: list[str]                  # one per sample_time (cycled if short)
    obj_type: str                       # may differ from truth (e.g. UNKNOWN)
    label_text: str
    type_conf_scale: float = 1.0


@dataclass
class TrueObject:
    gold_id: str
    true_type: str
    label_base: str
    waypoints: list[tuple[float, float, float]]   # (t, easting, northing)
    states: list[tuple[float, str]]               # (t_from, state) step fn
    relations: list[tuple[str, tuple]] = field(default_factory=list)
    # relations: (predicate, ("obj", gold_id) | ("landmark", lm_id))
    events: list[tuple[str, str]] = field(default_factory=list)   # (event_id, role)
    tracks: list[Track] = field(default_factory=list)
    dangling: bool = False

    def position_at(self, t: float) -> tuple[float, float]:
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

    def velocity_at(self, t: float) -> tuple[float, float]:
        """Return (speed_mps, heading_deg) from path derivative."""
        dt = 1.0
        e0, n0 = self.position_at(max(self.waypoints[0][0], t - dt))
        e1, n1 = self.position_at(min(self.waypoints[-1][0], t + dt))
        span = max(1e-6, (min(self.waypoints[-1][0], t + dt) - max(self.waypoints[0][0], t - dt)))
        de, dn = (e1 - e0) / span, (n1 - n0) / span
        speed = math.hypot(de, dn)
        heading = (math.degrees(math.atan2(de, dn))) % 360.0   # 0=N, 90=E
        return speed, heading

    def state_at(self, t: float) -> str:
        st = self.states[0][1]
        for t_from, s in self.states:
            if t >= t_from:
                st = s
        return st


@dataclass
class Landmark:
    landmark_id: str
    name: str
    easting: float
    northing: float


@dataclass
class EventDef:
    event_id: str
    name: str
    kind: str


# ---------------------------------------------------------------------------
# Fixed roster: scn_0001 (the spec-locked dev probe)
# ---------------------------------------------------------------------------
def landmarks_commons() -> dict[str, Landmark]:
    return {
        "lm_hill205": Landmark("lm_hill205", "Hill 205", BASE_E + 1200, BASE_N + 200),
        "lm_cross7": Landmark("lm_cross7", "Crossroad 7", BASE_E + 1500, BASE_N - 600),
        "lm_fordA": Landmark("lm_fordA", "Ford A", BASE_E + 700, BASE_N + 450),
    }


def events_commons() -> dict[str, EventDef]:
    return {
        "evt_advance": EventDef("evt_advance", "River-crossing advance", "maneuver"),
        "evt_fire_support": EventDef("evt_fire_support", "Indirect fire support", "fires"),
        "evt_air_defense": EventDef("evt_air_defense", "Point air defense", "air_defense"),
    }


def roster_scn_0001() -> list[TrueObject]:
    """Hand-authored roster reproducing the spec numbers exactly."""

    def line(e0, n0, de, dn):
        return [(0.0, BASE_E + e0, BASE_N + n0),
                (150.0, BASE_E + e0 + de / 2, BASE_N + n0 + dn / 2),
                (300.0, BASE_E + e0 + de, BASE_N + n0 + dn)]

    objs: list[TrueObject] = []

    # obj1 / id_0001 -- lead T-72, matched (A:3, B:3)
    objs.append(TrueObject(
        gold_id="id_0001", true_type="T-72", label_base="T-72 전차",
        waypoints=line(0, 0, 600, 300),
        states=[(0.0, "Moving")],
        relations=[("near", ("obj", "id_0002"))],
        events=[("evt_advance", "lead")],
        tracks=[
            Track("A", [10, 150, 290], ["EO_IR", "EO_IR", "RADAR"], "T-72", "T-72 전차"),
            Track("B", [42, 162, 282], ["RADAR", "SIGINT", "ACOUSTIC"], "T-72", "T-72 전차"),
        ],
    ))

    # obj2 / id_0002 -- BMP-2, matched (A:2, B:2)
    objs.append(TrueObject(
        gold_id="id_0002", true_type="BMP-2", label_base="BMP-2 보병전투차",
        waypoints=line(150, 120, 600, 300),
        states=[(0.0, "Moving")],
        relations=[("near", ("obj", "id_0001"))],
        events=[("evt_advance", "participant")],
        tracks=[
            Track("A", [20, 200], ["EO_IR", "RADAR"], "BMP-2", "BMP-2 보병전투차"),
            Track("B", [60, 220], ["RADAR", "HUMINT"], "BMP-2", "BMP-2 보병전투차"),
        ],
    ))

    # obj3 / id_0003 -- follower T-72, HARD POSITIVE (KG_B types it UNKNOWN). A:3, B:2
    objs.append(TrueObject(
        gold_id="id_0003", true_type="T-72", label_base="T-72 전차",
        waypoints=line(-200, -100, 600, 300),
        states=[(0.0, "Moving")],
        relations=[("follows", ("obj", "id_0001"))],
        events=[("evt_advance", "participant")],
        tracks=[
            Track("A", [15, 155, 295], ["EO_IR", "EO_IR", "RADAR"], "T-72", "T-72 전차"),
            # KG_B: weak classifier -> UNKNOWN type, vague label; relational
            # context (follows lead T-72, same advance event) carries the match.
            Track("B", [48, 250], ["ACOUSTIC", "RADAR"], "UNKNOWN", "미상 기갑차량", 0.6),
        ],
    ))

    # obj4 / id_0004 -- 2S1 SPG, matched, halts then fires. A:2, B:2
    objs.append(TrueObject(
        gold_id="id_0004", true_type="2S1", label_base="2S1 자주포",
        waypoints=[(0.0, BASE_E + 900, BASE_N - 400),
                   (120.0, BASE_E + 950, BASE_N - 380),
                   (300.0, BASE_E + 950, BASE_N - 380)],
        states=[(0.0, "Moving"), (120.0, "Halted"), (180.0, "Firing")],
        relations=[("firesAt", ("landmark", "lm_cross7"))],
        events=[("evt_fire_support", "shooter")],
        tracks=[
            Track("A", [30, 240], ["EO_IR", "RADAR"], "2S1", "2S1 자주포"),
            Track("B", [70, 260], ["SIGINT", "ACOUSTIC"], "2S1", "2S1 자주포"),
        ],
    ))

    # obj5 / id_0005 -- BTR-80, matched, screening flank ~1 km away. A:2, B:2
    objs.append(TrueObject(
        gold_id="id_0005", true_type="BTR-80", label_base="BTR-80 장갑차",
        waypoints=line(300, 900, 500, -200),
        states=[(0.0, "Moving")],
        relations=[("screens", ("obj", "id_0001"))],
        events=[("evt_advance", "screen")],
        tracks=[
            Track("A", [25, 235], ["RADAR", "EO_IR"], "BTR-80", "BTR-80 장갑차"),
            Track("B", [65, 255], ["RADAR", "HUMINT"], "BTR-80", "BTR-80 장갑차"),
        ],
    ))

    # obj6 -- ZPU-4 towed AA, DANGLING (A only), emplaced on Hill 205. A:3
    objs.append(TrueObject(
        gold_id="dang_A_zpu4", true_type="ZPU-4", label_base="ZPU-4 대공포",
        waypoints=[(0.0, BASE_E + 1200, BASE_N + 200), (300.0, BASE_E + 1200, BASE_N + 200)],
        states=[(0.0, "Emplaced"), (160.0, "Firing")],
        relations=[("emplacedAt", ("landmark", "lm_hill205"))],
        events=[("evt_air_defense", "shooter")],
        dangling=True,
        tracks=[
            Track("A", [35, 175, 285], ["EO_IR", "RADAR", "ACOUSTIC"], "ZPU-4", "ZPU-4 대공포"),
        ],
    ))

    # obj7 -- MO-120-RT towed mortar, DANGLING (B only), emplaced at Crossroad 7. B:3
    objs.append(TrueObject(
        gold_id="dang_B_mo120", true_type="MO-120-RT", label_base="MO-120-RT 박격포",
        waypoints=[(0.0, BASE_E + 1500, BASE_N - 600), (300.0, BASE_E + 1500, BASE_N - 600)],
        states=[(0.0, "Emplaced"), (140.0, "Firing")],
        relations=[("emplacedAt", ("landmark", "lm_cross7"))],
        events=[("evt_fire_support", "shooter")],
        dangling=True,
        tracks=[
            Track("B", [50, 150, 270], ["ACOUSTIC", "SIGINT", "RADAR"], "MO-120-RT", "MO-120-RT 박격포"),
        ],
    ))

    return objs


# ---------------------------------------------------------------------------
# Random roster for additional train/dev scenarios
# ---------------------------------------------------------------------------
TYPE_POOL = [
    ("T-72", "T-72 전차", "Moving"),
    ("BMP-2", "BMP-2 보병전투차", "Moving"),
    ("BTR-80", "BTR-80 장갑차", "Moving"),
    ("2S1", "2S1 자주포", "Moving"),
    ("ZPU-4", "ZPU-4 대공포", "Emplaced"),
    ("MO-120-RT", "MO-120-RT 박격포", "Emplaced"),
]

# Surface-form variants per type: same referent, different label text (KO/EN,
# abbreviations, word order). Picking one per object per scenario gives the
# learned text encoder genuine lexical variety instead of a single fixed string.
LABEL_VARIANTS = {
    "T-72": ["T-72 전차", "T-72 주력전차", "T72 tank", "전차 T-72", "T-72형 전차"],
    "BMP-2": ["BMP-2 보병전투차", "BMP-2 IFV", "BMP2 장갑차", "보병전투차 BMP-2"],
    "BTR-80": ["BTR-80 장갑차", "BTR-80 APC", "BTR80 병력수송장갑차", "장갑차 BTR-80"],
    "2S1": ["2S1 자주포", "2S1 Gvozdika", "2S1 자주곡사포", "자주포 2S1"],
    "ZPU-4": ["ZPU-4 대공포", "ZPU-4 견인대공포", "ZPU4 AA gun", "대공포 ZPU-4"],
    "MO-120-RT": ["MO-120-RT 박격포", "MO-120 견인박격포", "120mm 박격포", "박격포 MO-120-RT"],
}
# Vague labels used by a weak classifier when it reports UNKNOWN type.
UNKNOWN_LABELS = ["미상 기갑차량", "미상 차량", "정체불명 표적", "분류불가 기동표적"]


def _pick_label(rng: np.random.Generator, obj_type: str) -> str:
    variants = LABEL_VARIANTS.get(obj_type)
    if not variants:
        return obj_type
    return variants[int(rng.integers(0, len(variants)))]


def roster_random(rng: np.random.Generator, n_objects: int) -> list[TrueObject]:
    """Sample a roster preserving the difficulty pattern of scn_0001:
    near-duplicate same-type movers (wrong-merge bait), >=1 UNKNOWN hard
    positive, and >=1 dangling object per KG."""
    objs: list[TrueObject] = []
    n_objects = max(4, n_objects)
    n_dangling = max(2, n_objects // 4)
    n_matched = n_objects - n_dangling

    # Ensure at least two same-type movers (wrong-merge bait) and one hard positive.
    forced_types = ["T-72", "T-72", "BMP-2"]
    for i in range(n_matched):
        if i < len(forced_types):
            t = forced_types[i]
            default_state = "Moving"
        else:
            t, _, default_state = TYPE_POOL[rng.integers(0, 4)]   # movers only
        # One surface form per object, shared by both KG sides so the blocking
        # text gate (cosine >= theta) still links the cross-KG match. Variety is
        # across objects/scenarios, not across one object's two sides.
        label = _pick_label(rng, t)
        e0 = rng.uniform(-200, 1400)
        n0 = rng.uniform(-700, 1100)
        de = rng.uniform(300, 800) * (1 if rng.random() > 0.5 else -1)
        dn = rng.uniform(-400, 400)
        gold = f"id_{i + 1:04d}"
        # First mover leads; others relate to it for relational context.
        rels, evs = [], [("evt_advance", "participant")]
        if i == 0:
            rels = [("near", ("obj", "id_0002"))] if n_matched > 1 else []
        elif i == 1:
            rels = [("follows", ("obj", "id_0001"))]      # hard-positive carrier
        else:
            rels = [("near", ("obj", "id_0001"))]
        # KG_B type for i==1 (the hard positive) is UNKNOWN with a vague label.
        b_type = "UNKNOWN" if i == 1 else t
        b_label = (UNKNOWN_LABELS[int(rng.integers(0, len(UNKNOWN_LABELS)))]
                   if i == 1 else label)
        b_conf = 0.6 if i == 1 else 1.0
        ta = sorted(rng.choice([10, 80, 150, 220, 290], size=rng.integers(2, 4), replace=False).tolist())
        tb = sorted((rng.choice([40, 110, 180, 250], size=rng.integers(2, 4), replace=False) + 2).tolist())
        objs.append(TrueObject(
            gold_id=gold, true_type=t, label_base=label,
            waypoints=[(0.0, BASE_E + e0, BASE_N + n0),
                       (300.0, BASE_E + e0 + de, BASE_N + n0 + dn)],
            states=[(0.0, default_state)],
            relations=rels, events=evs,
            tracks=[
                Track("A", [float(x) for x in ta],
                      [KG_SOURCES["A"][int(rng.integers(0, 2))] for _ in ta], t, label),
                Track("B", [float(x) for x in tb],
                      [KG_SOURCES["B"][int(rng.integers(0, 4))] for _ in tb], b_type, b_label, b_conf),
            ],
        ))

    # Dangling objects: per scenario the count, owning KG, towed type, and
    # emplacement landmark/event all vary, so the abstain target differs across
    # scenarios (not a fixed A:ZPU-4 / B:MO-120-RT every time). At least one
    # dangling is forced into each KG so both sides exercise abstention.
    towed_types = ["ZPU-4", "MO-120-RT"]
    emplacements = [("lm_hill205", "evt_air_defense"), ("lm_cross7", "evt_fire_support")]
    forced_kgs = ["A", "B"]
    for j in range(n_dangling):
        kg = forced_kgs[j] if j < len(forced_kgs) else ("A" if rng.random() < 0.5 else "B")
        t = towed_types[int(rng.integers(0, len(towed_types)))]
        label = _pick_label(rng, t)
        lm, evt = emplacements[int(rng.integers(0, len(emplacements)))]
        e0 = rng.uniform(800, 1800)
        n0 = rng.uniform(-900, 600)
        n_t = int(rng.integers(2, 4))
        times = sorted(rng.choice([30, 100, 170, 240, 290], size=n_t, replace=False).tolist())
        fire_from = float(rng.choice([120, 140, 160, 180]))
        src_pool = KG_SOURCES[kg]
        objs.append(TrueObject(
            gold_id=f"dang_{kg}_{t.replace('-', '').lower()}_{j}", true_type=t, label_base=label,
            waypoints=[(0.0, BASE_E + e0, BASE_N + n0), (300.0, BASE_E + e0, BASE_N + n0)],
            states=[(0.0, "Emplaced"), (fire_from, "Firing")],
            relations=[("emplacedAt", ("landmark", lm))],
            events=[(evt, "shooter")],
            dangling=True,
            tracks=[Track(kg, [float(x) for x in times],
                          [src_pool[int(rng.integers(0, len(src_pool)))] for _ in times], t, label)],
        ))

    return objs


# ---------------------------------------------------------------------------
# Ambiguity roster: relation is necessary, position is not sufficient (M2)
# ---------------------------------------------------------------------------
def roster_ambiguity(rng: np.random.Generator, jitter: bool = False) -> list[TrueObject]:
    """A controlled probe where the two T-72s (id_0001 lead, id_0003 follower)
    travel the SAME path within sensor CEP, so position/type cannot separate
    them; only the relational context differs (id_0001 ``near`` the BMP-2,
    id_0003 ``follows`` id_0001). KG_B types id_0003 UNKNOWN (hard positive).

    Also includes a Destroyed T-72 *wreck* (dangling, A-side) sitting on the
    column: blocking pairs it with the live tanks (same type, nearby), and only
    the b_state term (Destroyed->Moving ~ 0) keeps it from being merged.

    ``jitter`` randomises positions/labels/times for the training variants while
    preserving the structure; ``jitter=False`` yields the fixed dev probe.
    """
    def J(scale):
        return float(rng.uniform(-scale, scale)) if jitter else 0.0

    ox, oy = J(300), J(300)            # whole-scene offset for train variants

    def col(e0, n0, de, dn):
        return [(0.0, BASE_E + e0 + ox, BASE_N + n0 + oy),
                (150.0, BASE_E + e0 + ox + de / 2, BASE_N + n0 + oy + dn / 2),
                (300.0, BASE_E + e0 + ox + de, BASE_N + n0 + oy + dn)]

    de, dn = 600.0, 250.0
    lead_label = _pick_label(rng, "T-72") if jitter else "T-72 전차"
    # The follower is made IDENTICAL to the lead in type, label, and mean path so
    # that position/type/text give NO consistent signal separating the two
    # T-72s -- only the relation ("follows" vs "near the BMP") distinguishes
    # them. This is what forces the relation encoder to matter; with it off, the
    # ambiguous pair cannot be fit and id_0003 wrong-merges/fragments.
    foll_label = lead_label
    bmp_label = _pick_label(rng, "BMP-2") if jitter else "BMP-2 보병전투차"
    b_unknown = (UNKNOWN_LABELS[int(rng.integers(0, len(UNKNOWN_LABELS)))]
                 if jitter else "미상 기갑차량")

    # Draw observation times and sources from shared pools so the two T-72s are
    # statistically EXCHANGEABLE on every non-relational channel (time, source,
    # type, label, position). Any spurious cue a relation-off model might key on
    # (e.g. "t=30 => follower") is thus randomised away per scenario, leaving the
    # relation as the only generalisable separator.
    A_pool = [10, 45, 90, 150, 205, 290]
    B_pool = [25, 70, 130, 185, 250]

    def a_times(k=3):
        return sorted(float(x) for x in rng.choice(A_pool, size=k, replace=False))

    def b_times(k=3):
        return sorted(float(x) for x in rng.choice(B_pool, size=k, replace=False))

    def a_src(k):
        return [KG_SOURCES["A"][int(rng.integers(0, 2))] for _ in range(k)]

    def b_src(k):
        return [KG_SOURCES["B"][int(rng.integers(0, len(KG_SOURCES["B"])))] for _ in range(k)]

    objs: list[TrueObject] = []

    # id_0001 -- lead T-72, near the BMP-2.
    ta1 = a_times(3)
    objs.append(TrueObject(
        gold_id="id_0001", true_type="T-72", label_base=lead_label,
        waypoints=col(0, 0, de, dn), states=[(0.0, "Moving")],
        relations=[("near", ("obj", "id_0002"))],
        events=[("evt_advance", "lead")],
        tracks=[
            Track("A", ta1, a_src(3), "T-72", lead_label),
            Track("B", b_times(3), b_src(3), "T-72", lead_label),
        ],
    ))
    # id_0002 -- BMP-2, ~160 m offset (the distinct anchor that makes id_0001
    # matchable, which in turn disambiguates id_0003 through the GNN).
    objs.append(TrueObject(
        gold_id="id_0002", true_type="BMP-2", label_base=bmp_label,
        waypoints=col(160, 60, de, dn), states=[(0.0, "Moving")],
        relations=[("near", ("obj", "id_0001"))],
        events=[("evt_advance", "participant")],
        tracks=[
            Track("A", [20, 200], ["EO_IR", "RADAR"], "BMP-2", bmp_label),
            Track("B", [60, 220], ["RADAR", "HUMINT"], "BMP-2", bmp_label),
        ],
    ))
    # id_0003 -- follower T-72 on the SAME mean path (~3 m offset, far below
    # sensor CEP) and interleaved in time -> positionally inseparable from
    # id_0001; identical type and label. Only "follows id_0001" tells them apart.
    objs.append(TrueObject(
        gold_id="id_0003", true_type="T-72", label_base=foll_label,
        waypoints=col(3, 2, de, dn), states=[(0.0, "Moving")],
        relations=[("follows", ("obj", "id_0001"))],
        events=[("evt_advance", "participant")],
        tracks=[
            # A-side: same count/time-pool/source-pool as id_0001 -> exchangeable.
            Track("A", a_times(3), a_src(3), "T-72", foll_label),
            # B-side: the hard positive -- UNKNOWN type, vague label.
            Track("B", b_times(2), b_src(2), "UNKNOWN", b_unknown, 0.6),
        ],
    ))
    # id_0004 -- 2S1 SPG, normal distinct match elsewhere.
    objs.append(TrueObject(
        gold_id="id_0004", true_type="2S1", label_base=_pick_label(rng, "2S1") if jitter else "2S1 자주포",
        waypoints=[(0.0, BASE_E + 900 + ox, BASE_N - 400 + oy),
                   (300.0, BASE_E + 940 + ox, BASE_N - 380 + oy)],
        states=[(0.0, "Halted"), (160.0, "Firing")],
        relations=[("firesAt", ("landmark", "lm_cross7"))],
        events=[("evt_fire_support", "shooter")],
        tracks=[
            Track("A", [35, 250], ["EO_IR", "RADAR"], "2S1", "2S1 자주포"),
            Track("B", [75, 260], ["SIGINT", "ACOUSTIC"], "2S1", "2S1 자주포"),
        ],
    ))
    # id_0005 -- BTR-80, screening flank, distinct.
    objs.append(TrueObject(
        gold_id="id_0005", true_type="BTR-80", label_base=_pick_label(rng, "BTR-80") if jitter else "BTR-80 장갑차",
        waypoints=col(400, 850, 400, -200), states=[(0.0, "Moving")],
        relations=[("screens", ("obj", "id_0001"))],
        events=[("evt_advance", "screen")],
        tracks=[
            Track("A", [25, 240], ["RADAR", "EO_IR"], "BTR-80", "BTR-80 장갑차"),
            Track("B", [65, 250], ["RADAR", "HUMINT"], "BTR-80", "BTR-80 장갑차"),
        ],
    ))
    # dangling A -- Destroyed T-72 wreck ON the column (b_state demo). Same type
    # as the live tanks and nearby, so blocking pairs it with them; only
    # b_state(Destroyed->Moving ~ 0) prevents a wrong merge.
    objs.append(TrueObject(
        gold_id="dang_A_wreck", true_type="T-72",
        label_base=_pick_label(rng, "T-72") if jitter else "파괴된 전차",
        waypoints=[(0.0, BASE_E + 300 + ox, BASE_N + 120 + oy),
                   (300.0, BASE_E + 300 + ox, BASE_N + 120 + oy)],
        states=[(0.0, "Destroyed")],
        relations=[("near", ("obj", "id_0001"))],
        events=[],
        dangling=True,
        tracks=[Track("A", [120, 230], ["EO_IR", "EO_IR"], "T-72",
                      "파괴된 전차" if not jitter else _pick_label(rng, "T-72"))],
    ))
    # dangling B -- MO-120-RT towed mortar, no counterpart.
    objs.append(TrueObject(
        gold_id="dang_B_mo120", true_type="MO-120-RT",
        label_base=_pick_label(rng, "MO-120-RT") if jitter else "MO-120-RT 박격포",
        waypoints=[(0.0, BASE_E + 1500 + ox, BASE_N - 600 + oy),
                   (300.0, BASE_E + 1500 + ox, BASE_N - 600 + oy)],
        states=[(0.0, "Emplaced"), (140.0, "Firing")],
        relations=[("emplacedAt", ("landmark", "lm_cross7"))],
        events=[("evt_fire_support", "shooter")],
        dangling=True,
        tracks=[Track("B", [50, 150, 270], ["ACOUSTIC", "SIGINT", "RADAR"],
                      "MO-120-RT", "MO-120-RT 박격포")],
    ))
    return objs


# ---------------------------------------------------------------------------
# Realisation: roster -> observations / entities / files
# ---------------------------------------------------------------------------
def _mgrs_stub(e: float, n: float) -> str:
    return f"52SCF{int(e) % 100000:05d}{int(n) % 100000:05d}"


def _iso(t: float) -> str:
    return (EPOCH + timedelta(seconds=t)).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Realised:
    scenario_id: str
    observations: list[dict]
    entities: dict[str, dict]            # entity_key "kg:local" -> entity dict
    landmarks: dict[str, Landmark]
    events: dict[str, EventDef]
    gold: dict[str, Any]


def realise(scenario_id: str, roster: list[TrueObject], rng: np.random.Generator,
            ontology: dict) -> Realised:
    landmarks = landmarks_commons()
    events = events_commons()

    # Assign local entity ids per KG and gather membership.
    ent_counter = {"A": 0, "B": 0}
    obj_local: dict[str, dict[str, str]] = {}   # gold_id -> {kg: local_id}
    for obj in roster:
        obj_local[obj.gold_id] = {}
        for tr in obj.tracks:
            ent_counter[tr.kg_id] += 1
            local_id = f"ent_{tr.kg_id}_{ent_counter[tr.kg_id]:03d}"
            obj_local[obj.gold_id][tr.kg_id] = local_id

    def resolve_target(kg: str, tgt: tuple) -> Optional[str]:
        kind, ref = tgt
        if kind == "landmark":
            return ref            # commons id, valid in any KG scope
        if kind == "obj":
            return obj_local.get(ref, {}).get(kg)   # only if observed in this KG
        return None

    observations: list[dict] = []
    entities: dict[str, dict] = {}
    src_cep = {s: ontology["sources"].get(s, {}).get("cep_m", 50.0) for s in ontology["sources"]}
    src_typerel = {s: ontology["sources"].get(s, {}).get("type_reliability", 0.5) for s in ontology["sources"]}
    src_rel = {s: ontology["sources"].get(s, {}).get("reliability", 0.5) for s in ontology["sources"]}

    obs_counter = 0
    for obj in roster:
        for tr in obj.tracks:
            kg = tr.kg_id
            local_id = obj_local[obj.gold_id][kg]
            ekey = f"{kg}:{local_id}"
            # Resolve entity-level relations for this KG scope.
            ent_rels = []
            for pred, tgt in obj.relations:
                rt = resolve_target(kg, tgt)
                if rt is not None:
                    ent_rels.append({"predicate": pred, "target_ref": rt,
                                     "confidence": round(float(0.8 + 0.2 * rng.random()), 3)})
            ent_events = [{"event_id": ev, "role": role} for ev, role in obj.events]
            entities[ekey] = {
                "local_entity_id": local_id, "kg_id": kg,
                "object_type": tr.obj_type, "relations": ent_rels,
                "events": ent_events, "obs_ids": [],
                "_gold_id": obj.gold_id, "_dangling": obj.dangling,
            }
            n = len(tr.sample_times)
            for i, t_true in enumerate(tr.sample_times):
                obs_counter += 1
                obs_id = f"obs_{kg}_{obs_counter:04d}"
                src = tr.sources[i % len(tr.sources)]
                cep = src_cep.get(src, 50.0)
                # Observed time = true time + clock skew + small jitter (within eps).
                t_obs = t_true + KG_CLOCK_OFFSET[kg] + float(rng.normal(0, 1.0))
                e_true, n_true = obj.position_at(t_true)
                # Position noise scaled by CEP (CEP ~ radius; sigma ~ cep/1.1774).
                sigma = cep / 1.1774
                e_obs = e_true + float(rng.normal(0, sigma))
                n_obs = n_true + float(rng.normal(0, sigma))
                state = obj.state_at(t_true)
                speed, heading = obj.velocity_at(t_true)
                if state in ("Emplaced", "Halted", "Destroyed"):
                    speed = 0.0
                type_conf = round(min(0.99, max(0.2,
                                src_typerel.get(src, 0.5) * tr.type_conf_scale + 0.05 * rng.random())), 3)
                state_conf = round(min(0.99, max(0.2, src_rel.get(src, 0.5) + 0.05 * rng.random())), 3)
                obs = {
                    "obs_id": obs_id, "kg_id": kg, "source": src,
                    "local_entity_id": local_id, "label_text": tr.label_text,
                    "type": tr.obj_type, "type_confidence": type_conf,
                    "state": state, "state_confidence": state_conf,
                    "time": _iso(t_obs),
                    "location": {"crs": CRS, "easting": round(e_obs, 2),
                                 "northing": round(n_obs, 2), "elevation": 50.0,
                                 "mgrs": _mgrs_stub(e_obs, n_obs)},
                    "cep_m": round(cep, 1),
                    "velocity": {"speed_mps": round(speed, 2), "heading_deg": round(heading, 1)},
                    "relations": ent_rels,
                    "events": ent_events,
                    "provenance": {"wasAttributedTo": src,
                                   "generatedAtTime": _iso(t_obs)},
                }
                observations.append(obs)
                entities[ekey]["obs_ids"].append(obs_id)

    # Sort observations by time then id for stable output.
    observations.sort(key=lambda o: (o["time"], o["obs_id"]))

    gold = build_gold(roster, obj_local, observations, entities)
    return Realised(scenario_id, observations, entities, landmarks, events, gold)


# ---------------------------------------------------------------------------
# Gold-label construction
# ---------------------------------------------------------------------------
def build_gold(roster, obj_local, observations, entities) -> dict[str, Any]:
    obs_by_gold: dict[str, list[str]] = {}
    for ekey, ent in entities.items():
        obs_by_gold.setdefault(ent["_gold_id"], []).extend(ent["obs_ids"])

    identities = []
    dangling_obs: list[str] = []
    dangling_entities: list[dict] = []
    obs_to_gold: dict[str, str] = {}

    for obj in roster:
        members = sorted(obs_by_gold.get(obj.gold_id, []))
        for oid in members:
            obs_to_gold[oid] = obj.gold_id
        member_local = obj_local[obj.gold_id]
        # true trajectory sampled along the path at observation times
        traj = []
        for oid in members:
            o = next(o for o in observations if o["obs_id"] == oid)
            traj.append({"time": o["time"],
                         "easting": o["location"]["easting"],
                         "northing": o["location"]["northing"]})
        traj.sort(key=lambda p: p["time"])
        if obj.dangling:
            dangling_obs.extend(members)
            for kg, lid in member_local.items():
                dangling_entities.append({"kg_id": kg, "local_entity_id": lid,
                                          "true_type": obj.true_type, "obs_ids": members})
        else:
            identities.append({
                "gold_identity_id": obj.gold_id,
                "true_type": obj.true_type,
                "member_observations": members,
                "member_local_entities": member_local,   # {kg: local_id}
                "true_trajectory": traj,
            })

    gold_identities = {
        "scenario_id": None,   # filled by caller
        "identities": identities,
        "dangling_observations": sorted(dangling_obs),
        "dangling_local_entities": dangling_entities,
    }

    # --- entity_level: A x B entity pairs, same/diff + hard-negative flag ---
    a_ents = [e for e in entities.values() if e["kg_id"] == "A"]
    b_ents = [e for e in entities.values() if e["kg_id"] == "B"]
    entity_pairs = []
    for ea in a_ents:
        for eb in b_ents:
            same = (ea["_gold_id"] == eb["_gold_id"]) and not ea["_dangling"]
            hard = (not same) and (ea["object_type"] == eb["object_type"]
                                   and ea["object_type"] != "UNKNOWN")
            entity_pairs.append({
                "kg_a_entity": ea["local_entity_id"],
                "kg_b_entity": eb["local_entity_id"],
                "label": "same" if same else "diff",
                "hard_negative_same_type": bool(hard),
            })
    entity_level = {"scenario_id": None, "pairs": entity_pairs}

    # --- observation_level: obs->gold + all cross-KG obs pair labels ---
    a_obs = [o for o in observations if o["kg_id"] == "A"]
    b_obs = [o for o in observations if o["kg_id"] == "B"]
    obs_pairs = []
    for oa in a_obs:
        for ob in b_obs:
            ga, gb = obs_to_gold.get(oa["obs_id"]), obs_to_gold.get(ob["obs_id"])
            same = (ga is not None and ga == gb
                    and oa["obs_id"] not in dangling_obs and ob["obs_id"] not in dangling_obs)
            obs_pairs.append({"obs_a": oa["obs_id"], "obs_b": ob["obs_id"],
                              "label": "same" if same else "diff"})
    observation_level = {
        "scenario_id": None,
        "assignment": [{"obs_id": oid, "gold_identity_id": g} for oid, g in sorted(obs_to_gold.items())],
        "cross_kg_pairs": obs_pairs,
    }

    # --- transition_level: within-identity consecutive positives + hard negs ---
    positives, negatives = [], []
    for obj in roster:
        members = sorted(obs_by_gold.get(obj.gold_id, []),
                         key=lambda oid: next(o["time"] for o in observations if o["obs_id"] == oid))
        for a, b in zip(members, members[1:]):
            positives.append({"src": a, "dst": b})
    # hard negatives: consecutive-in-time obs that belong to different golds
    ordered = sorted(observations, key=lambda o: o["time"])
    for a, b in zip(ordered, ordered[1:]):
        ga, gb = obs_to_gold.get(a["obs_id"]), obs_to_gold.get(b["obs_id"])
        if ga != gb:
            negatives.append({"src": a["obs_id"], "dst": b["obs_id"]})
    transition_level = {"scenario_id": None, "positives": positives,
                        "hard_negatives": negatives[: max(len(positives), 4)]}

    dangling_json = {"scenario_id": None,
                     "dangling_observations": sorted(dangling_obs),
                     "dangling_entities": dangling_entities}

    return {
        "gold_identities": gold_identities,
        "entity_level": entity_level,
        "observation_level": observation_level,
        "transition_level": transition_level,
        "dangling": dangling_json,
    }


# ---------------------------------------------------------------------------
# JSON-LD writers
# ---------------------------------------------------------------------------
def _entity_node(ent: dict) -> dict:
    kg = ent["kg_id"]
    node = {"@id": f"kg{kg}:{ent['local_entity_id']}", "@type": "Entity",
            "objectType": ent["object_type"]}
    # relation predicates as JSON-LD properties (grouped by predicate)
    for r in ent["relations"]:
        pred = r["predicate"]
        tref = r["target_ref"]
        target_id = (tref if tref.startswith("lm_") else f"kg{kg}:{tref}")
        node.setdefault(pred, []).append(target_id)
    if ent["events"]:
        node["participatesIn"] = [f"commons:{e['event_id']}" for e in ent["events"]]
    return node


def _obs_node(o: dict) -> dict:
    kg = o["kg_id"]
    node = {
        "@id": f"kg{kg}:{o['obs_id']}", "@type": "Observation",
        "observationOf": f"kg{kg}:{o['local_entity_id']}",
        "time": o["time"],
        "easting": o["location"]["easting"], "northing": o["location"]["northing"],
        "elevation": o["location"]["elevation"], "crs": o["location"]["crs"],
        "mgrs": o["location"]["mgrs"], "cepM": o["cep_m"],
        "objectType": o["type"], "typeConfidence": o["type_confidence"],
        "state": o["state"], "stateConfidence": o["state_confidence"],
        "labelText": o["label_text"], "source": o["source"],
        "speedMps": o["velocity"]["speed_mps"], "headingDeg": o["velocity"]["heading_deg"],
        "wasAttributedTo": o["source"], "generatedAtTime": o["provenance"]["generatedAtTime"],
    }
    return node


def write_jsonld(real: Realised, out_dir: Path) -> None:
    for kg in ("A", "B"):
        graph = []
        for ekey, ent in real.entities.items():
            if ent["kg_id"] == kg:
                graph.append(_entity_node(ent))
        for o in real.observations:
            if o["kg_id"] == kg:
                graph.append(_obs_node(o))
        doc = {"@context": "ontology/context.jsonld",
               "@graph": graph, "stkg:kgId": kg, "stkg:scenario": real.scenario_id}
        (out_dir / f"kg_{kg}.jsonld").write_text(
            json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    # commons: shared landmarks + events
    commons_graph = []
    for lm in real.landmarks.values():
        commons_graph.append({"@id": f"commons:{lm.landmark_id}", "@type": "Landmark",
                              "rdfs:label": lm.name, "easting": lm.easting,
                              "northing": lm.northing, "crs": CRS})
    for ev in real.events.values():
        commons_graph.append({"@id": f"commons:{ev.event_id}", "@type": "Event",
                              "rdfs:label": ev.name, "stkg:eventKind": ev.kind})
    commons = {"@context": "ontology/context.jsonld", "@graph": commons_graph}
    (out_dir / "commons.jsonld").write_text(
        json.dumps(commons, ensure_ascii=False, indent=2), encoding="utf-8")


def write_observations_jsonl(real: Realised, out_dir: Path) -> None:
    with open(out_dir / "observations.jsonl", "w", encoding="utf-8") as fh:
        for o in real.observations:
            fh.write(json.dumps(o, ensure_ascii=False) + "\n")


def write_labels(real: Realised, out_dir: Path) -> None:
    labels_dir = out_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    for key in ("gold_identities", "entity_level", "observation_level",
                "transition_level", "dangling"):
        obj = real.gold[key]
        obj["scenario_id"] = real.scenario_id
        (labels_dir / f"{key}.json").write_text(
            json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_manifest(real: Realised, out_dir: Path, seed: int, split: str) -> dict:
    n_a_ent = sum(1 for e in real.entities.values() if e["kg_id"] == "A")
    n_b_ent = sum(1 for e in real.entities.values() if e["kg_id"] == "B")
    n_match = len(real.gold["gold_identities"]["identities"])
    n_dangling_ent = len(real.gold["dangling"]["dangling_entities"])
    manifest = {
        "scenario_id": real.scenario_id, "seed": seed, "split": split,
        "crs": CRS, "epoch": EPOCH.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_observations": len(real.observations),
        "n_entities_A": n_a_ent, "n_entities_B": n_b_ent,
        "n_matched_identities": n_match,
        "n_dangling_entities": n_dangling_ent,
        "n_dangling_observations": len(real.gold["dangling"]["dangling_observations"]),
        "files": ["kg_A.jsonld", "kg_B.jsonld", "commons.jsonld",
                  "observations.jsonl", "manifest.json", "labels/"],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def generate_scenario(scenario_id: str, seed: int, n_objects: int, split: str) -> dict:
    rng = np.random.default_rng(seed)
    ontology = {
        "sources": yaml.safe_load((ONTOLOGY_DIR / "sources.yaml").read_text(encoding="utf-8")),
    }
    if scenario_id == "scn_0001":
        roster = roster_scn_0001()
    elif scenario_id == "scn_amb01":
        roster = roster_ambiguity(rng, jitter=False)     # fixed dev probe (M2)
    elif "amb" in scenario_id:
        roster = roster_ambiguity(rng, jitter=True)      # train/test variants
    else:
        roster = roster_random(rng, n_objects)
    real = realise(scenario_id, roster, rng, ontology)

    out_dir = DATA_ROOT / "scenarios" / scenario_id
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    write_jsonld(real, out_dir)
    write_observations_jsonl(real, out_dir)
    write_labels(real, out_dir)
    manifest = write_manifest(real, out_dir, seed, split)
    return manifest


def update_splits(assignments: dict[str, list[str]]) -> None:
    splits_dir = DATA_ROOT / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split, ids in assignments.items():
        (splits_dir / f"{split}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")


DEFAULT_SUITE = {
    # scn_0001 / scn_amb01 are the fixed dev probes; train/test generated under
    # leakage rules (disjoint seeds, §A.9.3). Kept small so CPU training is fast.
    # scn_tamb* are relation-necessary ambiguity training variants so the GNN can
    # learn to use relational context; scn_amb01 is the held-out ambiguity probe.
    "train": ([("scn_t{:03d}".format(i), 100 + i, 6 + (i % 4)) for i in range(1, 9)]
              + [("scn_tamb{:02d}".format(i), 500 + i, 8) for i in range(1, 11)]),
    "dev": ([("scn_0001", 20250606, 7), ("scn_amb01", 424242, 8)]
            + [("scn_amb{:02d}".format(i), 424242 + i, 8) for i in range(2, 6)]
            + [("scn_d{:03d}".format(i), 300 + i, 6 + (i % 3)) for i in range(1, 3)]),
    "test": ([("scn_e{:03d}".format(i), 900 + i, 6 + (i % 4)) for i in range(1, 4)]
             + [("scn_eamb{:02d}".format(i), 700 + i, 8) for i in range(1, 5)]),
}


def build_suite() -> None:
    assignments: dict[str, list[str]] = {"train": [], "dev": [], "test": []}
    for split, specs in DEFAULT_SUITE.items():
        for scenario_id, seed, n_obj in specs:
            m = generate_scenario(scenario_id, seed, n_obj, split)
            assignments[split].append(scenario_id)
            print(f"[{split}] {scenario_id}: obs={m['n_observations']} "
                  f"A={m['n_entities_A']} B={m['n_entities_B']} "
                  f"match={m['n_matched_identities']} dangling={m['n_dangling_entities']}")
    update_splits(assignments)
    print("splits written:", {k: len(v) for k, v in assignments.items()})


def main() -> None:
    ap = argparse.ArgumentParser(description="CITA-Net dataset generator")
    ap.add_argument("--scenario-id", default="scn_0001")
    ap.add_argument("--seed", type=int, default=20250606)
    ap.add_argument("--n-objects", type=int, default=7)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--build-suite", action="store_true",
                    help="regenerate the full default train/dev/test suite")
    args = ap.parse_args()

    if args.build_suite:
        build_suite()
        return
    m = generate_scenario(args.scenario_id, args.seed, args.n_objects, args.split)
    print(json.dumps(m, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
