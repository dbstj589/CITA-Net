"""Two-layer 10-day battle GT for Hill 395.

Upper layer (DIRECTOR): a source-based ten-day skeleton (daily loss/retake, D3 fog
window, echelon commitment, day/night) laid out as a schedule with provenance
tags ("record" = from sources, "interp" = designed to fill a gap).

Lower layer (UNIT RULES): each squad carries a combat-power s in [0,1] and acts on
threshold rules (mutual attrition, withdraw, merge, relief, boundary entry/exit).
The director sets attack timing/intensity to steer the DAILY outcome; individual
squad motion/attrition is the result of the rules.

Attrition is a simplification of Lanchester (1916) aimed-fire law — it provides the
DIRECTION of mutual losses only, not a prediction. Combat power s is an ontology
EXTENSION (not in the current ontology) — flagged "확인 필요". This is synthetic;
record items follow source narratives with approximate times/scales, interp items
(A-D) fill gaps by design. NOT a reconstruction of real battle causality.

The simulator time-steps at dt and emits the trajectory (with s), interval-stamped
relations/events, and reified transitions at meaningful threshold crossings only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .common import SIZE_BY_CATEGORY

# --- day/night (10-month Korea approximation; procedural simplification) ---
DUSK = 18 * 3600            # 18:00
DAWN = 6 * 3600 + 1800      # 06:30
DAY = 86400


def is_night(t):
    s = t % DAY
    return s >= DUSK or s < DAWN


@dataclass
class Agent:
    eid: str
    typ: str
    aff: str            # ROK / US / CCF
    unit: str
    role: str           # fwd / reserve / arty / mortar / tank / light / assault
    echelon: str        # parent echelon id (CCF) or platoon
    x: float
    y: float
    home: tuple
    s: float = 1.0
    state: str = "Holding"
    active: bool = True          # on-map / participating
    entered: bool = False
    t_enter: float = 0.0
    t_exit: float = None
    obj: tuple = None            # current move objective (x,y) or None
    alive: bool = True
    merged_into: str = None
    samples: list = field(default_factory=list)   # (t,x,y,state,speed,heading,s)
    transitions: list = field(default_factory=list)  # (t, kind, detail)


def _v_max(ont, typ, state):
    return ont.v_max(typ, state)


def build_schedule(W):
    """Absolute-time director schedule. t in seconds from 1952-10-06 00:00.
    Each op: dict(kind, t0, ...). provenance: record|interp-X."""
    def d(day):
        return day * DAY
    def night_of(day, hh, mm=0):
        return d(day) + hh * 3600 + mm * 60

    ops = []
    fog = []
    # background prep intensity handled continuously; here the discrete pulses.
    # D1 (day0): daytime preparatory bombardment, no infantry [record]
    ops.append(dict(kind="prep_bombard", t0=d(0) + 10 * 3600, t1=d(0) + 16 * 3600, prov="record",
                    note="D1 daytime preparatory fires, no infantry"))
    # D1 night: echelons 1&2, three waves on two axes, all repulsed [record]
    for k, hh in enumerate([19, 21, 23]):
        ops.append(dict(kind="attack", t0=night_of(0, hh), ech=[k % 7, (k + 1) % 7], axis=k % 3,
                        result="repulsed", prov="record", note=f"D1 night wave {k+1} (repulsed)"))
    # D2 (day1): daytime refit (low) [interp-B]; night 2 echelons -> LOSS, ~2h later retake [record]
    ops.append(dict(kind="refit", t0=d(1) + 8 * 3600, t1=d(1) + 16 * 3600, prov="interp-B", note="D2 refit"))
    ops.append(dict(kind="attack", t0=night_of(1, 20), ech=[2, 3], axis=0, result="take",
                    take_t=night_of(1, 21), prov="record", note="D2 night: crest lost"))
    ops.append(dict(kind="counterattack", t0=night_of(1, 22, 30), prep=1800, retake_t=night_of(1, 23, 30),
                    prov="record", note="D2 reserve counterattack retakes crest"))
    # D3 (day2): dawn attack + 1 echelon; fog 06:00-12:00 -> no friendly fire -> LOSS 08:10;
    #            17:00 reserve, 23:05 retake [record]
    fog.append((d(2) + 6 * 3600, d(2) + 12 * 3600))
    ops.append(dict(kind="attack", t0=d(2) + 5 * 3600, ech=[4, 5, 6], axis=1, result="take",
                    take_t=d(2) + 8 * 3600 + 600, prov="record", note="D3 dawn+fog: crest lost 08:10 (fog: no friendly fire)"))
    ops.append(dict(kind="counterattack", t0=d(2) + 17 * 3600, prep=2400, retake_t=d(2) + 23 * 3600 + 300,
                    prov="record", note="D3 17:00 reserve committed, 23:05 retake"))
    # D4 (day3): just-after-midnight wave -> crest+right ridge LOSS; daytime MAX prep then retake [record]
    ops.append(dict(kind="attack", t0=d(3) + 1800, ech=[0, 1, 2], axis=2, result="take",
                    take_t=d(3) + 2 * 3600, prov="record", note="D4 post-midnight: crest + right ridge lost"))
    ops.append(dict(kind="counterattack", t0=d(3) + 13 * 3600, prep=3600, retake_t=d(3) + 15 * 3600,
                    prep_max=True, prov="record", note="D4 daytime MAX preparatory fire then counterattack retake"))
    # D5 (day4): dawn large attack; fall back to 9-bu ridge -> MERGE with reinforcement -> retake [record]
    ops.append(dict(kind="attack", t0=d(4) + 4 * 3600, ech=[3, 4, 5, 6], axis=0, result="take",
                    take_t=d(4) + 5 * 3600, force_merge=True, prov="record", note="D5 dawn assault; defenders fall back, MERGE, retake"))
    ops.append(dict(kind="counterattack", t0=d(4) + 8 * 3600, prep=2400, retake_t=d(4) + 10 * 3600,
                    prov="record", note="D5 reinforced retake"))
    # D6 (day5): daytime quiet (artillery only) [interp-C]; night crest LOSS [record]
    ops.append(dict(kind="quiet", t0=d(5) + 8 * 3600, t1=d(5) + 16 * 3600, prov="interp-C", note="D6 daytime lull (artillery only)"))
    ops.append(dict(kind="attack", t0=night_of(5, 21), ech=[0, 1], axis=1, result="take",
                    take_t=night_of(5, 22), prov="record", note="D6 night: crest lost"))
    # D7 (day6): morning OVERWATCH attack (fresh passes tired) -> retake -> enemy counter -> LOSS [record]
    ops.append(dict(kind="counterattack", t0=d(6) + 6 * 3600, prep=1800, retake_t=d(6) + 8 * 3600,
                    overwatch=True, prov="record", note="D7 morning passage-of-lines counterattack retakes"))
    ops.append(dict(kind="attack", t0=d(6) + 14 * 3600, ech=[2, 3], axis=2, result="take",
                    take_t=d(6) + 15 * 3600, prov="record", note="D7 enemy counter: crest lost again"))
    # D8-9 (day7,8): nightly small loss, dawn retake, shrinking [interp-D]
    for day in (7, 8):
        ops.append(dict(kind="attack", t0=night_of(day, 22), ech=[day % 7], axis=day % 3, result="take",
                        take_t=night_of(day, 23), shrink=True, prov="interp-D", note=f"D{day+1} night small loss"))
        ops.append(dict(kind="counterattack", t0=d(day + 1) + 4 * 3600, prep=1500, retake_t=d(day + 1) + 5 * 3600,
                        shrink=True, prov="interp-D", note=f"D{day+1} dawn retake"))
    # D10 (day9): dawn surprise counterattack -> full retake -> Occupy northtip/ridge; CCF exit north [record]
    ops.append(dict(kind="counterattack", t0=d(9) + 4 * 3600, prep=1800, retake_t=d(9) + 6 * 3600,
                    final=True, prov="record", note="D10 dawn surprise counterattack, full recapture then pursue to ridge"))
    ops.append(dict(kind="withdraw_all", t0=d(9) + 8 * 3600, prov="record", note="D10 surviving CCF withdraw north off-map"))
    return ops, fog
