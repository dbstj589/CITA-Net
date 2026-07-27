"""Multi-phase, lifecycle-aware sector builder for long-window GT.

Motivation (pre-check finding): stretching a single 10-minute engagement over a
long window is physically unnatural. Here each sector runs P phases (config)
over the global window with:
  - phase rotation      even phase = CCF assault, odd phase = ROK counterattack
  - entry/exit          CCF assault WAVES enter at each phase start and exit after
                        a bounded lifetime (reinforcement built in)
  - attrition           a fraction of entities die (state -> Destroyed at t_die,
                        remaining as a wreck to the window end)
  - reinforcement       later phases add fresh CCF waves and ROK squads
Persistent ROK defenders hold the MLR/crest across all phases (state cycles
Holding/Engaging). One sector is built at a time (streaming); IDs/coords/time are
global (passed-in counters, global origin, shared window).

All motion feasibility is still guaranteed downstream by the trajectory sampler
(clamp to v_max*dt, static states frozen). Vocabulary comes from the ontology.
"""
from __future__ import annotations

import numpy as np

from .common import (AFFILIATION, LANDMARK_OFFSETS, SIZE_BY_CATEGORY, Landmark,
                     RelationEdge, Unit, WorldEntity, WorldEvent, roster_counts,
                     sector_origin)


def _pick(rng, pool):
    return pool[int(rng.integers(0, len(pool)))]


def _size_of(ont, t):
    return SIZE_BY_CATEGORY.get(ont.category(t), "unknown")


def build_sector_world(s, cfg, ontology, rng, eid_ctr, evid_ctr):
    """Build ONE sector. Returns (entities, units, landmarks, events, relations).
    eid_ctr/evid_ctr are 1-element lists used as global counters."""
    W = float(cfg["time"]["window_seconds"])
    P = int(cfg["time"].get("phases", 6))
    ph = W / P
    pitch = float(cfg["world_scope"].get("sector_pitch_m", 6000.0))
    per_sector = int(cfg["entities"]["per_sector"])
    p_attr = float(cfg["entities"].get("attrition_frac", 0.22))
    reinf = float(cfg["entities"].get("reinforce_frac", 0.5))   # wave size vs base CCF inf
    ox, oy = sector_origin(s, pitch)
    counts = roster_counts(per_sector)

    def neid():
        eid_ctr[0] += 1
        return f"e_{eid_ctr[0]:06d}"

    def nevid():
        evid_ctr[0] += 1
        return f"ev_{evid_ctr[0]:06d}"

    entities: list[WorldEntity] = []
    units: dict[str, Unit] = {}
    landmarks: dict[str, Landmark] = {}
    events: list[WorldEvent] = []
    relations: list[RelationEdge] = []

    # landmarks
    lm = {}
    for name, (dx, dy) in LANDMARK_OFFSETS.items():
        lid = f"lm_{name}_{s}"
        landmarks[lid] = Landmark(lid, name, ox + dx, oy + dy)
        lm[name] = lid

    # units
    rok_rgt = f"u_rok_rgt_{s}"; ccf_rgt = f"u_ccf_rgt_{s}"
    units[rok_rgt] = Unit(rok_rgt, "regiment", "ROK", "ROK Regiment (defence)")
    units[ccf_rgt] = Unit(ccf_rgt, "regiment", "CCF", "CCF Regiment (assault)")
    rok_bns, ccf_bns = [], []
    for i in range(max(1, counts["rok_inf"] // 9)):
        u = f"u_rok_bn{i}_{s}"; units[u] = Unit(u, "battalion", "ROK", f"ROK Infantry Bn {i}", rok_rgt); rok_bns.append(u)
    for i in range(max(1, counts["ccf_inf"] // 9)):
        u = f"u_ccf_bn{i}_{s}"; units[u] = Unit(u, "battalion", "CCF", f"CCF Infantry Bn {i}", ccf_rgt); ccf_bns.append(u)

    # per-phase events (scale with phases). participants filled after entities.
    phase_events = []          # (evid, kind, p, a, b, landmark)
    for p in range(P):
        a, b = p * ph, (p + 1) * ph
        kind = "Attack" if p % 2 == 0 else "Counterattack"
        e1 = nevid(); events.append(WorldEvent(e1, kind, s, a, b, lm["crest"], []))
        e2 = nevid(); events.append(WorldEvent(e2, "FireSupport", s, a, b, lm["valley"], []))
        phase_events.append({"main": (e1, kind), "fire": e2, "a": a, "b": b, "p": p})
    # sector-level Occupy + Illuminate (once)
    ev_occ = nevid(); events.append(WorldEvent(ev_occ, "Occupy", s, 0.0, W, lm["objA"], []))
    ev_ill = nevid(); events.append(WorldEvent(ev_ill, "Illuminate", s, 0.0, min(W, ph), lm["mlr"], []))
    ev_by_id = {e.evid: e for e in events}

    def add_ent(e):
        entities.append(e)
        for evid, _ in e.events:
            ev_by_id[evid].participants.append(e.eid)

    def die_or_exit(a, b, force_persist=False):
        """Return (t_exit, states_tail, exit_relpred). Attrition -> Destroyed wreck
        kept to window end; else withdraw and exit at b."""
        L = b - a
        if not force_persist and float(rng.random()) < p_attr:
            t_die = a + rng.uniform(0.5, 0.85) * L
            return W, [(t_die, "Destroyed")], None          # wreck remains to W
        return b, [(a + 0.8 * L, "Withdrawing")], "withdrawsFrom"

    # ---------------- persistent ROK defence (whole window) ----------------
    rok_state_cycle = []
    for p in range(P):
        rok_state_cycle.append((p * ph, "Engaging" if p % 2 == 1 else "Holding"))
    for _ in range(counts["rok_inf"]):
        bn = _pick(rng, rok_bns)
        e0 = ox + 1500 + rng.uniform(-400, 400); n0 = oy + rng.uniform(-500, 300)
        wp = [(0.0, e0, n0)]
        for p in range(1, P + 1):
            wp.append((min(W, p * ph), e0 + rng.uniform(-60, 60), n0 + rng.uniform(-60, 60)))
        st = [(0.0, "Holding")] + rok_state_cycle
        rels = [("partOf", bn, "unit"), ("emplacedAt", lm["mlr"], "landmark")]
        evs = [(ev_occ, "defence")]
        # counterattack participation on the first odd phase
        if P > 1:
            evs.append((phase_events[1]["main"][0], "counterattack"))
            rels.append(("occupies", lm["crest"], "landmark"))
        # attrition for some defenders
        t_exit, tail, _ = die_or_exit(0.0, W, force_persist=(float(rng.random()) > p_attr))
        if tail and tail[0][1] == "Destroyed":
            st = st + tail
        add_ent(WorldEntity(neid(), "ROK_Infantry", "ROK", _size_of(ontology, "ROK_Infantry"),
                            bn, s, wp, st, rels, evs, t_enter=0.0, t_exit=t_exit))

    # persistent ROK support: armour / artillery / mortar / AA / engineer / searchlight
    def emplaced_support(t, n_e, e_lo, e_hi, n_lo, n_hi, lmk, rgt, fire_state="Firing"):
        for _ in range(n_e):
            e0 = ox + rng.uniform(e_lo, e_hi); n0 = oy + rng.uniform(n_lo, n_hi)
            wp = [(0.0, e0, n0), (W, e0, n0)]
            st = [(0.0, "Emplaced")]
            for p in range(P):
                st.append((p * ph + 0.4 * ph, fire_state if p % 2 == 0 else "Emplaced"))
            rels = [("emplacedAt", lmk, "landmark"), ("supports", rgt, "unit"),
                    ("firesAt", lm["valley"], "landmark")]
            evs = [(phase_events[0]["fire"], "shooter")]
            add_ent(WorldEntity(neid(), t, AFFILIATION.get(t, "ROK"), _size_of(ontology, t),
                                rgt, s, wp, st, rels, evs, t_enter=0.0, t_exit=W))

    for _ in range(counts["tank"]):
        t = _pick(rng, ["US_Tank", "ROK_Tank"])
        e0 = ox + rng.uniform(100, 500); n0 = oy + rng.uniform(-600, 200)
        wp = [(0.0, e0, n0), (0.3 * W, e0 + 300, n0 + 60), (W, e0 + 300, n0 + 60)]
        st = [(0.0, "Moving"), (0.3 * W, "Halted"), (0.4 * W, "Firing")]
        rels = [("supports", rok_rgt, "unit"), ("firesAt", lm["valley"], "landmark"),
                ("screens", lm["mlr"], "landmark")]
        add_ent(WorldEntity(neid(), t, AFFILIATION[t], _size_of(ontology, t), rok_rgt, s, wp, st, rels,
                            [(phase_events[0]["fire"], "shooter")], t_enter=0.0, t_exit=W))
    emplaced_support("ROK_Artillery", counts["arty"], 800, 2400, -1400, -700, lm["mlr"], rok_rgt)
    emplaced_support("Mortar", counts["mortar"], 700, 2300, -1200, -500, lm["mlr"], rok_rgt)
    emplaced_support("CCF_AA", counts["aa"], 600, 2400, 1000, 1600, lm["northslope"], ccf_rgt)
    for _ in range(counts["engr"]):
        e0 = ox + rng.uniform(-300, 300); n0 = oy + rng.uniform(-200, 600)
        wp = [(0.0, e0, n0), (W, e0 + rng.uniform(-150, 150), n0 + rng.uniform(-150, 150))]
        add_ent(WorldEntity(neid(), "Engineer", "ROK", _size_of(ontology, "Engineer"), rok_rgt, s, wp,
                            [(0.0, "Moving")], [("near", lm["yokkok"], "landmark"), ("supports", rok_rgt, "unit")],
                            [(ev_occ, "engineer")], t_enter=0.0, t_exit=W))
    # searchlight (phase 0 only)
    e0 = ox + rng.uniform(200, 1200); n0 = oy + rng.uniform(-1400, -900)
    add_ent(WorldEntity(neid(), "Searchlight", AFFILIATION["Searchlight"], _size_of(ontology, "Searchlight"),
                        rok_rgt, s, [(0.0, e0, n0), (W, e0, n0)],
                        [(0.0, "Emplaced"), (0.4 * ph, "Firing")], [("supports", rok_rgt, "unit")],
                        [(ev_ill, "illuminate")], t_enter=0.0, t_exit=min(W, ph)))

    # ---------------- CCF assault waves (entry/exit/attrition/reinforcement) ----------------
    wave_n = max(3, round(counts["ccf_inf"] * reinf))
    defender_ids = [e.eid for e in entities if e.affiliation == "ROK"]
    for p in range(P):
        if p % 2 == 1:
            continue                          # attack waves launch on assault (even) phases
        a = p * ph
        life = min(W - a, 1.6 * ph)
        b = a + life
        pe = phase_events[p]
        lane_y = np.linspace(1700, 600, wave_n)
        prev = None
        for i in range(wave_n):
            e0 = ox + 1500 + rng.uniform(-250, 250)
            y = oy + float(lane_y[i]); adv = rng.uniform(-350, -150)
            y_end = y + (200 if i % 2 == 0 else -200)
            wp = [(a, e0, y), (a + 0.5 * life, e0 + adv / 2, (y + y_end) / 2), (b, e0 + adv, y_end)]
            st = [(a, "Approaching"), (a + 0.35 * life, "Engaging")]
            t_exit, tail, exitpred = die_or_exit(a, b)
            st = st + tail
            bn = _pick(rng, ccf_bns)
            rels = [("partOf", bn, "unit"), ("movesToward", lm["crest"], "landmark")]
            if prev is not None:
                rels.append(("follows", prev, "entity"))
            if defender_ids:
                rels.append(("engagedWith", _pick(rng, defender_ids), "entity"))
            if exitpred:
                rels.append((exitpred, lm["crest"], "landmark"))
            evs = [(pe["main"][0], "assault")]
            eid = neid()
            add_ent(WorldEntity(eid, "CCF_Infantry", "CCF", _size_of(ontology, "CCF_Infantry"),
                                bn, s, wp, st, rels, evs, t_enter=a, t_exit=t_exit))
            prev = eid
        # CCF vehicle column reinforcement on this phase
        for _ in range(max(1, counts["veh"])):
            e0 = ox + rng.uniform(-600, -200); n0 = oy + rng.uniform(200, 1400)
            wp = [(a, e0, n0), (b, e0 + rng.uniform(600, 1400), n0 + rng.uniform(-200, 200))]
            add_ent(WorldEntity(neid(), "VehicleColumn", "CCF", _size_of(ontology, "VehicleColumn"),
                                ccf_rgt, s, wp, [(a, "Moving")],
                                [("movesToward", lm["northtip"], "landmark"), ("reinforces", ccf_rgt, "unit")],
                                [(pe["main"][0], "reinforce")], t_enter=a, t_exit=b))

    # CCF artillery (far side, whole window fire support)
    emplaced_support("CCF_Artillery", counts["arty"], 800, 2400, 900, 1500, lm["northslope"], ccf_rgt)

    # static wrecks (pre-existing, Destroyed whole window)
    for _ in range(counts["wreck"]):
        t = _pick(rng, ["US_Tank", "CCF_Infantry", "VehicleColumn"])
        e0 = ox + rng.uniform(-100, 2400); n0 = oy + rng.uniform(-500, 1200)
        add_ent(WorldEntity(neid(), t, AFFILIATION[t], _size_of(ontology, t), None, s,
                            [(0.0, e0, n0), (W, e0, n0)], [(0.0, "Destroyed")],
                            [("near", lm["crest"], "landmark")], [], t_enter=0.0, t_exit=W))

    # relation edges with intervals (structural = whole active window of subject)
    ent_active = {e.eid: (e.t_enter, W if e.t_exit is None else e.t_exit) for e in entities}
    dyn_state = {"occupies": {"Occupying"}, "firesAt": {"Firing"}, "withdrawsFrom": {"Withdrawing"},
                 "engagedWith": {"Engaging"}}
    for e in entities:
        a0, b0 = ent_active[e.eid]
        for pred, target, kind in e.relations:
            if pred in dyn_state:
                t0 = next((float(tf) for tf, stt in e.states if stt in dyn_state[pred]), a0)
                start, end = max(a0, t0), b0
            else:
                start, end = a0, b0
            relations.append(RelationEdge(e.eid, pred, target, kind, start, end))

    return entities, units, landmarks, events, relations
