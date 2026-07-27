"""Real-scale, fixed-space sector builder: unit echelon hierarchy + multi-phase
tactical relations. Space is FIXED (a real ~6x8 km battlefield split into a small
grid); triple volume is grown by NON-spatial means (time density, relations,
events, reification), not by adding sectors.

Per sector this returns entities (each partOf a platoon), the full unit echelon
(regiment->battalion->company->platoon), landmarks, phase events, and
interval-stamped relations (structural partOf chain + per-phase tactical edges).
Lifecycle (entry/exit/attrition/withdrawal) reuses the phase model.
"""
from __future__ import annotations

import numpy as np

from .common import (AFFILIATION, LANDMARK_OFFSETS, SIZE_BY_CATEGORY, Landmark,
                     RelationEdge, Unit, WorldEntity, WorldEvent, roster_counts)


def _pick(rng, pool):
    return pool[int(rng.integers(0, len(pool)))]


def _echelon_tree(side, aff, s, units):
    """regiment -> 3 bn -> 3 co -> 3 pl. Returns list of platoon uids."""
    rgt = f"u_{side}_rgt_{s}"
    units[rgt] = Unit(rgt, "regiment", aff, f"{aff} Regiment", None)
    platoons = []
    for b in range(3):
        bn = f"u_{side}_b{b}_{s}"; units[bn] = Unit(bn, "battalion", aff, f"{aff} Bn {b}", rgt)
        for c in range(3):
            co = f"u_{side}_b{b}c{c}_{s}"; units[co] = Unit(co, "company", aff, f"{aff} Co {b}{c}", bn)
            for p in range(3):
                pl = f"u_{side}_b{b}c{c}p{p}_{s}"
                units[pl] = Unit(pl, "platoon", aff, f"{aff} Pl {b}{c}{p}", co)
                platoons.append(pl)
    return rgt, platoons


def build_sector_real(s, sector_origin_xy, cfg, ontology, rng, eid_ctr, evid_ctr):
    W = float(cfg["time"]["window_seconds"]); P = int(cfg["time"].get("phases", 6))
    ph = W / P
    per_sector = int(cfg["entities"]["per_sector"])
    p_attr = float(cfg["entities"].get("attrition_frac", 0.22))
    reinf = float(cfg["entities"].get("reinforce_frac", 0.5))
    tac_pp = int(cfg["entities"].get("tactical_per_phase", 3))
    # transit-bounded motion: entities RELOCATE only for `transit` seconds, then
    # hold (static). Defaults to the whole window (original behaviour for short
    # configs); long windows (10-day) set a small value so movers don't sit in a
    # moving state for days and blow up trajectory samples. min ensures <= phase.
    transit = min(ph, float(cfg["time"].get("transit_seconds", W)))
    ox, oy = sector_origin_xy
    counts = roster_counts(per_sector)

    def neid():
        eid_ctr[0] += 1; return f"e_{eid_ctr[0]:06d}"

    def nevid():
        evid_ctr[0] += 1; return f"ev_{evid_ctr[0]:06d}"

    entities: list[WorldEntity] = []
    units: dict[str, Unit] = {}
    landmarks: dict[str, Landmark] = {}
    events: list[WorldEvent] = []
    relations: list[RelationEdge] = []

    lm = {}
    for name, (dx, dy) in LANDMARK_OFFSETS.items():
        lid = f"lm_{name}_{s}"; landmarks[lid] = Landmark(lid, name, ox + dx, oy + dy); lm[name] = lid

    rok_rgt, rok_pls = _echelon_tree("rok", "ROK", s, units)
    ccf_rgt, ccf_pls = _echelon_tree("ccf", "CCF", s, units)
    size_of = lambda t: SIZE_BY_CATEGORY.get(ontology.category(t), "unknown")

    # per-phase events (distinct): main (attack/counter) + fire mission each phase,
    # plus a sector-level illuminate. Each is a separate node (no duplication).
    phase_events = []
    for p in range(P):
        a, b = p * ph, (p + 1) * ph
        main = nevid(); events.append(WorldEvent(main, "Attack" if p % 2 == 0 else "Counterattack", s, a, b, lm["crest"], []))
        fire = nevid(); events.append(WorldEvent(fire, "FireSupport", s, a, b, lm["valley"], []))
        occ = nevid(); events.append(WorldEvent(occ, "Occupy", s, a, b, lm["objA"], []))
        phase_events.append({"main": main, "fire": fire, "occ": occ, "a": a, "b": b, "p": p})
    ev_ill = nevid(); events.append(WorldEvent(ev_ill, "Illuminate", s, 0.0, min(W, ph), lm["mlr"], []))
    ev_by_id = {e.evid: e for e in events}

    def add(e):
        entities.append(e)
        for evid, _ in e.events:
            if evid in ev_by_id:
                ev_by_id[evid].participants.append(e.eid)

    def die_or_exit(a, b):
        L = b - a
        if float(rng.random()) < p_attr:
            # killed: becomes a Destroyed wreck (static) and remains to window end
            return W, [(a + rng.uniform(0.5, 0.85) * L, "Destroyed")], None
        # survives: a SHORT withdrawing burst (<= transit) just before exit, so a
        # moving state never spans a whole multi-hour phase.
        return b, [(max(a, b - min(transit, 0.5 * L)), "Withdrawing")], "withdrawsFrom"

    def fire_missions(e, lmk):
        """One DISTINCT fire-mission event per firing (even) phase -> grows the
        event category (growth mean c) without duplication."""
        for p in range(0, P, 2):
            a, b = p * ph, min(W, (p + 1) * ph)
            fm = nevid()
            events.append(WorldEvent(fm, "FireSupport", s, a, b, lmk, [e.eid]))
            e.events.append((fm, "shooter"))

    def tactical(e, active_phases, extra):
        """Emit up to tac_pp phase-stamped tactical relations per active phase."""
        for p in active_phases:
            a, b = p * ph, min(W, (p + 1) * ph)
            pool = list(extra)
            for pred, tgt, kind in pool[:tac_pp]:
                relations.append(RelationEdge(e.eid, pred, tgt, kind, a, b))

    # ---- persistent ROK defence ----
    for _ in range(counts["rok_inf"]):
        pl = _pick(rng, rok_pls)
        e0 = ox + 1500 + rng.uniform(-400, 400); n0 = oy + rng.uniform(-500, 300)
        wp = [(0.0, e0, n0), (W, e0, n0)]                     # defenders hold position
        # all-static schedule (Holding / Firing-in-place) -> sparse sampling; the
        # defender fights in place rather than relocating for days.
        st = [(0.0, "Holding")] + [(p * ph, "Firing" if p % 2 == 1 else "Holding") for p in range(P)]
        t_exit, tail, _ = die_or_exit(0.0, W)
        if tail and tail[0][1] == "Destroyed":
            st = st + tail
        e = WorldEntity(neid(), "ROK_Infantry", "ROK", size_of("ROK_Infantry"), pl, s, wp, st,
                        [("partOf", pl, "unit"), ("emplacedAt", lm["mlr"], "landmark")],
                        [(phase_events[0]["occ"], "defence")], t_enter=0.0, t_exit=t_exit)
        add(e)
        tactical(e, list(range(P)), [("occupies", lm["crest"], "landmark"),
                                     ("firesAt", lm["valley"], "landmark"),
                                     ("firesAt", lm["objA"], "landmark"),
                                     ("supports", rok_rgt, "unit"),
                                     ("screens", lm["mlr"], "landmark"),
                                     ("firesAt", lm["objB"], "landmark")])

    rok_ids = [e.eid for e in entities if e.affiliation == "ROK"]

    def support(t, n_e, e_lo, e_hi, n_lo, n_hi, lmk, rgt):
        for _ in range(n_e):
            e0 = ox + rng.uniform(e_lo, e_hi); n0 = oy + rng.uniform(n_lo, n_hi)
            st = [(0.0, "Emplaced")] + [(p * ph + 0.4 * ph, "Firing" if p % 2 == 0 else "Emplaced") for p in range(P)]
            e = WorldEntity(neid(), t, AFFILIATION.get(t, "ROK"), size_of(t), rgt, s,
                            [(0.0, e0, n0), (W, e0, n0)], st,
                            [("partOf", rgt, "unit"), ("emplacedAt", lmk, "landmark")],
                            [(phase_events[0]["fire"], "shooter")], t_enter=0.0, t_exit=W)
            add(e)
            fire_missions(e, lmk)
            tactical(e, list(range(P)),
                     [("firesAt", lm["valley"], "landmark"), ("firesAt", lm["objA"], "landmark"),
                      ("firesAt", lm["crest"], "landmark"), ("supports", rgt, "unit"),
                      ("screens", lm["mlr"], "landmark"), ("firesAt", lm["objB"], "landmark")])

    for _ in range(counts["tank"]):
        t = _pick(rng, ["US_Tank", "ROK_Tank"])
        e0 = ox + rng.uniform(2500, 3500); n0 = oy + rng.uniform(-600, 200)   # east plain (tank country)
        wp = [(0.0, e0, n0), (transit, e0 + 400, n0 + 60), (W, e0 + 400, n0 + 60)]
        st = [(0.0, "Moving"), (transit, "Halted"), (transit + 0.2 * transit, "Firing")]
        e = WorldEntity(neid(), t, AFFILIATION[t], size_of(t), rok_rgt, s, wp, st,
                        [("partOf", rok_rgt, "unit"), ("firesAt", lm["valley"], "landmark")],
                        [(phase_events[0]["fire"], "shooter")], t_enter=0.0, t_exit=W)
        add(e)
        fire_missions(e, lm["valley"])
        tactical(e, list(range(P)), [("screens", lm["mlr"], "landmark"),
                                     ("supports", rok_rgt, "unit"),
                                     ("firesAt", lm["objA"], "landmark"),
                                     ("firesAt", lm["valley"], "landmark"),
                                     ("firesAt", lm["objB"], "landmark"),
                                     ("screens", lm["crest"], "landmark")])
    support("ROK_Artillery", counts["arty"], 800, 2400, -1400, -700, lm["mlr"], rok_rgt)
    support("Mortar", counts["mortar"], 700, 2300, -1200, -500, lm["mlr"], rok_rgt)
    support("CCF_Artillery", counts["arty"], 800, 2400, 3000, 4000, lm["northslope"], ccf_rgt)
    support("CCF_AA", counts["aa"], 600, 2400, 2500, 3500, lm["northslope"], ccf_rgt)
    for _ in range(counts["engr"]):
        e0 = ox + rng.uniform(-300, 300); n0 = oy + rng.uniform(-200, 600)
        e = WorldEntity(neid(), "Engineer", "ROK", size_of("Engineer"), rok_rgt, s,
                        [(0.0, e0, n0), (transit, e0 + rng.uniform(-150, 150), n0 + rng.uniform(-150, 150)), (W, e0, n0)],
                        [(0.0, "Moving"), (transit, "Halted")],
                        [("partOf", rok_rgt, "unit"), ("near", lm["yokkok"], "landmark")],
                        [(phase_events[0]["occ"], "engineer")], t_enter=0.0, t_exit=W)
        add(e)
    e0 = ox + rng.uniform(200, 1200); n0 = oy + rng.uniform(-1400, -900)
    add(WorldEntity(neid(), "Searchlight", "US", size_of("Searchlight"), rok_rgt, s,
                    [(0.0, e0, n0), (W, e0, n0)], [(0.0, "Emplaced"), (0.4 * ph, "Firing")],
                    [("partOf", rok_rgt, "unit"), ("supports", rok_rgt, "unit")],
                    [(ev_ill, "illuminate")], t_enter=0.0, t_exit=min(W, ph)))

    # ---- CCF assault waves (entry/exit/attrition/reinforcement per assault phase) ----
    wave_n = max(3, round(counts["ccf_inf"] * reinf))
    for p in range(0, P, 2):
        a = p * ph; life = min(W - a, 1.6 * ph); b = a + life
        pe = phase_events[p]
        lane_y = np.linspace(3500, 1200, wave_n)         # from north (ridge) down to crest
        prev = None
        for i in range(wave_n):
            e0 = ox + 1500 + rng.uniform(-250, 250); y = oy + float(lane_y[i])
            adv = rng.uniform(-350, -150); y_end = y + (200 if i % 2 == 0 else -200)
            tr = min(transit, 0.6 * life)                    # assault transit, then hold
            wp = [(a, e0, y), (a + tr, e0 + adv, y_end), (b, e0 + adv, y_end)]
            st = [(a, "Approaching"), (a + tr, "Occupying")]  # move in, then hold in place
            t_exit, tail, exitpred = die_or_exit(a, b)
            st = st + tail
            pl = _pick(rng, ccf_pls)
            rels = [("partOf", pl, "unit"), ("movesToward", lm["crest"], "landmark")]
            if prev is not None:
                rels.append(("follows", prev, "entity"))
            eid = neid()
            e = WorldEntity(eid, "CCF_Infantry", "CCF", size_of("CCF_Infantry"), pl, s, wp, st, rels,
                            [(pe["main"], "assault")], t_enter=a, t_exit=t_exit)
            add(e)
            extra = []
            if rok_ids:
                extra += [("engagedWith", _pick(rng, rok_ids), "entity"),
                          ("engagedWith", _pick(rng, rok_ids), "entity")]
            extra += [("movesToward", lm["crest"], "landmark"),
                      ("movesToward", lm["objA"], "landmark"),
                      ("firesAt", lm["crest"], "landmark"),
                      ("occupies", lm["objB"], "landmark")]
            if exitpred:
                extra.append((exitpred, lm["crest"], "landmark"))
            tactical(e, [p], extra)
            prev = eid
        for _ in range(max(1, counts["veh"])):
            e0 = ox + rng.uniform(-600, -200); n0 = oy + rng.uniform(2000, 3500)
            tr = min(transit, 0.6 * (b - a))
            add(WorldEntity(neid(), "VehicleColumn", "CCF", size_of("VehicleColumn"), ccf_rgt, s,
                            [(a, e0, n0), (a + tr, e0 + rng.uniform(600, 1400), n0 + rng.uniform(-200, 200)), (b, e0, n0)],
                            [(a, "Moving"), (a + tr, "Halted")],
                            [("partOf", ccf_rgt, "unit"), ("reinforces", ccf_rgt, "unit"),
                             ("movesToward", lm["northtip"], "landmark")],
                            [(pe["main"], "reinforce")], t_enter=a, t_exit=b))

    for _ in range(counts["wreck"]):
        t = _pick(rng, ["US_Tank", "CCF_Infantry", "VehicleColumn"])
        e0 = ox + rng.uniform(-100, 2400); n0 = oy + rng.uniform(-500, 1200)
        add(WorldEntity(neid(), t, AFFILIATION[t], size_of(t), None, s,
                        [(0.0, e0, n0), (W, e0, n0)], [(0.0, "Destroyed")],
                        [("near", lm["crest"], "landmark")], [], t_enter=0.0, t_exit=W))

    # structural relations (partOf entity->platoon/regiment + emplacedAt/near etc.)
    # over the whole active window (dynamic tactical ones already added by tactical()).
    ent_active = {e.eid: (e.t_enter, W if e.t_exit is None else e.t_exit) for e in entities}
    for e in entities:
        a0, b0 = ent_active[e.eid]
        for pred, target, kind in e.relations:
            relations.append(RelationEdge(e.eid, pred, target, kind, a0, b0))

    return entities, units, landmarks, events, relations
