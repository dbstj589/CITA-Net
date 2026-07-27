"""Micro single-engagement scenario — readable, ~40 objects that actually move,
with an explicit CAUSAL CHAIN per assault wave:

   1) CCF squad Approaches up the approach lane toward the crest        (move)
   2) ROK observer/squad detects it   -> near / engagedWith             (interval @ contact)
   3) ROK requests support from artillery -> supports                   (overlaps 2->4)
   4) Artillery runs a FireSupport event on the lane -> firesAt         (interval after 3)
   5) CCF is stopped: Halted, then Withdrawing or Destroyed             (state change after 4)

Relation validFrom/validTo, event intervals and the participants' state-change
times are laid out in this causal order so a renderer shows one flowing chain.
Single area (no sectors), global coords aligned to the terrain peak. This is a
DESIGNED scenario logic, not a reconstruction of real battle causality.
"""
from __future__ import annotations

import numpy as np

from .common import (SIZE_BY_CATEGORY, Landmark, RelationEdge, Unit, WorldEntity,
                     WorldEvent)


def build_micro(cfg, ontology, terrain, rng):
    W = float(cfg["time"]["window_seconds"])
    ecfg = cfg["entities"]
    n_waves = int(ecfg.get("waves", 4))
    squads_per_wave = int(ecfg.get("squads_per_wave", 5))
    n_rok = int(ecfg.get("rok_infantry", 8))
    size_of = lambda t: SIZE_BY_CATEGORY.get(ontology.category(t), "unknown")

    # --- landmarks aligned to the ACTUAL terrain peak (argmax of the field) ---
    ei, ej = np.unravel_index(int(np.argmax(terrain.elevation)), terrain.elevation.shape)
    px = terrain.origin_e + ej * terrain.res
    py = terrain.origin_n + ei * terrain.res
    L = {
        "crest": (px, py),
        "mlr": (px, py - 700),               # ROK defence line, south of the crest
        "rear": (px, py - 1100),             # ROK artillery, further rear
        "ridge": (px, py + 400),             # ridge extends north
        "approach": (px, py + 900),          # enemy approach lane, north
        "approach_w": (px - 450, py + 750),
        "approach_e": (px + 450, py + 750),
        "valley": (px - 750, py + 150),
    }
    landmarks = {f"lm_{k}": Landmark(f"lm_{k}", k, x, y, terrain.elev_at(x, y)) for k, (x, y) in L.items()}
    lm = {k: f"lm_{k}" for k in L}

    units = {}
    rok = "u_rok_coy"; ccf = "u_ccf_coy"
    units[rok] = Unit(rok, "company", "ROK", "ROK Rifle Company (defence)")
    units[ccf] = Unit(ccf, "company", "CCF", "CCF Rifle Company (assault)")
    rok_pl = "u_rok_pl"; units[rok_pl] = Unit(rok_pl, "platoon", "ROK", "ROK Rifle Platoon", rok)
    ccf_pl = "u_ccf_pl"; units[ccf_pl] = Unit(ccf_pl, "platoon", "CCF", "CCF Rifle Platoon", ccf)

    entities, events, relations = [], [], []
    ec = [0]; vc = [0]
    def neid():
        ec[0] += 1; return f"e_{ec[0]:04d}"
    def nevid():
        vc[0] += 1; return f"ev_{vc[0]:04d}"
    ev_by = {}
    def add(e):
        entities.append(e)
        for evid, _ in e.events:
            if evid in ev_by:
                ev_by[evid].participants.append(e.eid)
    def rel(subj, pred, obj, kind, a, b):
        relations.append(RelationEdge(subj, pred, obj, kind, a, b))

    (cx, cy) = L["crest"]; (mx, my) = L["mlr"]; (ax, ay) = L["approach"]; (rx, ry) = L["rear"]

    # ---------- ROK defenders (persistent) ----------
    rok_ids = []
    obs = neid()                                     # forward observer on the crest
    add(WorldEntity(obs, "ROK_Infantry", "ROK", size_of("ROK_Infantry"), rok_pl, 0,
                    [(0.0, cx, cy), (W, cx, cy)], [(0.0, "Holding")],
                    [("partOf", rok_pl, "unit"), ("occupies", lm["crest"], "landmark")], [], 0.0, W))
    rok_ids.append(obs)
    for i in range(n_rok):
        e0 = mx + rng.uniform(-350, 350); n0 = my + rng.uniform(-120, 180)
        eid = neid(); rok_ids.append(eid)
        add(WorldEntity(eid, "ROK_Infantry", "ROK", size_of("ROK_Infantry"), rok_pl, 0,
                        [(0.0, e0, n0), (W, e0, n0)], [(0.0, "Holding")],
                        [("partOf", rok_pl, "unit"), ("emplacedAt", lm["mlr"], "landmark")], [], 0.0, W))
    # supporting weapons (emplaced, will fire on request)
    arty_ids, mortar_ids = [], []
    for _ in range(2):
        e0 = rx + rng.uniform(-300, 300); n0 = ry + rng.uniform(-150, 150)
        eid = neid(); arty_ids.append(eid)
        add(WorldEntity(eid, "ROK_Artillery", "ROK", size_of("ROK_Artillery"), rok, 0,
                        [(0.0, e0, n0), (W, e0, n0)], [(0.0, "Emplaced")],
                        [("partOf", rok, "unit"), ("emplacedAt", lm["rear"], "landmark"),
                         ("supports", rok_pl, "unit")], [], 0.0, W))
    for _ in range(2):
        e0 = mx + rng.uniform(-400, 400); n0 = my - 250 + rng.uniform(-100, 100)
        eid = neid(); mortar_ids.append(eid)
        add(WorldEntity(eid, "Mortar", "ROK", size_of("Mortar"), rok_pl, 0,
                        [(0.0, e0, n0), (W, e0, n0)], [(0.0, "Emplaced")],
                        [("partOf", rok_pl, "unit"), ("emplacedAt", lm["mlr"], "landmark"),
                         ("supports", rok_pl, "unit")], [], 0.0, W))
    # a screening tank on the flank
    tk = neid()
    add(WorldEntity(tk, "US_Tank", "US", size_of("US_Tank"), rok, 0,
                    [(0.0, cx + 700, my), (0.2 * W, cx + 500, my + 200), (W, cx + 500, my + 200)],
                    [(0.0, "Moving"), (0.2 * W, "Halted"), (0.25 * W, "Firing")],
                    [("partOf", rok, "unit"), ("screens", lm["mlr"], "landmark")], [], 0.0, W))
    # searchlight
    sl = neid()
    add(WorldEntity(sl, "Searchlight", "US", size_of("Searchlight"), rok, 0,
                    [(0.0, mx - 500, my - 300), (W, mx - 500, my - 300)],
                    [(0.0, "Emplaced"), (0.5 * W, "Firing")],
                    [("partOf", rok, "unit"), ("supports", rok, "unit")], [], 0.0, W))
    # pre-existing wrecks
    for _ in range(2):
        e0 = ax + rng.uniform(-300, 300); n0 = (ay + cy) / 2 + rng.uniform(-200, 200)
        add(WorldEntity(neid(), "CCF_Infantry", "CCF", size_of("CCF_Infantry"), None, 0,
                        [(0.0, e0, n0), (W, e0, n0)], [(0.0, "Destroyed")],
                        [("near", lm["crest"], "landmark")], [], 0.0, W))

    # ---------- CCF assault waves with the causal chain ----------
    wave_dt = W / n_waves
    chain_example = None
    for w in range(n_waves):
        t0 = w * wave_dt
        t_move_end = t0 + 2100.0          # 35 min approach
        t_detect = t0 + 1200.0            # observed ~20 min in
        t_fire0 = t0 + 1500.0             # fire mission starts ~25 min
        t_fire1 = t0 + 2700.0             # fire mission ends ~45 min
        t_exit = min(W, t0 + 3600.0)      # cycle ~60 min
        lane = [lm["approach"], lm["approach_w"], lm["approach_e"]][w % 3]
        lx, ly = L[lane[3:]]

        # fire-support event for this wave (the 4th link of the chain)
        fm = nevid(); events.append(WorldEvent(fm, "FireSupport", 0, t_fire0, t_fire1, lane, []))
        ev_by[fm] = events[-1]
        atk = nevid(); events.append(WorldEvent(atk, "Attack", 0, t0, t_move_end, lm["crest"], []))
        ev_by[atk] = events[-1]

        wave_squads = []
        for i in range(squads_per_wave):
            sx = lx + rng.uniform(-200, 200); sy = ly + rng.uniform(-150, 150)
            tx = cx + rng.uniform(-250, 250); ty = cy + 150 + rng.uniform(-100, 100)
            killed = rng.random() < 0.5
            if killed:
                st = [(t0, "Approaching"), (t_move_end, "Halted"), (t_fire1, "Destroyed")]
                t_ex = W                      # wreck remains
                wp = [(t0, sx, sy), (t_move_end, tx, ty), (W, tx, ty)]
            else:
                st = [(t0, "Approaching"), (t_move_end, "Halted"), (t_fire1, "Withdrawing")]
                t_ex = t_exit
                wp = [(t0, sx, sy), (t_move_end, tx, ty), (t_fire1, tx, ty), (t_ex, sx, sy)]
            eid = neid(); wave_squads.append(eid)
            defender = rok_ids[(w * squads_per_wave + i) % len(rok_ids)]
            e = WorldEntity(eid, "CCF_Infantry", "CCF", size_of("CCF_Infantry"), ccf_pl, 0, wp, st,
                            [("partOf", ccf_pl, "unit")], [(atk, "assault")], t0, t_ex)
            add(e)
            # link 1: approach (movement + movesToward)
            rel(eid, "movesToward", lm["crest"], "landmark", t0, t_move_end)
            # link 2: detected / engaged by a defender
            rel(defender, "engagedWith", eid, "entity", t_detect, t_fire1)
            rel(eid, "engagedWith", defender, "entity", t_detect, t_fire1)
            # link 5: stopped -> withdraws (survivors)
            if not killed:
                rel(eid, "withdrawsFrom", lm["crest"], "landmark", t_fire1, t_ex)

        # link 2b: forward observer watches the lane
        rel(obs, "near", lane, "landmark", t_detect, t_fire1)
        # link 4: artillery + mortars fire on the lane (after the request)
        for aid in arty_ids + mortar_ids:
            rel(aid, "firesAt", lane, "landmark", t_fire0, t_fire1)
            ev_by[fm].participants.append(aid)
            # attach the fire-mission event to the shooter
            for e in entities:
                if e.eid == aid:
                    e.events.append((fm, "shooter"))
        # link 3: the engaged ROK squad requests/relies on artillery support (overlaps 2->4)
        rel(defender, "supports", rok, "unit", t_detect, t_fire0)

        # ROK counterattack after the wave is repulsed (defenders actually move)
        if w % 2 == 1:
            cnt = nevid(); events.append(WorldEvent(cnt, "Counterattack", 0, t_fire1, t_exit, lm["crest"], []))
            ev_by[cnt] = events[-1]
            mover = rok_ids[1 + (w % (len(rok_ids) - 1))]
            for e in entities:
                if e.eid == mover:
                    e.waypoints = [(0.0, e.waypoints[0][1], e.waypoints[0][2]),
                                   (t_fire1, e.waypoints[0][1], e.waypoints[0][2]),
                                   (t_exit, cx, cy - 100), (W, cx, cy - 100)]
                    e.states = [(0.0, "Holding"), (t_fire1, "Occupying")]
                    e.events.append((cnt, "counterattack"))
                    ev_by[cnt].participants.append(mover)
                    rel(mover, "occupies", lm["crest"], "landmark", t_fire1, t_exit)

        if chain_example is None:
            chain_example = {"wave": w, "ccf_squad": wave_squads[0], "defender": rok_ids[0],
                             "shooters": arty_ids + mortar_ids, "fire_event": fm,
                             "t_move": [t0, t_move_end], "t_detect": [t_detect, t_fire1],
                             "t_fire": [t_fire0, t_fire1]}

    return entities, units, landmarks, events, relations, chain_example
