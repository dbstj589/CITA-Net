"""P1 — world entities, unit org, landmarks and event skeletons.

Roster ratios and movement patterns mirror the suite generator's Hill-395
sub-engagement (CCF assault from the north tip toward the crest, ROK defence on
the MLR/crest band to the south, armour screening the flanks, artillery/mortars
emplaced on defensible ground), but everything is produced in the GLOBAL frame
with global entity IDs and no sensor/observation notion (no tracks, no dangling,
no noise). Waypoints stay well within v_max*window; the sampler additionally
clamps every step, so motion feasibility holds by construction.
"""
from __future__ import annotations

import numpy as np

from .common import (AFFILIATION, EVENT_KINDS, LANDMARK_OFFSETS, SIZE_BY_CATEGORY,
                     Landmark, Unit, WorldEntity, WorldEvent, roster_counts,
                     sector_origin)


def _pick(rng, pool):
    return pool[int(rng.integers(0, len(pool)))]


def build_world(cfg: dict, ontology, rng: np.random.Generator):
    """Return (entities, units, landmarks, events). All IDs/coords/times global."""
    W = float(cfg["time"]["window_seconds"])
    f = W / 600.0                              # scale the canonical 600 s schedule
    scope = cfg["world_scope"]
    n_sectors = int(scope.get("n_sectors", 1)) if scope.get("mode", "tiled") == "tiled" else 1
    pitch = float(scope.get("sector_pitch_m", 6000.0))
    per_sector = int(cfg["entities"]["per_sector"])

    def size_of(t):
        return SIZE_BY_CATEGORY.get(ontology.category(t), "unknown")

    entities: list[WorldEntity] = []
    units: dict[str, Unit] = {}
    landmarks: dict[str, Landmark] = {}
    events: list[WorldEvent] = []

    ecount = [0]
    evcount = [0]

    def new_eid():
        ecount[0] += 1
        return f"e_{ecount[0]:05d}"

    def new_evid():
        evcount[0] += 1
        return f"ev_{evcount[0]:04d}"

    for s in range(n_sectors):
        ox, oy = sector_origin(s, pitch)
        counts = roster_counts(per_sector)

        # --- landmarks (global ids per sector) ---
        lm = {}
        for name, (dx, dy) in LANDMARK_OFFSETS.items():
            lid = f"lm_{name}_{s}"
            landmarks[lid] = Landmark(lid, name, ox + dx, oy + dy)
            lm[name] = lid

        # --- unit org ---
        rok_rgt = f"u_rok_rgt_{s}"; ccf_rgt = f"u_ccf_rgt_{s}"
        units[rok_rgt] = Unit(rok_rgt, "regiment", "ROK", "ROK Regiment (defence)")
        units[ccf_rgt] = Unit(ccf_rgt, "regiment", "CCF", "CCF Regiment (assault)")
        rok_bns, ccf_bns = [], []
        for i in range(max(1, counts["rok_inf"] // 9)):
            u = f"u_rok_bn{i}_{s}"; units[u] = Unit(u, "battalion", "ROK", f"ROK Infantry Bn {i}", rok_rgt); rok_bns.append(u)
        for i in range(max(1, counts["ccf_inf"] // 9)):
            u = f"u_ccf_bn{i}_{s}"; units[u] = Unit(u, "battalion", "CCF", f"CCF Infantry Bn {i}", ccf_rgt); ccf_bns.append(u)

        # --- events (first-class, fixed intervals within the window) ---
        ev = {}
        for kind, (a, b, lmk) in {
            "Attack":       (0.0,      300.0 * f, lm["crest"]),
            "Counterattack": (260.0 * f, W,        lm["crest"]),
            "FireSupport":  (240.0 * f, W,        lm["valley"]),
            "Occupy":       (260.0 * f, W,        lm["objA"]),
            "Illuminate":   (0.0,      200.0 * f, lm["mlr"]),
        }.items():
            evid = new_evid()
            events.append(WorldEvent(evid, kind, s, a, b, lmk, []))
            ev[kind] = evid

        def add(e):
            entities.append(e)

        # --- CCF infantry assault: same-type mass, north tip -> crest, crossing lanes ---
        prev = None
        lane_y = np.linspace(1700, 600, counts["ccf_inf"])
        for i in range(counts["ccf_inf"]):
            e0 = ox + 1500 + rng.uniform(-250, 250)
            y = oy + float(lane_y[i])
            adv = rng.uniform(-350, -150)
            y_end = y + (200 if i % 2 == 0 else -200)
            wp = [(0.0, e0, y), (300.0 * f, e0 + adv / 2, (y + y_end) / 2), (W, e0 + adv, y_end)]
            st = [(0.0, "Approaching"), (250.0 * f, "Engaging")]
            bn = _pick(rng, ccf_bns)
            rels = [("partOf", bn, "unit"), ("movesToward", lm["crest"], "landmark")]
            if prev is not None:
                rels.append(("follows", prev, "entity"))
            eid = new_eid()
            add(WorldEntity(eid, "CCF_Infantry", AFFILIATION["CCF_Infantry"], size_of("CCF_Infantry"),
                            bn, s, wp, st, rels, [(ev["Attack"], "assault")]))
            prev = eid

        # --- ROK infantry defence on the MLR/crest band ---
        for _ in range(counts["rok_inf"]):
            bn = _pick(rng, rok_bns)
            e0 = ox + 1500 + rng.uniform(-400, 400); n0 = oy + rng.uniform(-500, 300)
            if float(rng.random()) < 0.25:                       # counterattacking
                wp = [(0.0, e0, n0), (W, e0 + rng.uniform(-80, 80), n0 + rng.uniform(120, 320))]
                st = [(0.0, "Holding"), (260.0 * f, "Occupying")]
                rels = [("partOf", bn, "unit"), ("occupies", lm["crest"], "landmark")]
                evref = (ev["Counterattack"], "counterattack")
            else:                                                # holding the line
                wp = [(0.0, e0, n0), (W, e0 + rng.uniform(-40, 40), n0 + rng.uniform(-40, 40))]
                st = [(0.0, "Holding"), (300.0 * f, "Engaging")]
                rels = [("partOf", bn, "unit"), ("emplacedAt", lm["mlr"], "landmark")]
                evref = (ev["Occupy"], "defence")
            add(WorldEntity(new_eid(), "ROK_Infantry", "ROK", size_of("ROK_Infantry"),
                            bn, s, wp, st, rels, [evref]))

        # --- armour: screen flanks, fire the valley ---
        for _ in range(counts["tank"]):
            t = _pick(rng, ["US_Tank", "ROK_Tank"])
            e0 = ox + rng.uniform(100, 500); n0 = oy + rng.uniform(-600, 200)
            wp = [(0.0, e0, n0), (200.0 * f, e0 + 300, n0 + 60), (W, e0 + 300, n0 + 60)]
            st = [(0.0, "Moving"), (200.0 * f, "Halted"), (260.0 * f, "Firing")]
            rels = [("supports", rok_rgt, "unit"), ("firesAt", lm["valley"], "landmark"),
                    ("screens", lm["mlr"], "landmark")]
            add(WorldEntity(new_eid(), t, AFFILIATION[t], size_of(t), rok_rgt, s, wp, st, rels,
                            [(ev["FireSupport"], "shooter")]))

        # --- artillery: emplaced -> firing ---
        for _ in range(counts["arty"]):
            t = _pick(rng, ["ROK_Artillery", "US_Artillery", "CCF_Artillery"])
            far = t == "CCF_Artillery"
            e0 = ox + rng.uniform(800, 2400)
            n0 = oy + (rng.uniform(900, 1500) if far else rng.uniform(-1400, -700))
            wp = [(0.0, e0, n0), (W, e0, n0)]
            st = [(0.0, "Emplaced"), (240.0 * f, "Firing")]
            rgt = ccf_rgt if far else rok_rgt
            rels = [("emplacedAt", lm["northslope"] if far else lm["mlr"], "landmark"),
                    ("supports", rgt, "unit")]
            add(WorldEntity(new_eid(), t, AFFILIATION[t], size_of(t), rgt, s, wp, st, rels,
                            [(ev["FireSupport"], "shooter")]))

        # --- mortars ---
        for _ in range(counts["mortar"]):
            e0 = ox + rng.uniform(700, 2300); n0 = oy + rng.uniform(-1200, -500)
            wp = [(0.0, e0, n0), (W, e0, n0)]
            st = [(0.0, "Emplaced"), (260.0 * f, "Firing")]
            rels = [("emplacedAt", lm["mlr"], "landmark"), ("supports", rok_rgt, "unit")]
            add(WorldEntity(new_eid(), "Mortar", "ROK", size_of("Mortar"), rok_rgt, s, wp, st, rels,
                            [(ev["FireSupport"], "shooter")]))

        # --- AA (CCF towed / mobile quad-50) ---
        for _ in range(counts["aa"]):
            t = _pick(rng, ["CCF_AA", "QuadFifty"])
            own = t == "QuadFifty"
            e0 = ox + rng.uniform(600, 2400)
            n0 = oy + (rng.uniform(-300, 200) if own else rng.uniform(1000, 1600))
            if own:
                wp = [(0.0, e0, n0), (180.0 * f, e0 + 40, n0), (W, e0 + 40, n0)]
                st = [(0.0, "Halted"), (200.0 * f, "Firing")]
            else:
                wp = [(0.0, e0, n0), (W, e0, n0)]
                st = [(0.0, "Emplaced"), (240.0 * f, "Firing")]
            rgt = rok_rgt if own else ccf_rgt
            rels = [("emplacedAt", lm["mlr"] if own else lm["northslope"], "landmark"),
                    ("supports", rgt, "unit")]
            add(WorldEntity(new_eid(), t, AFFILIATION[t], size_of(t), rgt, s, wp, st, rels,
                            [(ev["FireSupport"], "shooter")]))

        # --- engineers ---
        for _ in range(counts["engr"]):
            e0 = ox + rng.uniform(-300, 300); n0 = oy + rng.uniform(-200, 600)
            wp = [(0.0, e0, n0), (W, e0 + rng.uniform(-150, 150), n0 + rng.uniform(-150, 150))]
            st = [(0.0, "Moving")]
            rels = [("near", lm["yokkok"], "landmark"), ("supports", rok_rgt, "unit")]
            add(WorldEntity(new_eid(), "Engineer", "ROK", size_of("Engineer"), rok_rgt, s, wp, st, rels,
                            [(ev["Occupy"], "engineer")]))

        # --- vehicle columns (CCF supply on the rear road) ---
        for _ in range(counts["veh"]):
            e0 = ox + rng.uniform(-600, -200); n0 = oy + rng.uniform(200, 1400)
            wp = [(0.0, e0, n0), (W, e0 + rng.uniform(1200, 2200), n0 + rng.uniform(-200, 200))]
            st = [(0.0, "Moving")]
            rels = [("movesToward", lm["northtip"], "landmark"), ("reinforces", ccf_rgt, "unit")]
            add(WorldEntity(new_eid(), "VehicleColumn", "CCF", size_of("VehicleColumn"), ccf_rgt, s, wp, st, rels,
                            [(ev["Attack"], "reinforce")]))

        # --- searchlight / illumination ---
        e0 = ox + rng.uniform(200, 1200); n0 = oy + rng.uniform(-1400, -900)
        add(WorldEntity(new_eid(), "Searchlight", AFFILIATION["Searchlight"], size_of("Searchlight"),
                        rok_rgt, s, [(0.0, e0, n0), (W, e0, n0)],
                        [(0.0, "Emplaced"), (200.0 * f, "Firing")],
                        [("supports", rok_rgt, "unit")], [(ev["Illuminate"], "illuminate")]))

        # --- destroyed wrecks (static, no events) ---
        for _ in range(counts["wreck"]):
            t = _pick(rng, ["US_Tank", "CCF_Infantry", "VehicleColumn"])
            e0 = ox + rng.uniform(-100, 2400); n0 = oy + rng.uniform(-500, 1200)
            add(WorldEntity(new_eid(), t, AFFILIATION[t], size_of(t), None, s,
                            [(0.0, e0, n0), (W, e0, n0)], [(0.0, "Destroyed")],
                            [("near", lm["crest"], "landmark")], []))

    # fill event participants from entity event refs
    ev_index = {e.evid: e for e in events}
    for ent in entities:
        for evid, _role in ent.events:
            ev_index[evid].participants.append(ent.eid)

    return entities, units, landmarks, events
