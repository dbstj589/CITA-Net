"""Simulation core for the two-layer 10-day battle GT (see scenario_battle10d).

v2: every agent accumulates a KEYFRAME timeline (t, x, y, state) across ALL the
director ops it participates in — a reused CCF echelon gets a fresh round-trip
(boundary -> approach -> crest -> withdraw off-map / destroyed) for EACH attack it
is committed to, and ROK reserves get advance/counterattack/9-bu round-trips. So
every attack/counterattack has real infantry motion, not just events/relations.
Combat power s is recorded per-sample; only meaningful threshold transitions are
reified. Friendly fire is clipped out of the D3 fog window.
"""
from __future__ import annotations

import numpy as np

from .common import Landmark, Unit, SIZE_BY_CATEGORY
from .scenario_battle10d import Agent, build_schedule, DAY


def _lm(terrain):
    ei, ej = np.unravel_index(int(np.argmax(terrain.elevation)), terrain.elevation.shape)
    px = terrain.origin_e + ej * terrain.res
    py = terrain.origin_n + ei * terrain.res
    ext_n = terrain.origin_n + terrain.res * (terrain.elevation.shape[0] - 1)
    by = min(ext_n - 30, py + 1500)                       # north boundary (edge)
    return px, py, {
        "crest": (px, py), "mlr": (px, py - 700), "rear": (px, py - 1100),
        "ninebu": (px + 250, py - 350), "ridge": (px, py + 400), "northtip": (px, py + 700),
        "approach": (px, py + 900), "approach_w": (px - 450, py + 750),
        "approach_e": (px + 450, py + 750), "valley": (px - 750, py + 150),
        "boundary": (px, by),
    }


def simulate(cfg, ontology, terrain, rng):
    W = float(cfg["time"]["window_seconds"]); dt = float(cfg["time"]["dt_seconds"])
    ec = cfg["entities"]
    fmin = float(ec.get("fire_mission_minutes", 0.0))     # >0 -> granular fire missions (knob 3)
    px, py, L = _lm(terrain)
    ops, fog = build_schedule(W)
    def fire_ok(t):
        return not any(a <= t <= b for a, b in fog)
    def clip_fog(a, b):
        """Return sub-intervals of [a,b] with the fog windows removed (friendly fire)."""
        segs = [(a, b)]
        for fa, fb in fog:
            out = []
            for sa, sb in segs:
                if fb <= sa or fa >= sb:
                    out.append((sa, sb))
                else:
                    if sa < fa:
                        out.append((sa, min(fa, sb)))
                    if sb > fb:
                        out.append((max(fb, sa), sb))
            segs = out
        return [(round(sa, 1), round(sb, 1)) for sa, sb in segs if sb - sa > 1.0]

    landmarks = {f"lm_{k}": Landmark(f"lm_{k}", k, x, y, terrain.elev_at(x, y)) for k, (x, y) in L.items()}
    lmid = {k: f"lm_{k}" for k in L}
    units = {}
    rok = "u_rok_regt"; units[rok] = Unit(rok, "regiment", "ROK", "ROK Regiment (defence)")
    rok_pl = "u_rok_pl"; units[rok_pl] = Unit(rok_pl, "platoon", "ROK", "ROK Rifle Platoon", rok)
    rok_res = "u_rok_reserve"; units[rok_res] = Unit(rok_res, "platoon", "ROK", "ROK Reserve Platoon", rok)
    ccf = "u_ccf_regt"; units[ccf] = Unit(ccf, "regiment", "CCF", "CCF Regiment (assault)")
    n_ech = int(ec.get("ccf_echelons", 7)); spe = int(ec.get("squads_per_echelon", 5))
    ech_units = []
    for i in range(n_ech):
        u = f"u_ccf_ech{i}"; units[u] = Unit(u, "company", "CCF", f"CCF Echelon {i}", ccf); ech_units.append(u)

    ecount = [0]
    def neid():
        ecount[0] += 1; return f"e_{ecount[0]:04d}"
    agents = {}
    def mk(typ, aff, unit, role, x, y, ech="-"):
        a = Agent(neid(), typ, aff, unit, role, ech, x, y, (x, y))
        a.kf = [(0.0, x, y, "Holding")]; a.s_kf = [(0.0, 1.0)]; a.dead_at = None; a.occ_pos = None
        agents[a.eid] = a; return a
    def kf(a, t, x, y, st):
        a.kf.append((round(float(t), 1), float(x), float(y), st))
    def skf(a, t):
        a.s_kf.append((round(float(t), 1), round(float(a.s), 3)))

    rok_fwd = [mk("ROK_Infantry", "ROK", rok_pl, "fwd", px + rng.uniform(-300, 300), py - 180 + rng.uniform(-150, 250)) for _ in range(int(ec.get("rok_fwd", 6)))]
    rok_reserve = [mk("ROK_Infantry", "ROK", rok_res, "reserve", px + rng.uniform(-250, 250), py - 1000 + rng.uniform(-120, 120)) for _ in range(int(ec.get("rok_reserve", 6)))]
    rok_arty = [mk("ROK_Artillery", "ROK", rok, "arty", px + rng.uniform(-250, 250), py - 1100 + rng.uniform(-120, 120)) for _ in range(2)]
    rok_mortar = [mk("Mortar", "ROK", rok_pl, "mortar", px + rng.uniform(-350, 350), py - 550 + rng.uniform(-100, 100)) for _ in range(2)]
    for a in rok_arty + rok_mortar:
        a.state = "Emplaced"; a.kf = [(0.0, a.x, a.y, "Emplaced")]
    tank = mk("US_Tank", "US", rok, "tank", px + 700, py - 200); tank.state = "Halted"; tank.kf = [(0.0, tank.x, tank.y, "Halted")]
    light = mk("Searchlight", "US", rok, "light", px - 500, py - 400); light.state = "Emplaced"; light.kf = [(0.0, light.x, light.y, "Emplaced")]

    axes = {0: "approach", 1: "approach_w", 2: "approach_e"}
    ccf_squads = {i: [] for i in range(n_ech)}
    bx, by = L["boundary"]
    for i in range(n_ech):
        for k in range(spe):
            a = mk("CCF_Infantry", "CCF", ech_units[i], "assault", bx + rng.uniform(-400, 400), by + 150 + rng.uniform(0, 300), ech=ech_units[i])
            a.state = "Holding"; a.kf = [(0.0, a.x, a.y, "Holding")]
            ccf_squads[i].append(a)
    ccf_arty = [mk("CCF_Artillery", "CCF", ccf, "arty", bx + rng.uniform(-300, 300), by + 200) for _ in range(2)]
    for a in ccf_arty:
        a.state = "Emplaced"; a.kf = [(0.0, a.x, a.y, "Emplaced")]
    for _ in range(2):
        a = mk("CCF_Infantry", "CCF", None, "wreck", px + rng.uniform(-300, 300), py + 400 + rng.uniform(-150, 150))
        a.state = "Destroyed"; a.s = 0.0; a.alive = False; a.kf = [(0.0, a.x, a.y, "Destroyed")]

    events, relations, transitions = [], [], []
    evc = [0]
    def nev(kind, a, b, lmk, parts, prov="record"):
        evc[0] += 1; eid = f"ev_{evc[0]:04d}"
        events.append(dict(id=eid, type=kind, interval=[round(a, 1), round(b, 1)], participants=list(parts), landmark=lmk, prov=prov)); return eid
    def rel(sub, pred, obj, kind, a, b):
        relations.append(dict(subject=sub, predicate=pred, object=obj, object_kind=kind, interval=[round(a, 1), round(b, 1)]))
    def rel_clip(sub, pred, obj, kind, a, b):     # friendly fire clipped out of fog
        for sa, sb in clip_fog(a, b):
            rel(sub, pred, obj, kind, sa, sb)
    def trans(eid, t, kind, frm, to, s):
        transitions.append(dict(entity=eid, t=round(t, 1), kind=kind, from_state=frm, to_state=to, s=round(s, 3)))

    def fire_missions(shooters, lmk, a, b, prov, friendly):
        """Emit fire as granular short missions (knob 3: source cites ~275k shells
        over 10 days -> mission-level granularity is realism, not padding). Friendly
        fire is clipped out of the fog window (interval level)."""
        segs = clip_fog(a, b) if friendly else [(a, b)]
        step = fmin * 60.0 if fmin > 0 else max(1.0, b - a)
        for sa, sb in segs:
            t = sa
            while t < sb - 1.0:
                te = min(sb, t + step)
                nev("FireSupport", t, te, lmk, [s.eid for s in shooters], prov)
                for s in shooters:
                    rel(s.eid, "firesAt", lmk, "landmark", t, te)
                t = te

    owner_changes = []; daily = {}
    def dsum(day):
        return daily.setdefault(day, dict(waves=0, losses=0, retakes=0, merges=0, reliefs=0, overwatch=0,
                                          ccf_loss=0.0, rok_loss=0.0, ccf_destroyed=0, rok_destroyed=0))
    APPROACH = 2100.0; ENGAGE = 2400.0

    def attrition(att, dfn, day, coeff_att, coeff_dfn, t0, t1):
        for a in att:
            d = dfn[int(rng.integers(0, len(dfn)))] if dfn else None
            loss_a = coeff_att * (0.5 + 0.5 * (d.s if d else 0.5))
            a.s = max(0.0, a.s - loss_a); dsum(day)["ccf_loss"] += loss_a; skf(a, t1)
            if d:
                loss_d = coeff_dfn * (0.4 + 0.6 * a.s)
                before = d.s; d.s = max(0.0, d.s - loss_d); dsum(day)["rok_loss"] += loss_d; skf(d, t1)
                rel(a.eid, "engagedWith", d.eid, "entity", t0, t1)
                rel(d.eid, "engagedWith", a.eid, "entity", t0, t1)
                if before > 0.5 >= d.s:
                    trans(d.eid, t1, "attrit", "Holding", "Withdrawing", d.s)
                if d.s <= 0.25 and d.alive:
                    d.alive = False; d.dead_at = t1; dsum(day)["rok_destroyed"] += 1
                    kf(d, t1, d.home[0], d.home[1], "Destroyed")
                    trans(d.eid, t1, "destroyed", "Holding", "Destroyed", d.s)

    for op in ops:
        if op.get("t0", 0.0) >= W:
            continue
        k = op["kind"]; day = int(op["t0"] // DAY)
        if k in ("prep_bombard", "refit", "quiet"):
            fire_missions(rok_arty, lmid["approach"], op["t0"], op["t1"], op["prov"], True)
            continue
        if k == "attack":
            dsum(day)["waves"] += 1
            axis = axes[op["axis"]]; lane = lmid[axis]; lx, ly = L[axis]
            t0 = op["t0"]; t_app = t0 + APPROACH; t_eng = t_app + ENGAGE
            atk_ev = nev("Attack", t0, t_eng, lmid["crest"], [], op["prov"])
            squads = [sq for i in op["ech"] for sq in ccf_squads[i] if sq.alive]
            op["squad_ids"] = [a.eid for a in squads]; op["window"] = [t0, t_eng]
            dfn = [a for a in rok_fwd if a.alive]
            if dfn:
                rel(dfn[0].eid, "near", lane, "landmark", t0 + 1200, t_eng)
                rel(dfn[0].eid, "supports", rok, "unit", t0 + 1200, t0 + 1500)
            fire = fire_ok(t0 + 1500)
            fire_missions(rok_arty + rok_mortar, lane, t0 + 1500, t_eng, op["prov"], True)
            rel_clip(tank.eid, "firesAt", lane, "landmark", t0 + 1200, t_eng)   # friendly -> clipped in fog
            coeff_att = 0.22 if fire else 0.13
            if op.get("shrink"):
                coeff_att *= 0.6
            for a in squads:
                skf(a, t0)
            attrition(squads, dfn, day, coeff_att, 0.14, t0 + 1200, t_eng)
            take = op["result"] == "take"; prev = None
            for j, a in enumerate(squads):
                a.entered = True; a.t_enter = min(a.t_enter or 0.0, t0)
                hx, hy = a.home
                sx, sy = lx + rng.uniform(-200, 200), ly + rng.uniform(-120, 120)
                tx, ty = px + rng.uniform(-250, 250), py + 120 + rng.uniform(-80, 80)
                rel(a.eid, "movesToward", lmid["crest"], "landmark", t0, t_app)
                if prev is not None:
                    rel(a.eid, "follows", prev, "entity", t0, t_app)
                prev = a.eid; s_end = a.s
                kf(a, t0, hx, hy, "Approaching")
                kf(a, t0 + 0.4 * APPROACH, sx, sy, "Approaching")
                if s_end <= 0.25:
                    kf(a, t_app, tx, ty, "Halted"); kf(a, t_eng, tx, ty, "Destroyed")
                    a.alive = False; a.dead_at = t_eng
                    trans(a.eid, t_eng, "destroyed", "Halted", "Destroyed", s_end); dsum(day)["ccf_destroyed"] += 1
                elif take and s_end > 0.4 and j < 3:
                    kf(a, t_app, tx, ty, "Occupying"); a.occ_pos = (tx, ty)
                    rel(a.eid, "occupies", lmid["crest"], "landmark", op.get("take_t", t_app), op.get("take_t", t_app) + 3600)
                else:
                    tex = t_eng + 3600
                    kf(a, t_app, tx, ty, "Halted"); kf(a, t_eng, tx, ty, "Withdrawing")
                    kf(a, tex, hx, hy, "Withdrawing"); kf(a, tex + 60, hx, hy, "Holding")   # back off-map, re-staged
                    rel(a.eid, "withdrawsFrom", lmid["crest"], "landmark", t_eng, tex)
                    trans(a.eid, t_eng, "withdraw", "Halted", "Withdrawing", s_end)
                a.events_join = getattr(a, "events_join", []) + [(atk_ev, "assault")]
            if take:
                tt = op.get("take_t", t_app); owner_changes.append((tt, "CCF")); dsum(day)["losses"] += 1
                # rule: the most-worn forward squads are relieved by reserves that
                # advance (reinforces); the relieved squads fall back to refit (s recovers).
                # rule: on every crest loss, 2 reserves advance to relieve the line
                # (reinforces); the most-worn alive forward squads fall back to refit.
                worn = sorted([d for d in dfn if d.alive], key=lambda d: d.s)
                for r_i in range(2):
                    res = rok_reserve[(int(tt // DAY) * 2 + r_i) % len(rok_reserve)]; rhx, rhy = res.home
                    rel(res.eid, "reinforces", rok_pl, "unit", tt, tt + 3600); dsum(day)["reliefs"] += 1
                    kf(res, tt, rhx, rhy, "Moving"); kf(res, tt + 1800, px + rng.uniform(-200, 200), py - 250, "Occupying")
                    kf(res, tt + 6 * 3600, rhx, rhy, "Holding")
                    if r_i < len(worn):
                        dfa = worn[r_i]
                        kf(dfa, tt, dfa.home[0], dfa.home[1], "Withdrawing")
                        kf(dfa, tt + 1800, px + rng.uniform(-200, 200), py - 950, "Holding")
                        kf(dfa, tt + 4 * 3600, dfa.home[0], dfa.home[1], "Holding")
                        dfa.s = min(1.0, dfa.s + 0.45); skf(dfa, tt + 4 * 3600)
            if op.get("force_merge"):
                # some forward defenders fall back to the 9-bu ridge then return (D5)
                for dfa in dfn[:2]:
                    tt = op.get("take_t", t_app)
                    kf(dfa, tt, dfa.home[0], dfa.home[1], "Withdrawing")
                    kf(dfa, tt + 1200, L["ninebu"][0], L["ninebu"][1], "Holding")
                wd = [a for a in squads if a.alive and a.s <= 0.6]
                for pp in range(0, min(4, len(wd) - 1), 2):
                    a, b = wd[pp], wd[pp + 1]; a.s = min(1.0, a.s + b.s); skf(a, t_eng + 3600)
                    tm = t_eng + 3600
                    rel(b.eid, "reinforces", a.eid, "entity", max(0.0, tm - 1800), tm); b._merged = tm
                    trans(b.eid, tm, "merge", "Withdrawing", "merged_end", b.s); dsum(day)["merges"] += 1
            continue
        if k == "counterattack":
            t0 = op["t0"]; prep = op.get("prep", 1800); rt = op.get("retake_t", t0 + 7200)
            fire_missions(rok_arty + rok_mortar, lmid["crest"], t0, t0 + prep, op["prov"], True)
            ca = nev("Counterattack", t0 + prep, rt, lmid["crest"], [], op["prov"])
            movers = [a for a in rok_reserve][:4]
            occupiers = [a for i in range(n_ech) for a in ccf_squads[i] if a.occ_pos is not None and a.alive]
            op["squad_ids"] = [a.eid for a in movers]; op["window"] = [t0 + prep, rt]
            prev = None
            for a in movers:
                rhx, rhy = a.home
                a.events_join = getattr(a, "events_join", []) + [(ca, "counterattack")]
                if op.get("overwatch") and prev is not None:
                    rel(a.eid, "follows", prev, "entity", t0 + prep, rt); dsum(day)["overwatch"] += 1
                prev = a.eid
                kf(a, t0 + prep, rhx, rhy, "Approaching")
                kf(a, rt, px + rng.uniform(-200, 200), py + rng.uniform(-80, 80), "Occupying")
                if op.get("final"):
                    kf(a, rt + 2400, L["northtip"][0], L["northtip"][1], "Occupying")
                    kf(a, W, L["northtip"][0], L["northtip"][1], "Occupying")
                else:
                    kf(a, rt + 6 * 3600, rhx, rhy, "Holding")
                rel(a.eid, "occupies", lmid["crest"], "landmark", rt, min(W, rt + 6 * 3600))
            attrition(occupiers, movers, day, 0.5, 0.12, t0 + prep, rt)
            for a in occupiers:
                ox_, oy_ = a.occ_pos
                if a.s <= 0.25:
                    a.alive = False; a.dead_at = rt; kf(a, rt, ox_, oy_, "Destroyed")
                    trans(a.eid, rt, "destroyed", "Occupying", "Destroyed", a.s); dsum(day)["ccf_destroyed"] += 1
                else:
                    tex = min(W, rt + 3600)
                    kf(a, rt, ox_, oy_, "Withdrawing"); kf(a, tex, bx + rng.uniform(-300, 300), by + 250, "Withdrawing")
                    kf(a, min(W, tex + 60), bx, by + 250, "Holding")
                    rel(a.eid, "withdrawsFrom", lmid["crest"], "landmark", rt, tex)
                    trans(a.eid, rt, "withdraw", "Occupying", "Withdrawing", a.s)
                a.occ_pos = None
            owner_changes.append((rt, "ROK")); dsum(day)["retakes"] += 1
            if op.get("final"):
                for a in movers[:2]:
                    nev("Occupy", rt, W, lmid["northtip"], [a.eid], op["prov"])
                    rel(a.eid, "occupies", lmid["northtip"], "landmark", rt + 3600, W)
            continue

    # continuous background artillery -> >=70% time coverage; nightly illumination
    n_days = int(np.ceil(W / DAY))
    for dd in range(n_days):
        a0 = dd * DAY + 2 * 3600; b0 = min(W, dd * DAY + 22 * 3600)
        if a0 >= W:
            break
        shooters = rok_arty if dd % 2 == 0 else ccf_arty
        friendly = shooters is rok_arty
        fire_missions(shooters, lmid["approach"], a0, b0, "record", friendly)
        ni0 = dd * DAY + 18 * 3600; ni1 = min(W, dd * DAY + 30 * 3600 + 1800)
        if ni0 < W:
            nev("Illuminate", ni0, ni1, lmid["mlr"], [light.eid], "record")
            rel(light.eid, "supports", rok, "unit", ni0, ni1)

    return dict(agents=agents, units=units, landmarks=landmarks, events=events, relations=relations,
                transitions=transitions, owner_changes=sorted(owner_changes), daily=daily, ops=ops,
                fog=fog, lmid=lmid, L=L, peak=(px, py), rok=rok, rok_pl=rok_pl, W=W, dt=dt, boundary=(bx, by))
