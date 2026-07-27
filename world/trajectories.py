"""P1 (motion) — dense trajectory + state time series with v_max enforcement.

Waypoints are interpolated at dt (default 1 s). Every step is clamped to
v_max(type, state)*dt and static states freeze the position with speed 0, so the
"impossible move" check passes by construction (required speed between any two
consecutive samples never exceeds v_max for the later sample's state).
Elevation for each sample is looked up from the terrain field (shared frame).
"""
from __future__ import annotations

import math

from .common import STATIC_STATES


def _interp(waypoints, t):
    if t <= waypoints[0][0]:
        return waypoints[0][1], waypoints[0][2]
    if t >= waypoints[-1][0]:
        return waypoints[-1][1], waypoints[-1][2]
    for (t0, e0, n0), (t1, e1, n1) in zip(waypoints, waypoints[1:]):
        if t0 <= t <= t1:
            a = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return e0 + a * (e1 - e0), n0 + a * (n1 - n0)
    return waypoints[-1][1], waypoints[-1][2]


def _state_at(states, t):
    st = states[0][1]
    for tf, s in states:
        if t >= tf:
            st = s
    return st


def _emit_rows(ent, times, ontology, terrain) -> list[dict]:
    """Emit trajectory rows at the given (sorted) times, clamping each step to
    v_max*gap (gap-aware, so variable spacing is fine) and freezing static
    states with speed 0."""
    rows = []
    e_prev, n_prev = _interp(ent.waypoints, times[0]); last_hd = 0.0
    for k, t in enumerate(times):
        state = _state_at(ent.states, t)
        if k == 0:
            e_cur, n_cur, speed, hd = e_prev, n_prev, 0.0, last_hd
        elif state in STATIC_STATES:
            e_cur, n_cur, speed, hd = e_prev, n_prev, 0.0, last_hd
        else:
            gap = max(1e-6, t - times[k - 1])
            e_t, n_t = _interp(ent.waypoints, t)
            de, dn = e_t - e_prev, n_t - n_prev
            dist = math.hypot(de, dn); vmax = ontology.v_max(ent.true_type, state)
            if dist > vmax * gap and dist > 0.0:
                sc = vmax * gap / dist; de, dn = de * sc, dn * sc; dist = vmax * gap
            e_cur, n_cur = e_prev + de, n_prev + dn
            speed = dist / gap
            hd = (math.degrees(math.atan2(de, dn)) % 360.0) if dist > 1e-9 else last_hd
        rows.append({"entity_id": ent.eid, "t": round(t, 3), "easting": round(e_cur, 3),
                     "northing": round(n_cur, 3), "elevation": round(terrain.elev_at(e_cur, n_cur), 3),
                     "state": state, "speed": round(speed, 4), "heading": round(hd, 2)})
        e_prev, n_prev, last_hd = e_cur, n_cur, hd
    return rows


def sample_sparse(entities, ontology, terrain, cfg) -> list[dict]:
    """Policy sampler: MOVING segments are densified at dt; STATIC segments
    (Emplaced/Halted/Holding/Occupying/Firing/Destroyed) record only their
    endpoints plus a coarse baseline interval — no repeated identical rows."""
    W = float(cfg["time"]["window_seconds"]); dt = float(cfg["time"]["dt_seconds"])
    static_dt = float(cfg["time"].get("static_baseline_seconds", max(dt, W / 6)))
    rows: list[dict] = []
    for ent in entities:
        lo = max(0.0, float(ent.t_enter)); hi = W if ent.t_exit is None else min(W, float(ent.t_exit))
        if hi <= lo:
            continue
        changes = sorted({lo, hi} | {float(tf) for tf, _ in ent.states if lo < tf < hi})
        times = {round(lo, 3), round(hi, 3)}
        for k0, k1 in zip(changes, changes[1:]):
            st = _state_at(ent.states, k0)
            step = dt if st not in STATIC_STATES else static_dt
            n = max(1, int(math.ceil((k1 - k0) / step)))
            for j in range(n + 1):
                times.add(round(min(k1, k0 + j * (k1 - k0) / n), 3))   # round -> strict monotonic
        rows.extend(_emit_rows(ent, sorted(times), ontology, terrain))
    return rows


def sample_trajectories(entities, ontology, terrain, cfg) -> list[dict]:
    W = float(cfg["time"]["window_seconds"])
    dt = float(cfg["time"]["dt_seconds"])
    n_steps = int(round(W / dt))
    times = [round(k * dt, 6) for k in range(n_steps + 1)]

    rows: list[dict] = []
    for ent in entities:
        lo = max(0.0, float(ent.t_enter))
        hi = W if ent.t_exit is None else min(W, float(ent.t_exit))
        active_times = [t for t in times if lo <= t <= hi]
        if not active_times:
            continue
        e_prev, n_prev = _interp(ent.waypoints, active_times[0])
        last_hd = 0.0
        for k, t in enumerate(active_times):
            state = _state_at(ent.states, t)
            if k == 0:
                e_cur, n_cur, speed, hd = e_prev, n_prev, 0.0, last_hd
            elif state in STATIC_STATES:
                e_cur, n_cur, speed, hd = e_prev, n_prev, 0.0, last_hd
            else:
                e_t, n_t = _interp(ent.waypoints, t)
                de, dn = e_t - e_prev, n_t - n_prev
                dist = math.hypot(de, dn)
                vmax = ontology.v_max(ent.true_type, state)
                max_step = vmax * dt
                if dist > max_step and dist > 0.0:
                    scale = max_step / dist
                    de, dn = de * scale, dn * scale
                    dist = max_step
                e_cur, n_cur = e_prev + de, n_prev + dn
                speed = dist / dt
                hd = (math.degrees(math.atan2(de, dn)) % 360.0) if dist > 1e-9 else last_hd
            rows.append({
                "entity_id": ent.eid,
                "t": t,
                "easting": round(e_cur, 3),
                "northing": round(n_cur, 3),
                "elevation": round(terrain.elev_at(e_cur, n_cur), 3),
                "state": state,
                "speed": round(speed, 4),
                "heading": round(hd, 2),
            })
            e_prev, n_prev, last_hd = e_cur, n_cur, hd
    return rows
