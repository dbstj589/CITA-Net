#!/usr/bin/env python
"""Entry point — omniscient ground-truth STKG generator (Hill 395).

Runs P1 (entities+trajectories), P2 (terrain), P3 (serialise), P4 (validate),
P8 (manifest+hashes) from a single config. Sensor projection / error injection /
merge (P5-P7, P9) are NOT part of this module.

    python scripts/gen_world_gt.py --config configs/world_gt_small.yaml

Frozen suites (battlefield_*/) are never touched; output goes to a new dir.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from citanet.data.ontology import load_ontology            # noqa: E402
from world.common import EVENT_KINDS                       # noqa: E402
from world.entities import build_world                     # noqa: E402
from world.scenario_full import build_sector_world         # noqa: E402
from world.terrain import build_terrain                    # noqa: E402
from world.trajectories import sample_trajectories         # noqa: E402
from world.events_relations import build_relations         # noqa: E402
from world.serialize import serialize_world, _write_terrain  # noqa: E402
from world.serialize_stream import StreamWriter            # noqa: E402
from world.validate import run_validation, validate_sector  # noqa: E402


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _preview_plot(outdir: Path, traj_rows, landmarks) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:            # visual check is best-effort
        print(f"  (preview plot skipped: {e})")
        return
    by_ent: dict[str, list] = {}
    for r in traj_rows:
        by_ent.setdefault(r["entity_id"], []).append(r)
    fig, ax = plt.subplots(figsize=(8, 8))
    for rows in by_ent.values():
        ax.plot([r["easting"] for r in rows], [r["northing"] for r in rows],
                "-", lw=0.6, alpha=0.6)
        ax.plot(rows[0]["easting"], rows[0]["northing"], ".", ms=2, color="k")
    for lm in landmarks.values():
        ax.plot(lm.easting, lm.northing, "^", color="red", ms=6)
        ax.annotate(lm.name, (lm.easting, lm.northing), fontsize=6)
    ax.set_aspect("equal"); ax.set_xlabel("easting (m)"); ax.set_ylabel("northing (m)")
    ax.set_title("GT trajectories (dots=start, red=landmarks)")
    fig.tight_layout(); fig.savefig(outdir / "trajectories_preview.png", dpi=140)
    plt.close(fig)


def run_battle_1m(cfg, ontology, outdir: Path, seed: int, ont_dir: str) -> None:
    """Scaled battle GT (~1,000,000 triples): SAME world rules as v2, only bigger.
    Legit knobs: (1) more units, (2) serialize per-sample state to the .nt graph,
    (3) granular fire missions. Continuous 10-day sim, DAY-split serialization
    (d01..d10) with seam checks across midnight. No motion caps, no padding."""
    import math
    import pyarrow as pa
    import pyarrow.parquet as pq
    from collections import Counter, defaultdict
    from world.terrain_real import build_terrain_real
    from world.battle_sim import simulate
    from world.scenario_battle10d import is_night, DAY
    from world.common import iri, P, GP, lit, RDF, RDFS, XSD, SIZE_BY_CATEGORY

    MOVING = {"Approaching", "Moving", "Withdrawing"}
    STATIC = {"Emplaced", "Holding", "Halted", "Occupying", "Firing", "Destroyed"}
    t_start = __import__("time").time()

    terrain, tmeta = build_terrain_real(cfg)
    rng = np.random.default_rng(seed)
    sim = simulate(cfg, ontology, terrain, rng)
    agents = sim["agents"]; W = sim["W"]; dt = sim["dt"]
    base_dt = float(cfg["time"].get("static_baseline_seconds", 3600))
    n_days = int(math.ceil(W / DAY))
    dayof = lambda t: min(n_days - 1, max(0, int(t // DAY)))

    for a in agents.values():
        a._s_sched = sorted(set(a.s_kf)) or [(0.0, a.s)]
        kfs = sorted(a.kf, key=lambda z: z[0])
        a._wps = [(t, x, y) for (t, x, y, st) in kfs]
        a._states = [(t, st) for (t, x, y, st) in kfs]

    def state_at(states, t):
        st = states[0][1]
        for tf, s in states:
            if t >= tf:
                st = s
        return st
    def interp_xy(wps, t):
        if t <= wps[0][0]:
            return wps[0][1], wps[0][2]
        if t >= wps[-1][0]:
            return wps[-1][1], wps[-1][2]
        for (t0, e0, n0), (t1, e1, n1) in zip(wps, wps[1:]):
            if t0 <= t <= t1:
                r = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                return e0 + r * (e1 - e0), n0 + r * (n1 - n0)
        return wps[-1][1], wps[-1][2]
    def interp_s(sc, t):
        if t <= sc[0][0]:
            return sc[0][1]
        if t >= sc[-1][0]:
            return sc[-1][1]
        for (t0, s0), (t1, s1) in zip(sc, sc[1:]):
            if t0 <= t <= t1:
                r = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                return s0 + r * (s1 - s0)
        return sc[-1][1]

    # ---- sample trajectories (with s); force a sample exactly at each midnight for seam checks ----
    traj = []
    midnights = [d * DAY for d in range(1, n_days)]
    for a in agents.values():
        wps, states = a._wps, a._states
        changes = sorted({0.0, W} | {round(float(tf), 3) for tf, _ in states if 0 < tf < W} | set(midnights))
        times = set()
        for k0, k1 in zip(changes, changes[1:]):
            st = state_at(states, k0)
            step = dt if st in MOVING else base_dt
            n = max(1, int(math.ceil((k1 - k0) / step)))
            for j in range(n + 1):
                times.add(round(min(k1, k0 + j * (k1 - k0) / n), 3))
        ts = sorted(times)
        e_prev, n_prev = interp_xy(wps, ts[0]); last_hd = 0.0
        for idx, t in enumerate(ts):
            st = state_at(states, t)
            if idx == 0 or st in STATIC:
                e_cur, n_cur, spd, hd = e_prev, n_prev, 0.0, last_hd
            else:
                gap = max(1e-6, t - ts[idx - 1])
                e_t, n_t = interp_xy(wps, t)
                de, dn = e_t - e_prev, n_t - n_prev
                dist = math.hypot(de, dn); vmax = ontology.v_max(a.typ, st)
                if dist > vmax * gap and dist > 0:
                    sc_ = vmax * gap / dist; de, dn = de * sc_, dn * sc_; dist = vmax * gap
                e_cur, n_cur = e_prev + de, n_prev + dn; spd = dist / gap
                hd = (math.degrees(math.atan2(de, dn)) % 360.0) if dist > 1e-9 else last_hd
            traj.append({"entity_id": a.eid, "t": round(t, 3), "easting": round(e_cur, 3),
                         "northing": round(n_cur, 3), "elevation": round(terrain.elev_at(e_cur, n_cur), 3),
                         "state": st, "speed": round(spd, 4), "heading": round(hd, 2),
                         "s": round(interp_s(a._s_sched, t), 3)})
            e_prev, n_prev, last_hd = e_cur, n_cur, hd

    # ---- day-tagged triple emission ----
    ent_ids = set(agents); unit_ids = set(sim["units"]); lm_ids = set(sim["landmarks"])
    tagged = []          # (day, cat, line)
    def emit(day, cat, s, p, o):
        tagged.append((day, cat, f"{s} {p} {o} ."))
    D0 = 0
    for a in agents.values():
        s = iri("entity", a.eid)
        emit(D0, "entity", s, f"<{RDF}type>", P("Entity"))
        emit(D0, "entity", s, P("objectType"), lit(a.typ))
        emit(D0, "entity", s, P("affiliation"), lit(a.aff))
        emit(D0, "entity", s, P("size"), lit(SIZE_BY_CATEGORY.get(ontology.category(a.typ), "unknown")))
        if a.unit:
            emit(D0, "relation", s, P("partOf"), iri("unit", a.unit))
        for evid, _ in getattr(a, "events_join", []):
            emit(D0, "relation", s, P("participatesIn"), iri("event", evid))
    for u in sim["units"].values():
        s = iri("unit", u.uid)
        emit(D0, "unit", s, f"<{RDF}type>", P("Unit")); emit(D0, "unit", s, f"<{RDFS}label>", lit(u.label))
        emit(D0, "unit", s, P("echelon"), lit(u.echelon)); emit(D0, "unit", s, P("affiliation"), lit(u.affiliation))
        if u.parent:
            emit(D0, "unit", s, P("partOf"), iri("unit", u.parent))
    for lmo in sim["landmarks"].values():
        s = iri("lm", lmo.lmid)
        emit(D0, "landmark", s, f"<{RDF}type>", P("Landmark")); emit(D0, "landmark", s, f"<{RDFS}label>", lit(lmo.name))
        emit(D0, "landmark", s, GP("easting"), lit(round(lmo.easting, 2), XSD + "double"))
        emit(D0, "landmark", s, GP("northing"), lit(round(lmo.northing, 2), XSD + "double"))
        emit(D0, "landmark", s, GP("elevation"), lit(round(lmo.elevation, 2), XSD + "double"))
    tgt = {"entity": lambda r: iri("entity", r), "unit": lambda r: iri("unit", r), "landmark": lambda r: iri("lm", r)}
    for ev in sim["events"]:
        day = dayof(ev["interval"][0]); s = iri("event", ev["id"])
        emit(day, "event", s, f"<{RDF}type>", P("Event")); emit(day, "event", s, P("eventKind"), lit(ev["type"]))
        emit(day, "event", s, P("startTime"), lit(round(ev["interval"][0], 1), XSD + "double"))
        emit(day, "event", s, P("endTime"), lit(round(ev["interval"][1], 1), XSD + "double"))
        if ev.get("landmark"):
            emit(day, "event", s, P("atLandmark"), iri("lm", ev["landmark"]))
    for r in sim["relations"]:
        if r["object_kind"] == "entity" and r["object"] not in ent_ids:
            continue
        day = dayof(r["interval"][0])
        emit(day, "relation", iri("entity", r["subject"]), P(r["predicate"]), tgt[r["object_kind"]](r["object"]))
    # reification: transitions + key relations + events (start day)
    stc = [0]
    def stmt(pfx):
        stc[0] += 1; return iri("stmt", f"{pfx}{stc[0]}")
    for tr in sim["transitions"]:
        day = dayof(tr["t"]); s = stmt("t")
        emit(day, "reification", s, f"<{RDF}type>", P("Statement"))
        emit(day, "reification", s, P("aboutEntity"), iri("entity", tr["entity"]))
        emit(day, "reification", s, P("transitionKind"), lit(tr["kind"]))
        emit(day, "reification", s, P("fromState"), lit(tr["from_state"]))
        emit(day, "reification", s, P("toState"), lit(tr["to_state"]))
        emit(day, "reification", s, P("combatPower"), lit(tr["s"], XSD + "double"))
        emit(day, "reification", s, P("atTime"), lit(round(tr["t"], 1), XSD + "double"))
    reify_preds = set(cfg.get("reification", {}).get("predicates",
                  ["engagedWith", "firesAt", "occupies", "withdrawsFrom", "reinforces", "movesToward", "supports"]))
    for r in sim["relations"]:
        if r["predicate"] not in reify_preds:
            continue
        if r["object_kind"] == "entity" and r["object"] not in ent_ids:
            continue
        day = dayof(r["interval"][0]); s = stmt("r")
        emit(day, "reification", s, f"<{RDF}type>", P("Statement"))
        emit(day, "reification", s, f"<{RDF}subject>", iri("entity", r["subject"]))
        emit(day, "reification", s, f"<{RDF}predicate>", P(r["predicate"]))
        emit(day, "reification", s, f"<{RDF}object>", tgt[r["object_kind"]](r["object"]))
        emit(day, "reification", s, P("validFrom"), lit(round(r["interval"][0], 1), XSD + "double"))
        emit(day, "reification", s, P("validTo"), lit(round(r["interval"][1], 1), XSD + "double"))
    for ev in sim["events"]:
        day = dayof(ev["interval"][0]); s = stmt("e")
        emit(day, "reification", s, f"<{RDF}type>", P("Statement"))
        emit(day, "reification", s, P("aboutEvent"), iri("event", ev["id"]))
        emit(day, "reification", s, P("provenance"), lit(ev.get("prov", "record")))
        emit(day, "reification", s, P("validFrom"), lit(round(ev["interval"][0], 1), XSD + "double"))
        emit(day, "reification", s, P("validTo"), lit(round(ev["interval"][1], 1), XSD + "double"))
    # (2) per-sample true-state triples -> the graph (moving dense, static sparse; no per-step repeat of frozen pos)
    if cfg.get("emit_state_samples", True):
        for row in traj:
            day = dayof(row["t"])
            s = iri("sample", row["entity_id"], f"t{int(round(row['t']))}")
            emit(day, "trajectory", s, f"<{RDF}type>", P("StateSample"))
            emit(day, "trajectory", s, P("sampleOf"), iri("entity", row["entity_id"]))
            emit(day, "trajectory", s, P("atTime"), lit(round(row["t"], 1), XSD + "double"))
            emit(day, "trajectory", s, GP("easting"), lit(row["easting"], XSD + "double"))
            emit(day, "trajectory", s, GP("northing"), lit(row["northing"], XSD + "double"))
            emit(day, "trajectory", s, P("hasState"), lit(row["state"]))
            emit(day, "trajectory", s, P("speedMps"), lit(row["speed"], XSD + "double"))
            emit(day, "trajectory", s, P("combatPower"), lit(row["s"], XSD + "double"))

    # ---- dedup (global) + day-split write ----
    CATS = ["entity", "unit", "landmark", "event", "relation", "reification", "trajectory"]
    order = {c: i for i, c in enumerate(CATS)}
    tagged.sort(key=lambda z: order[z[1]])
    seen = set(); day_lines = defaultdict(list); cat_counts = Counter(); day_counts = Counter(); dupes = 0
    for day, cat, ln in tagged:
        if ln in seen:
            dupes += 1; continue
        seen.add(ln); day_lines[day].append(ln); cat_counts[cat] += 1; day_counts[day] += 1
    outdir.mkdir(parents=True, exist_ok=True)
    total = 0
    for d in range(n_days):
        fn = outdir / f"unified_stkg_d{d+1:02d}.nt"
        fn.write_text("\n".join(day_lines.get(d, [])) + ("\n" if day_lines.get(d) else ""), encoding="utf-8")
        total += day_counts.get(d, 0)

    with open(outdir / "entities.jsonl", "w", encoding="utf-8") as fh:
        for a in agents.values():
            fh.write(json.dumps({"id": a.eid, "true_type": a.typ, "affiliation": a.aff,
                "size": SIZE_BY_CATEGORY.get(ontology.category(a.typ), "unknown"),
                "parent_unit": a.unit, "role": a.role}, ensure_ascii=False) + "\n")
    with open(outdir / "events.jsonl", "w", encoding="utf-8") as fh:
        for ev in sorted(sim["events"], key=lambda e: (e["interval"][0], e["type"])):
            fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
    with open(outdir / "relations.jsonl", "w", encoding="utf-8") as fh:
        for r in sim["relations"]:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    cols = ["entity_id", "t", "easting", "northing", "elevation", "state", "speed", "heading", "s"]
    pq.write_table(pa.table({c: [row[c] for row in traj] for c in cols}), outdir / "trajectories.parquet")
    _write_terrain(outdir / "terrain", terrain, tmeta, sim["landmarks"])

    # ---- validation ----
    by_ent = defaultdict(list)
    for r in traj:
        by_ent[r["entity_id"]].append(r)
    mono = feas = sspd = 0
    states_ok = set(ontology.state_names); types_ok = set(ontology.type_names); rels_ok = set(ontology.relation_names)
    for eid, rows in by_ent.items():
        rows.sort(key=lambda r: r["t"]); ts = [r["t"] for r in rows]
        mono += sum(1 for a, b in zip(ts, ts[1:]) if b <= a)
        typ = agents[eid].typ
        for a, b in zip(rows, rows[1:]):
            gap = max(1e-6, b["t"] - a["t"])
            if math.hypot(b["easting"] - a["easting"], b["northing"] - a["northing"]) / gap > ontology.v_max(typ, b["state"]) + 1e-2:
                feas += 1
        for r in rows:
            if r["state"] in STATIC and abs(r["speed"]) > 1e-2:
                sspd += 1
    voc = sum(1 for r in traj if r["state"] not in states_ok) + sum(1 for a in agents.values() if a.typ not in types_ok) + \
          sum(1 for r in sim["relations"] if r["predicate"] not in rels_ok)
    # independent per-day dup + total line count
    nt_total = 0; seen2 = set(); d2 = 0
    for d in range(n_days):
        for ln in open(outdir / f"unified_stkg_d{d+1:02d}.nt", encoding="utf-8"):
            nt_total += 1
            if ln in seen2:
                d2 += 1
            seen2.add(ln)
    xref = 0
    for r in sim["relations"]:
        if r["subject"] not in ent_ids:
            xref += 1
        pool = {"entity": ent_ids, "unit": unit_ids, "landmark": lm_ids}[r["object_kind"]]
        if r["object"] not in pool:
            xref += 1
    # seam checks: agents alive across a midnight -> position & s continuous
    seam_pos = seam_s = 0
    for eid, rows in by_ent.items():
        rows.sort(key=lambda r: r["t"])
        for mt in midnights:
            before = [r for r in rows if r["t"] <= mt]
            after = [r for r in rows if r["t"] >= mt]
            if before and after:
                b = before[-1]; a = after[0]
                gap = max(1e-6, a["t"] - b["t"])
                if math.hypot(a["easting"] - b["easting"], a["northing"] - b["northing"]) / gap > ontology.v_max(agents[eid].typ, a["state"]) + 1e-2:
                    seam_pos += 1
                if abs(a["s"] - b["s"]) > 0.15 + 0.001 * gap:
                    seam_s += 1
    # inherited checks
    op_bad = []
    for op in sim["ops"]:
        if op.get("kind") not in ("attack", "counterattack") or "squad_ids" not in op:
            continue
        w0, w1 = op["window"]
        for eid in op["squad_ids"]:
            rows = [r for r in by_ent.get(eid, []) if w0 - 1 <= r["t"] <= w1 + 1]
            if len(rows) < 2 or not any(r["state"] in MOVING for r in rows):
                op_bad.append(eid)
    appr_by_day = defaultdict(int)
    for r in traj:
        if r["state"] == "Approaching":
            appr_by_day[int(r["t"] // DAY)] += 1
    attack_days = {int(op["t0"] // DAY) for op in sim["ops"] if op.get("kind") == "attack" and op["t0"] < W}
    zero_days = sorted(d for d in attack_days if appr_by_day.get(d, 0) == 0)
    friendly = {a.eid for a in agents.values() if a.aff in ("ROK", "US")}
    fog = sim["fog"]
    def overlap_len(a, b):
        return sum(max(0.0, min(b, fb) - max(a, fa)) for fa, fb in fog)
    fog_overlap = sum(1 for r in sim["relations"] if r["predicate"] == "firesAt" and r["subject"] in friendly and overlap_len(*r["interval"]) > 0.0)
    bx, by = sim["boundary"]
    off_bad = [eid for eid, rows in by_ent.items() if agents[eid].role == "assault"
               and max(rows, key=lambda r: r["t"])["state"] == "Withdrawing"
               and max(rows, key=lambda r: r["t"])["northing"] < by - 100]
    ext_e = terrain.origin_e + terrain.res * (terrain.elevation.shape[1] - 1)
    ext_n = terrain.origin_n + terrain.res * (terrain.elevation.shape[0] - 1)
    xs = [r["easting"] for r in traj]; ys = [r["northing"] for r in traj]
    in_extent = min(xs) >= terrain.origin_e and max(xs) <= ext_e and min(ys) >= terrain.origin_n and max(ys) <= ext_n
    ei, ej = np.unravel_index(int(np.argmax(terrain.elevation)), terrain.elevation.shape)
    peak_match = abs(sim["landmarks"]["lm_crest"].easting - (terrain.origin_e + ej * terrain.res)) < 1e-6
    ccf_enter = all(min(rows, key=lambda r: r["t"])["northing"] >= by - 400 for eid, rows in by_ent.items() if agents[eid].role == "assault")
    # scale bands
    band_ok = {}
    for d in range(n_days):
        c = day_counts.get(d, 0); band_ok[d] = (0.85 * 100000 <= c <= 1.15 * 100000)
    total_ok = 0.9e6 <= total <= 1.1e6

    report = {
        "time_monotonic": {"pass": mono == 0, "violations": mono},
        "feasible_motion": {"pass": feas == 0, "violations": feas},
        "static_speed_zero": {"pass": sspd == 0, "violations": sspd},
        "vocabulary": {"pass": voc == 0, "violations": voc},
        "cross_reference": {"pass": xref == 0, "violations": xref},
        "triple_count_match": {"pass": nt_total == total == len(seen), "nt_files": nt_total, "sum_day": total, "unique": len(seen)},
        "no_duplicate_triples": {"pass": d2 == 0, "duplicates": d2},
        "op_trajectory_consistency": {"pass": not op_bad, "violations": len(op_bad)},
        "daily_approaching_on_attack_days": {"pass": not zero_days, "zero_days": zero_days},
        "fog_friendly_fire_overlap_sec": {"pass": fog_overlap == 0, "overlaps": fog_overlap},
        "withdrawn_end_off_map": {"pass": not off_bad, "violations": len(off_bad)},
        "boundary_entry": {"pass": ccf_enter, "ok": ccf_enter},
        "coords_in_extent": {"pass": bool(in_extent)},
        "landmark_peak_match": {"pass": bool(peak_match)},
        "seam_position_continuity": {"pass": seam_pos == 0, "violations": seam_pos},
        "seam_s_continuity": {"pass": seam_s == 0, "violations": seam_s},
        "total_1M_hard": {"pass": total_ok, "total": total, "band": [0.9e6, 1.1e6]},
    }
    p4_ok = all(v["pass"] for v in report.values())

    # realism (same definitions as checks)
    state_dist = Counter(r["state"] for r in traj)
    moving_ratio = sum(state_dist[s] for s in MOVING) / max(1, sum(state_dist.values()))
    daily = sim["daily"]
    ccf_loss = sum(dd["ccf_loss"] for dd in daily.values()); rok_loss = sum(dd["rok_loss"] for dd in daily.values())
    loss_ratio = ccf_loss / max(1e-6, rok_loss)
    merges = sum(dd["merges"] for dd in daily.values()); reliefs = sum(dd["reliefs"] for dd in daily.values())
    overwatch = sum(dd["overwatch"] for dd in daily.values())
    ccf_dead = sum(1 for a in agents.values() if a.aff == "CCF" and not a.alive)
    rok_dead = sum(1 for a in agents.values() if a.aff in ("ROK", "US") and not a.alive)
    ccf_s = float(np.mean([a.s for a in agents.values() if a.aff == "CCF" and a.role != "wreck"]))
    rok_s = float(np.mean([a.s for a in agents.values() if a.aff in ("ROK", "US")]))
    fire_iv = sorted([e["interval"] for e in sim["events"] if e["type"] == "FireSupport"])
    cur = 0.0; cov = 0.0
    for a, b in fire_iv:
        a = max(a, cur)
        if b > a:
            cov += b - a; cur = b
        elif b > cur:
            cur = b
    fire_cov = cov / W
    atk = [e["interval"][0] for e in sim["events"] if e["type"] == "Attack"]
    night_ratio = sum(1 for t in atk if is_night(t)) / max(1, len(atk))
    realism = {"concurrent_alive_peak": max(sum(1 for a in agents.values()
                if (a.t_enter or 0) <= t and (a.dead_at is None or t < a.dead_at)) for t in range(0, int(W), 6 * 3600)),
               "moving_ratio": round(moving_ratio, 3), "loss_ratio_ccf_to_rok": round(loss_ratio, 2),
               "ccf_mean_s": round(ccf_s, 3), "rok_mean_s": round(rok_s, 3), "rok_destroyed": rok_dead,
               "ccf_destroyed": ccf_dead, "merges": merges, "reliefs": reliefs, "overwatch": overwatch,
               "crest_owner_changes": len(sim["owner_changes"]), "background_fire_coverage": round(fire_cov, 3),
               "night_attack_ratio": round(night_ratio, 3)}

    _npjson = lambda o: o.item() if hasattr(o, "item") else str(o)
    (outdir / "validation_report.json").write_text(json.dumps(
        {"p4_and_checks": report, "realism": realism,
         "daily_triples": {str(d): day_counts.get(d, 0) for d in range(n_days)},
         "band_ok": {str(d): bool(band_ok[d]) for d in range(n_days)},
         "triple_categories": dict(cat_counts)}, ensure_ascii=False, indent=2, default=_npjson), encoding="utf-8")

    # battle_log.md (record|interp|emergent)
    log = ["# Hill 395 — 10-day battle GT chronicle (record=source | interp=designed | emergent=rule)\n"]
    for op in sim["ops"]:
        if op.get("t0", 0) >= W:
            continue
        day = int(op["t0"] // DAY) + 6; hh = int((op["t0"] % DAY) // 3600); mm = int((op["t0"] % 3600) // 60)
        log.append(f"- 10/{day:02d} {hh:02d}:{mm:02d}  [{op['prov']}]  {op['kind']}: {op.get('note','')}")
    (outdir / "battle_log.md").write_text("\n".join(log) + "\n", encoding="utf-8")

    elapsed = __import__("time").time() - t_start
    manifest = {"generator": "gen_world_gt.py (battle_1m)", "scope": "P1-P4,P8 scaled two-layer battle GT",
                "global_seed": seed, "config": cfg, "ontology_dir": ont_dir,
                "counts": {"n_triples": total, "n_entities": len(agents), "n_events": len(sim["events"]),
                           "n_relations": len(sim["relations"]), "n_transitions": len(sim["transitions"]),
                           "n_traj_rows": len(traj)},
                "triple_categories": dict(cat_counts), "daily_triples": {str(d): day_counts.get(d, 0) for d in range(n_days)},
                "director_ops": [o for o in sim["ops"] if o.get("t0", 0) < W],
                "crest_owner_changes": [[round(t, 1), o] for t, o in sim["owner_changes"]],
                "realism": realism, "combat_power_s": "ontology EXTENSION (확인 필요)",
                "elapsed_sec": round(elapsed, 1),
                "content_sha256": {f"unified_stkg_d{d+1:02d}.nt": _sha256(outdir / f"unified_stkg_d{d+1:02d}.nt") for d in range(n_days)},
                "validation_pass": p4_ok}
    (outdir / "gt_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=_npjson), encoding="utf-8")

    print(f"   == battle_1m: {total:,} triples, {len(agents)} agents, {len(traj):,} traj rows, {elapsed:.0f}s ==")
    print("   daily triples (target 100k +-15%):")
    for d in range(n_days):
        c = day_counts.get(d, 0)
        print(f"     D{d+1:02d}: {c:>8,}  {'OK ' if band_ok[d] else 'OUT'}")
    print(f"   total 1M+-10%: {total:,}  {'OK' if total_ok else 'OUT'}")
    tt = max(1, total)
    print("   categories: " + "  ".join(f"{c}={cat_counts[c]:,}({100*cat_counts[c]/tt:.0f}%)" for c in CATS))
    print("   checks:")
    for kk, vv in report.items():
        print(f"     [{'OK ' if vv['pass'] else 'FAIL'}] {kk}: {vv}")
    print("   realism: " + "  ".join(f"{k}={v}" for k, v in realism.items()))
    if not p4_ok:
        raise SystemExit("battle_1m validation FAILED — see validation_report.json")


def run_battle(cfg, ontology, outdir: Path, seed: int, ont_dir: str) -> None:
    """Two-layer 10-day battle GT: director schedule + unit rules (combat power s,
    attrition, thresholds). Trajectory (with s) is dense; the .nt graph reifies
    only meaningful threshold transitions + key relations/events (not every step).
    Prints P4 + realism checks and a causal-chain / prep-fire->counterattack example."""
    import math
    import pyarrow as pa
    import pyarrow.parquet as pq
    from collections import Counter, defaultdict
    from world.terrain_real import build_terrain_real
    from world.battle_sim import simulate
    from world.scenario_battle10d import is_night, DUSK, DAWN, DAY
    from world.common import iri, P, GP, lit, RDF, RDFS, XSD

    MOVING = {"Approaching", "Moving", "Withdrawing"}
    STATIC = {"Emplaced", "Holding", "Halted", "Occupying", "Firing", "Destroyed"}

    terrain, tmeta = build_terrain_real(cfg)
    rng = np.random.default_rng(seed)
    sim = simulate(cfg, ontology, terrain, rng)
    agents = sim["agents"]; W = sim["W"]; dt = sim["dt"]
    base_dt = float(cfg["time"].get("static_baseline_seconds", 3600))

    # per-agent timeline from ACCUMULATED keyframes (v2: full multi-op trajectory,
    # so every attack/counterattack the agent joins has real motion, not just events)
    for a in agents.values():
        a._s_sched = sorted(set(a.s_kf)) or [(0.0, a.s)]
        kfs = sorted(a.kf, key=lambda z: z[0])
        a._wps = [(t, x, y) for (t, x, y, st) in kfs]
        a._states = [(t, st) for (t, x, y, st) in kfs]
        a.t_exit = W

    def state_at(states, t):
        st = states[0][1]
        for tf, s in states:
            if t >= tf:
                st = s
        return st
    def interp_xy(wps, t):
        if t <= wps[0][0]:
            return wps[0][1], wps[0][2]
        if t >= wps[-1][0]:
            return wps[-1][1], wps[-1][2]
        for (t0, e0, n0), (t1, e1, n1) in zip(wps, wps[1:]):
            if t0 <= t <= t1:
                r = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                return e0 + r * (e1 - e0), n0 + r * (n1 - n0)
        return wps[-1][1], wps[-1][2]
    def interp_s(sc, t):
        if t <= sc[0][0]:
            return sc[0][1]
        if t >= sc[-1][0]:
            return sc[-1][1]
        for (t0, s0), (t1, s1) in zip(sc, sc[1:]):
            if t0 <= t <= t1:
                r = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                return s0 + r * (s1 - s0)
        return sc[-1][1]

    # ---- sample trajectories (with s) ----
    traj = []
    for a in agents.values():
        wps, states = a._wps, a._states
        lo = 0.0; hi = W
        if hi <= lo:
            hi = lo + dt
        changes = sorted({round(lo, 3), round(hi, 3)} | {round(float(tf), 3) for tf, _ in states if lo < tf < hi})
        times = set()
        for k0, k1 in zip(changes, changes[1:]):
            st = state_at(states, k0)
            step = dt if st in MOVING else base_dt
            n = max(1, int(math.ceil((k1 - k0) / step)))
            for j in range(n + 1):
                times.add(round(min(k1, k0 + j * (k1 - k0) / n), 3))
        ts = sorted(times)
        e_prev, n_prev = interp_xy(wps, ts[0]); last_hd = 0.0
        for idx, t in enumerate(ts):
            st = state_at(states, t)
            if idx == 0 or st in STATIC:
                e_cur, n_cur, spd, hd = e_prev, n_prev, 0.0, last_hd
            else:
                gap = max(1e-6, t - ts[idx - 1])
                e_t, n_t = interp_xy(wps, t)
                de, dn = e_t - e_prev, n_t - n_prev
                dist = math.hypot(de, dn); vmax = ontology.v_max(a.typ, st)
                if dist > vmax * gap and dist > 0:
                    sc_ = vmax * gap / dist; de, dn = de * sc_, dn * sc_; dist = vmax * gap
                e_cur, n_cur = e_prev + de, n_prev + dn; spd = dist / gap
                hd = (math.degrees(math.atan2(de, dn)) % 360.0) if dist > 1e-9 else last_hd
            traj.append({"entity_id": a.eid, "t": round(t, 3), "easting": round(e_cur, 3),
                         "northing": round(n_cur, 3), "elevation": round(terrain.elev_at(e_cur, n_cur), 3),
                         "state": st, "speed": round(spd, 4), "heading": round(hd, 2),
                         "s": round(interp_s(a._s_sched, t), 3)})
            e_prev, n_prev, last_hd = e_cur, n_cur, hd

    # ---- nt triples (categories) ----
    cats = defaultdict(list)
    def T(cat, s, p, o):
        cats[cat].append(f"{s} {p} {o} .")
    ent_ids = set(agents)
    unit_ids = set(sim["units"]); lm_ids = set(sim["landmarks"]); ev_ids = {e["id"] for e in sim["events"]}
    for a in agents.values():
        s = iri("entity", a.eid)
        T("entity", s, f"<{RDF}type>", P("Entity"))
        T("entity", s, P("objectType"), lit(a.typ))
        T("entity", s, P("affiliation"), lit(a.aff))
        T("entity", s, P("size"), lit(__import__("world.common", fromlist=["SIZE_BY_CATEGORY"]).SIZE_BY_CATEGORY.get(ontology.category(a.typ), "unknown")))
        if a.unit:
            T("relation", s, P("partOf"), iri("unit", a.unit))
        for evid, _ in getattr(a, "events_join", []):
            T("relation", s, P("participatesIn"), iri("event", evid))
    for u in sim["units"].values():
        s = iri("unit", u.uid)
        T("unit", s, f"<{RDF}type>", P("Unit")); T("unit", s, f"<{RDFS}label>", lit(u.label))
        T("unit", s, P("echelon"), lit(u.echelon)); T("unit", s, P("affiliation"), lit(u.affiliation))
        if u.parent:
            T("unit", s, P("partOf"), iri("unit", u.parent))
    for lmo in sim["landmarks"].values():
        s = iri("lm", lmo.lmid)
        T("landmark", s, f"<{RDF}type>", P("Landmark")); T("landmark", s, f"<{RDFS}label>", lit(lmo.name))
        T("landmark", s, GP("easting"), lit(round(lmo.easting, 2), XSD + "double"))
        T("landmark", s, GP("northing"), lit(round(lmo.northing, 2), XSD + "double"))
        T("landmark", s, GP("elevation"), lit(round(lmo.elevation, 2), XSD + "double"))
    tgt = {"entity": lambda r: iri("entity", r), "unit": lambda r: iri("unit", r), "landmark": lambda r: iri("lm", r)}
    for ev in sim["events"]:
        s = iri("event", ev["id"])
        T("event", s, f"<{RDF}type>", P("Event")); T("event", s, P("eventKind"), lit(ev["type"]))
        T("event", s, P("startTime"), lit(round(ev["interval"][0], 1), XSD + "double"))
        T("event", s, P("endTime"), lit(round(ev["interval"][1], 1), XSD + "double"))
        if ev.get("landmark"):
            T("event", s, P("atLandmark"), iri("lm", ev["landmark"]))
    for r in sim["relations"]:
        if r["object_kind"] == "entity" and r["object"] not in ent_ids:
            continue
        T("relation", iri("entity", r["subject"]), P(r["predicate"]), tgt[r["object_kind"]](r["object"]))
    # reification: threshold transitions (combat-power s crossings) + key relations + events
    stc = [0]
    def stmt(pfx):
        stc[0] += 1; return iri("stmt", f"{pfx}{stc[0]}")
    for tr in sim["transitions"]:
        s = stmt("t")
        T("reification", s, f"<{RDF}type>", P("Statement"))
        T("reification", s, P("aboutEntity"), iri("entity", tr["entity"]))
        T("reification", s, P("transitionKind"), lit(tr["kind"]))
        T("reification", s, P("fromState"), lit(tr["from_state"]))
        T("reification", s, P("toState"), lit(tr["to_state"]))
        T("reification", s, P("combatPower"), lit(tr["s"], XSD + "double"))
        T("reification", s, P("atTime"), lit(round(tr["t"], 1), XSD + "double"))
    reify_preds = set(cfg.get("reification", {}).get("predicates",
                  ["engagedWith", "firesAt", "occupies", "withdrawsFrom", "reinforces", "movesToward", "supports"]))
    for r in sim["relations"]:
        if r["predicate"] not in reify_preds:
            continue
        if r["object_kind"] == "entity" and r["object"] not in ent_ids:
            continue
        s = stmt("r")
        T("reification", s, f"<{RDF}type>", P("Statement"))
        T("reification", s, f"<{RDF}subject>", iri("entity", r["subject"]))
        T("reification", s, f"<{RDF}predicate>", P(r["predicate"]))
        T("reification", s, f"<{RDF}object>", tgt[r["object_kind"]](r["object"]))
        T("reification", s, P("validFrom"), lit(round(r["interval"][0], 1), XSD + "double"))
        T("reification", s, P("validTo"), lit(round(r["interval"][1], 1), XSD + "double"))
    for ev in sim["events"]:
        s = stmt("e")
        T("reification", s, f"<{RDF}type>", P("Statement"))
        T("reification", s, P("aboutEvent"), iri("event", ev["id"]))
        T("reification", s, P("eventKind"), lit(ev["type"]))
        T("reification", s, P("validFrom"), lit(round(ev["interval"][0], 1), XSD + "double"))
        T("reification", s, P("validTo"), lit(round(ev["interval"][1], 1), XSD + "double"))
        T("reification", s, P("provenance"), lit(ev.get("prov", "record")))

    CATS = ["entity", "unit", "landmark", "event", "relation", "reification"]
    seen = set(); out_lines = []; cat_counts = {c: 0 for c in CATS}; dupes = 0
    for c in CATS:
        for ln in cats[c]:
            if ln in seen:
                dupes += 1; continue
            seen.add(ln); out_lines.append(ln); cat_counts[c] += 1
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "unified_stkg.nt").write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    # jsonl + parquet(with s) + terrain
    with open(outdir / "entities.jsonl", "w", encoding="utf-8") as fh:
        for a in agents.values():
            fh.write(json.dumps({"id": a.eid, "true_type": a.typ, "affiliation": a.aff,
                "size": __import__("world.common", fromlist=["SIZE_BY_CATEGORY"]).SIZE_BY_CATEGORY.get(ontology.category(a.typ), "unknown"),
                "parent_unit": a.unit, "role": a.role}, ensure_ascii=False) + "\n")
    with open(outdir / "events.jsonl", "w", encoding="utf-8") as fh:
        for ev in sorted(sim["events"], key=lambda e: (e["type"], e["interval"][0])):
            fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
    with open(outdir / "relations.jsonl", "w", encoding="utf-8") as fh:
        for r in sim["relations"]:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    cols = ["entity_id", "t", "easting", "northing", "elevation", "state", "speed", "heading", "s"]
    pq.write_table(pa.table({c: [row[c] for row in traj] for c in cols}), outdir / "trajectories.parquet")
    _write_terrain(outdir / "terrain", terrain, tmeta, sim["landmarks"])

    # ---- P4 validation (in-memory) ----
    by_ent = defaultdict(list)
    for r in traj:
        by_ent[r["entity_id"]].append(r)
    mono = feas = sspd = 0
    states_ok = set(ontology.state_names); types_ok = set(ontology.type_names); rels_ok = set(ontology.relation_names)
    for eid, rows in by_ent.items():
        rows.sort(key=lambda r: r["t"])
        ts = [r["t"] for r in rows]
        mono += sum(1 for a, b in zip(ts, ts[1:]) if b <= a)
        typ = agents[eid].typ
        for a, b in zip(rows, rows[1:]):
            gap = max(1e-6, b["t"] - a["t"])
            if math.hypot(b["easting"] - a["easting"], b["northing"] - a["northing"]) / gap > ontology.v_max(typ, b["state"]) + 1e-2:
                feas += 1
        for r in rows:
            if r["state"] in STATIC and abs(r["speed"]) > 1e-2:
                sspd += 1
    voc = sum(1 for r in traj if r["state"] not in states_ok) + sum(1 for a in agents.values() if a.typ not in types_ok) + \
          sum(1 for r in sim["relations"] if r["predicate"] not in rels_ok)
    n_lines = sum(1 for _ in open(outdir / "unified_stkg.nt", encoding="utf-8"))
    xref = 0
    for r in sim["relations"]:
        if r["subject"] not in ent_ids:
            xref += 1
        pool = {"entity": ent_ids, "unit": unit_ids, "landmark": lm_ids}[r["object_kind"]]
        if r["object"] not in pool:
            xref += 1
    report = {
        "time_monotonic": {"pass": mono == 0, "violations": mono},
        "feasible_motion": {"pass": feas == 0, "violations": feas},
        "static_speed_zero": {"pass": sspd == 0, "violations": sspd},
        "vocabulary": {"pass": voc == 0, "violations": voc},
        "cross_reference": {"pass": xref == 0, "violations": xref},
        "triple_count": {"pass": n_lines == len(out_lines), "nt_lines": n_lines, "manifest": len(out_lines)},
        "no_duplicate_triples": {"pass": True, "duplicates": 0},
    }
    # independent dup check
    seen2 = set(); d2 = 0
    for ln in open(outdir / "unified_stkg.nt", encoding="utf-8"):
        if ln in seen2:
            d2 += 1
        seen2.add(ln)
    report["no_duplicate_triples"] = {"pass": d2 == 0, "duplicates": d2}

    # ---- v2 layer-consistency checks (close the audit gaps) ----
    from world.scenario_battle10d import DAY as _DAY, is_night as _isnight
    # op<->trajectory: every participating squad has >=2 samples in the op window incl >=1 moving
    op_bad = []
    for op in sim["ops"]:
        if op.get("kind") not in ("attack", "counterattack") or "squad_ids" not in op:
            continue
        w0, w1 = op["window"]
        for eid in op["squad_ids"]:
            rows = [r for r in by_ent.get(eid, []) if w0 - 1 <= r["t"] <= w1 + 1]
            if len(rows) < 2 or not any(r["state"] in ("Approaching", "Moving", "Withdrawing") for r in rows):
                op_bad.append((op.get("note", op["kind"])[:24], eid))
    report["op_trajectory_consistency"] = {"pass": not op_bad, "violations": len(op_bad), "examples": op_bad[:4]}
    # daily Approaching sample counts; attack-day must be > 0
    appr_by_day = defaultdict(int)
    for r in traj:
        if r["state"] == "Approaching":
            appr_by_day[int(r["t"] // _DAY)] += 1
    attack_days = {int(op["t0"] // _DAY) for op in sim["ops"] if op.get("kind") == "attack" and op["t0"] < W}
    zero_days = sorted(d for d in attack_days if appr_by_day.get(d, 0) == 0)
    report["daily_approaching_on_attack_days"] = {"pass": not zero_days, "zero_days": zero_days}
    # fog window: friendly firesAt INTERVAL overlap == 0 (interval level, not just starts)
    friendly = {a.eid for a in agents.values() if a.aff in ("ROK", "US")}
    fog = sim["fog"]
    fog_overlap = 0
    for r in sim["relations"]:
        if r["predicate"] == "firesAt" and r["subject"] in friendly:
            a, b = r["interval"]
            if any(a < fb and b > fa for fa, fb in fog):
                fog_overlap += 1
    report["fog_friendly_fire_overlap"] = {"pass": fog_overlap == 0, "overlaps": fog_overlap}
    # withdrawn squads end off-map (last sample beyond the north boundary)
    bx, by = sim["boundary"]
    off_bad = []
    for eid, rows in by_ent.items():
        if agents[eid].role != "assault":
            continue
        last = max(rows, key=lambda r: r["t"])
        if last["state"] == "Withdrawing" and last["northing"] < by - 100:
            off_bad.append(eid)
    report["withdrawn_end_off_map"] = {"pass": not off_bad, "on_map_at_end": off_bad[:4]}
    p4_ok = all(v["pass"] for v in report.values())

    # ---- realism / dynamism checks ----
    state_dist = Counter(r["state"] for r in traj)
    moving_ratio = sum(state_dist[s] for s in MOVING) / max(1, sum(state_dist.values()))
    ccf_final = [a.s for a in agents.values() if a.aff == "CCF" and a.role != "wreck"]
    rok_final = [a.s for a in agents.values() if a.aff in ("ROK", "US")]
    ccf_dead = sum(1 for a in agents.values() if a.aff == "CCF" and not a.alive)
    rok_dead = sum(1 for a in agents.values() if a.aff in ("ROK", "US") and not a.alive)
    daily = sim["daily"]
    ccf_loss = sum(d["ccf_loss"] for d in daily.values()); rok_loss = sum(d["rok_loss"] for d in daily.values())
    loss_ratio = ccf_loss / max(1e-6, rok_loss)
    merges = sum(d["merges"] for d in daily.values()); reliefs = sum(d["reliefs"] for d in daily.values())
    overwatch = sum(d["overwatch"] for d in daily.values())
    owner_changes = sim["owner_changes"]
    # background fire coverage
    fire_iv = sorted([e["interval"] for e in sim["events"] if e["type"] == "FireSupport"])
    covered = 0.0; cur = 0.0
    for a, b in fire_iv:
        a = max(a, cur);
        if b > a:
            covered += b - a; cur = b
        elif b > cur:
            cur = b
    fire_cov = covered / W
    # night attack ratio
    atk_starts = [e["interval"][0] for e in sim["events"] if e["type"] == "Attack"]
    night_atk = sum(1 for t in atk_starts if is_night(t)) / max(1, len(atk_starts))
    # D3 fog: friendly FireSupport count in fog window == 0
    fog = sim["fog"][0] if sim["fog"] else (0, 0)
    fog_fire = sum(1 for e in sim["events"] if e["type"] == "FireSupport" and fog[0] <= e["interval"][0] <= fog[1])
    # boundary entry/exit
    bx, by = sim["L"]["boundary"]
    ccf_assault = [a for a in agents.values() if a.role == "assault"]
    enter_ok = all(ag.samples_mission[0][0][2] >= by - 100 for ag in ccf_assault if getattr(ag, "samples_mission", None))
    # coords in extent
    ext_e = terrain.origin_e + terrain.res * (terrain.elevation.shape[1] - 1)
    ext_n = terrain.origin_n + terrain.res * (terrain.elevation.shape[0] - 1)
    xs = [r["easting"] for r in traj]; ys = [r["northing"] for r in traj]
    in_extent = min(xs) >= terrain.origin_e and max(xs) <= ext_e and min(ys) >= terrain.origin_n and max(ys) <= ext_n
    # landmark peak match
    ei, ej = np.unravel_index(int(np.argmax(terrain.elevation)), terrain.elevation.shape)
    peak_xy = (terrain.origin_e + ej * terrain.res, terrain.origin_n + ei * terrain.res)
    peak_match = abs(sim["landmarks"]["lm_crest"].easting - peak_xy[0]) < 1e-6

    realism = {
        "concurrent_alive_peak": None, "moving_ratio": round(moving_ratio, 3),
        "ccf_mean_s": round(float(np.mean(ccf_final)), 3), "rok_mean_s": round(float(np.mean(rok_final)), 3),
        "ccf_destroyed": ccf_dead, "rok_destroyed": rok_dead,
        "loss_ratio_ccf_to_rok": round(loss_ratio, 2), "merges": merges, "reliefs": reliefs, "overwatch": overwatch,
        "crest_owner_changes": len(owner_changes), "background_fire_coverage": round(fire_cov, 3),
        "night_attack_ratio": round(night_atk, 3), "fog_window_friendly_fire": fog_fire,
        "ccf_enter_from_boundary": bool(enter_ok), "coords_in_extent": bool(in_extent), "landmark_peak_match": bool(peak_match),
    }
    # concurrent alive
    probe = [k * 3600 for k in range(int(W / 3600) + 1)]
    life = {a.eid: (a.t_enter, W if a.t_exit is None else a.t_exit,
                    next((tr["t"] for tr in sim["transitions"] if tr["entity"] == a.eid and tr["kind"] == "destroyed"), None))
            for a in agents.values()}
    alive_series = [sum(1 for (lo, hi, dd) in life.values() if lo <= t <= hi and (dd is None or t < dd)) for t in probe]
    realism["concurrent_alive_peak"] = max(alive_series)
    (outdir / "validation_report.json").write_text(json.dumps({"p4": report, "realism": realism}, ensure_ascii=False, indent=2), encoding="utf-8")

    _preview_plot(outdir, traj, sim["landmarks"])

    # ---- battle_log.md ----
    log = ["# Hill 395 — 10-day battle GT chronicle (synthetic; record=source-based, interp=designed gap-fill)\n"]
    for op in sim["ops"]:
        day = int(op["t0"] // DAY) + 6
        hh = int((op["t0"] % DAY) // 3600); mm = int((op["t0"] % 3600) // 60)
        log.append(f"- 10/{day:02d} {hh:02d}:{mm:02d}  [{op['prov']}]  {op['kind']}: {op.get('note','')}")
    (outdir / "battle_log.md").write_text("\n".join(log) + "\n", encoding="utf-8")

    tot = len(out_lines)
    manifest = {"generator": "gen_world_gt.py (battle)", "scope": "P1-P4,P8 (two-layer director+rules 10-day battle GT)",
                "global_seed": seed, "config": cfg, "ontology_dir": ont_dir,
                "counts": {"n_triples": tot, "n_entities": len(agents), "n_events": len(sim["events"]),
                           "n_relations": len(sim["relations"]), "n_transitions": len(sim["transitions"]),
                           "n_traj_rows": len(traj)},
                "triple_categories": cat_counts, "director_ops": sim["ops"],
                "provenance_map": {op.get("note", str(i)): op["prov"] for i, op in enumerate(sim["ops"])},
                "daily_summary": {str(k): v for k, v in sorted(daily.items())},
                "crest_owner_changes": [[round(t, 1), o] for t, o in owner_changes],
                "realism": realism, "combat_power_s": "ontology EXTENSION (not in ontology) — 확인 필요",
                "ontology_vocab": {"types": ontology.type_names, "states": ontology.state_names, "relations": ontology.relation_names},
                "content_sha256": {f: _sha256(outdir / f) for f in
                    ["unified_stkg.nt", "entities.jsonl", "trajectories.parquet", "events.jsonl", "relations.jsonl"] if (outdir / f).exists()},
                "validation_pass": p4_ok}
    (outdir / "gt_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- console report ----
    print("   P4:")
    for kk, vv in report.items():
        print(f"     [{'OK ' if vv['pass'] else 'FAIL'}] {kk}: {vv}")
    print(f"   triples={tot:,}  categories={cat_counts}")
    print(f"   entities={len(agents)} events={len(sim['events'])} relations={len(sim['relations'])} transitions={len(sim['transitions'])} traj_rows={len(traj)}")
    print("   realism:")
    for kk, vv in realism.items():
        print(f"     {kk}: {vv}")
    print(f"   daily Approaching samples: {dict(sorted(appr_by_day.items()))}")
    print(f"   attack days: {sorted(attack_days)}  zero-Approaching attack days: {zero_days}")
    print(f"   states={dict(state_dist)}")
    print(f"   crest owner timeline: {[(round(t/DAY,2), o) for t, o in owner_changes][:16]}")
    # causal chain + prep->counter example from D4 (day3)
    d4 = [op for op in sim["ops"] if op.get("prep_max")]
    print("   PREP-FIRE -> COUNTERATTACK (D4):")
    for op in d4[:1]:
        print(f"     prep FireSupport [{op['t0']}~{op['t0']+op['prep']}] BEFORE Counterattack [{op['t0']+op['prep']}~{op['retake_t']}]  (prep precedes: {op['t0']+op['prep'] <= op['retake_t']})")
    if not p4_ok:
        raise SystemExit("P4 FAILED — see validation_report.json")


def run_micro(cfg, ontology, outdir: Path, seed: int, ont_dir: str) -> None:
    """Small single-engagement GT (no sectors): ~40 objects that actually move,
    with an explicit causal chain (approach -> detect -> request -> fire ->
    halt/withdraw). Triple count is a RESULT, not a target. Readability checks
    (concurrent alive, moving ratio, one causal chain) are printed."""
    from world.terrain_real import build_terrain_real
    from world.scenario_micro import build_micro
    from world.trajectories import sample_sparse
    from world.serialize_real import StreamWriterReal, CATEGORIES

    reify_preds = cfg.get("reification", {}).get("predicates",
                  ["engagedWith", "firesAt", "movesToward", "occupies", "withdrawsFrom", "supports", "screens"])
    reify_cap = int(cfg.get("reification", {}).get("per_sector_cap", 100000))

    terrain, terrain_meta = build_terrain_real(cfg)
    rng = np.random.default_rng(seed)
    entities, units, landmarks, events, relations, chain = build_micro(cfg, ontology, terrain, rng)
    for lm in landmarks.values():
        lm.elevation = terrain.elev_at(lm.easting, lm.northing)
    traj = sample_sparse(entities, ontology, terrain, cfg)

    writer = StreamWriterReal(outdir, reify_preds, reify_cap)
    writer.write_sector(entities, units, landmarks, events, relations, traj)
    totals, cat_counts = writer.close()
    _write_terrain(outdir / "terrain", terrain, terrain_meta, landmarks)

    # dup check
    seen = set(); n_lines = 0; n_dupes = 0
    for ln in open(outdir / "unified_stkg.nt", encoding="utf-8"):
        n_lines += 1
        if ln in seen:
            n_dupes += 1
        else:
            seen.add(ln)

    v = validate_sector(cfg, ontology, entities, units, landmarks, events, relations, traj)
    report = {}
    for k in ["time_monotonic", "feasible_motion", "static_speed_zero",
              "required_attributes", "vocabulary", "cross_reference"]:
        report[k] = {"pass": len(v.get(k, [])) == 0, "violations": len(v.get(k, [])), "examples": v.get(k, [])[:3]}
    report["triple_count"] = {"pass": n_lines == totals["n_triples"], "nt_lines": n_lines, "manifest_triples": totals["n_triples"]}
    report["no_duplicate_triples"] = {"pass": n_dupes == 0, "duplicates": n_dupes}
    integrity_ok = all(x["pass"] for x in report.values())
    report["_all_pass"] = integrity_ok
    (outdir / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    _preview_plot(outdir, traj, landmarks)

    # readability / dynamism checks
    W = float(cfg["time"]["window_seconds"]); dt = float(cfg["time"]["dt_seconds"])
    from collections import defaultdict, Counter
    life = {}
    for r in traj:
        life.setdefault(r["entity_id"], [1e18, -1, None])
        life[r["entity_id"]][0] = min(life[r["entity_id"]][0], r["t"])
        life[r["entity_id"]][1] = max(life[r["entity_id"]][1], r["t"])
        if r["state"] == "Destroyed" and life[r["entity_id"]][2] is None:
            life[r["entity_id"]][2] = r["t"]
    probe = [k * dt for k in range(int(W / dt) + 1)]
    alive_series = [sum(1 for (a, b, dd) in life.values() if a <= t <= b and (dd is None or t < dd)) for t in probe]
    peak_alive = max(alive_series)
    state_dist = Counter(r["state"] for r in traj)
    moving = sum(state_dist[s] for s in ("Approaching", "Moving", "Withdrawing"))
    moving_ratio = moving / max(1, sum(state_dist.values()))
    force = Counter("enemy" if e.affiliation == "CCF" else ("friendly" if e.affiliation in ("ROK", "US", "UN") else "other") for e in entities)
    cat = Counter(ontology.category(e.true_type) for e in entities)

    # landmark<->peak coincidence
    ei, ej = np.unravel_index(int(np.argmax(terrain.elevation)), terrain.elevation.shape)
    peak_xy = (terrain.origin_e + ej * terrain.res, terrain.origin_n + ei * terrain.res)
    crest = landmarks["lm_crest"]
    peak_match = abs(crest.easting - peak_xy[0]) < 1e-6 and abs(crest.northing - peak_xy[1]) < 1e-6
    # coords within extent?
    xs = [r["easting"] for r in traj]; ys = [r["northing"] for r in traj]
    ext_e = terrain.origin_e + terrain.res * (terrain.elevation.shape[1] - 1)
    ext_n = terrain.origin_n + terrain.res * (terrain.elevation.shape[0] - 1)
    in_extent = (min(xs) >= terrain.origin_e and max(xs) <= ext_e and min(ys) >= terrain.origin_n and max(ys) <= ext_n)

    manifest = {"generator": "gen_world_gt.py (micro)", "scope": "P1-P4,P8 (sensor-neutral GT, readable single engagement)",
                "global_seed": seed, "config": cfg, "counts": totals, "triple_categories": cat_counts,
                "readability": {"concurrent_alive_peak": peak_alive, "moving_ratio": round(moving_ratio, 3),
                                "landmark_peak_match": bool(peak_match), "coords_in_extent": bool(in_extent)},
                "ontology_dir": ont_dir,
                "ontology_vocab": {"types": ontology.type_names, "states": ontology.state_names,
                                   "relations": ontology.relation_names, "event_kinds": EVENT_KINDS},
                "content_sha256": {f: _sha256(outdir / f) for f in
                    ["unified_stkg.nt", "entities.jsonl", "trajectories.parquet", "events.jsonl",
                     "relations.jsonl", "terrain/elevation.npy", "terrain/landmarks.json"] if (outdir / f).exists()},
                "validation_pass": integrity_ok}
    (outdir / "gt_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("   validation:")
    for k, x in report.items():
        if k != "_all_pass":
            print(f"     [{'OK ' if x['pass'] else 'FAIL'}] {k}: {x}")
    tot = totals["n_triples"]
    print(f"   triple categories ({tot:,}):")
    for c in CATEGORIES:
        print(f"     {c:18s} {cat_counts[c]:>7,}  ({100*cat_counts[c]/max(1,tot):4.1f}%)")
    print(f"   entities={totals['n_entities']} force={dict(force)} category={dict(cat)}")
    print(f"   concurrent alive: peak={peak_alive}  |  moving-state ratio={moving_ratio:.2f}")
    print(f"   states={dict(state_dist)}")
    print(f"   coords_in_extent={in_extent}  landmark_peak_match={peak_match}  peak_local=({peak_xy[0]-terrain.origin_e:.0f},{peak_xy[1]-terrain.origin_n:.0f}) elev={float(terrain.elevation[ei,ej]):.0f}m")
    if chain:
        print("   CAUSAL CHAIN example (one wave):")
        print(f"     1) CCF squad {chain['ccf_squad']} Approaching lane   t={chain['t_move'][0]:.0f}~{chain['t_move'][1]:.0f}s")
        print(f"     2) defender  {chain['defender']} engagedWith it       t={chain['t_detect'][0]:.0f}~{chain['t_detect'][1]:.0f}s")
        print(f"     3) defender requests support (supports unit)          t={chain['t_move'][0]+1200:.0f}~{chain['t_fire'][0]:.0f}s")
        print(f"     4) shooters {chain['shooters']} FireSupport ev {chain['fire_event']} firesAt lane  t={chain['t_fire'][0]:.0f}~{chain['t_fire'][1]:.0f}s")
        print(f"     5) CCF squad {chain['ccf_squad']} -> Halted then Withdrawing/Destroyed  t>{chain['t_fire'][1]:.0f}s")
    print(f"== triples={tot:,} entities={totals['n_entities']} traj_rows={totals['n_traj_rows']} dupes_removed={totals['n_dupes_removed']} -> {outdir} ==")
    if not integrity_ok:
        raise SystemExit("P4 integrity FAILED — see validation_report.json")


def run_real(cfg, ontology, outdir: Path, seed: int, ont_dir: str) -> None:
    """Real-scale, FIXED-space GT. Space is a single ~6x8 km battlefield on a small
    grid; the >=300k triple floor is met by non-spatial growth (time density,
    relations, events, reification), never by adding sectors. Streaming + dedup."""
    from world.terrain_real import build_terrain_real
    from world.scenario_real import build_sector_real
    from world.trajectories import sample_sparse
    from world.serialize_real import StreamWriterReal, CATEGORIES
    from world.common import BASE_E, BASE_N

    floor = int(cfg.get("target_triples_floor", 300000))
    ws = cfg["world_scope"]
    cols = int(ws["cols"]); rows = int(ws["rows"]); pitch = float(ws["sector_pitch_m"])
    n_sectors = cols * rows
    reify_preds = cfg.get("reification", {}).get("predicates",
                  ["engagedWith", "firesAt", "movesToward", "occupies", "withdrawsFrom",
                   "supports", "screens", "reinforces"])
    reify_cap = int(cfg.get("reification", {}).get("per_sector_cap", 100000))

    terrain, terrain_meta = build_terrain_real(cfg)
    print(f"   terrain: {terrain.elevation.shape} peak={terrain_meta['elevation_max']:.0f}m "
          f"area={terrain_meta['area_width_m']:.0f}x{terrain_meta['area_height_m']:.0f}m")
    print(f"   grid: {cols}x{rows}={n_sectors} sectors, pitch={pitch}m")

    writer = StreamWriterReal(outdir, reify_preds, reify_cap)
    rng = np.random.default_rng(seed); eid_ctr = [0]; evid_ctr = [0]
    all_landmarks = {}; agg = {}; examples = {}; first_traj = None; first_lms = None

    for s in range(n_sectors):
        ox = BASE_E + (s % cols) * pitch; oy = BASE_N + (s // cols) * pitch
        ents, units, lms, evs, rels = build_sector_real(s, (ox, oy), cfg, ontology, rng, eid_ctr, evid_ctr)
        for lm in lms.values():
            lm.elevation = terrain.elev_at(lm.easting, lm.northing)
        traj = sample_sparse(ents, ontology, terrain, cfg)
        writer.write_sector(ents, units, lms, evs, rels, traj)
        v = validate_sector(cfg, ontology, ents, units, lms, evs, rels, traj)
        for k, viol in v.items():
            agg[k] = agg.get(k, 0) + len(viol)
            if viol and k not in examples:
                examples[k] = viol[:3]
        all_landmarks.update(lms)
        if s == 0:
            first_traj = traj; first_lms = dict(lms)

    totals, cat_counts = writer.close()
    _write_terrain(outdir / "terrain", terrain, terrain_meta, all_landmarks)

    # duplicate verification (global): count non-unique nt lines
    seen = set(); n_lines = 0; n_dupes = 0
    for ln in open(outdir / "unified_stkg.nt", encoding="utf-8"):
        n_lines += 1
        if ln in seen:
            n_dupes += 1
        else:
            seen.add(ln)

    report = {}
    for k in ["time_monotonic", "feasible_motion", "static_speed_zero",
              "required_attributes", "vocabulary", "cross_reference"]:
        report[k] = {"pass": agg.get(k, 0) == 0, "violations": agg.get(k, 0), "examples": examples.get(k, [])}
    report["triple_count"] = {"pass": n_lines == totals["n_triples"],
                              "nt_lines": n_lines, "manifest_triples": totals["n_triples"]}
    report["no_duplicate_triples"] = {"pass": n_dupes == 0, "duplicates": n_dupes}
    report["triple_floor"] = {"pass": totals["n_triples"] >= floor, "floor": floor,
                              "actual": totals["n_triples"]}
    integrity_ok = all(report[k]["pass"] for k in report if k != "triple_floor")
    report["_all_pass"] = integrity_ok and report["triple_floor"]["pass"]
    (outdir / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if first_traj:
        _preview_plot(outdir, first_traj, first_lms)

    means = {"(a) time_density": cat_counts["trajectory_state"], "(b) relations": cat_counts["relation"],
             "(c) events": cat_counts["event"], "(d) reification": cat_counts["reification"],
             "structural(entity/unit/landmark)": cat_counts["entity_identity"] + cat_counts["unit_echelon"] + cat_counts["landmark_other"]}

    data_files = ["unified_stkg.nt", "entities.jsonl", "trajectories.parquet", "events.jsonl",
                  "relations.jsonl", "terrain/elevation.npy", "terrain/concealment.npy",
                  "terrain/terrain_meta.json", "terrain/landmarks.json"]
    hashes = {f: _sha256(outdir / f) for f in data_files if (outdir / f).exists()}
    manifest = {"generator": "gen_world_gt.py (real/streaming)",
                "scope": "P1-P4,P8 (sensor-neutral GT, real-scale fixed space)",
                "global_seed": seed, "config": cfg, "counts": totals,
                "triple_categories": cat_counts, "growth_means_contribution": means,
                "grid": {"cols": cols, "rows": rows, "sectors": n_sectors, "pitch_m": pitch},
                "area_km": {"ew": terrain_meta["area_width_m"] / 1000, "ns": terrain_meta["area_height_m"] / 1000},
                "ontology_dir": ont_dir,
                "ontology_vocab": {"types": ontology.type_names, "states": ontology.state_names,
                                   "relations": ontology.relation_names, "event_kinds": EVENT_KINDS},
                "content_sha256": hashes, "validation_pass": report["_all_pass"]}
    (outdir / "gt_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("   validation:")
    for k, v in report.items():
        if k != "_all_pass":
            print(f"     [{'OK ' if v['pass'] else 'FAIL'}] {k}: {v}")
    tot = totals["n_triples"]
    print(f"   triple categories ({tot:,} total):")
    for c in CATEGORIES:
        print(f"     {c:18s} {cat_counts[c]:>9,}  ({100*cat_counts[c]/max(1,tot):4.1f}%)")
    print(f"   growth means: " + "  ".join(f"{k}={v:,}" for k, v in means.items()))
    print(f"== triples={tot:,} sectors={n_sectors} entities={totals['n_entities']:,} "
          f"traj_rows={totals['n_traj_rows']:,} dupes_removed={totals['n_dupes_removed']:,} -> {outdir} ==")
    if not integrity_ok:
        raise SystemExit("P4 integrity FAILED — see validation_report.json")


def run_full_streaming(cfg, ontology, outdir: Path, seed: int, ont_dir: str) -> None:
    """target_triples-driven, sector-streaming generation for the full battle GT.
    Density is probed from one sector, the sector count is set to hit the target,
    then sectors are generated/written/validated one at a time (bounded memory)."""
    target = int(cfg["target_triples"]); tol = float(cfg.get("tolerance", 0.05))

    # --- density probe: build sector 0 (throwaway rng) to size the run ---
    from world.serialize import nt_lines
    pe, pu, pl, pv, pr = build_sector_world(0, cfg, ontology, np.random.default_rng(seed), [0], [0])
    per_sector_triples = len(nt_lines(pe, pu, pl, pv, pr))
    n_sectors = max(1, round(target / per_sector_triples))
    cfg["world_scope"]["n_sectors"] = n_sectors
    print(f"   probe: {per_sector_triples} triples/sector, {len(pe)} entities/sector "
          f"-> n_sectors={n_sectors} for target {target:,}")

    # --- terrain sized to the chosen grid ---
    terrain, terrain_meta = build_terrain(cfg)

    writer = StreamWriter(outdir)
    rng = np.random.default_rng(seed); eid_ctr = [0]; evid_ctr = [0]
    all_landmarks: dict = {}
    agg = {}                                   # check -> total violations
    examples: dict = {}
    first_traj = None; first_lms = None
    W = float(cfg["time"]["window_seconds"])

    for s in range(n_sectors):
        ents, units, lms, evs, rels = build_sector_world(s, cfg, ontology, rng, eid_ctr, evid_ctr)
        for lm in lms.values():
            lm.elevation = terrain.elev_at(lm.easting, lm.northing)
        traj = sample_trajectories(ents, ontology, terrain, cfg)
        writer.write_sector(ents, units, lms, evs, rels, traj)
        v = validate_sector(cfg, ontology, ents, units, lms, evs, rels, traj)
        for k, viol in v.items():
            agg[k] = agg.get(k, 0) + len(viol)
            if viol and k not in examples:
                examples[k] = viol[:3]
        all_landmarks.update(lms)
        if s == 0:
            first_traj = traj; first_lms = dict(lms)
        if (s + 1) % 50 == 0 or s == n_sectors - 1:
            print(f"   sector {s + 1}/{n_sectors}  triples so far={writer.totals['n_triples']:,}")

    totals = writer.close()

    # terrain files (written once; spans the whole map)
    _write_terrain(outdir / "terrain", terrain, terrain_meta, all_landmarks)

    # P4 report: aggregated per-sector checks + triple count + tolerance
    nt_lines_count = sum(1 for _ in open(outdir / "unified_stkg.nt", encoding="utf-8"))
    report = {}
    for k in ["time_monotonic", "feasible_motion", "static_speed_zero",
              "required_attributes", "vocabulary", "cross_reference"]:
        report[k] = {"pass": agg.get(k, 0) == 0, "violations": agg.get(k, 0),
                     "examples": examples.get(k, [])}
    report["triple_count"] = {"pass": nt_lines_count == totals["n_triples"],
                              "nt_lines": nt_lines_count, "manifest_triples": totals["n_triples"]}
    lo, hi = target * (1 - tol), target * (1 + tol)
    within = lo <= totals["n_triples"] <= hi
    report["triple_target"] = {"pass": within, "target": target, "tolerance": tol,
                               "actual": totals["n_triples"], "range": [lo, hi]}
    integrity_ok = all(report[k]["pass"] for k in report if k != "triple_target")
    report["_all_pass"] = integrity_ok and within
    (outdir / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if first_traj is not None:
        _preview_plot(outdir, first_traj, first_lms)   # preview from sector 0

    data_files = ["unified_stkg.nt", "entities.jsonl", "trajectories.parquet",
                  "events.jsonl", "relations.jsonl", "terrain/elevation.npy",
                  "terrain/concealment.npy", "terrain/terrain_meta.json", "terrain/landmarks.json"]
    hashes = {f: _sha256(outdir / f) for f in data_files if (outdir / f).exists()}
    manifest = {"generator": "gen_world_gt.py (full/streaming)",
                "scope": "P1-P4,P8 (sensor-neutral GT, multi-phase, streaming)",
                "global_seed": seed, "config": cfg, "counts": totals, "ontology_dir": ont_dir,
                "per_sector_triples_probe": per_sector_triples, "n_sectors": n_sectors,
                "ontology_vocab": {"types": ontology.type_names, "states": ontology.state_names,
                                   "relations": ontology.relation_names, "event_kinds": EVENT_KINDS},
                "content_sha256": hashes, "validation_pass": report["_all_pass"]}
    (outdir / "gt_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("   validation:")
    for k, v in report.items():
        if k == "_all_pass":
            continue
        print(f"     [{'OK ' if v['pass'] else 'FAIL'}] {k}: {v}")
    print(f"== triples={totals['n_triples']:,} sectors={n_sectors} entities={totals['n_entities']:,} "
          f"traj_rows={totals['n_traj_rows']:,}  (no observations; true state only)")
    print(f"   validation_pass={report['_all_pass']}  -> {outdir} ==")
    if not integrity_ok:
        raise SystemExit("P4 integrity FAILED — see validation_report.json")


def main() -> None:
    ap = argparse.ArgumentParser(description="Hill 395 world-GT generator (P1-P4, P8)")
    ap.add_argument("--config", default=str(REPO / "configs" / "world_gt_small.yaml"))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    seed = int(cfg["global_seed"])
    outdir = Path(cfg["output_dir"])
    if not outdir.is_absolute():
        outdir = REPO / outdir
    # safety: never write into a frozen suite
    if any(part.startswith("battlefield_") for part in outdir.parts):
        raise SystemExit(f"refusing to write into a frozen suite path: {outdir}")

    ont_dir = cfg.get("ontology_dir", "data/battlefield_hill395_large/ontology")
    ontology = load_ontology(str(REPO / ont_dir) if not Path(ont_dir).is_absolute() else ont_dir)
    print(f"== world-GT: seed={seed} out={outdir} ontology={ont_dir} ==")
    print(f"   vocab: types={len(ontology.type_names)} states={len(ontology.state_names)} "
          f"relations={len(ontology.relation_names)}")

    if cfg.get("mode") == "battle_1m":
        run_battle_1m(cfg, ontology, outdir, seed, ont_dir)
        return
    if cfg.get("mode") == "battle":
        run_battle(cfg, ontology, outdir, seed, ont_dir)
        return
    if cfg.get("mode") == "micro":
        run_micro(cfg, ontology, outdir, seed, ont_dir)
        return
    if cfg.get("mode") == "real":
        run_real(cfg, ontology, outdir, seed, ont_dir)
        return
    if "target_triples" in cfg:
        run_full_streaming(cfg, ontology, outdir, seed, ont_dir)
        return

    rng = np.random.default_rng(seed)

    # P1 entities + org + landmarks + events
    entities, units, landmarks, events = build_world(cfg, ontology, rng)
    # P2 terrain, then stamp landmark elevations from the field
    terrain, terrain_meta = build_terrain(cfg)
    for lm in landmarks.values():
        lm.elevation = terrain.elev_at(lm.easting, lm.northing)
    # P1 motion
    traj_rows = sample_trajectories(entities, ontology, terrain, cfg)
    # P1 relations with intervals
    relations = build_relations(entities, cfg)

    print(f"   built: entities={len(entities)} units={len(units)} landmarks={len(landmarks)} "
          f"events={len(events)} relations={len(relations)} traj_rows={len(traj_rows)}")

    # P3 serialise
    outdir.mkdir(parents=True, exist_ok=True)
    counts = serialize_world(outdir, entities, units, landmarks, events, relations,
                             traj_rows, terrain, terrain_meta)

    # P4 validate
    art = {"entities": entities, "units": units, "landmarks": landmarks,
           "events": events, "relations": relations, "traj_rows": traj_rows}
    report, ok = run_validation(cfg, ontology, art, outdir / "unified_stkg.nt", counts["n_triples"])
    (outdir / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # visual check
    _preview_plot(outdir, traj_rows, landmarks)

    # P8 manifest + content hashes
    data_files = ["unified_stkg.nt", "entities.jsonl", "trajectories.parquet",
                  "events.jsonl", "relations.jsonl", "terrain/elevation.npy",
                  "terrain/concealment.npy", "terrain/terrain_meta.json",
                  "terrain/landmarks.json"]
    hashes = {f: _sha256(outdir / f) for f in data_files if (outdir / f).exists()}
    manifest = {"generator": "gen_world_gt.py", "scope": "P1-P4,P8 (sensor-neutral GT)",
                "global_seed": seed, "config": cfg, "counts": counts,
                "ontology_dir": ont_dir,
                "ontology_vocab": {"types": ontology.type_names,
                                   "states": ontology.state_names,
                                   "relations": ontology.relation_names,
                                   "event_kinds": EVENT_KINDS},
                "content_sha256": hashes, "validation_pass": ok}
    (outdir / "gt_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("   validation:")
    for k, v in report.items():
        if k == "_all_pass":
            continue
        flag = "OK " if v["pass"] else "FAIL"
        print(f"     [{flag}] {k}: {v}")
    print(f"== triples={counts['n_triples']}  validation_pass={ok}  -> {outdir} ==")
    if not ok:
        raise SystemExit("P4 validation FAILED — see validation_report.json")


if __name__ == "__main__":
    main()
