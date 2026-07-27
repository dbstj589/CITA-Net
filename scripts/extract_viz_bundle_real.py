#!/usr/bin/env python
"""Extract a lightweight visualisation bundle from the real-scale world GT.

Read-only on the GT (trajectories.parquet, entities.jsonl, events.jsonl,
terrain/, gt_manifest.json). unified_stkg.nt / validation_report.json are NOT
used. relations.jsonl is used only if --relation-snapshot is passed (default off).
No rendering / no model here.

    python scripts/extract_viz_bundle_real.py --gt data/hill395_world_gt_real --frames 720
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

# stated CONVENTION (ROK/UN defender viewpoint), not a guess:
AFFIL_TO_FORCE = {"ROK": "friendly", "US": "friendly", "UN": "friendly",
                  "CCF": "enemy", "NA": "other", "UNK": "other"}
AFFIL_TO_FORCE_NOTE = ("viewpoint convention (ROK/UN defender): ROK,US,UN->friendly, "
                       "CCF->enemy, NA/UNK->other. Stated perspective, not a guess.")
ALIVE_RULE = "alive = (t within entity's active window) AND (state != 'Destroyed')"
B2_THRESHOLD = 500


def sector_of(parent_unit, x, y, cols, rows, pitch):
    if parent_unit:                                   # units end with _<sector>
        try:
            return int(str(parent_unit).rsplit("_", 1)[-1])
        except ValueError:
            pass
    col = min(cols - 1, max(0, int(x // pitch)))       # wrecks (no unit): by position
    row = min(rows - 1, max(0, int(y // pitch)))
    return row * cols + col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="data/hill395_world_gt_real")
    ap.add_argument("--frames", type=int, default=720)
    ap.add_argument("--overview-frames", type=int, default=240)
    ap.add_argument("--relation-snapshot", action="store_true")
    args = ap.parse_args()

    gt = Path(args.gt); out = gt / "viz_bundle"; out.mkdir(parents=True, exist_ok=True)
    F = int(args.frames)
    notes = []

    manifest = json.load(open(gt / "gt_manifest.json", encoding="utf-8"))
    cfg = manifest["config"]; W = float(cfg["time"]["window_seconds"])
    dt = float(cfg["time"]["dt_seconds"]); epoch = cfg["time"]["epoch"]
    grid = manifest["grid"]; cols, rows, pitch = grid["cols"], grid["rows"], grid["pitch_m"]

    tmeta = json.load(open(gt / "terrain" / "terrain_meta.json", encoding="utf-8"))
    off_e, off_n = float(tmeta["origin_e"]), float(tmeta["origin_n"])
    res, nx, ny = float(tmeta["res_m"]), int(tmeta["nx"]), int(tmeta["ny"])
    elev = np.load(gt / "terrain" / "elevation.npy")
    np.save(out / "terrain_elevation.npy", elev)
    map_extent = {"xmin": 0.0, "xmax": res * (nx - 1), "ymin": 0.0, "ymax": res * (ny - 1)}

    ont_dir = manifest.get("ontology_dir", "data/battlefield_hill395_large/ontology")
    classes = yaml.safe_load(open(Path(ont_dir) / "classes.yaml", encoding="utf-8"))
    type_to_category = {t: v.get("category", "unknown") for t, v in classes.items()}

    # --- entities ---
    ents = [json.loads(l) for l in open(gt / "entities.jsonl", encoding="utf-8")]
    ent_out, ent_sector, ent_force = [], {}, {}
    for e in ents:
        cat = type_to_category.get(e["true_type"], "unknown")
        force = AFFIL_TO_FORCE.get(e.get("affiliation", "UNK"), "other")
        ent_force[e["id"]] = force
        ent_out.append({"entity_id": e["id"], "true_type": e["true_type"], "category": cat,
                        "affiliation": e.get("affiliation", "UNK"), "force": force,
                        "parent_unit": e.get("parent_unit")})

    # --- trajectories: per-entity samples ---
    tab = pq.read_table(gt / "trajectories.parquet").to_pydict()
    samples = defaultdict(list)
    for i in range(len(tab["entity_id"])):
        samples[tab["entity_id"][i]].append((tab["t"][i], tab["easting"][i], tab["northing"][i],
                                             tab["state"][i]))
    for eid in samples:
        samples[eid].sort()
    # sector per entity (unit-based; wrecks by position)
    pu = {e["entity_id"]: e["parent_unit"] for e in ent_out}
    for eid, ss in samples.items():
        ent_sector[eid] = sector_of(pu.get(eid), ss[0][1] - off_e, ss[0][2] - off_n, cols, rows, pitch)
    for e in ent_out:
        e["sector_id"] = ent_sector.get(e["entity_id"], 0)

    # per-entity lifecycle: enter/exit/destroyed_t
    life = {}
    for eid, ss in samples.items():
        ts = [s[0] for s in ss]
        dt_death = next((s[0] for s in ss if s[3] == "Destroyed"), None)
        life[eid] = (ts[0], ts[-1], dt_death)

    # ---------- (A) stats: concurrent alive at original GT steps ----------
    n_steps = int(round(W / dt))
    probe = [round(k * dt, 3) for k in range(n_steps + 1)]
    alive_series, sector_peak = [], defaultdict(int)
    for t in probe:
        alive_ids = [eid for eid, (a, b, dd) in life.items()
                     if a <= t <= b and (dd is None or t < dd)]
        alive_series.append(len(alive_ids))
        sc = Counter(ent_sector[eid] for eid in alive_ids)
        for s, c in sc.items():
            sector_peak[s] = max(sector_peak[s], c)
    peak = max(alive_series); peak_t = probe[int(np.argmax(alive_series))]
    med = float(np.median(alive_series)); mean = float(np.mean(alive_series))

    cat_dist = Counter(e["category"] for e in ent_out)
    force_dist = Counter(e["force"] for e in ent_out)
    state_dist = Counter(tab["state"])                       # over original samples
    sector_count = Counter(ent_sector.values())
    xs = [x - off_e for x in tab["easting"]]; ys = [y - off_n for y in tab["northing"]]
    bbox = {"xmin": min(xs), "xmax": max(xs), "ymin": min(ys), "ymax": max(ys)}
    if (bbox["xmin"] < map_extent["xmin"] or bbox["xmax"] > map_extent["xmax"]
            or bbox["ymin"] < map_extent["ymin"] or bbox["ymax"] > map_extent["ymax"]):
        notes.append("확인 필요: entity trajectory bbox extends OUTSIDE the terrain extent "
                     "(sector placement offsets exceed the 2km pitch). Terrain elevation lookups "
                     "clamp to grid edges; renderer should use trajectory_bbox for the canvas.")

    make_b2 = peak > B2_THRESHOLD

    stats = {"concurrent_alive": {"peak": peak, "peak_time_s": peak_t, "median": med,
                                  "mean": round(mean, 1), "series_step_s": dt,
                                  "series": alive_series},
             "category_distribution": dict(cat_dist), "force_distribution": dict(force_dist),
             "state_distribution": dict(state_dist),
             "coord_bbox_local_m": bbox, "map_extent_local_m": map_extent,
             "base_offset": {"easting": off_e, "northing": off_n},
             "per_sector": {str(s): {"entities": sector_count.get(s, 0),
                                     "peak_alive": sector_peak.get(s, 0)} for s in range(cols * rows)},
             "b2_overview_built": make_b2, "b2_threshold": B2_THRESHOLD,
             "n_events": int(manifest["counts"]["n_events"]),
             "n_entities": len(ent_out)}
    (out / "viz_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- (B1) trajectories resampled to F frames (active frames only) ----------
    def interp(ss, t):
        if t <= ss[0][0]:
            return ss[0][1], ss[0][2], ss[0][3]
        if t >= ss[-1][0]:
            return ss[-1][1], ss[-1][2], ss[-1][3]
        lo = 0
        for k in range(len(ss) - 1):
            if ss[k][0] <= t <= ss[k + 1][0]:
                lo = k; break
        (t0, e0, n0, s0), (t1, e1, n1, _s1) = ss[lo], ss[lo + 1]
        a = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        return e0 + a * (e1 - e0), n0 + a * (n1 - n0), s0     # state = floor

    fr, idc, xc, yc, stc, alc, secc = [], [], [], [], [], [], []
    for eid, ss in samples.items():
        a, b, _ = life[eid]
        sec = ent_sector[eid]
        for f in range(F):
            tf = (f / (F - 1)) * W if F > 1 else 0.0
            if tf < a - 1e-6 or tf > b + 1e-6:
                continue
            e_, n_, state = interp(ss, tf)
            fr.append(f); idc.append(eid); xc.append(round(e_ - off_e, 3)); yc.append(round(n_ - off_n, 3))
            stc.append(state); alc.append(state != "Destroyed"); secc.append(sec)
    pq.write_table(pa.table({"frame": pa.array(fr, pa.int32()), "entity_id": idc,
                             "x": xc, "y": yc, "state": stc,
                             "alive": pa.array(alc, pa.bool_()),
                             "sector_id": pa.array(secc, pa.int32())}),
                   out / "viz_trajectories.parquet")

    with open(out / "viz_entities.jsonl", "w", encoding="utf-8") as fh:
        for e in ent_out:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")

    # events -> frames, sorted by type
    lm_name = {lm["id"]: lm["name"] for lm in json.load(open(gt / "terrain" / "landmarks.json", encoding="utf-8"))}
    def s2f(t):
        return int(round((t / W) * (F - 1))) if F > 1 else 0
    evs = [json.loads(l) for l in open(gt / "events.jsonl", encoding="utf-8")]
    evs_out = sorted(({"type": e["type"], "start_frame": s2f(e["interval"][0]),
                       "end_frame": s2f(e["interval"][1]), "participants": e["participants"],
                       "landmark": lm_name.get(e["landmark"], e["landmark"])} for e in evs),
                     key=lambda r: (r["type"], r["start_frame"]))
    with open(out / "viz_events.jsonl", "w", encoding="utf-8") as fh:
        for r in evs_out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # landmarks (local), sector origins (local)
    lms = json.load(open(gt / "terrain" / "landmarks.json", encoding="utf-8"))
    lm_out = [{"name": lm["name"], "x": round(lm["easting"] - off_e, 2),
               "y": round(lm["northing"] - off_n, 2)} for lm in lms]
    sector_origins = [{"sector": s, "x": (s % cols) * pitch, "y": (s // cols) * pitch}
                      for s in range(cols * rows)]

    meta = {"source_gt": str(gt), "global_seed": manifest.get("global_seed"),
            "time": {"epoch": epoch, "window_seconds": W, "dt_seconds": dt, "frames": F,
                     "resample": (f"per-entity resample of sparse GT samples -> {F} frames within each "
                                  "entity's active window: linear interp of (x,y); state via floor "
                                  "(nearest-earlier) sample; rows only for active frames")},
            "map": {"unit": "meters", "base_offset": {"easting": off_e, "northing": off_n,
                    "note": "subtract from GT UTM coords for local x,y; equals terrain origin so "
                            "points align with terrain_elevation.npy"},
                    "extent": map_extent, "extent_source": "global terrain grid (origin + res*(n-1))",
                    "trajectory_bbox": bbox, "elevation_file": "terrain_elevation.npy",
                    "elevation_shape": list(elev.shape), "elevation_res_m": res},
            "sectors": {"cols": cols, "rows": rows, "sector_pitch_m": pitch,
                        "sector_origins_local": sector_origins,
                        "note": "sector_id from parent_unit suffix; wrecks (no unit) by position"},
            "landmarks": lm_out,
            "landmarks_note": "landmark names are per-sector procedural labels; they do NOT coincide "
                              "with the global terrain peak/ridge features (확인 필요).",
            "type_to_category": type_to_category,
            "affiliation_to_force": {"mapping": AFFIL_TO_FORCE, "note": AFFIL_TO_FORCE_NOTE},
            "alive_rule": ALIVE_RULE,
            "counts": {"entities_total": len(ent_out), "by_force": dict(force_dist),
                       "by_category": dict(cat_dist)},
            "notes_and_checks": notes or ["none"]}
    (out / "viz_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- (B2) overview density grid (conditional) ----------
    if make_b2:
        Fo = int(args.overview_frames)
        ev_active_by_frame = defaultdict(int)
        for e in evs:
            for f in range(s2f(e["interval"][0]), s2f(e["interval"][1]) + 1):
                ev_active_by_frame[f] += 1
        rows_o = []
        for fo in range(Fo):
            tf = (fo / (Fo - 1)) * W if Fo > 1 else 0.0
            alive_ids = [eid for eid, (a, b, dd) in life.items() if a <= tf <= b and (dd is None or tf < dd)]
            per = defaultdict(lambda: [0, 0, 0])
            for eid in alive_ids:
                s = ent_sector[eid]; per[s][0] += 1
                if ent_force[eid] == "enemy":
                    per[s][1] += 1
                elif ent_force[eid] == "friendly":
                    per[s][2] += 1
            fmain = int(round(fo / (Fo - 1) * (F - 1))) if Fo > 1 else 0
            for s in range(cols * rows):
                ac, en, fr_ = per.get(s, [0, 0, 0])
                rows_o.append({"frame": fo, "sector_id": s, "col": s % cols, "row": s // cols,
                               "alive_count": ac, "enemy_count": en, "friendly_count": fr_,
                               "event_active": ev_active_by_frame.get(fmain, 0)})
        cols_o = ["frame", "sector_id", "col", "row", "alive_count", "enemy_count", "friendly_count", "event_active"]
        pq.write_table(pa.table({c: [r[c] for r in rows_o] for c in cols_o}), out / "overview_grid.parquet")

    # ---------- console report ----------
    print(f"== viz bundle (real) -> {out} ==")
    print(f"  concurrent alive: peak={peak} @ t={peak_t}s  median={med}  mean={mean:.1f}")
    print(f"  -> B2 overview_grid: {'BUILT' if make_b2 else f'SKIPPED (peak {peak} <= {B2_THRESHOLD})'}")
    print(f"  category={dict(cat_dist)}")
    print(f"  force={dict(force_dist)}")
    print(f"  state={dict(state_dist)}")
    print(f"  coord bbox local(m)={bbox}  base_offset=({off_e},{off_n})")
    print(f"  viz_trajectories rows={len(fr)}  F={F}  (interp x,y; floor state; active frames only)")
    print(f"  per-sector entities/peak_alive:")
    for s in range(cols * rows):
        print(f"     sector {s}: entities={sector_count.get(s,0)} peak_alive={sector_peak.get(s,0)}")
    print(f"  events={len(evs)}  (B3 relations_snapshot: {'built' if args.relation_snapshot else 'SKIPPED'})")
    print(f"  notes/확인필요: {meta['notes_and_checks']}; landmarks!=terrain-peak (see meta)")


if __name__ == "__main__":
    main()
