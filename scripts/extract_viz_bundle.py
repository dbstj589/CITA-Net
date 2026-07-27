#!/usr/bin/env python
"""Extract a lightweight visualisation bundle from a world-GT directory.

Read-only on the GT (uses entities.jsonl, trajectories.parquet, events.jsonl,
terrain/, gt_manifest.json; ignores unified_stkg.nt / relations.jsonl /
validation_report.json). Writes data/<gt>/viz_bundle/ with frame-resampled
trajectories, entity table, events, terrain elevation and a self-describing
viz_meta.json. No rendering, no model, no video here.

    python scripts/extract_viz_bundle.py --gt data/hill395_world_gt --frames 720
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

# Force mapping is a stated CONVENTION (ROK/UN defender viewpoint), not a guess:
# ROK and US/UN are the defenders (friendly); CCF is the attacker (enemy).
AFFIL_TO_FORCE = {"ROK": "friendly", "US": "friendly", "UN": "friendly",
                  "CCF": "enemy", "NA": "other", "UNK": "other"}
AFFIL_TO_FORCE_NOTE = ("viewpoint convention (ROK/UN defender): ROK,US,UN->friendly, "
                       "CCF->enemy, NA/UNK->other. Not a guess; stated perspective.")
ALIVE_RULE = "alive = (state != 'Destroyed') at that frame (Destroyed = wreck/debris, not a live actor)"


def _floor_idx(tf: float, tmax: int) -> tuple[int, float]:
    i = int(np.floor(tf))
    if i >= tmax:
        return tmax, 0.0
    return i, tf - i


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="data/hill395_world_gt")
    ap.add_argument("--frames", type=int, default=720)
    args = ap.parse_args()

    gt = Path(args.gt)
    out = gt / "viz_bundle"
    out.mkdir(parents=True, exist_ok=True)
    F = int(args.frames)
    notes: list[str] = []      # "확인 필요" / assumptions

    manifest = json.load(open(gt / "gt_manifest.json", encoding="utf-8"))
    cfg = manifest["config"]
    window = float(cfg["time"]["window_seconds"])
    dt = float(cfg["time"]["dt_seconds"])
    epoch = cfg["time"]["epoch"]

    # --- ontology type -> category ---
    ont_dir = manifest.get("ontology_dir", "data/battlefield_hill395_large/ontology")
    classes = yaml.safe_load(open(Path(ont_dir) / "classes.yaml", encoding="utf-8"))
    type_to_category = {t: v.get("category", "unknown") for t, v in classes.items()}

    # --- entities ---
    ent_records = [json.loads(l) for l in open(gt / "entities.jsonl", encoding="utf-8")]
    ent_out = []
    for e in ent_records:
        ty = e["true_type"]
        cat = type_to_category.get(ty)
        if cat is None:
            cat = "unknown"; notes.append(f"type '{ty}' missing in classes.yaml -> category=unknown")
        aff = e.get("affiliation", "UNK")
        force = AFFIL_TO_FORCE.get(aff, "other")
        ent_out.append({"entity_id": e["id"], "true_type": ty, "category": cat,
                        "affiliation": aff, "force": force})
    ent_by_id = {e["entity_id"]: e for e in ent_out}
    with open(out / "viz_entities.jsonl", "w", encoding="utf-8") as fh:
        for e in ent_out:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")

    # --- terrain elevation + local frame offset ---
    tmeta = json.load(open(gt / "terrain" / "terrain_meta.json", encoding="utf-8"))
    off_e, off_n = float(tmeta["origin_e"]), float(tmeta["origin_n"])   # local-frame offset
    res = float(tmeta["res_m"]); nx = int(tmeta["nx"]); ny = int(tmeta["ny"])
    elev = np.load(gt / "terrain" / "elevation.npy")
    np.save(out / "terrain_elevation.npy", elev)
    map_extent = {"xmin": 0.0, "xmax": res * (nx - 1), "ymin": 0.0, "ymax": res * (ny - 1)}

    # --- trajectories: resample to F frames (GT dt=1s, 601 samples -> upsample) ---
    tab = pq.read_table(gt / "trajectories.parquet").to_pydict()
    eids = sorted(set(tab["entity_id"]))
    times = sorted(set(tab["t"]))
    tmax = len(times) - 1                                   # index of last sample (t=window)
    ti = {v: k for k, v in enumerate(times)}
    ei = {v: k for k, v in enumerate(eids)}
    Ea = np.full((len(eids), len(times)), np.nan)
    Na = np.full((len(eids), len(times)), np.nan)
    Sa = np.empty((len(eids), len(times)), dtype=object)
    for k in range(len(tab["entity_id"])):
        i = ei[tab["entity_id"][k]]; j = ti[tab["t"][k]]
        Ea[i, j] = tab["easting"][k]; Na[i, j] = tab["northing"][k]; Sa[i, j] = tab["state"][k]

    resample = (f"upsample {len(times)} GT samples (dt={dt}s) -> {F} frames: linear interp of "
                f"(easting,northing); state via floor (nearest-earlier) sample")
    if F <= len(times):
        resample = (f"downsample {len(times)} GT samples -> {F} frames: nearest-earlier sample "
                    f"(state exact, position at sampled time)")

    frame_col, id_col, x_col, y_col, st_col, al_col = [], [], [], [], [], []
    for f in range(F):
        tf_sec = (f / (F - 1)) * window if F > 1 else 0.0
        tf_idx = tf_sec / dt
        i0, frac = _floor_idx(tf_idx, tmax)
        i1 = min(i0 + 1, tmax)
        for i, eid in enumerate(eids):
            x = Ea[i, i0] + frac * (Ea[i, i1] - Ea[i, i0]) - off_e
            y = Na[i, i0] + frac * (Na[i, i1] - Na[i, i0]) - off_n
            state = Sa[i, i0]
            frame_col.append(f); id_col.append(eid)
            x_col.append(round(float(x), 3)); y_col.append(round(float(y), 3))
            st_col.append(state); al_col.append(state != "Destroyed")

    pq.write_table(pa.table({
        "frame": pa.array(frame_col, pa.int32()),
        "entity_id": id_col, "x": x_col, "y": y_col,
        "state": st_col, "alive": pa.array(al_col, pa.bool_()),
    }), out / "viz_trajectories.parquet")

    traj_bbox = {"xmin": float(min(x_col)), "xmax": float(max(x_col)),
                 "ymin": float(min(y_col)), "ymax": float(max(y_col))}

    # --- landmarks (local frame) ---
    lms = json.load(open(gt / "terrain" / "landmarks.json", encoding="utf-8"))
    lm_out = [{"name": lm["name"], "x": round(lm["easting"] - off_e, 2),
               "y": round(lm["northing"] - off_n, 2)} for lm in lms]
    lm_name_by_id = {lm["id"]: lm["name"] for lm in lms}

    # --- events -> frames ---
    ev_path = gt / "events.jsonl"
    have_events = ev_path.exists() and sum(1 for _ in open(ev_path)) > 0
    if have_events:
        def sec_to_frame(t):
            return int(round((t / window) * (F - 1))) if F > 1 else 0
        with open(out / "viz_events.jsonl", "w", encoding="utf-8") as fh:
            for l in open(ev_path, encoding="utf-8"):
                e = json.loads(l)
                a, b = e["interval"]
                lmk = e.get("landmark")
                fh.write(json.dumps({
                    "type": e["type"],
                    "start_frame": sec_to_frame(a), "end_frame": sec_to_frame(b),
                    "participants": e.get("participants", []),
                    "landmark": lm_name_by_id.get(lmk, lmk),
                }, ensure_ascii=False) + "\n")
    else:
        notes.append("events.jsonl 없음 -> viz_events.jsonl 생성 안 함")

    # --- sectors ---
    ws = cfg["world_scope"]
    n_sectors = int(ws.get("n_sectors", 1)) if ws.get("mode") == "tiled" else 1
    pitch = float(ws.get("sector_pitch_m", 6000.0))
    BASE_E_GLOBAL, BASE_N_GLOBAL = 300000.0, 4238000.0        # generator constant
    cols = min(8, n_sectors); rows = int(np.ceil(n_sectors / 8))
    sector_origins = [{"sector": s,
                       "x": (BASE_E_GLOBAL + (s % 8) * pitch) - off_e,
                       "y": (BASE_N_GLOBAL + (s // 8) * pitch) - off_n} for s in range(n_sectors)]

    # --- counts ---
    from collections import Counter
    force_counts = Counter(e["force"] for e in ent_out)
    cat_counts = Counter(e["category"] for e in ent_out)

    meta = {
        "source_gt": str(gt), "global_seed": manifest.get("global_seed"),
        "time": {"epoch": epoch, "window_seconds": window, "dt_seconds": dt,
                 "frames": F, "resample": resample},
        "map": {"unit": "meters",
                "base_offset": {"easting": off_e, "northing": off_n,
                                "note": "subtracted from GT UTM coords to get local x,y; "
                                        "equals terrain grid origin so points align with "
                                        "terrain_elevation.npy"},
                "extent": map_extent,
                "extent_source": "terrain grid (origin + res*(n-1)); manifest has no explicit map range",
                "trajectory_bbox": traj_bbox,
                "elevation_file": "terrain_elevation.npy",
                "elevation_shape": list(elev.shape), "elevation_res_m": res},
        "sectors": {"mode": ws.get("mode"), "n_sectors": n_sectors, "cols": cols, "rows": rows,
                    "sector_pitch_m": pitch, "sector_origins_local": sector_origins},
        "landmarks": lm_out,
        "type_to_category": type_to_category,
        "affiliation_to_force": {"mapping": AFFIL_TO_FORCE, "note": AFFIL_TO_FORCE_NOTE},
        "alive_rule": ALIVE_RULE,
        "counts": {"entities_total": len(ent_out),
                   "by_force": dict(force_counts), "by_category": dict(cat_counts)},
        "notes_and_checks": notes if notes else ["none"],
    }
    (out / "viz_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- console report ---
    print(f"== viz bundle -> {out} ==")
    print(f"  viz_trajectories.parquet  ({F} frames x {len(eids)} entities = {F*len(eids)} rows)")
    print(f"  viz_entities.jsonl        ({len(ent_out)} entities)")
    print(f"  viz_events.jsonl          ({'yes' if have_events else 'NONE'})")
    print(f"  terrain_elevation.npy     (shape {elev.shape}, res {res} m)")
    print(f"  viz_meta.json")
    print(f"  entities total={len(ent_out)}  by_force={dict(force_counts)}  by_category={dict(cat_counts)}")
    print(f"  frames F={F}  resample: {resample}")
    print(f"  BASE offset (subtracted) = ({off_e}, {off_n})   local map extent = {map_extent}")
    print(f"  trajectory bbox (local)  = {traj_bbox}")
    print(f"  events: {'present' if have_events else 'NONE'}   elevation field: present")
    print(f"  '확인 필요'/notes: {meta['notes_and_checks']}")


if __name__ == "__main__":
    main()
