#!/usr/bin/env python
"""Animate a world-GT scenario as it unfolds over the time window.

Reads data/<gt>/ (trajectories.parquet, entities.jsonl, events.jsonl,
terrain/) and renders an animated GIF: terrain elevation as background,
landmarks, and every entity moving in real (compressed) time, coloured by
affiliation, with a fading tail. Firing entities flash a yellow star and the
active events are listed in the title. Read-only — touches no training run.

    python scripts/viz_world_gt.py --gt data/hill395_world_gt --stride 5 --fps 12
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import pyarrow.parquet as pq

AFFIL_COLOR = {"CCF": "#d7191c", "ROK": "#2c7bb6", "US": "#1a9641", "NA": "#888888", "UNK": "#999999"}
STATIC_STATES = {"Emplaced", "Halted", "Destroyed", "Holding", "Occupying", "Firing"}


def load(gt: Path):
    t = pq.read_table(gt / "trajectories.parquet").to_pydict()
    ents = {json.loads(l)["id"]: json.loads(l)
            for l in open(gt / "entities.jsonl", encoding="utf-8")}
    events = [json.loads(l) for l in open(gt / "events.jsonl", encoding="utf-8")]
    elev = np.load(gt / "terrain" / "elevation.npy")
    meta = json.loads((gt / "terrain" / "terrain_meta.json").read_text(encoding="utf-8"))
    lms = json.loads((gt / "terrain" / "landmarks.json").read_text(encoding="utf-8"))
    return t, ents, events, elev, meta, lms


def build_arrays(t):
    """Return eids, times(sorted unique), and (n_ent, n_t) arrays for e/n/state."""
    eids = sorted(set(t["entity_id"]))
    times = sorted(set(t["t"]))
    ti = {v: k for k, v in enumerate(times)}
    ei = {v: k for k, v in enumerate(eids)}
    ne, nt = len(eids), len(times)
    E = np.full((ne, nt), np.nan); N = np.full((ne, nt), np.nan)
    S = np.empty((ne, nt), dtype=object)
    for k in range(len(t["entity_id"])):
        i = ei[t["entity_id"][k]]; j = ti[t["t"][k]]
        E[i, j] = t["easting"][k]; N[i, j] = t["northing"][k]; S[i, j] = t["state"][k]
    return eids, np.array(times), E, N, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="data/hill395_world_gt")
    ap.add_argument("--stride", type=int, default=5, help="seconds between frames")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--tail", type=int, default=45, help="trail length (seconds)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gt = Path(args.gt)
    t, ents, events, elev, meta, lms = load(gt)
    eids, times, E, N, S = build_arrays(t)
    affil = np.array([ents[e]["affiliation"] for e in eids])
    colors = np.array([AFFIL_COLOR.get(a, "#999999") for a in affil])
    types = np.array([ents[e]["true_type"] for e in eids])
    sizes = np.array([70 if ("Tank" in ty or "Vehicle" in ty) else
                      55 if ("Artillery" in ty or ty in ("Mortar", "CCF_AA", "QuadFifty")) else 30
                      for ty in types])

    frames = list(range(0, len(times), args.stride))
    if frames[-1] != len(times) - 1:
        frames.append(len(times) - 1)

    e0 = meta["origin_e"]; n0 = meta["origin_n"]; res = meta["res_m"]
    nx = meta["nx"]; ny = meta["ny"]
    extent = [e0, e0 + res * (nx - 1), n0, n0 + res * (ny - 1)]

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(elev, origin="lower", extent=extent, cmap="terrain", alpha=0.75, aspect="equal")
    for lm in lms:
        ax.plot(lm["easting"], lm["northing"], "^", color="black", ms=7, zorder=5)
        ax.annotate(lm["name"], (lm["easting"], lm["northing"]), fontsize=7, zorder=6)
    ax.set_xlabel("easting (m)"); ax.set_ylabel("northing (m)")

    # legend (affiliations + firing)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", ls="", mfc=c, mec="k", ms=8, label=a)
               for a, c in [("CCF (assault)", AFFIL_COLOR["CCF"]),
                            ("ROK (defence)", AFFIL_COLOR["ROK"]),
                            ("US/UN", AFFIL_COLOR["US"])]]
    handles.append(Line2D([0], [0], marker="*", ls="", mfc="yellow", mec="k", ms=13, label="Firing"))
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9)

    scat = ax.scatter(E[:, 0], N[:, 0], s=sizes, c=colors, edgecolors="k",
                      linewidths=0.4, zorder=4)
    fire = ax.scatter([], [], s=160, marker="*", facecolors="yellow",
                      edgecolors="k", linewidths=0.5, zorder=7)
    tails = [ax.plot([], [], "-", color=colors[i], lw=0.8, alpha=0.5, zorder=3)[0]
             for i in range(len(eids))]
    title = ax.set_title("")

    def frame(fi):
        j = frames[fi]
        tt = times[j]
        scat.set_offsets(np.c_[E[:, j], N[:, j]])
        lo = max(0, j - args.tail)
        for i, ln in enumerate(tails):
            ln.set_data(E[i, lo:j + 1], N[i, lo:j + 1])
        firing = [(E[i, j], N[i, j]) for i in range(len(eids)) if S[i, j] == "Firing"]
        fire.set_offsets(np.array(firing) if firing else np.empty((0, 2)))
        active = [e["type"] for e in events if e["interval"][0] <= tt <= e["interval"][1]]
        mm, ss = divmod(int(tt), 60)
        title.set_text(f"Hill 395 GT — t = {mm:02d}:{ss:02d}   active: {', '.join(active) or '-'}")
        return [scat, fire, title, *tails]

    anim = FuncAnimation(fig, frame, frames=len(frames), interval=1000 / args.fps, blit=False)
    out = Path(args.out) if args.out else gt / "scenario.gif"
    anim.save(out, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"wrote {out}  ({len(frames)} frames @ {args.fps}fps, stride {args.stride}s, "
          f"{len(eids)} entities)")


if __name__ == "__main__":
    main()
