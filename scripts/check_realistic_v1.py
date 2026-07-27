"""Integrity + gold-candidate-coverage gates for realistic-v1 (§4a, §4b).

Read-only. Gate a: every observation is partitioned exactly once into (a matched
gold identity | dangling); vocabulary (type/state/source/relation) has zero
violations vs the suite ontology. Gate b: after grid blocking, the fraction of
gold cross-KG within-window pairs retained (survival) >= 0.98; swept over several
dt_max_s so the correct blocking window can be chosen and reported.

Usage: python scripts/check_realistic_v1.py [--splits dev test] [--dt 180 600 900 1200]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from citanet.data.ontology import UNKNOWN_TYPE, load_ontology
from citanet.data.stream import load_sector_observations
from citanet.data.blocking_grid import (generate_candidates_grid,
                                         gold_cross_kg_same_pairs)
import json


def read_split(root, split):
    return (root / "splits" / f"{split}.txt").read_text(encoding="utf-8").split()


def gate_a(root, sids, ont):
    types_ok = set(ont.type_names) | {UNKNOWN_TYPE}
    states_ok = set(ont.state_names)
    sources_ok = set(ont.source_names)
    rels_ok = set(getattr(ont, "relation_names", []) or [])
    bad = {"type": 0, "state": 0, "source": 0, "relation": 0}
    part_fail = 0
    n_obs_total = 0
    for sid in sids:
        sec = root / "sectors" / sid
        obs = [json.loads(l) for l in open(sec / "observations.jsonl", encoding="utf-8") if l.strip()]
        n_obs_total += len(obs)
        all_ids = {o["obs_id"] for o in obs}
        gold = json.loads((sec / "labels" / "gold_identities.json").read_text(encoding="utf-8"))
        matched = set()
        for idn in gold["identities"]:
            matched.update(idn["member_observations"])
        dang = set(gold["dangling_observations"])
        # partition: matched ∪ dangling == all, disjoint
        if (matched | dang) != all_ids or (matched & dang):
            part_fail += 1
        for o in obs:
            if o["type"] not in types_ok:
                bad["type"] += 1
            if o["state"] not in states_ok:
                bad["state"] += 1
            if o["source"] not in sources_ok:
                bad["source"] += 1
            for r in o.get("relations", []):
                if rels_ok and r["predicate"] not in rels_ok:
                    bad["relation"] += 1
    ok = part_fail == 0 and sum(bad.values()) == 0
    return ok, part_fail, bad, n_obs_total


def gate_b(root, sids, ont, dt_list, blk):
    # per dt_max_s: aggregate survival across sectors
    out = {}
    for dt in dt_list:
        kept_hits = 0
        gold_total = 0
        for sid in sids:
            sec = root / "sectors" / sid
            obs = load_sector_observations(sec)
            gold = json.loads((sec / "labels" / "gold_identities.json").read_text(encoding="utf-8"))
            gpairs = gold_cross_kg_same_pairs(obs, gold, dt_max_s=dt)
            if not gpairs:
                continue
            pairs = generate_candidates_grid(
                obs, ont, dt_max_s=dt, theta_text=blk["theta_text"],
                r_err_floor_m=blk["r_err_floor_m"], reach_vmax_mult=blk["reach_vmax_mult"],
                reach_extra_m=blk["reach_extra_m"], cell_size_m=blk["grid_cell_m"],
                use_text_gate=blk["use_text_gate"], type_by_category=blk["type_by_category"])
            obs_sorted = sorted(obs, key=lambda o: (o.t, o.obs_id))
            kept = {frozenset((obs_sorted[p.i].obs_id, obs_sorted[p.j].obs_id)) for p in pairs}
            kept_hits += sum(1 for g in gpairs if g in kept)
            gold_total += len(gpairs)
        out[dt] = (kept_hits / gold_total) if gold_total else float("nan"), gold_total
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/realistic_v1")
    ap.add_argument("--splits", nargs="+", default=["dev", "test"])
    ap.add_argument("--dt", nargs="+", type=float, default=[450, 700])
    ap.add_argument("--r-err-floor", type=float, default=300.0,
                    help="blocking spatial slack; 300 chosen to absorb 5x ACOUSTIC noise (§4b)")
    args = ap.parse_args()
    root = REPO / args.data_root
    ont = load_ontology(root / "ontology")
    sids = [sid for sp in args.splits for sid in read_split(root, sp)]
    print(f"realistic-v1 gates on splits={args.splits} ({len(sids)} sectors)")

    # m1 blocking params (mirror configs/realistic_v1/m1.yaml)
    blk = {"theta_text": 0.3, "r_err_floor_m": args.r_err_floor, "grid_cell_m": 400.0,
           "use_text_gate": False, "type_by_category": True,
           "reach_vmax_mult": 1.0, "reach_extra_m": 0.0}

    print("\n[gate a] integrity")
    ok, part_fail, bad, n_obs = gate_a(root, sids, ont)
    print(f"  observations checked     = {n_obs:,}")
    print(f"  partition failures       = {part_fail} (sectors where matched∪dangling != all obs)")
    print(f"  vocab violations         = {bad}")
    print(f"  GATE a: {'PASS' if ok else 'FAIL'}")

    print("\n[gate b] gold cross-KG within-window pair survival after blocking")
    res = gate_b(root, sids, ont, args.dt, blk)
    print(f"  {'dt_max_s':>9s} {'survival':>9s} {'gold_pairs':>11s}")
    best = None
    for dt in args.dt:
        surv, tot = res[dt]
        flag = " <= 0.98 PASS" if surv >= 0.98 else ""
        print(f"  {dt:9.0f} {surv:9.4f} {tot:11d}{flag}")
        if surv >= 0.98 and best is None:
            best = dt
    if best is not None:
        print(f"  GATE b: PASS at dt_max_s={best:.0f} (smallest tested achieving >=0.98) -> "
              f"use this in configs/realistic_v1/m1.yaml")
    else:
        print("  GATE b: FAIL at all tested dt_max_s -> widen further and re-run")


if __name__ == "__main__":
    main()
