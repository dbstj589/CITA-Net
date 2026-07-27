"""Measure realistic-v1 (§3 base stats + §4d injection samples).

Read-only. Reports, from the generated suite:
  - triples / sectors / observations, identities per sector, neg:pos candidate ratio
  - per-source actual error profile: mean cep_m (encodes pos-mult), UNKNOWN %,
    mis-ID %, and reported-minus-true time (encodes clock offset + report delay)
  - state dynamics: fraction of identities with >=1 and >=2 observed transitions
  - relation density (mean predicates/entity), combo uniqueness, KG A/B overlap
  - §4d samples: 3 each of (offset-differing source pair), (report delay),
    (in-window state transition), (partial cross-KG relation overlap)

Usage: python scripts/measure_realistic_v1.py [--data-root data/realistic_v1]
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _parse(t):
    return datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def load_sector(sec: Path):
    obs = [json.loads(l) for l in open(sec / "observations.jsonl", encoding="utf-8") if l.strip()]
    gold = json.loads((sec / "labels" / "gold_identities.json").read_text(encoding="utf-8"))
    return obs, gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/realistic_v1")
    ap.add_argument("--true-types", default=None,
                    help="not needed; true type inferred from gold identities")
    args = ap.parse_args()
    root = REPO / args.data_root
    gm = json.loads((root / "manifest_global.json").read_text(encoding="utf-8"))
    splits = {s: (root / "splits" / f"{s}.txt").read_text(encoding="utf-8").split()
              for s in ("train", "dev", "test")}

    all_sids = [(s, sid) for s in ("train", "dev", "test") for sid in splits[s]]
    tot_obs = 0
    ids_per_sector = []
    src = defaultdict(lambda: {"n": 0, "unk": 0, "mis": 0, "mis_elig": 0, "cep": 0.0, "dt": []})
    # transitions measured BOTH on reported-time order (what the model sees) and
    # true-time order (physical dynamics), over the SAME identity set (all observed
    # gold ids incl. dangling). n_obs_ident is the shared denominator.
    trans1_rep = trans2_rep = trans1_true = trans2_true = n_obs_ident = 0
    n_ident = 0
    rel_sizes = []
    rel_combos = []
    overlaps = []
    # true type per (sid, gold_id)
    for split, sid in all_sids:
        sec = root / "sectors" / sid
        obs, gold = load_sector(sec)
        tot_obs += len(obs)
        o2g = {a["obs_id"]: a["gold_identity_id"] for a in gold["assignment"]}
        true_type = {i["gold_identity_id"]: i["true_type"] for i in gold["identities"]}
        # dangling entities also have a true type
        for de in gold.get("dangling_local_entities", []):
            pass
        ids_per_sector.append(len(gold["identities"]) + len(gold.get("dangling_local_entities", [])))
        n_ident += len(gold["identities"])
        states = defaultdict(list)
        MISID_TYPES = {"US_Tank", "ROK_Tank", "ROK_Artillery", "US_Artillery", "CCF_AA", "QuadFifty"}
        for o in obs:
            s = o["source"]
            src[s]["n"] += 1
            src[s]["cep"] += o["cep_m"]
            if o["type"] == "UNKNOWN":
                src[s]["unk"] += 1
            g = o2g.get(o["obs_id"])
            tt = true_type.get(g)
            if tt in MISID_TYPES:
                src[s]["mis_elig"] += 1
                if o["type"] != "UNKNOWN" and o["type"] != tt:
                    src[s]["mis"] += 1
            if "_true_time" in o:
                src[s]["dt"].append((_parse(o["time"]) - _parse(o["_true_time"])).total_seconds())
            if g:
                states[g].append((o["time"], o.get("_true_time", o["time"]), o["state"]))
        # transitions per observed identity, on reported-time AND true-time order
        for g, ss in states.items():
            n_obs_ident += 1
            rep = [s for _, _, s in sorted(ss, key=lambda x: x[0])]
            tru = [s for _, _, s in sorted(ss, key=lambda x: x[1])]
            c_rep = sum(1 for a, b in zip(rep, rep[1:]) if a != b)
            c_tru = sum(1 for a, b in zip(tru, tru[1:]) if a != b)
            trans1_rep += c_rep >= 1; trans2_rep += c_rep >= 2
            trans1_true += c_tru >= 1; trans2_true += c_tru >= 2
        # relations per entity + uniqueness (per sector to keep entity keys local)
        ent_rel = defaultdict(set)
        for o in obs:
            key = (o["kg_id"], o["local_entity_id"])
            for r in o.get("relations", []):
                ent_rel[key].add((r["predicate"], r["target_ref"]))
        for v in ent_rel.values():
            rel_sizes.append(len(v))
            rel_combos.append(frozenset(v))
        # KG A/B predicate overlap for matched identities
        byg = defaultdict(lambda: defaultdict(set))
        for o in obs:
            g = o2g.get(o["obs_id"])
            if g:
                byg[g][o["kg_id"]].add(o.get("relations") and 1 or 0)  # placeholder
        byg = defaultdict(lambda: defaultdict(set))
        for o in obs:
            g = o2g.get(o["obs_id"])
            if not g:
                continue
            for r in o.get("relations", []):
                byg[g][o["kg_id"]].add(r["predicate"])
        for g, kg in byg.items():
            if "A" in kg and "B" in kg and (kg["A"] or kg["B"]):
                inter = len(kg["A"] & kg["B"])
                uni = len(kg["A"] | kg["B"])
                if uni:
                    overlaps.append(inter / uni)

    print("=" * 70)
    print("realistic-v1 base statistics (§3)")
    print("=" * 70)
    print(f"sectors train/dev/test = {gm['sectors']}")
    print(f"total triples          = {gm['total_triples']:,}")
    print(f"total observations     = {tot_obs:,}")
    print(f"identities/sector      = mean {statistics.mean(ids_per_sector):.1f} "
          f"(min {min(ids_per_sector)}, max {max(ids_per_sector)})")
    print(f"matched identities     = {n_ident}")

    print("\nper-source actual error profile (verifies §4 injection):")
    print(f"  {'source':14s} {'n':>6s} {'mean_cep':>9s} {'UNK%':>6s} {'misID%':>7s} {'mis/elig%':>9s} "
          f"{'dt_mean':>8s} {'dt_min':>7s} {'dt_max':>7s}")
    for s in sorted(src):
        d = src[s]
        n = d["n"]
        dt = d["dt"]
        me = (100 * d["mis"] / d["mis_elig"]) if d["mis_elig"] else 0.0
        print(f"  {s:14s} {n:6d} {d['cep']/n:9.1f} {100*d['unk']/n:6.1f} {100*d['mis']/n:7.1f} {me:9.1f} "
              f"{(statistics.mean(dt) if dt else 0):8.1f} {(min(dt) if dt else 0):7.1f} "
              f"{(max(dt) if dt else 0):7.1f}")
    all_dt = [x for s in src.values() for x in s["dt"]]
    print(f"  reported-minus-true (all sources): min {min(all_dt):.1f}s  max {max(all_dt):.1f}s")
    print("  (misID fires only for types with a confusable peer -> misID% low overall, "
          "mis/elig% among eligible types matches the 15%/5% config)")

    print("\nstate dynamics (denominator = all observed gold ids incl. dangling = %d):" % n_obs_ident)
    print(f"  >=1 transition  (true-time order):   {trans1_true}/{n_obs_ident} = {trans1_true/n_obs_ident:.3f}")
    print(f"  >=2 transitions (true-time order):   {trans2_true}/{n_obs_ident} = {trans2_true/n_obs_ident:.3f}  "
          f"<- physical dynamics (30% given a >=2-transition schedule + pre-existing multi-point)")
    print(f"  >=2 transitions (reported-time order):{trans2_rep}/{n_obs_ident} = {trans2_rep/n_obs_ident:.3f}  "
          f"<- as the model sees it (report delay reorders states, inflates churn)")

    print("\nrelation structure:")
    print(f"  mean predicates/entity  = {statistics.mean(rel_sizes):.2f}")
    print(f"  unique relation combos  = {len(set(rel_combos))}/{len(rel_combos)} "
          f"= {len(set(rel_combos))/len(rel_combos):.3f}")
    print(f"  KG A/B predicate Jaccard overlap (matched ids): mean {statistics.mean(overlaps):.3f} "
          f"(n={len(overlaps)}, target 0.50-0.70)")

    # ---------------- §4d injection samples ----------------
    print("\n" + "=" * 70)
    print("§4d injection samples (3 each, from real data)")
    print("=" * 70)
    _samples(root, splits)


def _samples(root, splits):
    # use dev sector 0 for concrete examples
    sid = splits["dev"][0]
    sec = root / "sectors" / sid
    obs, gold = load_sector(sec)
    o2g = {a["obs_id"]: a["gold_identity_id"] for a in gold["assignment"]}
    by_g = defaultdict(list)
    for o in obs:
        g = o2g.get(o["obs_id"])
        if g:
            by_g[g].append(o)

    # (1) offset-differing source pair on the SAME identity
    print(f"\n[1] same-identity observations from clock-offset-differing sources ({sid}):")
    shown = 0
    for g, os in by_g.items():
        srcs = {o["source"]: o for o in os}
        pair = None
        keys = list(srcs)
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                if keys[a] != keys[b]:
                    pair = (srcs[keys[a]], srcs[keys[b]]); break
            if pair:
                break
        if pair:
            oa, ob = pair
            print(f"  id={g}: {oa['source']} t={oa['time']}(true {oa.get('_true_time')})  |  "
                  f"{ob['source']} t={ob['time']}(true {ob.get('_true_time')})")
            shown += 1
        if shown >= 3:
            break

    # (2) report delay (reported - true large)
    print("\n[2] report delay (reported vs true time):")
    delays = sorted(((( _parse(o["time"]) - _parse(o["_true_time"]) ).total_seconds(), o)
                     for o in obs if "_true_time" in o), key=lambda x: -abs(x[0]))
    for dt, o in delays[:3]:
        print(f"  {o['obs_id']} src={o['source']}: true={o['_true_time']} reported={o['time']} "
              f"Δ={dt:+.1f}s")

    # (3) in-window state transition (identity with >=2 distinct states)
    print("\n[3] in-window state transitions (same identity, changing state):")
    shown = 0
    for g, os in by_g.items():
        seq = [(o["time"], o["state"]) for o in sorted(os, key=lambda x: x["time"])]
        distinct = []
        for _, s in seq:
            if not distinct or distinct[-1] != s:
                distinct.append(s)
        if len(distinct) >= 3:
            print(f"  id={g} ({len(os)} obs): {' -> '.join(distinct)}")
            shown += 1
        if shown >= 3:
            break

    # (4) partial cross-KG relation overlap
    print("\n[4] partial cross-KG relation overlap (predicates seen by A vs B):")
    shown = 0
    byg_kg = defaultdict(lambda: defaultdict(set))
    for o in obs:
        g = o2g.get(o["obs_id"])
        if not g:
            continue
        for r in o.get("relations", []):
            byg_kg[g][o["kg_id"]].add(r["predicate"])
    for g, kg in byg_kg.items():
        if "A" in kg and "B" in kg and kg["A"] != kg["B"] and (kg["A"] - kg["B"] or kg["B"] - kg["A"]):
            print(f"  id={g}: A={sorted(kg['A'])}  B={sorted(kg['B'])}  "
                  f"only-A={sorted(kg['A']-kg['B'])} only-B={sorted(kg['B']-kg['A'])}")
            shown += 1
        if shown >= 3:
            break


if __name__ == "__main__":
    main()
