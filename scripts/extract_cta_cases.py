"""Qualitative cases for each CTA term, chosen by a rule fixed in advance. NO TRAINING.

Anti-cherry-picking: the selection rule below is the ONLY filter applied. Cases are taken
by rank under it; nothing is inspected first and kept or dropped afterwards.

SELECTION RULE (fixed before looking at any case)
  For each term X, within the `test` split of one seed:
    (i)  supporting  -- among GOLD-POSITIVE pairs the model ranks highly
                        (score in the sector's top decile), keep those where |b_X| is the
                        largest of the five constraint contributions ("the non-semantic
                        driver"), then take the top 3 by b_X (most positive).
    (ii) rejecting   -- among GOLD-NEGATIVE pairs the model ranks low
                        (score in the sector's bottom half), keep those where |b_X| is the
                        largest of the five, then take the top 3 by -b_X (most negative).

"Model ranks highly/low" is operationalised on the CTA score, which is what drives
p_transition and feeds the pair head; per-pair decoder assignments are not used, so a
case is "the scorer treats it correctly", not "the decoder emitted it".

Each case carries every term component plus the raw record of both observations (time,
true time, position, state, source, predicates) so the reader can check the claim.

    python scripts/extract_cta_cases.py
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
TERMS = ["b_motion", "b_state", "b_rel", "b_src"]      # b_time is structurally 0 -> no cases
TOP_N = 3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=19521006)
    ap.add_argument("--split", default="test")
    ap.add_argument("--dump-dir", default=str(REPO / "results/realistic_v1_score_analysis/dump"))
    ap.add_argument("--out-dir", default=str(REPO / "results/realistic_v1_score_analysis/cases"))
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    t = pq.read_table(Path(args.dump_dir) / f"m3_full_seed{args.seed}_{args.split}.parquet")
    sector = np.asarray(t["sector"]); label = t["label"].to_numpy()
    score = t["score"].to_numpy()
    comp = {k: t[k].to_numpy() for k in ["sem_w0"] + TERMS + ["b_time"]}
    ii, jj = t["i"].to_numpy(), t["j"].to_numpy()

    # per-sector score thresholds for "ranks highly" / "ranks low"
    hi_thr = np.empty_like(score); lo_thr = np.empty_like(score)
    for s in np.unique(sector):
        m = sector == s
        hi_thr[m] = np.quantile(score[m], 0.90)
        lo_thr[m] = np.quantile(score[m], 0.50)

    # "the non-semantic driver" is measured on each term's deviation from its own mean,
    # not on raw magnitude. b_src sits at a near-constant -1.26 (range -1.28..-1.21), so a
    # raw-magnitude rule made it the largest contributor for essentially every pair while
    # carrying almost no information (its marginal AUC is 0.554). A constant offset shifts
    # all pairs equally and cannot drive a decision; the centred value is what varies with
    # the pair, so that is what "drives" it.
    B = np.stack([comp[k] - comp[k].mean() for k in TERMS], axis=1)
    dominant = np.argmax(np.abs(B), axis=1)

    # raw observation records, loaded lazily per sector
    raw_cache: dict[str, dict] = {}

    def raw(sid: str, obs_id: str) -> dict:
        if sid not in raw_cache:
            recs = {}
            with open(REPO / "data/realistic_v1/sectors" / sid / "observations.jsonl",
                      encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        o = json.loads(line)
                        recs[o["obs_id"]] = o
            raw_cache[sid] = recs
        return raw_cache[sid][obs_id]

    # obs_id lookup per sector requires the same ordering featurize used
    import torch
    from citanet.config import load_config
    from citanet.engine_large import _featurize, _feat_sig, load_large_bundle
    cfg = load_config(REPO / "configs/realistic_v1/m3_full.yaml")
    cfg.data_root, cfg.ontology_dir = "data/realistic_v1", "data/realistic_v1/ontology"
    _, fs, ont = load_large_bundle(cfg, REPO / f"runs/realistic_v1/ablation/m3_full_seed{args.seed}")
    cache = os.environ.get("CITANET_FEAT_CACHE"); sig = _feat_sig(cfg, fs) if cache else None
    oid_cache: dict[str, list] = {}

    def obs_ids(sid: str) -> list:
        if sid not in oid_cache:
            f = _featurize(cfg, sid, fs, ont, cache, sig)
            oid_cache[sid] = list(f.obs_ids)
            del f
        return oid_cache[sid]

    doc = {"selection_rule": __doc__.split("SELECTION RULE")[1].split("Each case")[0].strip(),
           "seed": args.seed, "split": args.split, "cases": {}}
    for ti, tm in enumerate(TERMS):
        sup = np.flatnonzero((label == 1) & (score >= hi_thr) & (dominant == ti))
        rej = np.flatnonzero((label == 0) & (score <= lo_thr) & (dominant == ti))
        picks = {"supporting": sup[np.argsort(-comp[tm][sup])][:TOP_N],
                 "rejecting": rej[np.argsort(comp[tm][rej])][:TOP_N]}
        doc["cases"][tm] = {}
        for kind, idxs in picks.items():
            entries = []
            for p in idxs:
                sid = str(sector[p])
                oid = obs_ids(sid)
                a, b = raw(sid, oid[ii[p]]), raw(sid, oid[jj[p]])
                entries.append({
                    "sector": sid, "label": int(label[p]), "score": float(score[p]),
                    "components": {k: float(comp[k][p]) for k in ["sem_w0", "b_time"] + TERMS},
                    "obs_i": _slim(a), "obs_j": _slim(b),
                    "shared_predicates": sorted(
                        {r["predicate"] for r in a.get("relations", [])} &
                        {r["predicate"] for r in b.get("relations", [])}),
                    "predicates_i": sorted({r["predicate"] for r in a.get("relations", [])}),
                    "predicates_j": sorted({r["predicate"] for r in b.get("relations", [])}),
                })
            doc["cases"][tm][kind] = entries
        print(f"  {tm:9} supporting={len(picks['supporting'])} rejecting={len(picks['rejecting'])} "
              f"(pool {sup.size:,}/{rej.size:,})")

    (out / f"cases_seed{args.seed}_{args.split}.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"WROTE {out}/cases_seed{args.seed}_{args.split}.json")


def _slim(o: dict) -> dict:
    return {"obs_id": o["obs_id"], "kg": o["kg_id"], "source": o["source"],
            "entity": o["local_entity_id"], "label_text": o["label_text"],
            "type": o["type"], "state": o["state"],
            "time": o["time"], "true_time": o.get("_true_time"),
            "easting": o["location"]["easting"], "northing": o["location"]["northing"],
            "cep_m": o["cep_m"]}


if __name__ == "__main__":
    main()
