"""Dump per-candidate-pair CTA term components for the trained m3_full runs. NO TRAINING.

Score-level analysis of realistic-v1: for every cross-KG candidate pair we record the
gold label, the six CTA contributions, the total score, and the covariates needed to
stratify by the uncertainty each term targets. Everything is read off already-trained
checkpoints -- no weights change, no decoding, no new seeds.

CTA composes as (model/cta.py:101)

    score = w0*sim_sem + b_time + b_motion + b_state + b_rel + b_src

so `sem_w0` below is w0*sim_sem, i.e. the semantic contribution actually entering the
score (not the raw cosine). Every written row is checked against the model's own score
to 1e-6; a violation aborts the dump.

Only cross-KG pairs are dumped: `pair_label` is defined only where `pair_cross` is true
(featurize.py), and the entity-alignment question is a cross-KG one.

Ground-truth stratification covariates (§4 of the analysis plan). What the generator
does and does not store, verified on the data:
  * `_true_time` IS in observations.jsonl but is dropped by data/stream.py's Observation
    dataclass, so it is re-joined here by obs_id straight from the jsonl.
  * true (noise-free) position and explicit state-transition timestamps are NOT stored
    anywhere, so `gt_state_trans` is DERIVED from each entity's (true_time, state)
    sequence rather than read off a field. The derivation is documented at its call site.

    python scripts/dump_cta_scores.py                      # all 5 seeds, dev+test
    python scripts/dump_cta_scores.py --seeds 19521006 --splits dev
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from citanet.config import load_config
from citanet.data.stream import read_split
from citanet.engine_large import _featurize, _feat_sig, _sector_dir, load_large_bundle

REPO = Path(__file__).resolve().parents[1]
SEEDS = [19521006, 19521007, 19521008, 19521009, 19521010]
TOL = 1e-6


def parse_iso(s: str) -> float:
    from datetime import datetime
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def sector_truth(sector_dir: Path):
    """Per-observation ground truth joined by obs_id, plus per-entity state history.

    Returns (truth, ent_hist, rel_share) where
      truth[obs_id]  = (time_error_s, entity_key, state)
      ent_hist[key]  = sorted [(true_t, state), ...] for that local entity
      rel_share[key] = fraction of entities in this sector sharing this entity's exact
                       predicate set -- low share == distinctive "formation fingerprint"
    """
    truth, ent_hist, ent_preds = {}, defaultdict(list), {}
    with open(sector_dir / "observations.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            o = json.loads(line)
            key = f"{o['kg_id']}:{o['local_entity_id']}"
            tt = parse_iso(o["_true_time"]) if "_true_time" in o else None
            rt = parse_iso(o["time"])
            truth[o["obs_id"]] = (abs(rt - tt) if tt is not None else float("nan"),
                                  key, o["state"], tt)
            if tt is not None:
                ent_hist[key].append((tt, o["state"]))
            ent_preds.setdefault(key, set()).update(
                r["predicate"] for r in o.get("relations", []))
    for k in ent_hist:
        ent_hist[k].sort()
    combo_count = Counter(frozenset(v) for v in ent_preds.values())
    n_ent = max(1, len(ent_preds))
    rel_share = {k: combo_count[frozenset(v)] / n_ent for k, v in ent_preds.items()}
    return truth, ent_hist, rel_share


def gt_state_trans(hist, t_lo: float, t_hi: float) -> int:
    """1 if this entity's TRUE state changes strictly inside (t_lo, t_hi).

    The generator stores no transition timestamps, so a transition is defined as two
    temporally adjacent observations of the same entity carrying different states, with
    the later one falling inside the interval. This is a derived quantity, not a field.
    """
    prev = None
    for t, s in hist:
        if prev is not None and s != prev and t_lo < t < t_hi:
            return 1
        prev = s
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--splits", nargs="+", default=["dev", "test"])
    ap.add_argument("--variant", default="m3_full")
    ap.add_argument("--runs-dir", default=str(REPO / "runs" / "realistic_v1" / "ablation"))
    ap.add_argument("--config", default=str(REPO / "configs" / "realistic_v1" / "m3_full.yaml"))
    ap.add_argument("--out-dir", default=str(REPO / "results" / "realistic_v1_score_analysis" / "dump"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    cfg.data_root, cfg.ontology_dir = "data/realistic_v1", "data/realistic_v1/ontology"

    # source metadata -> error / reliability classes for the motion & src strata
    ontology_dir = Path(cfg.ontology_dir)
    import yaml
    src_meta = yaml.safe_load((ontology_dir / "sources.yaml").read_text(encoding="utf-8"))
    print(f"== dumping CTA components: variant={args.variant} seeds={args.seeds} "
          f"splits={args.splits} ==")
    print(f"   feat_cache={os.environ.get('CITANET_FEAT_CACHE') or '(off!)'}  out={out_dir}")

    for seed in args.seeds:
        run_dir = Path(args.runs_dir) / f"{args.variant}_seed{seed}"
        model, fs, ont = load_large_bundle(cfg, run_dir)
        cache = os.environ.get("CITANET_FEAT_CACHE")
        sig = _feat_sig(cfg, fs) if cache else None
        dev_t = next(model.parameters()).device
        w0 = float(model.cta.w0.detach())
        # per-source arrays aligned with fs.sources indices
        cep = np.array([float(src_meta[s]["cep_m"]) for s in fs.sources], dtype=np.float32)
        rel = np.array([float(src_meta[s]["reliability"]) for s in fs.sources], dtype=np.float32)
        cep_hi = float(np.median(cep))
        rel_hi = float(np.median(rel))

        for split in args.splits:
            t0 = time.time()
            cols = defaultdict(list)
            for sid in read_split(cfg.data_root, split):
                feats = _featurize(cfg, sid, fs, ont, cache, sig).to(dev_t)
                with torch.no_grad(), torch.device(dev_t):
                    out = model(feats)
                c = out.cta
                m = feats.pair_cross                       # labels valid only on cross pairs
                sem_w0 = (w0 * c.sim_sem)[m]
                parts = {"sem_w0": sem_w0, "b_time": c.b_time[m], "b_motion": c.b_motion[m],
                         "b_state": c.b_state[m], "b_rel": c.b_rel[m], "b_src": c.b_src[m]}
                score = c.score[m]
                # --- fidelity: every row must recompose to the model's own score ---
                recon = sum(parts.values())
                dmax = float((recon - score).abs().max()) if score.numel() else 0.0
                if dmax > TOL:
                    raise SystemExit(f"FIDELITY FAIL {sid} seed{seed}: max|sum-score|={dmax:.3e} > {TOL}")

                i = feats.pair_i[m].to("cpu").numpy().astype(np.int32)
                j = feats.pair_j[m].to("cpu").numpy().astype(np.int32)
                truth, ent_hist, rel_share = sector_truth(_sector_dir(cfg, sid))
                oid = feats.obs_ids
                ti = np.array([truth[oid[k]][0] for k in i], dtype=np.float32)
                tj = np.array([truth[oid[k]][0] for k in j], dtype=np.float32)
                tt_i = np.array([truth[oid[k]][3] for k in i], dtype=np.float64)
                tt_j = np.array([truth[oid[k]][3] for k in j], dtype=np.float64)
                key_i = [truth[oid[k]][1] for k in i]
                st = np.array([gt_state_trans(ent_hist[key_i[n]],
                                              min(tt_i[n], tt_j[n]), max(tt_i[n], tt_j[n]))
                               for n in range(len(i))], dtype=np.int8)
                si = feats.p_src_i[m].to("cpu").numpy()
                sj = feats.p_src_j[m].to("cpu").numpy()

                cols["sector"].append(np.full(len(i), sid, dtype=object))
                cols["i"].append(i); cols["j"].append(j)
                cols["label"].append(feats.pair_label[m].to("cpu").numpy().astype(np.int8))
                for k, v in parts.items():
                    cols[k].append(v.to("cpu").numpy().astype(np.float32))
                cols["score"].append(score.to("cpu").numpy().astype(np.float32))
                # covariates for the strata
                cols["p_dt"].append(feats.p_dt[m].to("cpu").numpy().astype(np.float32))
                cols["p_dist"].append(feats.p_dist[m].to("cpu").numpy().astype(np.float32))
                cols["p_state_compat"].append(feats.p_state_compat[m].to("cpu").numpy().astype(np.float32))
                cols["p_rel_jaccard"].append(feats.p_rel_jaccard[m].to("cpu").numpy().astype(np.float32))
                cols["src_i"].append(si.astype(np.int8)); cols["src_j"].append(sj.astype(np.int8))
                cols["time_err_sum"].append((ti + tj).astype(np.float32))       # time stratum
                cols["both_low_cep"].append(((cep[si] <= cep_hi) & (cep[sj] <= cep_hi)
                                             ).astype(np.int8))                 # motion stratum
                cols["gt_state_trans"].append(st)                               # state stratum
                cols["rel_share_min"].append(np.minimum(
                    [rel_share[truth[oid[k]][1]] for k in i],
                    [rel_share[truth[oid[k]][1]] for k in j]).astype(np.float32))  # rel stratum
                cols["both_high_rel"].append(((rel[si] >= rel_hi) & (rel[sj] >= rel_hi)
                                              ).astype(np.int8))                # src stratum
                del feats, out, c
            tab = pa.table({k: pa.array(np.concatenate(v)) for k, v in cols.items()})
            path = out_dir / f"{args.variant}_seed{seed}_{split}.parquet"
            pq.write_table(tab, path, compression="zstd")
            mb = path.stat().st_size / 1e6
            print(f"  [seed {seed} {split}] rows={tab.num_rows:,} pos={int(np.concatenate(cols['label']).sum()):,} "
                  f"-> {path.name} {mb:.0f} MB  ({time.time()-t0:.0f}s)")
        del model


if __name__ == "__main__":
    main()
