"""realistic-v1 second-pass scoring: re-decode every run with the REVIVE rule -- NO RETRAIN.

Why a new script instead of scripts/ablation_revive.py (experiment #7): that one
hardcodes CFG_DIR=configs/ablation, RUNS=runs/ablation and
ABL_JSONL=results/ablation/results.jsonl, has no --data-root/--ontology-dir, and
its fidelity check expects a NESTED baseline schema ({"variant":..,"dev":{..}})
while runs/realistic_v1/ablation/results.jsonl is FLAT ({"model":..,"split":..,..}),
so the check would silently no-op. Existing scripts are left untouched; the decoder
itself is imported, not reimplemented.

Three deliberate differences from the experiment-#5/#7 harnesses:

1. FEATURIZE CACHE. Those call data.stream.featurize_sector directly, bypassing
   engine_large._featurize -- the only place CITANET_FEAT_CACHE is honoured. On
   realistic-v1 that costs ~40 s/sector every time. We go through _featurize, so a
   warm cache makes it ~5 s/sector. Output is byte-identical either way.

2. NO SPLIT-WIDE RAM CACHE. Those hold every sector's forward output in memory to
   sweep many margins. realistic-v1 peaks at ~1.1-1.2 GB per sector, so caching a
   12-13 sector split risks OOM. We need only two margins, so each sector is
   decoded at both and then released -- memory stays at one sector.

3. RECOVERS THE MISSING DENOMINATOR. eval.evaluate_full computes
   wrong_merge_rate = wrong / len(result.identities) (eval.py:136) but does not
   return len(result.identities), and n_pred_matches is a strict subset of it
   (A+B identities only), so the denominator cannot be reconstructed from
   metrics.json. We record n_pred_identities and n_wrong_merges here. At margin=0
   the revive decode is mathematically identical to decode_full, so the margin-0
   numbers ARE the standard-decode denominators -- exact, not an approximation.

Fidelity gate: margin=0 must reproduce the recorded metrics.json dev/test P/R/F1
within 5e-3. Unlike the older scripts (which only warn), a failure ABORTS -- a
mismatch means the re-decode is not the same object as the recorded run and no
interpretation is allowed.

    python scripts/revive_realistic_v1.py                      # all 6 variants, all finished seeds
    python scripts/revive_realistic_v1.py --variants m3_full --seeds 19521006
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))          # so `import redecode_revive` resolves
from redecode_revive import decode_full_revive     # noqa: E402  (reuse, do not reimplement)

from citanet.config import load_config             # noqa: E402
from citanet.data.stream import read_split         # noqa: E402
from citanet.engine_large import (                 # noqa: E402
    _featurize, _feat_sig, _sector_dir, load_large_bundle)
from citanet.eval import aggregate, evaluate_full  # noqa: E402

VARIANTS = ["m3_full", "no_motion", "no_time", "no_state", "no_rel", "no_src"]
SPLITS = ["dev", "test"]
FIDELITY_TOL = 5e-3


def _pred_identity_counts(result, feats, sector_dir: Path) -> tuple[int, int]:
    """(n_pred_identities, n_wrong_merges) -- the wrong_merge_rate denominator and
    numerator that evaluate_full computes inline and discards (eval.py:121-136).
    Recomputed here with the identical rule so the ratio reproduces exactly."""
    gold = json.loads((Path(sector_dir) / "labels" / "gold_identities.json")
                      .read_text(encoding="utf-8"))
    ent_gold = {}
    for i in gold["identities"]:
        for kg, lid in i["member_local_entities"].items():
            ent_gold[(kg, lid)] = i["gold_identity_id"]
    wrong = 0
    for idn in result.identities:
        gids = set()
        for oid, _ in idn.member_obs:
            kg, lid = feats.entity_of[oid].split(":", 1)
            g = ent_gold.get((kg, lid))
            if g is not None:
                gids.add(g)
        if len(gids) > 1:
            wrong += 1
    return len(result.identities), wrong


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="default: every seed whose runs are all finished")
    ap.add_argument("--margin", type=float, default=0.70,
                    help="revive margin (experiment #7 value; do NOT re-tune)")
    ap.add_argument("--runs-dir", default=str(REPO / "runs" / "realistic_v1" / "ablation"))
    ap.add_argument("--config-dir", default=str(REPO / "configs" / "realistic_v1"))
    ap.add_argument("--data-root", default="data/realistic_v1")
    ap.add_argument("--ontology-dir", default="data/realistic_v1/ontology")
    ap.add_argument("--out-dir", default=str(REPO / "results" / "realistic_v1_ablation"))
    ap.add_argument("--tag", default="n5")
    ap.add_argument("--no-abort-on-fidelity", action="store_true",
                    help="record the mismatch and continue instead of aborting (diagnostics only)")
    args = ap.parse_args()

    runs = Path(args.runs_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    margins = [0.0, args.margin]

    seeds = args.seeds
    if not seeds:
        found = {}
        for p in runs.glob("*_seed*/metrics.json"):
            m = re.fullmatch(r"(.+)_seed(\d+)", p.parent.name)
            if m and m.group(1) in args.variants:
                found.setdefault(int(m.group(2)), set()).add(m.group(1))
        seeds = sorted(s for s, v in found.items() if set(args.variants) <= v)
    if not seeds:
        raise SystemExit("no seed has all requested variants finished")

    print(f"== revive re-scoring: variants={args.variants} seeds={seeds} margins={margins} ==")
    print(f"   runs={runs}  feat_cache={os.environ.get('CITANET_FEAT_CACHE') or '(off!)'}")

    rows, per_sector_rows, fidelity = [], [], []
    for variant in args.variants:
        cfg = load_config(Path(args.config_dir) / f"{variant}.yaml")
        cfg.data_root, cfg.ontology_dir = args.data_root, args.ontology_dir
        for seed in seeds:
            run_dir = runs / f"{variant}_seed{seed}"
            if not (run_dir / "model.pt").exists():
                raise SystemExit(f"missing checkpoint {run_dir/'model.pt'} -- no retrain, aborting")
            cfg.seed = seed
            model, fs, ont = load_large_bundle(cfg, run_dir)
            recorded = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            cache_dir = os.environ.get("CITANET_FEAT_CACHE")
            sig = _feat_sig(cfg, fs) if cache_dir else None
            dev_t = next(model.parameters()).device

            for split in SPLITS:
                per = {mg: [] for mg in margins}
                revived = {mg: 0 for mg in margins}
                for sid in read_split(cfg.data_root, split):
                    feats = _featurize(cfg, sid, fs, ont, cache_dir, sig).to(dev_t)
                    sd = _sector_dir(cfg, sid)
                    with torch.no_grad(), torch.device(dev_t):
                        out = model(feats)
                        for mg in margins:                    # decode both, then release
                            res = decode_full_revive(out, feats, ont, margin=mg)
                            m = evaluate_full(res, feats, sd)
                            n_ident, n_wrong = _pred_identity_counts(res, feats, sd)
                            m["n_pred_identities"] = n_ident      # <- recovered denominator
                            m["n_wrong_merges"] = n_wrong
                            m["scenario_id"] = sid
                            per[mg].append(m)
                            revived[mg] += getattr(res, "n_revived", 0)
                            per_sector_rows.append(
                                {"variant": variant, "seed": seed, "split": split,
                                 "margin": mg, **m})
                            del res
                    del feats, out
                for mg in margins:
                    agg = aggregate(per[mg])
                    rows.append({"variant": variant, "seed": seed, "split": split,
                                 "margin": mg, "n_revived": revived[mg], **agg})
                # --- fidelity: margin 0 must reproduce the recorded run ---
                a0 = next(r for r in rows if r["variant"] == variant and r["seed"] == seed
                          and r["split"] == split and r["margin"] == 0.0)
                rec = recorded[split]["aggregate"]
                deltas = {k: a0[k] - rec[k] for k in ("precision", "recall", "f1")}
                ok = all(abs(v) < FIDELITY_TOL for v in deltas.values())
                fidelity.append({"variant": variant, "seed": seed, "split": split,
                                 "ok": ok, **{f"d_{k}": v for k, v in deltas.items()}})
                flag = "OK " if ok else "FAIL"
                print(f"  [{variant} {seed} {split}] {flag} margin0 vs recorded "
                      f"dP={deltas['precision']:+.5f} dR={deltas['recall']:+.5f} "
                      f"dF1={deltas['f1']:+.5f} | revived@{args.margin}={revived[args.margin]}")
                if not ok and not args.no_abort_on_fidelity:
                    _dump(out_dir, args.tag, rows, per_sector_rows, fidelity)
                    raise SystemExit(
                        f"FIDELITY FAIL ({variant} seed{seed} {split}): margin-0 re-decode does not "
                        f"reproduce the recorded metrics within {FIDELITY_TOL}. Deltas={deltas}. "
                        "Interpretation is not allowed -- stopping. Partial output was written.")
            del model

    _dump(out_dir, args.tag, rows, per_sector_rows, fidelity)
    nbad = sum(1 for f in fidelity if not f["ok"])
    print(f"\nfidelity: {len(fidelity)-nbad}/{len(fidelity)} PASS"
          + ("" if nbad == 0 else f"  ** {nbad} FAIL **"))


def _dump(out_dir: Path, tag: str, rows, per_sector_rows, fidelity) -> None:
    (out_dir / f"revive_results_{tag}.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    if per_sector_rows:
        with open(out_dir / f"revive_per_sector_{tag}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(per_sector_rows[0].keys()))
            w.writeheader(); w.writerows(per_sector_rows)
    with open(out_dir / f"revive_fidelity_{tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fidelity[0].keys()))
        w.writeheader(); w.writerows(fidelity)
    print(f"WROTE -> {out_dir}/revive_{{results,per_sector,fidelity}}_{tag}.*")


if __name__ == "__main__":
    main()
