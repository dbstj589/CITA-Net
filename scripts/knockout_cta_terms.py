"""Analysis C -- inference-time knockout of a single CTA term. NO TRAINING.

Weights stay exactly as trained (m3_full); one term's contribution is zeroed at inference
and the sector is re-decoded. Because the comparison is made INSIDE one set of weights,
the seed-to-seed training noise that dominated study 1 cancels -- that is the whole point
of this analysis.

Mechanism: CTA reads `self.enabled` (a plain set built from cfg.cta.enabled_terms) on
every forward and zeroes any term not in it, in BOTH the score and the feature matrix fed
to the pair head (model/cta.py:89-99). Discarding a name from that set at runtime is
therefore an exact knockout, not an approximation, and it propagates the whole way:
score -> p_transition -> decoder context -> assign, and feature_matrix -> pair_logits ->
dangling evidence.

Fidelity gate: knocking out NOTHING must reproduce the recorded metrics.json exactly.
Knocking out `time` is a second, free null check -- b_time is structurally 0 on this data
(blocking emits only forward-in-time candidates), so its knockout must also be a no-op.

Interpretation is deliberately limited: the terms were trained together and have adapted
to each other, so this measures each term's MARGINAL contribution at inference in an
already-co-adapted model. It is not the same quantity as its contribution during training.

    python scripts/knockout_cta_terms.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from redecode_revive import decode_full_revive          # noqa: E402

from citanet.config import load_config                  # noqa: E402
from citanet.data.stream import read_split               # noqa: E402
from citanet.decode import decode_full                   # noqa: E402
from citanet.engine_large import (                       # noqa: E402
    _featurize, _feat_sig, _sector_dir, load_large_bundle)
from citanet.eval import aggregate, evaluate_full        # noqa: E402

SEEDS = [19521006, 19521007, 19521008, 19521009, 19521010]
TERMS = ["time", "motion", "state", "rel", "src"]
SPLITS = ["dev", "test"]
REVIVE_MARGIN = 0.70
TOL = 5e-3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--terms", nargs="+", default=TERMS)
    ap.add_argument("--runs-dir", default=str(REPO / "runs/realistic_v1/ablation"))
    ap.add_argument("--config", default=str(REPO / "configs/realistic_v1/m3_full.yaml"))
    ap.add_argument("--out-dir", default=str(REPO / "results/realistic_v1_score_analysis"))
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    cfg.data_root, cfg.ontology_dir = "data/realistic_v1", "data/realistic_v1/ontology"
    rows = []
    print(f"== inference-time knockout: seeds={args.seeds} terms={args.terms} ==")

    for seed in args.seeds:
        run_dir = Path(args.runs_dir) / f"m3_full_seed{seed}"
        model, fs, ont = load_large_bundle(cfg, run_dir)
        recorded = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        cache = os.environ.get("CITANET_FEAT_CACHE")
        sig = _feat_sig(cfg, fs) if cache else None
        dev_t = next(model.parameters()).device
        full_enabled = set(model.cta.enabled)

        for ko in ["__none__"] + list(args.terms):
            model.cta.enabled = set(full_enabled)
            if ko != "__none__":
                model.cta.enabled.discard(ko)
            for split in SPLITS:
                per = {"standard": [], "revive": []}
                for sid in read_split(cfg.data_root, split):
                    feats = _featurize(cfg, sid, fs, ont, cache, sig).to(dev_t)
                    sd = _sector_dir(cfg, sid)
                    with torch.no_grad(), torch.device(dev_t):
                        o = model(feats)
                        per["standard"].append(evaluate_full(decode_full(o, feats, ont), feats, sd))
                        per["revive"].append(evaluate_full(
                            decode_full_revive(o, feats, ont, margin=REVIVE_MARGIN), feats, sd))
                    del feats, o
                for dec in ("standard", "revive"):
                    rows.append({"seed": seed, "knockout": ko, "split": split,
                                 "decode": dec, **aggregate(per[dec])})
                if ko == "__none__" and split in recorded:      # fidelity gate
                    a = next(r for r in rows if r["seed"] == seed and r["knockout"] == "__none__"
                             and r["split"] == split and r["decode"] == "standard")
                    d = {k: a[k] - recorded[split]["aggregate"][k] for k in ("precision", "recall", "f1")}
                    ok = all(abs(v) < TOL for v in d.values())
                    print(f"  [seed {seed} {split}] identity gate {'OK' if ok else 'FAIL'} "
                          f"dP={d['precision']:+.5f} dR={d['recall']:+.5f} dF1={d['f1']:+.5f}")
                    if not ok:
                        _dump(out, rows)
                        raise SystemExit(f"IDENTITY GATE FAIL seed{seed} {split}: {d} -- stopping")
            f1s = " ".join(f"{sp}={next(r['f1'] for r in rows if r['seed']==seed and r['knockout']==ko and r['split']==sp and r['decode']=='standard'):.4f}"
                           for sp in SPLITS)
            print(f"  [seed {seed}] knockout={ko:9} standard F1 {f1s}")
        del model
    _dump(out, rows)


def _dump(out: Path, rows) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(out / "knockout_deltas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(f"WROTE {out/'knockout_deltas.csv'}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
