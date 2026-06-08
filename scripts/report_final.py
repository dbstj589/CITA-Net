#!/usr/bin/env python
"""Train the canonical Full CITA-Net, evaluate dev+test, and save the combined
metrics table to runs/cita_full/metrics.json. Also emits the scn_0001 Part-B
output JSON.

Combines the full (Sinkhorn) decoder metrics (P/R/F1, Wrong-Merge, Fragmentation,
Trajectory-Consistency, Impossible-Transition, dangling P/R) with the Hits@1 /
MRR ranking metrics from the pairwise affinity path.

    python scripts/report_final.py
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from citanet.config import load_config
from citanet.decode import decode_entities, decode_full
from citanet.engine import (
    featurize_split,
    read_split,
    save_bundle,
    scenario_dir,
    train,
)
from citanet.eval import aggregate, evaluate_decode, evaluate_full
from citanet.output_schema import validate_output
from citanet.serialize import build_part_b

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = REPO_ROOT / "runs" / "cita_full"

TABLE_KEYS = ["precision", "recall", "f1", "hits@1", "mrr",
              "wrong_merge_rate", "fragmentation_rate",
              "trajectory_consistency_rate", "impossible_transition_rate",
              "dangling_precision", "dangling_recall"]


@torch.no_grad()
def eval_split(cfg, model, ontology, feats, ids):
    per = []
    for f, sid in zip(feats, ids):
        out = model(f)
        full = evaluate_full(decode_full(out, f, ontology), f, scenario_dir(cfg, sid))
        rank = evaluate_decode(decode_entities(out, f), scenario_dir(cfg, sid))
        full["hits@1"] = rank["hits@1"]
        full["mrr"] = rank["mrr"]
        full["scenario_id"] = sid
        per.append(full)
    return aggregate(per), per


def main() -> None:
    cfg = load_config(REPO_ROOT / "configs" / "cita_full.yaml")
    print("== training canonical cita_full ==")
    bundle = train(cfg, verbose=False)
    save_bundle(bundle, RUN_DIR)

    results = {}
    for split in ("dev", "test"):
        feats = bundle.dev_feats if split == "dev" else \
            featurize_split(cfg, bundle.fs, bundle.ontology, "test")
        ids = read_split(cfg.data_root, split)
        agg, per = eval_split(cfg, bundle.model, bundle.ontology, feats, ids)
        results[split] = {"aggregate": agg, "per_scenario": per}

    (RUN_DIR / "metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # print combined table
    print("\n" + "=" * 78)
    print("CITA-Net (Full) -- dev/test aggregate metrics  [runs/cita_full/metrics.json]")
    print("=" * 78)
    print(f"  {'metric':30} {'dev':>10} {'test':>10}")
    for k in TABLE_KEYS:
        print(f"  {k:30} {results['dev']['aggregate'][k]:>10.4f} "
              f"{results['test']['aggregate'][k]:>10.4f}")

    # scn_0001 Part-B sample
    f0 = dict(zip(read_split(cfg.data_root, "dev"), bundle.dev_feats))["scn_0001"]
    with torch.no_grad():
        out0 = bundle.model(f0)
    doc = build_part_b("scn_0001", cfg, decode_full(out0, f0, bundle.ontology), f0)
    validate_output(doc)
    (RUN_DIR / "output_scn_0001.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nscn_0001 Part-B output -> {RUN_DIR / 'output_scn_0001.json'} (schema OK)")


if __name__ == "__main__":
    main()
