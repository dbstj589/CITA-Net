#!/usr/bin/env python
"""Evaluate a trained CITA-Net run on a split and print/save metrics.

    python scripts/evaluate.py --config configs/cita_lite.yaml --split dev
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from citanet.config import load_config
from citanet.engine import (
    evaluate,
    featurize_split,
    load_bundle,
    read_split,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _print_table(per: list[dict]) -> None:
    cols = ["scenario_id", "precision", "recall", "f1", "hits@1", "mrr",
            "wrong_merge_rate", "dangling_precision", "dangling_recall"]
    print("  ".join(f"{c:>12.12}" for c in cols))
    for m in per:
        row = []
        for c in cols:
            v = m[c]
            row.append(f"{v:>12.12}" if isinstance(v, str) else f"{v:>12.4f}")
        print("  ".join(row))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--run", default=None, help="run dir (default runs/<stage>)")
    ap.add_argument("--match-threshold", type=float, default=0.5)
    ap.add_argument("--dangling-threshold", type=float, default=0.5)
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_dir = Path(args.run) if args.run else REPO_ROOT / "runs" / cfg.stage
    model, fs, ontology = load_bundle(cfg, run_dir)
    feats = featurize_split(cfg, fs, ontology, args.split)
    ids = read_split(cfg.data_root, args.split)

    agg, per = evaluate(cfg, model, feats, ids,
                        args.match_threshold, args.dangling_threshold)
    _print_table(per)
    print("\n== aggregate (macro avg) ==")
    for k, v in agg.items():
        print(f"  {k:22s} {v:.4f}")

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"metrics_{args.split}.json").write_text(
        json.dumps({"aggregate": agg, "per_scenario": per}, ensure_ascii=False, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
