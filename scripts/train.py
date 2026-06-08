#!/usr/bin/env python
"""Train CITA-Net for a given stage config and save the run.

    python scripts/train.py --config configs/cita_lite.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from citanet.config import config_to_dict, load_config
from citanet.engine import evaluate, read_split, save_bundle, train

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None, help="run dir (default runs/<stage>)")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    out_dir = Path(args.out) if args.out else REPO_ROOT / "runs" / cfg.stage

    print(f"== training stage={cfg.stage} device={cfg.resolved_device()} ==")
    bundle = train(cfg)
    save_bundle(bundle, out_dir)

    dev_ids = read_split(cfg.data_root, "dev")
    agg, per = evaluate(cfg, bundle.model, bundle.dev_feats, dev_ids)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dev_metrics.json").write_text(
        json.dumps({"config": config_to_dict(cfg), "aggregate": agg, "per_scenario": per},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n== dev aggregate ==")
    for k, v in agg.items():
        print(f"  {k:22s} {v:.4f}")
    print(f"\nsaved run to {out_dir}")


if __name__ == "__main__":
    main()
