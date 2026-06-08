#!/usr/bin/env python
"""Run a trained Full CITA-Net on a scenario and emit the Part-B output JSON.

    python scripts/predict.py --config configs/cita_full.yaml --scenario scn_0001
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from citanet.config import load_config
from citanet.decode import decode_full
from citanet.engine import load_bundle, scenario_dir
from citanet.model.featurize import featurize_scenario
from citanet.output_schema import validate_output
from citanet.serialize import build_part_b

REPO_ROOT = Path(__file__).resolve().parents[1]


def predict(cfg, model, fs, ontology, scenario_id: str) -> dict:
    feats = featurize_scenario(scenario_dir(cfg, scenario_id), fs, ontology, cfg)
    with torch.no_grad():
        out = model(feats)
    result = decode_full(out, feats, ontology)
    doc = build_part_b(scenario_id, cfg, result, feats)
    validate_output(doc)            # raises if the document is malformed
    return doc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--run", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_dir = Path(args.run) if args.run else REPO_ROOT / "runs" / cfg.stage
    model, fs, ontology = load_bundle(cfg, run_dir)
    doc = predict(cfg, model, fs, ontology, args.scenario)

    out_path = Path(args.out) if args.out else run_dir / f"output_{args.scenario}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"schema validation: OK")
    print(f"identities={doc['stats']['n_identities']} dangling={doc['stats']['n_dangling']} "
          f"impossible_transitions={doc['stats']['n_impossible_transitions']}")
    for idn in doc["identities"]:
        print(f"  {idn['global_id']}  type={idn['type']:8} "
              f"local={idn['local_entity_ids']}  match_conf={idn['match_confidence']:.3f}")
    for d in doc["dangling"]:
        print(f"  [dangling] {d['kg_id']}:{d['local_entity_id']} -> abstain ({d['reason']})")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
