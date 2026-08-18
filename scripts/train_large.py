#!/usr/bin/env python
"""Train + evaluate Full CITA-Net on the LARGE sector-sharded suite (streaming).

    python scripts/train_large.py                     # subsampled sectors/epoch
    python scripts/train_large.py --full-sectors      # all train sectors/epoch
    python scripts/train_large.py --sectors-per-epoch 16 --epochs 40

Reports dev/test full metrics and the streaming peak memory, writes
runs/cita_full_large/metrics.json, and emits a Part-B sample for one dev sector.
"""
from __future__ import annotations

import argparse
import json
import tracemalloc
from pathlib import Path

import torch

from citanet.config import load_config
from citanet.decode import decode_full
from citanet.engine_large import (
    evaluate_large,
    save_large_bundle,
    train_large,
)
from citanet.data.stream import featurize_sector, read_split
from citanet.output_schema import validate_output
from citanet.serialize import build_part_b

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "cita_full_large"
TABLE = ["precision", "recall", "f1", "wrong_merge_rate", "fragmentation_rate",
         "trajectory_consistency_rate", "impossible_transition_rate",
         "dangling_precision", "dangling_recall"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "cita_full_large.yaml"))
    ap.add_argument("--sectors-per-epoch", type=int, default=12)
    ap.add_argument("--full-sectors", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--eval-sectors", type=int, default=None, help="cap eval sectors")
    ap.add_argument("--seed", type=int, default=None,
                    help="override config seed (controls init + sector shuffle order)")
    ap.add_argument("--run-dir", default=None,
                    help="output dir for model bundle, metrics.json, Part-B sample "
                         "(default runs/cita_full_large)")
    ap.add_argument("--data-root", default=None,
                    help="override cfg.data_root (e.g. a robustness difficulty suite)")
    ap.add_argument("--ontology-dir", default=None,
                    help="override cfg.ontology_dir (defaults alongside --data-root suite)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.seed is not None:
        cfg.seed = args.seed
    if args.data_root is not None:
        cfg.data_root = args.data_root
    if args.ontology_dir is not None:
        cfg.ontology_dir = args.ontology_dir
    run_dir = Path(args.run_dir) if args.run_dir else DEFAULT_RUN_DIR

    print(f"== training {cfg.stage}  (full_sectors={args.full_sectors}, "
          f"sectors/epoch={'all' if args.full_sectors else args.sectors_per_epoch}, "
          f"epochs={cfg.train.epochs}) ==")
    bundle = train_large(cfg, sectors_per_epoch=args.sectors_per_epoch,
                         full_sectors=args.full_sectors)
    save_large_bundle(bundle, run_dir)

    # streaming evaluation with peak-memory measurement
    results = {}
    for split in ("dev", "test"):
        tracemalloc.start()
        agg, per = evaluate_large(cfg, bundle.model, bundle.fs, bundle.ontology,
                                  split, max_sectors=args.eval_sectors)
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        agg["streaming_peak_mb"] = round(peak / 1e6, 2)
        results[split] = {"aggregate": agg, "per_scenario": per}

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                          encoding="utf-8")

    print("\n" + "=" * 70)
    print("Large-suite Full CITA-Net -- dev/test aggregate (streaming)")
    print("=" * 70)
    print(f"  {'metric':30} {'dev':>10} {'test':>10}")
    for k in TABLE:
        dv, tv = results['dev']['aggregate'].get(k), results['test']['aggregate'].get(k)
        # M1/M2 (greedy decode, no trajectory) legitimately lack the trajectory
        # metrics -> print n/a rather than KeyError.
        ds = f"{dv:>10.4f}" if isinstance(dv, (int, float)) else f"{'n/a':>10}"
        ts = f"{tv:>10.4f}" if isinstance(tv, (int, float)) else f"{'n/a':>10}"
        print(f"  {k:30} {ds} {ts}")
    print(f"  {'streaming_peak_mb':30} {results['dev']['aggregate']['streaming_peak_mb']:>10.2f} "
          f"{results['test']['aggregate']['streaming_peak_mb']:>10.2f}")

    # Part-B sample for one dev sector. The serialized Part-B trajectory output is
    # an M3 product (Sinkhorn decode); skip it cleanly when the decoder is off.
    if bundle.model.decoder is not None:
        sid = read_split(cfg.data_root, "dev")[0]
        # Part-B serialization does host-side conversions, so run this one-sector
        # sample on CPU: featurize_sector yields CPU tensors and after GPU training
        # the model lives on cuda. The bundle is already saved (above), so moving
        # the model here is harmless. On a CPU run .to("cpu") is a no-op, leaving
        # the original code path byte-identical.
        bundle.model.to("cpu")
        feats = featurize_sector(Path(cfg.data_root) / "sectors" / sid, bundle.fs, bundle.ontology, cfg)
        with torch.no_grad():
            out = bundle.model(feats)
        doc = build_part_b(sid, cfg, decode_full(out, feats, bundle.ontology), feats)
        validate_output(doc)
        (run_dir / f"output_{sid}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                                                    encoding="utf-8")
        print(f"\nPart-B sample ({sid}): identities={doc['stats']['n_identities']} "
              f"dangling={doc['stats']['n_dangling']} (schema OK) -> {run_dir / f'output_{sid}.json'}")
    else:
        print("\nPart-B sample skipped (no M3 decoder in this ablation variant).")


if __name__ == "__main__":
    main()
