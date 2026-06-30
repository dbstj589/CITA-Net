#!/usr/bin/env python
"""Robustness (difficulty) sweep: variant x difficulty-dataset x seed, resumable.

Trains the SAME 3 ablation variants (m3_full / no_motion / no_time, all Sinkhorn
decoder -> comparable) on each generated difficulty suite under data/robustness/,
and APPENDS one JSON record per cell to results/robustness/results.jsonl
immediately (interrupted sweep resumes by skipping completed cells).

    python scripts/run_robustness_hill395.py                 # 4 datasets x 3 variants x 3 seeds = 36
    python scripts/run_robustness_hill395.py --datasets noise2x noise4x

The frozen baseline column D0 (noise 1x, dangling 0.2) is NOT retrained here --
it is the prior ablation sweep (results/ablation/results.jsonl); summarize_
robustness.py merges it. The frozen data and the configs/ablation/*.yaml are
never modified; only --data-root/--ontology-dir are overridden at the CLI.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
CFG_DIR = REPO / "configs" / "ablation"
DATA_DIR = REPO / "data" / "robustness"
RUNS_DIR = REPO / "runs" / "robustness"
RESULTS = REPO / "results" / "robustness"
RESULTS_JSONL = RESULTS / "results.jsonl"

# dataset -> (noise_mult, dangling_ratio) for provenance in each record
DATASETS = {"noise2x": (2.0, 0.2), "noise4x": (4.0, 0.2),
            "dang035": (1.0, 0.35), "dang050": (1.0, 0.5)}
VARIANTS = ["m3_full", "no_motion", "no_time"]
DEFAULT_SEEDS = [19521006, 19521007, 19521008]


def load_done() -> set[tuple[str, str, int]]:
    done = set()
    if RESULTS_JSONL.exists():
        for ln in RESULTS_JSONL.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                done.add((r["dataset"], r["variant"], int(r["seed"])))
    return done


def run_cmd(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as lf:
        proc = subprocess.run(cmd, cwd=REPO, stdout=lf, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-25:])
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n--- log tail ---\n{tail}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()

    for ds in args.datasets:
        if not (DATA_DIR / ds / "manifest_global.json").exists():
            sys.exit(f"missing dataset {DATA_DIR / ds} -- run gen_dataset_hill395.py --build-suite first")

    RESULTS.mkdir(parents=True, exist_ok=True)
    done = load_done()
    cells = [(ds, v, s) for ds in args.datasets for v in args.variants for s in args.seeds]
    todo = [c for c in cells if c not in done]
    print(f"== robustness sweep: {len(args.datasets)} datasets x {len(args.variants)} variants x "
          f"{len(args.seeds)} seeds = {len(cells)} cells; {len(done)} done, {len(todo)} to run ==")
    print(f"   datasets: {args.datasets}\n   variants: {args.variants}\n   seeds: {args.seeds}  epochs: {args.epochs}")

    for n, (ds, variant, seed) in enumerate(todo, 1):
        nm, dr = DATASETS[ds]
        run_dir = RUNS_DIR / f"{ds}_{variant}_seed{seed}"
        cfg = CFG_DIR / f"{variant}.yaml"
        data_root = DATA_DIR / ds
        t0 = time.time()
        print(f"\n[{n}/{len(todo)}] {ds} / {variant} / seed={seed} -> {run_dir.relative_to(REPO)}", flush=True)

        run_cmd([PY, "scripts/train_large.py", "--config", str(cfg),
                 "--data-root", str(data_root), "--ontology-dir", str(data_root / "ontology"),
                 "--seed", str(seed), "--run-dir", str(run_dir),
                 "--full-sectors", "--epochs", str(args.epochs)],
                run_dir / "train.log")

        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        d, t = metrics["dev"]["aggregate"], metrics["test"]["aggregate"]
        rec = {"dataset": ds, "noise_mult": nm, "dangling_ratio": dr,
               "variant": variant, "seed": seed, "epochs": args.epochs,
               "elapsed_s": round(time.time() - t0, 1), "dev": d, "test": t,
               "run_dir": str(run_dir.relative_to(REPO))}
        with open(RESULTS_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"    done in {rec['elapsed_s']:.0f}s  dev_f1={d['f1']:.4f} test_f1={t['f1']:.4f} "
              f"test_R={t['recall']:.4f} impossible={t.get('impossible_transition_rate')}", flush=True)

    print(f"\n== robustness sweep complete: {len(todo)} runs; results -> {RESULTS_JSONL.relative_to(REPO)} ==")
    print("   next: python scripts/summarize_robustness.py")


if __name__ == "__main__":
    main()
