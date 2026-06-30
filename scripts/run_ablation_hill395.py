#!/usr/bin/env python
"""Run the Hill 395 ablation sweep: variants x seeds, sequential, resumable.

For each (variant, seed) it trains via scripts/train_large.py into its own run
dir, extracts dev/test metrics via scripts/extract_hill395_results.py
(--metrics-only), and APPENDS one JSON record to results/ablation/results.jsonl
immediately -- so an interrupted sweep resumes by skipping completed cells.

    python scripts/run_ablation_hill395.py                  # 1A + 1B  (8 variants)
    python scripts/run_ablation_hill395.py --variants all   # + 1C loss (10 variants)
    python scripts/run_ablation_hill395.py --epochs 40 --seeds 19521006 19521007 19521008

Data (data/battlefield_hill395_large) is never touched; blocking is fixed by the
generated configs. Configs come from configs/ablation/ (run gen_ablation_configs.py
first).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable                       # the venv python running this driver
CFG_DIR = REPO / "configs" / "ablation"
RUNS_DIR = REPO / "runs" / "ablation"
RESULTS = REPO / "results" / "ablation"
RESULTS_JSONL = RESULTS / "results.jsonl"
BASELINE_RUN = REPO / "runs" / "cita_full_large"   # restore the overwritten baseline

VARIANTS_1A = ["m1", "m2", "m3_full"]
VARIANTS_1B = ["no_time", "no_motion", "no_state", "no_rel", "no_src"]
VARIANTS_1C = ["no_ltraj", "no_lassign"]
DEFAULT_SEEDS = [19521006, 19521007, 19521008]


def load_done() -> set[tuple[str, int]]:
    done = set()
    if RESULTS_JSONL.exists():
        for ln in RESULTS_JSONL.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            done.add((r["variant"], int(r["seed"])))
    return done


def run_cmd(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as lf:
        proc = subprocess.run(cmd, cwd=REPO, stdout=lf,
                              stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-25:])
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n--- log tail ---\n{tail}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", choices=["core", "all", "loss"], default="core",
                    help="core=1A+1B (8); all=+1C (10); loss=1C only (2)")
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()

    if args.variants == "core":
        variants = VARIANTS_1A + VARIANTS_1B
    elif args.variants == "all":
        variants = VARIANTS_1A + VARIANTS_1B + VARIANTS_1C
    else:
        variants = VARIANTS_1C

    missing = [v for v in variants if not (CFG_DIR / f"{v}.yaml").exists()]
    if missing:
        sys.exit(f"missing configs {missing} -- run: python scripts/gen_ablation_configs.py")

    RESULTS.mkdir(parents=True, exist_ok=True)
    done = load_done()
    cells = [(v, s) for v in variants for s in args.seeds]
    todo = [c for c in cells if c not in done]
    print(f"== ablation sweep: {len(variants)} variants x {len(args.seeds)} seeds "
          f"= {len(cells)} cells; {len(done)} done, {len(todo)} to run ==")
    print(f"   variants: {variants}")
    print(f"   seeds:    {args.seeds}   epochs: {args.epochs}")

    for n, (variant, seed) in enumerate(todo, 1):
        run_dir = RUNS_DIR / f"{variant}_seed{seed}"
        cfg = CFG_DIR / f"{variant}.yaml"
        t0 = time.time()
        print(f"\n[{n}/{len(todo)}] {variant} seed={seed} -> {run_dir.relative_to(REPO)}", flush=True)

        run_cmd([PY, "scripts/train_large.py", "--config", str(cfg), "--seed", str(seed),
                 "--run-dir", str(run_dir), "--full-sectors", "--epochs", str(args.epochs)],
                run_dir / "train.log")

        # train_large.py writes dev/test aggregate metrics (decoder-aware) here.
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        d, t = metrics["dev"]["aggregate"], metrics["test"]["aggregate"]
        rec = {"variant": variant, "seed": seed, "stage": cfg.stem,
               "epochs": args.epochs, "elapsed_s": round(time.time() - t0, 1),
               "dev": d, "test": t,
               "run_dir": str(run_dir.relative_to(REPO))}
        with open(RESULTS_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"    done in {rec['elapsed_s']:.0f}s  "
              f"dev_f1={d['f1']:.4f} test_f1={t['f1']:.4f} "
              f"test_P={t['precision']:.4f} test_R={t['recall']:.4f}", flush=True)

        # Restore the baseline bundle that the earlier mis-run overwrote: the
        # m3_full @ base-seed run reproduces configs/cita_full_hill395.yaml exactly.
        if variant == "m3_full" and seed == DEFAULT_SEEDS[0]:
            shutil.copytree(run_dir, BASELINE_RUN, dirs_exist_ok=True)
            print(f"    restored baseline bundle -> {BASELINE_RUN.relative_to(REPO)}", flush=True)

    print(f"\n== sweep complete: {len(todo)} runs; results -> {RESULTS_JSONL.relative_to(REPO)} ==")
    print("   next: python scripts/summarize_ablation.py")


if __name__ == "__main__":
    main()
