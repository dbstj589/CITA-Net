#!/usr/bin/env bash
# Convergence check: 2 arms x 10 seeds x 80 epochs = 20 runs.  bash scripts/run_conv80.sh
# Pre-registration: docs/PREREG_realistic_v1_conv80.md
#
# Regime A (deterministic), identical to study 2, so the 40-vs-80 comparison is paired on
# both seed and regime. Seeds are the first 10 of study 2's, so the 40-epoch side is taken
# from study 2 and never re-run.
#
# Stop-on-failure is absolute (user instruction 2026-08-05): any rc != 0 halts the whole
# sweep rather than skipping the run, so a systematic fault cannot quietly accumulate
# damaged runs while nobody is watching.
set -u
cd /workspace/CITA-Net || exit 1

export CITANET_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export CITANET_FEAT_CACHE="$PWD/runs/realistic_v1/feat_cache"
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6

OUT=runs/realistic_v1_conv80/runs
mkdir -p "$OUT"
LOG="$OUT/../sweep.log"
ARMS="m3_full no_rel"
SEEDS="19521011 19521012 19521013 19521014 19521015 19521016 19521017 19521018 19521019 19521020"

say() { echo "== [conv80] $* $(date -Is) ==" | tee -a "$LOG"; }
say "START regime=A pid=$$ arms='$ARMS' seeds=10 epochs=80"

for s in $SEEDS; do
  for a in $ARMS; do
    d="$OUT/${a}_seed${s}"
    if [ -f "$d/COMPLETED" ]; then
      echo ">> [$a/$s] SKIP (completion marker)" | tee -a "$LOG"; continue
    fi
    rm -rf "$d"
    t0=$(date +%s)
    echo ">> [$a/$s] START $(date +%H:%M:%S)" | tee -a "$LOG"
    .venv/bin/python scripts/train_large.py --config "configs/realistic_v1_conv80/${a}.yaml" \
      --seed "$s" --run-dir "$d" --full-sectors --epochs 80 \
      --data-root data/realistic_v1 --ontology-dir data/realistic_v1/ontology \
      > "$OUT/${a}_seed${s}.log" 2>&1
    rc=$?; t1=$(date +%s)
    echo ">> [$a/$s] DONE rc=$rc elapsed=$((t1-t0))s" | tee -a "$LOG"
    if [ $rc -ne 0 ]; then
      say "FAILED $a/$s rc=$rc -- STOPPING ENTIRE SWEEP (no skip-and-continue)"
      exit $rc
    fi
    printf 'regime=A epochs=80 rc=0 elapsed=%s\n' "$((t1-t0))" > "$d/COMPLETED"
  done
  say "seed $s complete (2/2)"
done
say "ALL 20 RUNS COMPLETE"
