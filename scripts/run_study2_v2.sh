#!/usr/bin/env bash
# Study 2 v2: 3 arms x 15 seeds = 45 runs.  bash scripts/run_study2_v2.sh [regime]
#   regime = A (deterministic) | B (non-deterministic).  Required, no default -- the
#   regime is decided by the smoke test and every run must use the same one.
#
# Skip logic uses a COMPLETION MARKER written only after a run exits rc=0. A run that
# died midway can leave metrics.json behind (train_large writes it before the Part-B
# step), so metrics.json alone must never be treated as "finished".
#
# Regime A additionally enforces the placebo gate after every seed: no_time is a null
# manipulation (b_time is structurally 0), so under determinism its dF1 must be exactly
# 0. A non-zero value means the deterministic regime is not holding and the whole sweep
# stops -- pre-registered in docs/PREREG_realistic_v1_study2_v2.md section 4.A.
set -u
cd /workspace/CITA-Net || exit 1

REGIME="${1:-}"
case "$REGIME" in
  A) export CITANET_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 ;;
  B) : ;;
  *) echo "usage: $0 A|B   (regime must be given explicitly)"; exit 2 ;;
esac

export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export CITANET_FEAT_CACHE="$PWD/runs/realistic_v1/feat_cache"
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
# NOT set (pre-registered): CITANET_TRAIN_MAX_PAIRS, CITANET_EVAL_ON_CPU

OUT=runs/realistic_v1_study2/runs
mkdir -p "$OUT"
ARMS="m3_full no_rel no_time"
SEEDS="19521011 19521012 19521013 19521014 19521015 19521016 19521017 19521018 \
19521019 19521020 19521021 19521022 19521023 19521024 19521025"
LOG="$OUT/../sweep.log"

say() { echo "== [study2] $* $(date -Is) ==" | tee -a "$LOG"; }
f1of() {  # f1of <dir> <split>
  .venv/bin/python -c "import json,sys;print(json.load(open(sys.argv[1]+'/metrics.json'))[sys.argv[2]]['aggregate']['f1'])" "$1" "$2"
}

say "START regime=$REGIME pid=$$ arms='$ARMS' seeds=15"
for s in $SEEDS; do
  for a in $ARMS; do
    d="$OUT/${a}_seed${s}"
    if [ -f "$d/COMPLETED" ]; then
      echo ">> [$a/$s] SKIP (completion marker present)" | tee -a "$LOG"; continue
    fi
    rm -rf "$d"                       # never resume onto a partial run dir
    t0=$(date +%s)
    echo ">> [$a/$s] START $(date +%H:%M:%S)" | tee -a "$LOG"
    .venv/bin/python scripts/train_large.py --config "configs/realistic_v1/${a}.yaml" \
      --seed "$s" --run-dir "$d" --full-sectors --epochs 40 \
      --data-root data/realistic_v1 --ontology-dir data/realistic_v1/ontology \
      > "$OUT/${a}_seed${s}.log" 2>&1
    rc=$?; t1=$(date +%s)
    echo ">> [$a/$s] DONE rc=$rc elapsed=$((t1-t0))s" | tee -a "$LOG"
    if [ $rc -ne 0 ]; then say "FAILED $a/$s rc=$rc -- stopping"; exit $rc; fi
    printf 'regime=%s rc=0 elapsed=%s\n' "$REGIME" "$((t1-t0))" > "$d/COMPLETED"
  done

  # ---- regime A: placebo gate (pre-registered stop condition) ----
  if [ "$REGIME" = A ]; then
    for sp in dev test; do
      base=$(f1of "$OUT/m3_full_seed${s}" "$sp"); plac=$(f1of "$OUT/no_time_seed${s}" "$sp")
      if [ "$base" != "$plac" ]; then
        say "PLACEBO GATE VIOLATION seed=$s split=$sp m3_full=$base no_time=$plac -- stopping"
        exit 3
      fi
    done
    echo ">> [seed $s] placebo gate OK (no_time == m3_full exactly, dev+test)" | tee -a "$LOG"
  fi
  say "seed $s complete (3/3)"
done
say "ALL 45 RUNS COMPLETE"
