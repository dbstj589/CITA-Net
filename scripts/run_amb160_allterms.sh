#!/usr/bin/env bash
# amb160 conditional-validity run for the REMAINING four constraint terms
# (no_time / no_state / no_rel / no_src), 5 seeds each = 20 runs.
#
# Settings are IDENTICAL to the existing amb160 main run (see
# scripts/run_amb160_main_seed910_slots160.sh): data/hard_ambiguity/amb160,
# num_slots=160 configs, epochs=40, --full-sectors, decode_full scoring, dev+test.
# m3_full and no_motion are NOT re-run -- their existing 5-seed results in
# runs/hard_ambiguity/amb160_slots160/ are reused for comparison.
#
# Executed as TWO parallel lanes (6 torch threads each on a 12-logical-core CPU),
# so each process sees the same thread count as the original sequential runs.
#   usage: bash scripts/run_amb160_allterms.sh <lane:A|B>
#     lane A: no_time, no_rel     lane B: no_state, no_src
# Resumable: a run whose metrics.json already exists is skipped.
set -u
LANE="${1:?usage: run_amb160_allterms.sh <A|B>}"

PY=.venv/Scripts/python.exe
DATA=data/hard_ambiguity/amb160
ONT=$DATA/ontology
CFGDIR=configs/amb160_allterms
OUT=runs/hard_ambiguity/amb160_allterms
mkdir -p "$OUT"

export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6

case "$LANE" in
  A) VARIANTS=(no_time no_rel) ;;
  B) VARIANTS=(no_state no_src) ;;
  *) echo "lane must be A or B"; exit 2 ;;
esac

SEEDS=(19521006 19521007 19521008 19521009 19521010)
SWEEP_LOG=$OUT/sweep_lane${LANE}.log
JSONL=$OUT/results_lane${LANE}.jsonl

echo "== amb160 all-terms lane $LANE START $(date +%Y-%m-%d_%H:%M:%S) : ${VARIANTS[*]} ==" | tee -a "$SWEEP_LOG"
for m in "${VARIANTS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    RUN=$OUT/${m}_seed${SEED}
    LOG=${RUN}.log
    if [ -f "$RUN/metrics.json" ]; then
      echo ">> [$m / $SEED] SKIP (metrics.json exists)" | tee -a "$SWEEP_LOG"
      continue
    fi
    echo ">> [$m / $SEED] START $(date +%H:%M:%S) -> $RUN" | tee -a "$SWEEP_LOG"
    $PY scripts/train_large.py --config "$CFGDIR/${m}.yaml" --seed "$SEED" \
      --run-dir "$RUN" --full-sectors --epochs 40 \
      --data-root "$DATA" --ontology-dir "$ONT" > "$LOG" 2>&1
    rc=$?
    echo ">> [$m / $SEED] DONE rc=$rc $(date +%H:%M:%S)" | tee -a "$SWEEP_LOG"
    if [ $rc -ne 0 ]; then
      echo "!! [$m / $SEED] FAILED rc=$rc -- stopping lane $LANE" | tee -a "$SWEEP_LOG"
      exit $rc
    fi
    # incremental append of the finished run's dev/test aggregates
    $PY - "$m" "$SEED" "$RUN/metrics.json" "$JSONL" <<'PYEOF'
import json, sys
model, seed, mpath, jsonl = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
d = json.load(open(mpath))
with open(jsonl, "a", encoding="utf-8") as f:
    for split in ("dev", "test"):
        f.write(json.dumps({"model": model, "seed": seed, "split": split,
                            **d[split]["aggregate"]}) + "\n")
print("appended", model, seed, "->", jsonl)
PYEOF
  done
done
echo "== amb160 all-terms lane $LANE ALL DONE $(date +%Y-%m-%d_%H:%M:%S) ==" | tee -a "$SWEEP_LOG"
