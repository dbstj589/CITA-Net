#!/usr/bin/env bash
# Stage-2: add seeds 19521007 & 19521008 for n=3. IDENTICAL settings to seed
# 19521006 (amb160, num_slots=160 configs, epochs=40, full-sectors, decode_full,
# dev+test). 6 runs total, sequential (RAM-bound ~3.2GB/run). seed 19521006 is
# reused as-is (not re-run).
set -u
PY=.venv/Scripts/python.exe
DATA=data/hard_ambiguity/amb160
ONT=$DATA/ontology
OUT=runs/hard_ambiguity/amb160_slots160
mkdir -p "$OUT"
SWEEP_LOG=$OUT/sweep_seed78.log

declare -A CFG=(
  [m3_full]=configs/amb160_main/m3_full.yaml
  [no_motion]=configs/amb160_main/no_motion.yaml
  [term_gating]=configs/amb160_main/term_gating.yaml
)
ORDER=(m3_full no_motion term_gating)
SEEDS=(19521007 19521008)

echo "== amb160 slots160 sweep seeds 07/08 START $(date +%Y-%m-%d_%H:%M:%S) ==" | tee "$SWEEP_LOG"
for SEED in "${SEEDS[@]}"; do
  for m in "${ORDER[@]}"; do
    RUN=$OUT/${m}_seed${SEED}
    LOG=${RUN}.log
    echo ">> [seed $SEED / $m] START $(date +%H:%M:%S) -> $RUN" | tee -a "$SWEEP_LOG"
    $PY scripts/train_large.py --config "${CFG[$m]}" --seed "$SEED" \
      --run-dir "$RUN" --full-sectors --epochs 40 \
      --data-root "$DATA" --ontology-dir "$ONT" > "$LOG" 2>&1
    rc=$?
    echo ">> [seed $SEED / $m] DONE rc=$rc $(date +%H:%M:%S)" | tee -a "$SWEEP_LOG"
    if [ $rc -ne 0 ]; then
      echo "!! [seed $SEED / $m] FAILED rc=$rc -- stopping sweep" | tee -a "$SWEEP_LOG"
      exit $rc
    fi
  done
done
echo "== amb160 slots160 sweep seeds 07/08 ALL DONE $(date +%Y-%m-%d_%H:%M:%S) ==" | tee -a "$SWEEP_LOG"
