#!/usr/bin/env bash
# Stage-1 main training on amb160, seed 19521006, num_slots=160 (slot-starvation
# fix: amb160 has ~110 identities/sector, the default 48-slot decoder collapsed).
# New configs/run-dirs only; 48-slot results left untouched (unused). epochs=40
# for consistency; decode_full scoring (decoder on); dev+test evaluated. Sequential
# (RAM-bound ~3.2GB/run).
set -u
PY=.venv/Scripts/python.exe
SEED=19521006
DATA=data/hard_ambiguity/amb160
ONT=$DATA/ontology
OUT=runs/hard_ambiguity/amb160_slots160
mkdir -p "$OUT"
SWEEP_LOG=$OUT/sweep_seed${SEED}.log

declare -A CFG=(
  [m3_full]=configs/amb160_main/m3_full.yaml
  [no_motion]=configs/amb160_main/no_motion.yaml
  [term_gating]=configs/amb160_main/term_gating.yaml
)
ORDER=(m3_full no_motion term_gating)

echo "== amb160 slots160 sweep seed=$SEED START $(date +%H:%M:%S) ==" | tee "$SWEEP_LOG"
for m in "${ORDER[@]}"; do
  RUN=$OUT/${m}_seed${SEED}
  LOG=${RUN}.log
  echo ">> [$m] START $(date +%H:%M:%S) -> $RUN" | tee -a "$SWEEP_LOG"
  $PY scripts/train_large.py --config "${CFG[$m]}" --seed "$SEED" \
    --run-dir "$RUN" --full-sectors --epochs 40 \
    --data-root "$DATA" --ontology-dir "$ONT" > "$LOG" 2>&1
  rc=$?
  echo ">> [$m] DONE rc=$rc $(date +%H:%M:%S)" | tee -a "$SWEEP_LOG"
  if [ $rc -ne 0 ]; then
    echo "!! [$m] FAILED rc=$rc -- stopping sweep" | tee -a "$SWEEP_LOG"
    exit $rc
  fi
done
echo "== amb160 slots160 sweep seed=$SEED ALL DONE $(date +%H:%M:%S) ==" | tee -a "$SWEEP_LOG"
