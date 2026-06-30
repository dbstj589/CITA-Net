#!/usr/bin/env bash
# Stage-1 main training on amb160 (hard ambiguity), seed 19521006 only.
# 3 models, sequential (RAM-bound: ~3.2GB/run, 5GB free). epochs=40 for
# consistency with all prior amb/ablation experiments. decode_full scoring
# (built into evaluate_large when decoder is on). dev+test both evaluated.
set -u
PY=.venv/Scripts/python.exe
SEED=19521006
DATA=data/hard_ambiguity/amb160
ONT=$DATA/ontology
SWEEP_LOG=runs/hard_ambiguity/amb160/sweep_seed${SEED}.log

declare -A CFG=(
  [m3_full]=configs/ablation/m3_full.yaml
  [no_motion]=configs/ablation/no_motion.yaml
  [term_gating]=configs/term_gating.yaml
)
ORDER=(m3_full no_motion term_gating)

echo "== amb160 main sweep seed=$SEED START $(date +%H:%M:%S) ==" | tee "$SWEEP_LOG"
for m in "${ORDER[@]}"; do
  RUN=runs/hard_ambiguity/amb160/${m}_seed${SEED}
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
echo "== amb160 main sweep seed=$SEED ALL DONE $(date +%H:%M:%S) ==" | tee -a "$SWEEP_LOG"
