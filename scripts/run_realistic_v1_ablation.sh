#!/usr/bin/env bash
# realistic-v1 constraint-term conditional-validity MAIN run.
# Full CITA-Net (graph+decoder) ablation: 6 variants (m3_full/no_motion/no_time/
# no_state/no_rel/no_src) on data/realistic_v1. Sequential, resumable (a run whose
# metrics.json exists is skipped), per-run done log, incremental results.jsonl.
#   usage: bash scripts/run_realistic_v1_ablation.sh <seed> [variant ...]
#     default variant list = all 6.  epochs=40, --full-sectors, decode_full, dev+test.
set -u
SEED="${1:?usage: run_realistic_v1_ablation.sh <seed> [variant ...]}"
shift || true
VARIANTS=("$@")
if [ ${#VARIANTS[@]} -eq 0 ]; then
  VARIANTS=(m3_full no_motion no_time no_state no_rel no_src)
fi

# portable venv python: Linux (.venv/bin) or Windows (.venv/Scripts), else PATH
if   [ -x .venv/bin/python ];         then PY=.venv/bin/python
elif [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe
else PY=python; fi
DATA=data/realistic_v1
ONT=$DATA/ontology
CFGDIR=configs/realistic_v1
OUT=runs/realistic_v1/ablation
mkdir -p "$OUT"
# Featurization disk cache (see engine_large._featurize): blocking+assemble is
# ~58s/sector and deterministic in (data, blocking, vocab) -> compute once, reuse
# across every epoch AND every variant/seed (~5s/sector load). Shared dir so all
# 30 runs hit the same cache. Results are byte-identical to the uncached path.
export CITANET_FEAT_CACHE="$PWD/runs/realistic_v1/feat_cache"
mkdir -p "$CITANET_FEAT_CACHE"
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONIOENCODING=utf-8

SWEEP_LOG=$OUT/sweep_seed${SEED}.log
JSONL=$OUT/results.jsonl

echo "== realistic-v1 ablation seed $SEED START $(date +%Y-%m-%d_%H:%M:%S) : ${VARIANTS[*]} ==" | tee -a "$SWEEP_LOG"
for m in "${VARIANTS[@]}"; do
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
    echo "!! [$m / $SEED] FAILED rc=$rc -- stopping" | tee -a "$SWEEP_LOG"
    exit $rc
  fi
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
echo "== realistic-v1 ablation seed $SEED ALL DONE $(date +%Y-%m-%d_%H:%M:%S) ==" | tee -a "$SWEEP_LOG"
