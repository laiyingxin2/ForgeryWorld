#!/bin/bash
# Full-scale CHEAP run of all 5 methods (all gemini-2.5-flash = w1_cheap).
#   pool   = pool_scut_asian (2266 Asian ID headshots)  -- PRIMARY base
#   scale  = 10 rounds x 16 briefs x 2 rollouts  (bigger than weak-start 8x6x2 / p8 5x8x2)
#   M1/M2/M4 : self-evolution vs FROZEN strong detector :8001
#   M3       : co-evolution on weak vanilla detector :8006 (reset to base at start)
#   M5       : population co-evolution (K=3) on weak vanilla detector :8007
# GPU plan (avoid GPU0 = only 30G free): M1->2  M2->1  M4->2  M3->6  M5->4
set -u
PROJ=/data/disk4/lyx_ICML/self_evolution_forgery
PY=/cpfs01/bob_workspace/miniconda3/envs/fakevlm/bin/python
VANILLA=/cpfs01/bob_workspace/students/lyx/Model_download/llava-hf/llava-1.5-7b-hf
TAG="full_$(date +%Y%m%d_%H%M)"
OUT=$PROJ/outputs/full_run/$TAG
mkdir -p "$OUT"/{m1,m2,m4}
echo "OUT=$OUT" | tee "$OUT/META.txt"

ROUNDS=10; BRIEFS=16; ROLLOUTS=2
POOL=( $PROJ/data/pool_scut_asian/*.png )
echo "pool size = ${#POOL[@]}" | tee -a "$OUT/META.txt"
EP_STRONG=http://localhost:8001/v1
DONE=/tmp/full_run_done.log; : > "$DONE"

# ---- M1/M2/M4: --rounds 1 shell loop (state persists in --out dir = self-evolution) ----
run_orch() {  # $1=label $2=mode $3=cuda $4=outdir
  local L=$1 MODE=$2 CVD=$3 OD=$4
  echo "[$L] start (mode=$MODE, GPU=$CVD)" | tee -a "$OD/run.log"
  for R in $(seq 0 $((ROUNDS-1))); do
    CUDA_VISIBLE_DEVICES=$CVD "$PY" orchestrator.py \
      --mode $MODE --rounds 1 --briefs $BRIEFS --rollouts $ROLLOUTS \
      --multi-agent-preset w1_cheap \
      --tier2-backend fakevlm_local --fakevlm-endpoint $EP_STRONG \
      --src-pool "${POOL[@]}" --out "$OD" >> "$OD/run.log" 2>&1
    [ -f "$OD/reports/r0_${MODE}.json" ] && cp "$OD/reports/r0_${MODE}.json" "$OD/reports/round_${R}_${MODE}.json"
    echo "[$L] round $R done (rc=$?)" >> "$OD/run.log"
  done
  echo "$L DONE" >> "$DONE"
}

run_m4() {  # method4 has its own orchestrator (flash by default)
  local CVD=$1 OD=$2
  echo "[M4] start (GPU=$CVD)" | tee -a "$OD/run.log"
  for R in $(seq 0 $((ROUNDS-1))); do
    CUDA_VISIBLE_DEVICES=$CVD "$PY" method4_orchestrator.py \
      --rounds 1 --briefs $BRIEFS --rollouts $ROLLOUTS \
      --tier2-backend fakevlm_local --fakevlm-endpoint $EP_STRONG \
      --src-pool "${POOL[@]}" --out "$OD" >> "$OD/run.log" 2>&1
    echo "[M4] round $R done (rc=$?)" >> "$OD/run.log"
  done
  echo "M4 DONE" >> "$DONE"
}

cd "$PROJ/src"
( run_orch M1 v1 2 "$OUT/m1" ) &
sleep 10
( run_orch M2 v2 1 "$OUT/m2" ) &
sleep 10
( run_m4 2 "$OUT/m4" ) &
sleep 10

# ---- M3: single-lineage co-evolution on weak vanilla :8006, all GPU6 ----
( CUDA_VISIBLE_DEVICES=6 "$PY" "$PROJ/scripts/coevo/run_coevolution_v2.py" \
    --out "$OUT/M3" --endpoint http://localhost:8006/v1 \
    --rounds $ROUNDS --briefs $BRIEFS --rollouts $ROLLOUTS \
    --src-pool "${POOL[@]}" --preset w1_cheap \
    --detector-base "$VANILLA" --train-gpu 6 \
    --lora-r 8 --epochs 1 --defender-period 2 --bypass-floor 0.0 \
    --guard-real-floor 0.80 --guard-real-frac 0.25 \
    > "$OUT/M3.log" 2>&1 ; echo "M3 DONE" >> "$DONE" ) &
sleep 10

# ---- M5: population co-evolution (K=3) on weak vanilla :8007, all GPU4 ----
( CUDA_VISIBLE_DEVICES=4 "$PY" "$PROJ/scripts/coevo/run_coevolution_m5.py" \
    --out "$OUT/M5" --endpoint http://localhost:8007/v1 \
    --rounds $ROUNDS --lineages 3 --briefs $BRIEFS --rollouts $ROLLOUTS \
    --src-pool "${POOL[@]}" --preset w1_cheap \
    --detector-base "$VANILLA" --train-gpu 4 \
    --epochs 1 --guard-real-floor 0.80 --guard-real-frac 0.25 \
    > "$OUT/M5.log" 2>&1 ; echo "M5 DONE" >> "$DONE" ) &

wait
echo "ALL 5 COMPLETE" | tee -a "$DONE"
# clean valid re-score for the frozen-detector methods
"$PY" "$PROJ/scripts/coevo/eval_clean_coverage.py" \
  M1="$OUT/m1" M2="$OUT/m2" M4="$OUT/m4" > "$OUT/clean_coverage.json" 2>"$OUT/clean_coverage.err" || true
echo "DONE. results under $OUT" | tee -a "$DONE"
