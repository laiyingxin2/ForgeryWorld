#!/bin/bash
# P8-M1M2M4: Run M1, M2, M4 in parallel against the faithful 8001 detector.
#
# M3 (true co-evolution) is already complete — see:
#   outputs/m3_coevolution/m3coevo_20260621_1112/m3/coevo/armsrace_curve.json
#   R0=0%, R1=0%, R2=17%  (Non-Decreasing guard held 1.0 throughout)
#
# M1/M2/M4 design:
#   tier2-backend = fakevlm_local (bypass check via FakeVLM 8001 = ground truth)
#   attacker reasoning = Gemini-2.5-flash via viviai (already hardcoded in orchestrator)
#   5 rounds per method (state accumulates = self-evolution)
#   parallel: M1 || M2 || M4

PROJ=/data/disk4/lyx_ICML/self_evolution_forgery
PY=/cpfs01/bob_workspace/miniconda3/envs/fakevlm/bin/python
EP=http://localhost:8001/v1
TAG="p8_$(date +%Y%m%d_%H%M)"
OUT=$PROJ/outputs/p8_faithful/$TAG
mkdir -p "$OUT"/{m1,m2,m4}
echo "OUT=$OUT  EP=$EP" | tee "$OUT/META.txt"

SRC=(
  $PROJ/data/real_faces/0_row0_real.png
  $PROJ/data/real_faces/0_row1_real.png
  $PROJ/data/real_faces/0_row2_real.png
  $PROJ/data/real_faces/0_row3_real.png
  $PROJ/data/real_faces/1_row0_real.png
  $PROJ/data/real_faces/1_row1_real.png
)
N_BRIEFS=8
N_ROLLOUTS=2

run_m1() {
  local OUT_DIR=$1
  echo "[M1] starting 5-round run (mode=v1, frozen skills)" | tee -a "$OUT_DIR/run.log"
  for R in 0 1 2 3 4; do
    cd "$PROJ/src"
    "$PY" orchestrator.py \
      --mode v1 --rounds 1 \
      --briefs $N_BRIEFS --rollouts $N_ROLLOUTS \
      --multi-agent-preset w6_full \
      --tier2-backend fakevlm_local --fakevlm-endpoint $EP \
      --src-pool "${SRC[@]}" --out "$OUT_DIR" \
      >> "$OUT_DIR/run.log" 2>&1
    echo "[M1] round $R done (rc=$?)" | tee -a "$OUT_DIR/run.log"
  done
  echo "[M1] DONE" | tee -a "$OUT_DIR/run.log"
}

run_m2() {
  local OUT_DIR=$1
  echo "[M2] starting 5-round run (mode=v2, self-evolving)" | tee -a "$OUT_DIR/run.log"
  for R in 0 1 2 3 4; do
    cd "$PROJ/src"
    "$PY" orchestrator.py \
      --mode v2 --rounds 1 \
      --briefs $N_BRIEFS --rollouts $N_ROLLOUTS \
      --multi-agent-preset w6_full \
      --tier2-backend fakevlm_local --fakevlm-endpoint $EP \
      --src-pool "${SRC[@]}" --out "$OUT_DIR" \
      >> "$OUT_DIR/run.log" 2>&1
    echo "[M2] round $R done (rc=$?)" | tee -a "$OUT_DIR/run.log"
  done
  echo "[M2] DONE" | tee -a "$OUT_DIR/run.log"
}

run_m4() {
  local OUT_DIR=$1
  echo "[M4] starting 5-round run (method4_orchestrator)" | tee -a "$OUT_DIR/run.log"
  for R in 0 1 2 3 4; do
    cd "$PROJ/src"
    "$PY" method4_orchestrator.py \
      --rounds 1 --briefs $N_BRIEFS --rollouts $N_ROLLOUTS \
      --tier2-backend fakevlm_local --fakevlm-endpoint $EP \
      --src-pool "${SRC[@]}" --out "$OUT_DIR" \
      >> "$OUT_DIR/run.log" 2>&1
    echo "[M4] round $R done (rc=$?)" | tee -a "$OUT_DIR/run.log"
  done
  echo "[M4] DONE" | tee -a "$OUT_DIR/run.log"
}

echo "Launching M1, M2, M4 in parallel..."
run_m1 "$OUT/m1" &  PID_M1=$!
run_m2 "$OUT/m2" &  PID_M2=$!
run_m4 "$OUT/m4" &  PID_M4=$!
echo "M1 PID=$PID_M1  M2 PID=$PID_M2  M4 PID=$PID_M4" | tee -a "$OUT/META.txt"

wait
echo "=== ALL DONE ===" | tee -a "$OUT/META.txt"
echo "Computing metrics..."
"$PY" "$PROJ/scripts/p6_selfevo_metrics.py" "$OUT" 2>&1 | tee "$OUT/metrics.json"
echo "Metrics written to $OUT/metrics.json"
