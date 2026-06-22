#!/bin/bash
# Speed boost: parallelize M2/M3/M4 alongside the already-running P5 M1.
# All 4 share single vLLM (async concurrent handles it).
# F2 fix in orchestrator code already applied.

PROJ=/data/disk4/lyx_ICML/self_evolution_forgery
PY=/cpfs01/bob_workspace/miniconda3/envs/fakevlm/bin/python

# use SAME OUT dir as P5 (so curves aggregator catches everything)
OUT=$(ls -dt $PROJ/outputs/p5_after_fixes/p5_* | head -1)
echo "Using OUT=$OUT"

SRC=(
  $PROJ/data/real_faces/0_row0_real.png
  $PROJ/data/real_faces/0_row2_real.png
  $PROJ/data/real_faces/1_row1_real.png
)
N_BRIEFS=5
N_ROLLOUTS=1

run_method() {
  local LABEL=$1 MODE=$2 OUT_DIR=$3
  for R in 0 1 2 3 4; do
    SRC_FACE="${SRC[$((R % ${#SRC[@]}))]}"
    cd $PROJ/src
    if [ "$LABEL" = "M4" ]; then
      $PY method4_orchestrator.py \
        --rounds 1 --briefs $N_BRIEFS --rollouts $N_ROLLOUTS \
        --tier2-backend fakevlm_local \
        --src-pool $SRC_FACE --out $OUT_DIR >> $OUT_DIR/run.log 2>&1
    else
      $PY orchestrator.py \
        --mode $MODE --rounds 1 --briefs $N_BRIEFS --rollouts $N_ROLLOUTS \
        --multi-agent-preset w6_full --tier2-backend fakevlm_local \
        --src-pool $SRC_FACE --out $OUT_DIR >> $OUT_DIR/run.log 2>&1
      [ -f $OUT_DIR/reports/r0_${MODE}.json ] && \
        cp $OUT_DIR/reports/r0_${MODE}.json $OUT_DIR/reports/round_${R}_${MODE}.json
    fi
  done
  echo "$LABEL DONE" >> /tmp/p5_parallel_done.log
}

# Launch M2/M3/M4 in parallel (M1 already running from p5_after_fixes.sh)
(run_method M2 v2 $OUT/m2) &
echo "M2 PID: $!"
(run_method M3 v2 $OUT/m3) &
echo "M3 PID: $!"
(run_method M4 v_m4 $OUT/m4) &
echo "M4 PID: $!"

echo "M2/M3/M4 launched in parallel; M1 from p5 sequential script still running too"
echo "All 4 share vLLM port 8000 (async concurrent)"
