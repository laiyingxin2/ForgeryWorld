#!/bin/bash
# P8: 4-method self-evolution re-run against the FAITHFUL detector.
#
# Root cause fixed (2026-06-21): the Tier-2 detector was serving the WRONG checkpoint
# (multi_20260329 multi-task — collapses to "Real" on the raw prompt). All P7 bypass
# numbers were measured against it and are void. The PUBLISHED raw-completion model
# (llava-1.5-7b-fakevlm) is now served via LLM.generate() behind an OpenAI-compatible
# API on GPU7:8001 (scripts/fakevlm_raw_server.py + staged ckpt fakevlm_correct_ckpt/).
# Gold-validated: real 90% / fake 100% / bal 95% (vs wrong-ckpt 8000: 30/90/60).
#
# All 4 methods now point --fakevlm-endpoint at :8001 (8000, the user's shared server,
# is untouched). Same 6-face src pool every round (the faces the CORRECTED detector
# trusts as real) so the only thing changing R0->R4 is the agent's accumulated state.

PROJ=/data/disk4/lyx_ICML/self_evolution_forgery
PY=/cpfs01/bob_workspace/miniconda3/envs/fakevlm/bin/python
EP=http://localhost:8001/v1
TAG="p8_$(date +%Y%m%d_%H%M)"
OUT=$PROJ/outputs/p8_faithful/$TAG
mkdir -p $OUT/{m1,m2,m3,m4}
echo "OUT=$OUT  EP=$EP" | tee $OUT/META.txt

# 6 detector-trusted-real sources (verified against the corrected 8001 detector).
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

run_method() {
  local LABEL=$1 MODE=$2 OUT_DIR=$3
  for R in 0 1 2 3 4; do
    cd $PROJ/src
    if [ "$LABEL" = "M4" ]; then
      $PY method4_orchestrator.py \
        --rounds 1 --briefs $N_BRIEFS --rollouts $N_ROLLOUTS \
        --tier2-backend fakevlm_local --fakevlm-endpoint $EP \
        --src-pool "${SRC[@]}" --out $OUT_DIR >> $OUT_DIR/run.log 2>&1
    else
      $PY orchestrator.py \
        --mode $MODE --rounds 1 --briefs $N_BRIEFS --rollouts $N_ROLLOUTS \
        --multi-agent-preset w6_full --tier2-backend fakevlm_local --fakevlm-endpoint $EP \
        --src-pool "${SRC[@]}" --out $OUT_DIR >> $OUT_DIR/run.log 2>&1
      [ -f $OUT_DIR/reports/r0_${MODE}.json ] && \
        cp $OUT_DIR/reports/r0_${MODE}.json $OUT_DIR/reports/round_${R}_${MODE}.json
    fi
    echo "$LABEL R$R done" >> /tmp/p8_progress.log
  done
  echo "$LABEL DONE" >> /tmp/p8_done.log
}

: > /tmp/p8_progress.log
: > /tmp/p8_done.log
(run_method M1 v1 $OUT/m1) &  echo "M1 PID: $!"
(run_method M2 v2 $OUT/m2) &  echo "M2 PID: $!"
(run_method M3 v2 $OUT/m3) &  echo "M3 PID: $!"
(run_method M4 v_m4 $OUT/m4) & echo "M4 PID: $!"
echo "All 4 launched in parallel. detector=$EP  OUT=$OUT"
wait
echo "ALL 4 COMPLETE" >> /tmp/p8_done.log
echo "DONE. Metrics: $PY $PROJ/scripts/p6_selfevo_metrics.py $OUT"
