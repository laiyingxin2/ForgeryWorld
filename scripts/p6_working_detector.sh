#!/bin/bash
# P6: Re-run all 4 methods with the FIXED FakeVLM detector.
# Root bug (now fixed in fakevlm_judge_real.py): literal "<image>" token in the
# prompt made vLLM emit empty output -> tier2 always is_fake=False -> constant
# detector signal -> NO learning gradient for any method. Prompt now matches the
# fakeclue training format ("Does the image looks real/fake?") and the parser is
# validated 0/5000 mismatch vs gold pred_label.
#
# All 4 share one vLLM (async concurrent). M1/M4 fast, M2/M3 (v2 full stack) slow.
# 5 rounds x 5 briefs x 1 rollout. All use FakeVLM+LoRA tier2 (tier3 off).

PROJ=/data/disk4/lyx_ICML/self_evolution_forgery
PY=/cpfs01/bob_workspace/miniconda3/envs/fakevlm/bin/python
TAG="p6_$(date +%Y%m%d_%H%M)"
OUT=$PROJ/outputs/p6_working_detector/$TAG
mkdir -p $OUT/{m1,m2,m3,m4}
echo "OUT=$OUT" | tee $OUT/META.txt

# Source faces the FIXED detector classifies as REAL (clean baseline: detector
# trusts the source, so a forgery that stays "real" is a genuine bypass).
SRC=(
  $PROJ/data/real_faces/0_row0_real.png
  $PROJ/data/real_faces/1_row0_real.png
  $PROJ/data/real_faces/1_row4_real.png
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
    echo "$LABEL R$R done" >> /tmp/p6_progress.log
  done
  echo "$LABEL DONE" >> /tmp/p6_done.log
}

: > /tmp/p6_progress.log
: > /tmp/p6_done.log
(run_method M1 v1 $OUT/m1) &  echo "M1 PID: $!"
(run_method M2 v2 $OUT/m2) &  echo "M2 PID: $!"
(run_method M3 v2 $OUT/m3) &  echo "M3 PID: $!"
(run_method M4 v_m4 $OUT/m4) & echo "M4 PID: $!"
echo "All 4 launched in parallel on shared vLLM. OUT=$OUT"
wait
echo "ALL 4 COMPLETE" >> /tmp/p6_done.log
