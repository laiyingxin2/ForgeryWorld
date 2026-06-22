#!/bin/bash
# P7: Definitive 4-method self-evolution run with ALL THREE bugs fixed.
#  Bug#1 (detector empty): literal <image> token removed earlier.
#  Bug#2 (no-op chains): orchestrator.py now dispatches real ops via OPERATOR_REGISTRY.
#  Bug#3 (detector off-distribution): removed the off-distribution system prompt and
#    replaced the parser with FakeVLM's official eval_vllm.py protocol (first-sentence
#    real/fake, 'real' checked first → handles negation). Re-validated 0.9890 on the
#    5000-sample fakeclue gold (matches published 98.9% / deepfake 0.9649).
#
# Experimental-design fix: every round uses the SAME 3-face src pool (random pick per
# rollout) so the ONLY thing changing across R0->R4 is the agent's accumulated state
# (seed library / skills / markov) — a clean self-evolution signal, not a face-difficulty
# confound. All 4 share one vLLM (async concurrent).

PROJ=/data/disk4/lyx_ICML/self_evolution_forgery
PY=/cpfs01/bob_workspace/miniconda3/envs/fakevlm/bin/python
TAG="p7_$(date +%Y%m%d_%H%M)"
OUT=$PROJ/outputs/p7_definitive/$TAG
mkdir -p $OUT/{m1,m2,m3,m4}
echo "OUT=$OUT" | tee $OUT/META.txt

# Consistent src pool passed EVERY round. Under the CORRECTED detector only these
# 2 of the 10 real_faces are classified real (the other 8 are OOD false-positives:
# 318x322 RGBA crops vs FakeVLM's FF++ 256x256 training). A forgery is a meaningful
# "bypass" only if it starts from a detector-trusted-real source and stays real.
SRC=(
  $PROJ/data/real_faces/0_row0_real.png
  $PROJ/data/real_faces/1_row0_real.png
)
N_BRIEFS=6
N_ROLLOUTS=1

run_method() {
  local LABEL=$1 MODE=$2 OUT_DIR=$3
  for R in 0 1 2 3 4; do
    cd $PROJ/src
    if [ "$LABEL" = "M4" ]; then
      $PY method4_orchestrator.py \
        --rounds 1 --briefs $N_BRIEFS --rollouts $N_ROLLOUTS \
        --tier2-backend fakevlm_local \
        --src-pool "${SRC[@]}" --out $OUT_DIR >> $OUT_DIR/run.log 2>&1
    else
      $PY orchestrator.py \
        --mode $MODE --rounds 1 --briefs $N_BRIEFS --rollouts $N_ROLLOUTS \
        --multi-agent-preset w6_full --tier2-backend fakevlm_local \
        --src-pool "${SRC[@]}" --out $OUT_DIR >> $OUT_DIR/run.log 2>&1
      [ -f $OUT_DIR/reports/r0_${MODE}.json ] && \
        cp $OUT_DIR/reports/r0_${MODE}.json $OUT_DIR/reports/round_${R}_${MODE}.json
    fi
    echo "$LABEL R$R done" >> /tmp/p7_progress.log
  done
  echo "$LABEL DONE" >> /tmp/p7_done.log
}

: > /tmp/p7_progress.log
: > /tmp/p7_done.log
(run_method M1 v1 $OUT/m1) &  echo "M1 PID: $!"
(run_method M2 v2 $OUT/m2) &  echo "M2 PID: $!"
(run_method M3 v2 $OUT/m3) &  echo "M3 PID: $!"
(run_method M4 v_m4 $OUT/m4) & echo "M4 PID: $!"
echo "All 4 launched in parallel on shared vLLM. OUT=$OUT"
wait
echo "ALL 4 COMPLETE" >> /tmp/p7_done.log
echo "DONE. Aggregate: $PY $PROJ/scripts/p6_aggregate.py $OUT"
