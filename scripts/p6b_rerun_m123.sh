#!/bin/bash
# P6b: Re-run ONLY M1/M2/M3 with the FIXED orchestrator.py op dispatch.
# Root bug #2 (now fixed): orchestrator.py mocked every non-API op as identity-pass
# (liveportrait/facevid2vid/gfpgan/inswapper_128/simswap_256 all phantom names) ->
# forged image == source -> arcface=1.0 -> PSEUDO_BYPASS_REJECTED -> 0% bypass,
# no learning gradient. Now wired to the unified OPERATOR_REGISTRY (same lib M4 uses)
# and operator_list advertises only real registry keys.
#
# Reuses the existing P6 OUT dir (M4 already done + valid there). Wipes m1/m2/m3
# contaminated state for a clean R0->R4 self-evolution restart.

PROJ=/data/disk4/lyx_ICML/self_evolution_forgery
PY=/cpfs01/bob_workspace/miniconda3/envs/fakevlm/bin/python
OUT=$PROJ/outputs/p6_working_detector/p6_20260620_2051
echo "Reusing OUT=$OUT (keeping valid m4)"

# Wipe contaminated m1/m2/m3 (built from no-op mocked trajectories)
for m in m1 m2 m3; do rm -rf $OUT/$m; mkdir -p $OUT/$m; done

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
    $PY orchestrator.py \
      --mode $MODE --rounds 1 --briefs $N_BRIEFS --rollouts $N_ROLLOUTS \
      --multi-agent-preset w6_full --tier2-backend fakevlm_local \
      --src-pool $SRC_FACE --out $OUT_DIR >> $OUT_DIR/run.log 2>&1
    [ -f $OUT_DIR/reports/r0_${MODE}.json ] && \
      cp $OUT_DIR/reports/r0_${MODE}.json $OUT_DIR/reports/round_${R}_${MODE}.json
    echo "$LABEL R$R done" >> /tmp/p6b_progress.log
  done
  echo "$LABEL DONE" >> /tmp/p6b_done.log
}

: > /tmp/p6b_progress.log
: > /tmp/p6b_done.log
(run_method M1 v1 $OUT/m1) &  echo "M1 PID: $!"
(run_method M2 v2 $OUT/m2) &  echo "M2 PID: $!"
(run_method M3 v2 $OUT/m3) &  echo "M3 PID: $!"
echo "M1/M2/M3 re-launched in parallel on shared vLLM. OUT=$OUT"
wait
echo "M123 RERUN COMPLETE" >> /tmp/p6b_done.log
