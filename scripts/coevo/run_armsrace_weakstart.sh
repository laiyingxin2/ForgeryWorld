#!/usr/bin/env bash
# WEAK-START arms-race (the "expensive" phase, user-approved). Unlike the v2 grid
# (strong FakeVLM base -> attacker floored at ~0 bypass by capability-gap, 2505.20162),
# here BOTH the served detector AND the per-round LoRA train on the VANILLA llava-1.5-7b
# base (a NAIVE detector, ~random on forgery). So at r0 forgeries BYPASS easily (high
# ASR); each round trains a LoRA on the SAME vanilla base -> the detector hardens ->
# ASR drops round by round = the CHASE/MART-style suppression curve we could not get
# from the strong base. (Active-Attacks 2509.21947 weak-start curriculum.)
#
# Two parallel variants on two dedicated weak servers (8006/8007):
#   W1 clean    : K=1 (train every round), rank-8 ep1, bypass-floor=0  -> cleanest
#                 "defender closes the gap" monotone suppression (MART -85% / CHASE -76%).
#   W2 throttle : K=2 (TTUR throttle),     rank-8 ep1, bypass-floor=0  -> attacker gets
#                 frozen-detector rounds between updates -> oscillating descent.
# bypass-floor is DISABLED on purpose: weak-start WANTS the detector to suppress ASR
# (that IS the result); the floor (preserve-a-gap) is a strong-base anti-collapse lever.
set -uo pipefail
PROJ=/data/disk4/lyx_ICML/self_evolution_forgery
PY=/cpfs01/bob_workspace/miniconda3/envs/fakevlm/bin/python
DRIVER=$PROJ/scripts/coevo/run_coevolution_v2.py
VANILLA=/cpfs01/bob_workspace/students/lyx/Model_download/llava-hf/llava-1.5-7b-hf
cd $PROJ
STAMP=$(date +%Y%m%d_%H%M)

for port in 8006 8007; do
  code=$(curl -s -o /dev/null -w "%{http_code}" -m5 http://localhost:$port/health || echo 000)
  [ "$code" != "200" ] && { echo "FATAL: :$port not up ($code)"; exit 1; }
done

# Larger QC'd real pool (60 SCUT-curated) -> the detector learns "real" robustly and
# the held-out real guard has ~15 samples (vs 2 with the 6-western pool), so the
# real-acc floor 0.80 is meaningful instead of brittle 2/2.
mapfile -t POOL < <(ls $PROJ/data/pool_scut_curated/*.png)

run() {  # name endpoint train_gpu period
  local NAME=$1 EP=$2 GPU=$3 K=$4
  local OUT=$PROJ/outputs/weakstart/${NAME}_${STAMP}; mkdir -p $OUT
  echo "$(date +%H:%M:%S) START $NAME ep=$EP gpu=$GPU K=$K -> $OUT"
  $PY $DRIVER --proj $PROJ --py $PY --out $OUT --endpoint $EP \
      --rounds 8 --briefs 6 --rollouts 2 --train-gpu $GPU \
      --detector-base $VANILLA \
      --defender-period $K --lora-r 8 --epochs 1 --bypass-floor 0.0 \
      --guard-real-floor 0.80 --guard-real-frac 0.25 \
      --replay-per-round 8 --src-pool "${POOL[@]}" > $OUT/driver.log 2>&1
  echo "$(date +%H:%M:%S) END $NAME (rc=$?) -> $OUT/m3/coevo/armsrace_curve_v2.json"
}

run W1_clean    http://localhost:8006/v1 4 1 &
P1=$!
run W2_throttle http://localhost:8007/v1 6 2 &
P2=$!
echo "launched W1(pid $P1,:8006,gpu4,K=1) W2(pid $P2,:8007,gpu6,K=2)"
wait $P1 $P2
echo "=== WEAK-START ARMS-RACE DONE $(date +%H:%M:%S) ==="
