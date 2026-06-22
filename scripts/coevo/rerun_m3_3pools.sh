#!/usr/bin/env bash
# Real Method-3 co-evolution (run_coevolution.py) across the SAME 3 base pools as the
# M1/M2/M4 :8001 rerun, so M3 becomes comparable in the 4-method table.
#   - Detector = TRAINABLE FakeVLM on :8002 (LoRA hot-reload). NOT the :8001 frozen one.
#   - Each round: attack -> isolate forgeries -> build data (replay+reals) -> train LoRA
#     (GPU1, warm-start) -> Non-Decreasing guard -> hot-reload. Logs armsrace_curve.json.
# Serial across pools (one LoRA train at a time) to avoid GPU contention with the rerun.
set -uo pipefail
PROJ=/data/disk4/lyx_ICML/self_evolution_forgery
PY=/cpfs01/bob_workspace/miniconda3/envs/fakevlm/bin/python
DRIVER=$PROJ/scripts/coevo/run_coevolution.py
cd $PROJ

# Each pool gets its OWN trainable detector server so per-round LoRA hot-reloads do not
# clobber each other (the server has a single "defender" LoRA slot). 3 servers, 3 ports.
for port in 8002 8003 8004; do
  code=$(curl -s -o /dev/null -w "%{http_code}" -m5 http://localhost:$port/health || echo 000)
  [ "$code" != "200" ] && { echo "FATAL: :$port (M3 trainable detector) not up (got $code). Start fakevlm_lora_server.py."; exit 1; }
  echo "  :$port /health=200 OK"
done

# Western (same 6 detector-trusted real faces as p8)
WESTERN=( $PROJ/data/real_faces/0_row0_real.png $PROJ/data/real_faces/0_row1_real.png \
          $PROJ/data/real_faces/0_row2_real.png $PROJ/data/real_faces/0_row3_real.png \
          $PROJ/data/real_faces/1_row0_real.png $PROJ/data/real_faces/1_row1_real.png )
mapfile -t ASIAN < <(ls $PROJ/data/pool_asian_kyc/*.png)
mapfile -t SCUT  < <(ls $PROJ/data/pool_scut_curated/*.png)

# Parallel across pools: VRAM has headroom (each card ~90GB+ free, LoRA train ~35GB),
# so stack all 3 concurrently, each pinned to a distinct train GPU. Attack inference all
# hits the shared :8002 server; --train-gpu only places the per-round LoRA training.
run_pool() {
  local NAME=$1; local GPU=$2; local EP=$3; shift 3
  local SRC=("$@")
  local TAG="m3_${NAME}_$(date +%Y%m%d_%H%M)"
  local OUT=$PROJ/outputs/m3_coevolution/$TAG
  mkdir -p $OUT
  echo "===== $(date +%H:%M:%S) M3 co-evo START pool=$NAME gpu=$GPU ep=$EP (n=${#SRC[@]}) -> $OUT ====="
  $PY $DRIVER --proj $PROJ --py $PY --out $OUT --endpoint $EP \
      --rounds 3 --briefs 8 --rollouts 2 --train-gpu $GPU \
      --replay-per-round 8 --guard-drop 0.15 --epochs 3 \
      --src-pool "${SRC[@]}" > $OUT/driver.log 2>&1
  echo "===== $(date +%H:%M:%S) M3 co-evo END pool=$NAME (rc=$?) curve: $OUT/m3/coevo/armsrace_curve.json ====="
}

# pool -> (train GPU, own detector endpoint). Train GPUs (1/3/4) are disjoint from the
# detector-server GPUs (6/7/5), so training and inference never contend on one card.
run_pool western 1 http://localhost:8002/v1 "${WESTERN[@]}" &
PID_W=$!
run_pool asian   3 http://localhost:8003/v1 "${ASIAN[@]}"   &
PID_A=$!
run_pool scut    4 http://localhost:8004/v1 "${SCUT[@]}"    &
PID_S=$!
echo "launched western(pid $PID_W,gpu1,:8002) asian(pid $PID_A,gpu3,:8003) scut(pid $PID_S,gpu4,:8004)"
wait $PID_W $PID_A $PID_S
echo "=== ALL 3 M3 CO-EVO POOLS DONE $(date +%H:%M:%S) ==="
