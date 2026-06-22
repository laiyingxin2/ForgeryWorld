#!/usr/bin/env bash
# v2 anti-collapse ablation grid (cheap proof-of-effect). Three runs in PARALLEL,
# each on its own trainable detector server + train GPU, compared against the v1
# collapse baseline (m3_*_20260621_1933). Modest budget: briefs=6 rollouts=2.
#   A  throttle-ONLY  : K=3, STRONG defender (r16,ep3), no bypass-floor  -> isolate throttle
#   C  full-v2        : K=3, WEAK defender (r8,ep1), bypass-floor 0.10    -> all levers
#   D  full-v2 + OOD  : same as C but on the SCUT Asian pool (reopen gap via OOD)
set -uo pipefail
PROJ=/data/disk4/lyx_ICML/self_evolution_forgery
PY=/cpfs01/bob_workspace/miniconda3/envs/fakevlm/bin/python
DRIVER=$PROJ/scripts/coevo/run_coevolution_v2.py
cd $PROJ
STAMP=$(date +%Y%m%d_%H%M)

for port in 8002 8003 8004; do
  code=$(curl -s -o /dev/null -w "%{http_code}" -m5 http://localhost:$port/health || echo 000)
  [ "$code" != "200" ] && { echo "FATAL: :$port not up ($code)"; exit 1; }
done

WESTERN=( $PROJ/data/real_faces/0_row0_real.png $PROJ/data/real_faces/0_row1_real.png \
          $PROJ/data/real_faces/0_row2_real.png $PROJ/data/real_faces/0_row3_real.png \
          $PROJ/data/real_faces/1_row0_real.png $PROJ/data/real_faces/1_row1_real.png )
mapfile -t SCUT < <(ls $PROJ/data/pool_scut_curated/*.png)

run() {  # name gpu endpoint period lora_r epochs floor "src..."
  local NAME=$1 GPU=$2 EP=$3 K=$4 R=$5 E=$6 FLOOR=$7; shift 7; local SRC=("$@")
  local OUT=$PROJ/outputs/m3v2/${NAME}_${STAMP}; mkdir -p $OUT
  echo "$(date +%H:%M:%S) START $NAME gpu=$GPU ep=$EP K=$K r=$R ep=$E floor=$FLOOR n=${#SRC[@]} -> $OUT"
  $PY $DRIVER --proj $PROJ --py $PY --out $OUT --endpoint $EP \
      --rounds 6 --briefs 6 --rollouts 2 --train-gpu $GPU \
      --defender-period $K --lora-r $R --epochs $E --bypass-floor $FLOOR \
      --replay-per-round 8 --src-pool "${SRC[@]}" > $OUT/driver.log 2>&1
  echo "$(date +%H:%M:%S) END $NAME (rc=$?) -> $OUT/m3/coevo/armsrace_curve_v2.json"
}

run v2A_throttle_western 1 http://localhost:8002/v1 3 16 3 0.0  "${WESTERN[@]}" &
PA=$!
run v2C_full_western     3 http://localhost:8003/v1 3  8 1 0.10 "${WESTERN[@]}" &
PC=$!
run v2D_full_scut        4 http://localhost:8004/v1 3  8 1 0.10 "${SCUT[@]}" &
PD=$!
echo "launched A(pid $PA,:8002) C(pid $PC,:8003) D(pid $PD,:8004)"
wait $PA $PC $PD
echo "=== ALL v2 ABLATION RUNS DONE $(date +%H:%M:%S) ==="
