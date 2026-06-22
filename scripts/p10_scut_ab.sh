#!/bin/bash
# P10: paper-grade A/B — clean Asian ID-photo base pool (SCUT-FBP5500) vs the faithful :8001 detector.
# Mirrors p8_m1m2m4.sh / p9_asian_ab.sh EXACTLY (same detector :8001, same briefs/rollouts, same M2=v2),
# changing ONLY the --src-pool to data/pool_scut_curated (60 balanced AF/AM clean frontal ID headshots).
# This is the paper number (SCUT academic license, visually QC'd sharp/frontal); p9 (asian_kyc, ND license)
# was the quick smoke. Compare bypass curve: Western (p8 M2) vs asian_kyc (p9) vs SCUT-Asian (here).

PROJ=/data/disk4/lyx_ICML/self_evolution_forgery
PY=/cpfs01/bob_workspace/miniconda3/envs/fakevlm/bin/python
EP=http://localhost:8001/v1
TAG="p10_scut_$(date +%Y%m%d_%H%M)"
OUT=$PROJ/outputs/p10_scut_ab/$TAG
mkdir -p "$OUT/m2"
echo "OUT=$OUT  EP=$EP  POOL=scut_curated" | tee "$OUT/META.txt"

# Clean Asian ID-photo base pool (SCUT-FBP5500 Asian, balanced 30 AF + 30 AM)
mapfile -t SRC < <(ls "$PROJ"/data/pool_scut_curated/*.png)
echo "src faces: ${#SRC[@]}" | tee -a "$OUT/META.txt"

N_BRIEFS=8
N_ROLLOUTS=2

echo "[M2-scut] starting 3-round run (mode=v2, self-evolving, SCUT-Asian pool)" | tee -a "$OUT/m2/run.log"
for R in 0 1 2; do
  cd "$PROJ/src"
  "$PY" orchestrator.py \
    --mode v2 --rounds 1 \
    --briefs $N_BRIEFS --rollouts $N_ROLLOUTS \
    --multi-agent-preset w6_full \
    --tier2-backend fakevlm_local --fakevlm-endpoint $EP \
    --src-pool "${SRC[@]}" --out "$OUT/m2" \
    >> "$OUT/m2/run.log" 2>&1
  echo "[M2-scut] round $R done (rc=$?)" | tee -a "$OUT/m2/run.log"
done
echo "[M2-scut] DONE" | tee -a "$OUT/m2/run.log"
