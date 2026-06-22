#!/bin/bash
# P9: A/B test — does swapping the base face pool to ASIAN faces re-open the bypass gap?
# Mirrors p8_m1m2m4.sh EXACTLY (same detector :8001, same briefs/rollouts, same M2=v2),
# changing ONLY the --src-pool (Western CelebA-Spoof -> asian-kyc crops).
# Compare bypass curve here vs p8_faithful M2 (Western) on the same :8001 detector.

PROJ=/data/disk4/lyx_ICML/self_evolution_forgery
PY=/cpfs01/bob_workspace/miniconda3/envs/fakevlm/bin/python
EP=http://localhost:8001/v1
TAG="p9_asian_$(date +%Y%m%d_%H%M)"
OUT=$PROJ/outputs/p9_asian_ab/$TAG
mkdir -p "$OUT/m2"
echo "OUT=$OUT  EP=$EP  POOL=asian_kyc" | tee "$OUT/META.txt"

# Asian base pool (frontal crops from UniqueData/asian-kyc, East/South Asian)
mapfile -t SRC < <(ls "$PROJ"/data/pool_asian_kyc/*.png)
echo "src faces: ${#SRC[@]}" | tee -a "$OUT/META.txt"

N_BRIEFS=8
N_ROLLOUTS=2

echo "[M2-asian] starting 3-round run (mode=v2, self-evolving, ASIAN pool)" | tee -a "$OUT/m2/run.log"
for R in 0 1 2; do
  cd "$PROJ/src"
  "$PY" orchestrator.py \
    --mode v2 --rounds 1 \
    --briefs $N_BRIEFS --rollouts $N_ROLLOUTS \
    --multi-agent-preset w6_full \
    --tier2-backend fakevlm_local --fakevlm-endpoint $EP \
    --src-pool "${SRC[@]}" --out "$OUT/m2" \
    >> "$OUT/m2/run.log" 2>&1
  echo "[M2-asian] round $R done (rc=$?)" | tee -a "$OUT/m2/run.log"
done
echo "[M2-asian] DONE" | tee -a "$OUT/m2/run.log"
