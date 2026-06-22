#!/bin/bash
# P1-C: 100+ traj across 3 methods (formal experiments for paper).
#
# Method 1 (v1 simple): 50 traj with gemini tier-2 detector
# Method 2 (v2 full):   50 traj with gemini + 6-LLM fan-out + 3-checker + L8
# Method 3 (v2 + Lv5):  30 traj with FakeVLM+LoRA tier-2 (defender from P1-A R2)
#
# All runs:
#   - --multi-agent-preset w6_full       (real 6-LLM)
#   - L8 memory hierarchy active         (auto, no flag)
#   - ChromaDB sem-dedupe active         (auto, no flag)
#   - 7-metric Tier-1 (real libs)        (auto, no flag)
#   - SimSwap + InSwapper local attack ops available
#
# Output: outputs/p1c_large_runs/<tag>/{m1,m2,m3}/

set -e
cd /data/disk4/lyx_ICML/self_evolution_forgery
PROJ=/data/disk4/lyx_ICML/self_evolution_forgery
PY=/cpfs01/bob_workspace/miniconda3/envs/fakevlm/bin/python
TAG="p1c_$(date +%Y%m%d_%H%M)"
OUT=$PROJ/outputs/p1c_large_runs/$TAG
mkdir -p $OUT/{m1,m2,m3}

# rotate src pool for diversity
SRC=(
  $PROJ/data/real_faces/0_row0_real.png
  $PROJ/data/real_faces/0_row2_real.png
  $PROJ/data/real_faces/1_row1_real.png
  $PROJ/data/real_faces/1_row3_real.png
)

cd $PROJ/src

run_method () {
  local TAG_M=$1 MODE=$2 BACKEND=$3 BRIEFS=$4 ROLLOUTS=$5 OUT_DIR=$6 SRC_FACE=$7
  echo ""
  echo "═══════════════════════════════════════════════════════════════"
  echo "═══  $TAG_M  mode=$MODE backend=$BACKEND  briefs=$BRIEFS rollouts=$ROLLOUTS"
  echo "═══════════════════════════════════════════════════════════════"
  $PY orchestrator.py \
    --mode $MODE \
    --rounds 1 --briefs $BRIEFS --rollouts $ROLLOUTS \
    --multi-agent-preset w6_full \
    --tier2-backend $BACKEND \
    --src-pool $SRC_FACE \
    --out $OUT_DIR 2>&1 | tee $OUT_DIR/run.log | grep -E \
    "bypass|brief|round|setter|chain=|family|cost|fakevlm|reasoning_bank|L4|L5" | tail -80
}

# ── Method 1: v1 simple baseline ──
run_method "Method 1 (v1 internal-article port)" v1 viviai 10 5 $OUT/m1 ${SRC[0]}

# ── Method 2: v2 full 10-layer ──
run_method "Method 2 (v2 paper-based 10-layer)" v2 viviai 10 5 $OUT/m2 ${SRC[1]}

# ── Method 3: v2 + FakeVLM+LoRA defender (from P1-A R2 LoRA) ──
run_method "Method 3 (v2 + FakeVLM+LoRA)" v2 fakevlm_local 6 5 $OUT/m3 ${SRC[2]}

# ── Final aggregation ──
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "═══  P1-C SUMMARY"
echo "═══════════════════════════════════════════════════════════════"
SUMMARY=$OUT/summary.json
$PY -c "
import json
from pathlib import Path
out = {}
for m in ('m1', 'm2', 'm3'):
    rpts = list(Path('$OUT/' + m).glob('reports/r*.json'))
    if not rpts: continue
    r = json.loads(rpts[0].read_text())
    diag = r.get('diagnosis', {})
    out[m] = {
        'global_bypass_rate': diag.get('global_bypass_rate'),
        'family_bypass_rates': diag.get('family_bypass_rates'),
        'total_cost_usd': r.get('total_cost_usd'),
        'baseline': r.get('baseline'),
    }
Path('$SUMMARY').write_text(json.dumps(out, ensure_ascii=False, indent=2))
print(json.dumps(out, ensure_ascii=False, indent=2))
"
echo ""
echo "  Summary written: $SUMMARY"
