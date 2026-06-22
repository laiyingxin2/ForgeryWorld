#!/bin/bash
# P2 self-evolution test: ≥3 round per method, compute round-over-round curve.
#
# Tests whether the 3 methods (post-Tier 1/2 patches) actually self-evolve:
#   - bypass_rate should respond to round (better attacker → higher bypass)
#   - seed_lib should grow (more chains learned)
#   - skill_doc length should grow (more constraints accumulated)
#   - family_agents experience_log should grow
#
# Does NOT include M3 LoRA retrain (use scripts/p1a_multi_round/run_multi_round.sh
# for that; this script focuses on attacker self-evolution against a fixed detector).

set -e
cd /data/disk4/lyx_ICML/self_evolution_forgery
PROJ=/data/disk4/lyx_ICML/self_evolution_forgery
PY=/cpfs01/bob_workspace/miniconda3/envs/fakevlm/bin/python
TAG="p2se_$(date +%Y%m%d_%H%M)"
OUT_BASE=$PROJ/outputs/p2_self_evolution_test/$TAG
mkdir -p $OUT_BASE

# fix src face pool (rotate per round for variety)
SRC=(
  $PROJ/data/real_faces/0_row0_real.png
  $PROJ/data/real_faces/0_row2_real.png
  $PROJ/data/real_faces/1_row1_real.png
)

run_method() {
  local METHOD=$1 MODE=$2 BACKEND=$3 N_ROUNDS=$4 BRIEFS=$5 ROLLOUTS=$6
  local METHOD_DIR=$OUT_BASE/$METHOD
  mkdir -p $METHOD_DIR
  echo ""
  echo "═══ $METHOD: $N_ROUNDS rounds × $BRIEFS briefs × $ROLLOUTS rollouts (mode=$MODE backend=$BACKEND) ═══"
  for R in $(seq 0 $((N_ROUNDS-1))); do
    SRC_FACE="${SRC[$((R % ${#SRC[@]}))]}"
    echo ""
    echo "── Round $R (src=$(basename $SRC_FACE)) ──"
    cd $PROJ/src
    $PY orchestrator.py \
      --mode $MODE \
      --rounds 1 --briefs $BRIEFS --rollouts $ROLLOUTS \
      --multi-agent-preset w6_full \
      --tier2-backend $BACKEND \
      --src-pool $SRC_FACE \
      --out $METHOD_DIR 2>&1 | tee -a $METHOD_DIR/run.log | grep -E \
      "ROUND|bypass_rate=|catch_rate|broadcast|ui_voyager|family_agents|4-evolution|seed_library|coevo|gan_upgrade|END|cost" | tail -15
    cd $PROJ
    # rename report so we keep per-round (orchestrator overwrites r0_v?.json)
    if [ -f $METHOD_DIR/reports/r0_$MODE.json ]; then
      cp $METHOD_DIR/reports/r0_$MODE.json $METHOD_DIR/reports/round_${R}_${MODE}.json
    fi
  done
}

# Method 1: v1 + all Tier 1 + Tier 2 (Markov / seed_lib / 4-evo / co-evo / VideoWeaver)
run_method m1_se v1 viviai 3 4 2 &
M1_PID=$!

# Method 2: v2 + all Tier 1 + Tier 2 (10-layer + 9-agent + 4-dim planner + co-evo)
run_method m2_se v2 viviai 3 4 2 &
M2_PID=$!

# Wait for M1 + M2 (run in parallel since both use viviai, no GPU contention)
wait $M1_PID
wait $M2_PID

# Method 3: v2 + fakevlm_local (needs vLLM exclusively)
run_method m3_se v2 fakevlm_local 3 4 2

# ── Aggregate self-evolution metrics ──
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "═══ SELF-EVOLUTION CURVE SUMMARY ═══"
echo "═══════════════════════════════════════════════════════════════"
$PY -c "
import json, sqlite3
from pathlib import Path

base = Path('$OUT_BASE')
curves = {}
for m in ('m1_se', 'm2_se', 'm3_se'):
    method_dir = base / m
    if not method_dir.exists(): continue
    rounds = []
    for r in range(3):
        # try renamed report first
        rpt_files = list(method_dir.glob(f'reports/round_{r}_*.json'))
        if not rpt_files: continue
        rpt = json.loads(rpt_files[0].read_text())
        rounds.append({
            'round': r,
            'bypass_rate': rpt.get('diagnosis', {}).get('global_bypass_rate'),
            'cost_usd': rpt.get('total_cost_usd'),
            'weak_families': rpt.get('diagnosis', {}).get('weak_families', []),
        })
    # seed library growth
    seed_db = method_dir / 'seed_library_v2/seeds.db'
    if not seed_db.exists():
        seed_db = method_dir / 'seed_library_v1/seeds.db'
    n_seeds = 0
    if seed_db.exists():
        c = sqlite3.connect(str(seed_db))
        n_seeds = c.execute('SELECT COUNT(*) FROM seed_chains').fetchone()[0]
        c.close()
    # family_agents experience growth
    fa_dir = method_dir / 'family_agents'
    fa_total_exp = 0
    if fa_dir.exists():
        for fj in fa_dir.glob('*.json'):
            try:
                d = json.loads(fj.read_text())
                fa_total_exp += int(d.get('total_attempts', 0))
            except: pass
    # videoweaver skills
    vw_db = method_dir / 'videoweaver_skills/skills.db'
    n_vw_comp = n_vw_creator = 0
    if vw_db.exists():
        c = sqlite3.connect(str(vw_db))
        n_vw_comp = c.execute('SELECT COUNT(*) FROM composition_skills').fetchone()[0]
        n_vw_creator = c.execute('SELECT COUNT(*) FROM creator_skills').fetchone()[0]
        c.close()
    curves[m] = {
        'round_curve': rounds,
        'n_seeds_total': n_seeds,
        'family_agents_total_experience': fa_total_exp,
        'videoweaver_compositions': n_vw_comp,
        'videoweaver_creators': n_vw_creator,
    }

out_path = base / 'self_evolution_curves.json'
out_path.write_text(json.dumps(curves, ensure_ascii=False, indent=2))
print('Curves written →', out_path)
print()
for m, c in curves.items():
    print(f'  {m}:')
    for r in c['round_curve']:
        rate = r['bypass_rate']
        rate_str = f'{rate:.2%}' if rate is not None else '?'
        print(f'    R{r[\"round\"]}: bypass={rate_str}  cost=\${r.get(\"cost_usd\", 0):.4f}')
    print(f'    n_seeds        = {c[\"n_seeds_total\"]}')
    print(f'    family_agents experience = {c[\"family_agents_total_experience\"]}')
    print(f'    VideoWeaver compositions = {c[\"videoweaver_compositions\"]}')
    print(f'    VideoWeaver creators     = {c[\"videoweaver_creators\"]}')
    print()
"
