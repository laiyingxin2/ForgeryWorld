#!/bin/bash
# P3: Re-run all 4 methods with the 30-op registry, compute learning curves.
#
# Each method: 3 rounds × 3 briefs × 1 rollout = 9 traj
# M1+M2+M4 use viviai detector (parallel OK, share rate limit)
# M3 uses FakeVLM_local (vLLM exclusive, sequential after others)
#
# Output: per-method learning curve + comparison table + final markdown.

set -e
cd /data/disk4/lyx_ICML/self_evolution_forgery
PROJ=/data/disk4/lyx_ICML/self_evolution_forgery
PY=/cpfs01/bob_workspace/miniconda3/envs/fakevlm/bin/python
TAG="p3_$(date +%Y%m%d_%H%M)"
OUT=$PROJ/outputs/p3_four_methods_compare/$TAG
mkdir -p $OUT/{m1,m2,m3,m4}

SRC=(
  $PROJ/data/real_faces/0_row0_real.png
  $PROJ/data/real_faces/0_row2_real.png
  $PROJ/data/real_faces/1_row1_real.png
)

# fixed total: 3 round × 3 brief × 1 rollout = 9 traj per method
N_ROUNDS=3
N_BRIEFS=3
N_ROLLOUTS=1

run_method() {
  local LABEL=$1 MODE=$2 BACKEND=$3 OUT_DIR=$4 CMD=$5
  echo ""
  echo "═══════════════════════════════════════════════════════════════"
  echo "═══  $LABEL  (mode=$MODE backend=$BACKEND)"
  echo "═══════════════════════════════════════════════════════════════"
  cd $PROJ/src
  for R in $(seq 0 $((N_ROUNDS-1))); do
    SRC_FACE="${SRC[$((R % ${#SRC[@]}))]}"
    echo ""
    echo "── $LABEL Round $R (src=$(basename $SRC_FACE)) ──"
    if [ "$LABEL" = "M4" ]; then
      $PY method4_orchestrator.py \
        --rounds 1 --briefs $N_BRIEFS --rollouts $N_ROLLOUTS \
        --tier2-backend $BACKEND \
        --src-pool $SRC_FACE \
        --out $OUT_DIR 2>&1 | tee -a $OUT_DIR/run.log | grep -E \
        "ROUND|bypass|cluster|pareto|family|cost|SUMMARY" | tail -10
    else
      $PY orchestrator.py \
        --mode $MODE --rounds 1 --briefs $N_BRIEFS --rollouts $N_ROLLOUTS \
        --multi-agent-preset w6_full --tier2-backend $BACKEND \
        --src-pool $SRC_FACE --out $OUT_DIR 2>&1 | tee -a $OUT_DIR/run.log \
        | grep -E "ROUND|bypass|family|cost|broadcast|seed_lib|4-evolution|END" | tail -10
    fi
    # save per-round report (orchestrator overwrites)
    if [ "$LABEL" != "M4" ]; then
      if [ -f $OUT_DIR/reports/r0_${MODE}.json ]; then
        cp $OUT_DIR/reports/r0_${MODE}.json $OUT_DIR/reports/round_${R}_${MODE}.json
      fi
    fi
  done
}

# Launch M1 + M2 + M4 in parallel (all viviai)
(run_method M1 v1 viviai $OUT/m1) &
M1_PID=$!
(run_method M2 v2 viviai $OUT/m2) &
M2_PID=$!
(run_method M4 v_m4 viviai $OUT/m4) &
M4_PID=$!

wait $M1_PID; wait $M2_PID; wait $M4_PID

# M3 uses FakeVLM_local; only if vLLM is up
echo ""
echo "═══ check vLLM up before M3 ═══"
if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
  run_method M3 v2 fakevlm_local $OUT/m3
else
  echo "  vLLM not up — skip M3 (run later with: bash this_script.sh m3-only)"
fi

# ─── aggregate ───
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "═══ P3 4-METHOD COMPARISON SUMMARY"
echo "═══════════════════════════════════════════════════════════════"
$PY -c "
import json, sqlite3
from pathlib import Path

base = Path('$OUT')
curves = {}
for m in ('m1', 'm2', 'm3', 'm4'):
    mdir = base / m
    if not mdir.exists(): continue
    rounds = []
    if m == 'm4':
        # Method 4 has method4_summary.json
        msum = mdir / 'method4_summary.json'
        if msum.exists():
            d = json.loads(msum.read_text())
            for r in d.get('rounds', []):
                rounds.append({
                    'round': r['round'],
                    'bypass_rate': r.get('bypass_rate'),
                    'cost': sum(rr.get('cost', 0) for rr in r.get('results', [])),
                    'n': r.get('n'),
                })
    else:
        # M1/M2/M3 have per-round reports
        for R in range(3):
            rpts = list(mdir.glob(f'reports/round_{R}_*.json'))
            if not rpts: continue
            d = json.loads(rpts[0].read_text())
            diag = d.get('diagnosis', {})
            rounds.append({
                'round': R,
                'bypass_rate': diag.get('global_bypass_rate'),
                'cost': d.get('total_cost_usd'),
                'weak_families': diag.get('weak_families', []),
            })
    # seed_lib / Pareto pool growth
    n_pool = 0
    for db_name in ('seed_library_v2/seeds.db', 'seed_library_v1/seeds.db', 'pareto.db'):
        db = mdir / db_name
        if db.exists():
            try:
                c = sqlite3.connect(str(db))
                table = 'pareto_snippets' if db_name == 'pareto.db' else 'seed_chains'
                n_pool = c.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
                c.close()
                break
            except: pass
    curves[m] = {'rounds': rounds, 'n_pool': n_pool}

out_path = base / 'four_methods_curves.json'
out_path.write_text(json.dumps(curves, ensure_ascii=False, indent=2))
print('Curves →', out_path)
print()
print(f'{\"method\":6s}  {\"R0\":>8s}  {\"R1\":>8s}  {\"R2\":>8s}  {\"Δ R2-R0\":>10s}  {\"pool\":>5s}')
print('-' * 55)
for m, c in curves.items():
    rs = c['rounds']
    r0 = rs[0]['bypass_rate'] if len(rs)>0 else None
    r1 = rs[1]['bypass_rate'] if len(rs)>1 else None
    r2 = rs[2]['bypass_rate'] if len(rs)>2 else None
    delta = (r2 - r0) if (r0 is not None and r2 is not None) else None
    fmt = lambda v: f'{v:7.2%}' if v is not None else '       ?'
    delta_s = f'{delta:+8.2%}' if delta is not None else '        ?'
    print(f'{m:6s}  {fmt(r0)}  {fmt(r1)}  {fmt(r2)}  {delta_s}  {c[\"n_pool\"]:>5d}')
"
