#!/bin/bash
# P4: Fair 4-method comparison — ALL methods use FakeVLM+LoRA detector
# (vs P3 where M1/M2/M4 used gemini and only M3 used FakeVLM,导致不可比).
#
# 3 rounds × 3 briefs × 1 rollout = 9 traj per method
# M3 is special (uses defender LoRA + co-evo on same), others use plain FakeVLM
# All 4 must go SEQUENTIAL because they all hit the same vLLM endpoint.

set -e
cd /data/disk4/lyx_ICML/self_evolution_forgery
PROJ=/data/disk4/lyx_ICML/self_evolution_forgery
PY=/cpfs01/bob_workspace/miniconda3/envs/fakevlm/bin/python
TAG="p4_$(date +%Y%m%d_%H%M)"
OUT=$PROJ/outputs/p4_fair_compare/$TAG
mkdir -p $OUT/{m1,m2,m3,m4}

SRC=(
  $PROJ/data/real_faces/0_row0_real.png
  $PROJ/data/real_faces/0_row2_real.png
  $PROJ/data/real_faces/1_row1_real.png
)

N_ROUNDS=3
N_BRIEFS=3
N_ROLLOUTS=1

run_method() {
  local LABEL=$1 MODE=$2 OUT_DIR=$3
  echo ""
  echo "═══════════════════════════════════════════════════════════════"
  echo "═══  $LABEL  (mode=$MODE, ALL use FakeVLM+LoRA)"
  echo "═══════════════════════════════════════════════════════════════"
  cd $PROJ/src
  for R in $(seq 0 $((N_ROUNDS-1))); do
    SRC_FACE="${SRC[$((R % ${#SRC[@]}))]}"
    echo ""
    echo "── $LABEL Round $R (src=$(basename $SRC_FACE)) ──"
    if [ "$LABEL" = "M4" ]; then
      $PY method4_orchestrator.py \
        --rounds 1 --briefs $N_BRIEFS --rollouts $N_ROLLOUTS \
        --tier2-backend fakevlm_local \
        --src-pool $SRC_FACE --out $OUT_DIR 2>&1 | tee -a $OUT_DIR/run.log \
        | grep -E "ROUND|bypass|cluster|pareto|family|cost|SUMMARY" | tail -10
    else
      $PY orchestrator.py \
        --mode $MODE --rounds 1 --briefs $N_BRIEFS --rollouts $N_ROLLOUTS \
        --multi-agent-preset w6_full \
        --tier2-backend fakevlm_local \
        --src-pool $SRC_FACE --out $OUT_DIR 2>&1 | tee -a $OUT_DIR/run.log \
        | grep -E "ROUND|bypass|family|cost|broadcast|seed_lib|4-evolution|END" | tail -10
      if [ -f $OUT_DIR/reports/r0_${MODE}.json ]; then
        cp $OUT_DIR/reports/r0_${MODE}.json $OUT_DIR/reports/round_${R}_${MODE}.json
      fi
    fi
  done
}

# Sequential since all 4 share single vLLM
run_method M1 v1 $OUT/m1
run_method M2 v2 $OUT/m2
run_method M3 v2 $OUT/m3   # M3 = v2 + fakevlm (Lv5 自进化)
run_method M4 v_m4 $OUT/m4

# Final aggregation
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "═══ P4 FAIR 4-METHOD COMPARISON (all FakeVLM+LoRA detector)"
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
        # M4 stores cumulative in data_flow + summary overwrites; read DB
        db = mdir / 'data_flow_m4.db'
        if db.exists():
            c = sqlite3.connect(str(db))
            # group by round (which is always 0 in M4) — use insertion order via rowid
            all_rows = list(c.execute('SELECT round_id, sandbox_pass FROM trajectories ORDER BY rowid'))
            c.close()
            n_per_round = max(len(all_rows) // 3, 1)
            for R in range(3):
                slice_rows = all_rows[R*n_per_round:(R+1)*n_per_round]
                if not slice_rows: continue
                bp = sum(1 for r in slice_rows if r[1])
                rounds.append({
                    'round': R, 'n': len(slice_rows),
                    'bypass_rate': bp / max(len(slice_rows), 1),
                })
    else:
        for R in range(3):
            rpts = list(mdir.glob(f'reports/round_{R}_*.json'))
            if not rpts: continue
            d = json.loads(rpts[0].read_text())
            diag = d.get('diagnosis', {})
            rounds.append({
                'round': R,
                'bypass_rate': diag.get('global_bypass_rate'),
                'cost': d.get('total_cost_usd'),
                'family_bypass_rates': diag.get('family_bypass_rates', {}),
                'weak_families': diag.get('weak_families', []),
            })
    # pool size
    n_pool = 0
    for db_name in ('seed_library_v2/seeds.db', 'seed_library_v1/seeds.db', 'pareto.db'):
        db = mdir / db_name
        if db.exists():
            try:
                c = sqlite3.connect(str(db))
                table = 'pareto_snippets' if 'pareto' in db_name else 'seed_chains'
                n_pool = c.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
                c.close(); break
            except: pass
    curves[m] = {'rounds': rounds, 'n_pool': n_pool}

out = base / 'p4_fair_curves.json'
out.write_text(json.dumps(curves, ensure_ascii=False, indent=2))
print('curves →', out)
print()
print(f'{\"method\":6s}  {\"R0\":>8s}  {\"R1\":>8s}  {\"R2\":>8s}  {\"Δ\":>10s}  {\"pool\":>5s}')
print('-' * 60)
for m, c in curves.items():
    rs = c['rounds']
    def fmt(i):
        if i >= len(rs): return '       ?'
        bp = rs[i].get('bypass_rate')
        return f'{bp:7.2%}' if bp is not None else '       ?'
    r0 = rs[0].get('bypass_rate') if len(rs) > 0 else None
    r2 = rs[2].get('bypass_rate') if len(rs) > 2 else None
    delta = (r2 - r0) if (r0 is not None and r2 is not None) else None
    delta_s = f'{delta:+8.2%}' if delta is not None else '        ?'
    print(f'{m:6s}  {fmt(0)}  {fmt(1)}  {fmt(2)}  {delta_s}  {c[\"n_pool\"]:>5d}')
"
