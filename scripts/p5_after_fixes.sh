#!/bin/bash
# P5: After F2 fix (force seed-library top-k in setter prompt) + F1 (n=15 traj/round) + F4 (5 rounds).
# All 4 methods on FakeVLM+LoRA. Sequential.

set -e
cd /data/disk4/lyx_ICML/self_evolution_forgery
PROJ=/data/disk4/lyx_ICML/self_evolution_forgery
PY=/cpfs01/bob_workspace/miniconda3/envs/fakevlm/bin/python
TAG="p5_$(date +%Y%m%d_%H%M)"
OUT=$PROJ/outputs/p5_after_fixes/$TAG
mkdir -p $OUT/{m1,m2,m3,m4}

SRC=(
  $PROJ/data/real_faces/0_row0_real.png
  $PROJ/data/real_faces/0_row2_real.png
  $PROJ/data/real_faces/1_row1_real.png
)

N_ROUNDS=5             # ↑ F4: 5 rounds (was 3) → see saturation curve
N_BRIEFS=5             # ↑ F1: 5 briefs/round (was 3)
N_ROLLOUTS=1           # keep at 1 → 5×1 = 5 traj/round (vs 3) and ×5 round = 25 total

run_method() {
  local LABEL=$1 MODE=$2 OUT_DIR=$3
  echo ""
  echo "═══════════════════════════════════════════════════════════════"
  echo "═══  $LABEL  (FakeVLM+LoRA, F2+F1+F4 fixes)"
  echo "═══════════════════════════════════════════════════════════════"
  cd $PROJ/src
  for R in $(seq 0 $((N_ROUNDS-1))); do
    SRC_FACE="${SRC[$((R % ${#SRC[@]}))]}"
    echo ""
    echo "── $LABEL Round $R ──"
    if [ "$LABEL" = "M4" ]; then
      $PY method4_orchestrator.py \
        --rounds 1 --briefs $N_BRIEFS --rollouts $N_ROLLOUTS \
        --tier2-backend fakevlm_local \
        --src-pool $SRC_FACE --out $OUT_DIR 2>&1 | tee -a $OUT_DIR/run.log \
        | grep -E "bypass|cluster|pareto|family|cost|SUMMARY|F2" | tail -10
    else
      $PY orchestrator.py \
        --mode $MODE --rounds 1 --briefs $N_BRIEFS --rollouts $N_ROLLOUTS \
        --multi-agent-preset w6_full --tier2-backend fakevlm_local \
        --src-pool $SRC_FACE --out $OUT_DIR 2>&1 | tee -a $OUT_DIR/run.log \
        | grep -E "bypass|family|cost|broadcast|seed_lib|4-evolution|END|F2" | tail -10
      if [ -f $OUT_DIR/reports/r0_${MODE}.json ]; then
        cp $OUT_DIR/reports/r0_${MODE}.json $OUT_DIR/reports/round_${R}_${MODE}.json
      fi
    fi
  done
}

run_method M1 v1 $OUT/m1
run_method M2 v2 $OUT/m2
run_method M3 v2 $OUT/m3
run_method M4 v_m4 $OUT/m4

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "═══ P5 AFTER-FIX 4-METHOD CURVE"
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
        db = mdir / 'data_flow_m4.db'
        if db.exists():
            c = sqlite3.connect(str(db))
            all_rows = list(c.execute('SELECT round_id, sandbox_pass FROM trajectories ORDER BY rowid'))
            c.close()
            per_round_n = max(len(all_rows) // 5, 1)
            for R in range(5):
                slice_rows = all_rows[R*per_round_n:(R+1)*per_round_n]
                if not slice_rows: continue
                bp = sum(1 for r in slice_rows if r[1])
                rounds.append({'round': R, 'n': len(slice_rows),
                               'bypass_rate': bp / max(len(slice_rows), 1)})
    else:
        for R in range(5):
            rpts = list(mdir.glob(f'reports/round_{R}_*.json'))
            if not rpts: continue
            d = json.loads(rpts[0].read_text())
            diag = d.get('diagnosis', {})
            rounds.append({'round': R, 'bypass_rate': diag.get('global_bypass_rate'),
                           'family_rates': diag.get('family_bypass_rates', {})})
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

out = base / 'p5_after_fix_curves.json'
out.write_text(json.dumps(curves, ensure_ascii=False, indent=2))
print('curves →', out)
print()
print(f'{\"method\":6s} {\"R0\":>7s} {\"R1\":>7s} {\"R2\":>7s} {\"R3\":>7s} {\"R4\":>7s} {\"Δ\":>9s} {\"pool\":>5s} {\"monotone?\":>10s}')
print('-' * 80)
for m, c in curves.items():
    rs = c['rounds']
    def fmt(i):
        if i >= len(rs): return '      ?'
        bp = rs[i].get('bypass_rate')
        return f'{bp:6.2%}' if bp is not None else '      ?'
    r0 = rs[0].get('bypass_rate') if len(rs) > 0 else None
    rn = rs[-1].get('bypass_rate') if rs else None
    delta = (rn - r0) if (r0 is not None and rn is not None) else None
    delta_s = f'{delta:+7.2%}' if delta is not None else '       ?'
    bps = [r.get('bypass_rate', 0) for r in rs]
    mono = '✅' if all(bps[i+1] >= bps[i] for i in range(len(bps)-1)) else '❌'
    print(f'{m:6s} {fmt(0)} {fmt(1)} {fmt(2)} {fmt(3)} {fmt(4)} {delta_s} {c[\"n_pool\"]:>5d} {mono:>10s}')
"
