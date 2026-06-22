#!/usr/bin/env python
"""Aggregate P6 (working-detector) 4-method self-evolution curves.

Usage: python p6_aggregate.py <OUT_DIR>
Reads m1/m2/m3 round reports + m4 data_flow DB, prints R0-R4 bypass + monotone.
"""
import json, sqlite3, sys
from pathlib import Path

base = Path(sys.argv[1])
curves = {}
for m in ('m1', 'm2', 'm3', 'm4'):
    mdir = base / m
    if not mdir.exists():
        continue
    rounds = []
    if m == 'm4':
        db = mdir / 'data_flow_m4.db'
        if db.exists():
            c = sqlite3.connect(str(db))
            all_rows = list(c.execute(
                'SELECT round_id, sandbox_pass FROM trajectories ORDER BY rowid'))
            c.close()
            per = max(len(all_rows) // 5, 1)
            for R in range(5):
                sl = all_rows[R * per:(R + 1) * per]
                if not sl:
                    continue
                bp = sum(1 for r in sl if r[1])
                rounds.append({'round': R, 'n': len(sl),
                               'bypass_rate': bp / max(len(sl), 1)})
    else:
        for R in range(5):
            rpts = list(mdir.glob(f'reports/round_{R}_*.json'))
            if not rpts:
                continue
            d = json.loads(rpts[0].read_text())
            diag = d.get('diagnosis', {})
            rounds.append({'round': R,
                           'bypass_rate': diag.get('global_bypass_rate'),
                           'family_rates': diag.get('family_bypass_rates', {})})
    n_pool = 0
    for db_name in ('seed_library_v2/seeds.db', 'seed_library_v1/seeds.db', 'pareto.db'):
        db = mdir / db_name
        if db.exists():
            try:
                c = sqlite3.connect(str(db))
                table = 'pareto_snippets' if 'pareto' in db_name else 'seed_chains'
                n_pool = c.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
                c.close()
                break
            except Exception:
                pass
    curves[m] = {'rounds': rounds, 'n_pool': n_pool}

out = base / 'p6_curves.json'
out.write_text(json.dumps(curves, ensure_ascii=False, indent=2))
print('curves ->', out)
print()
hdr = f'{"method":6s} {"R0":>7s} {"R1":>7s} {"R2":>7s} {"R3":>7s} {"R4":>7s} {"delta":>9s} {"pool":>5s} {"monotone":>9s}'
print(hdr)
print('-' * 78)
for m, c in curves.items():
    rs = c['rounds']
    def fmt(i):
        if i >= len(rs):
            return '      ?'
        bp = rs[i].get('bypass_rate')
        return f'{bp:6.1%}' if bp is not None else '      ?'
    r0 = rs[0].get('bypass_rate') if rs else None
    rn = rs[-1].get('bypass_rate') if rs else None
    delta = (rn - r0) if (r0 is not None and rn is not None) else None
    delta_s = f'{delta:+7.1%}' if delta is not None else '       ?'
    bps = [r.get('bypass_rate', 0) or 0 for r in rs]
    mono = 'YES' if all(bps[i + 1] >= bps[i] for i in range(len(bps) - 1)) else 'no'
    print(f'{m:6s} {fmt(0)} {fmt(1)} {fmt(2)} {fmt(3)} {fmt(4)} {delta_s} {c["n_pool"]:>5d} {mono:>9s}')
