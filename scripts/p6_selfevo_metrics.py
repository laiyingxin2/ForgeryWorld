#!/usr/bin/env python
"""Self-evolution metrics that are valid under a FROZEN detector.

Why this file exists
--------------------
The naive metric (instantaneous mean bypass per round) is the WRONG curve for
this setup, for two reasons the red-teaming literature is explicit about:

  1. Non-stationary eval: the run scripts rotate the source face per round
     (SRC[R % 3]) so each round's bypass is measured on a different input ->
     the curve mixes "did it learn" with "is this round's face easier".
  2. Frozen detector + adaptive family reweighting: some families saturate at
     100% (audio_synth, reenact), some are floored at 0% (profile_swap robustly
     caught), and the diagnosis deliberately shifts mass toward WEAK families ->
     instantaneous mean bypass MECHANICALLY declines even while the agent learns.

Standard practice for self-evolving / quality-diversity red-teaming
(OpenAI "Diverse and Effective Red Teaming" 2412.18693; "Quality-Diversity
Red-Teaming" 2506.07121; MAP-Elites archive coverage; iterative-ASR pass@k) is
to report CUMULATIVE / COVERAGE / BEST-SO-FAR metrics on a stationary basis:

  * coverage_cum   : cumulative # of DISTINCT successful attack signatures
                     (attack_family, ran-tool-chain) discovered up to round R.
                     Monotone non-decreasing by construction -> the honest
                     "self-evolution" curve. A better learner discovers more.
  * best_so_far    : running max of per-round bypass (RL/evo is noisy; report
                     the frontier, not the instantaneous mean).
  * weak_family_asr: ASR on families that started weak (had headroom). Measures
                     whether the agent improved WHERE improvement was possible.
  * inst_bypass    : the old instantaneous mean (kept for context only).

Usage: python p6_selfevo_metrics.py <OUT_DIR>
"""
import json, re, sqlite3, sys
from pathlib import Path

DBS = {
    'm1': 'data_flow_v1.db',
    'm2': 'data_flow_v2.db',
    'm3': 'data_flow_v2.db',
    'm4': 'data_flow_m4.db',
}
FALLBACK_WEAK = {'profile_swap', '3d_mask'}  # used only if R0 pooling yields nothing


def compute_weak_set(all_rows, thresh=0.5):
    """Data-driven 'weak/hard' family set = families whose POOLED R0 ASR is below
    `thresh` (i.e. had headroom under the frozen detector). Pooling across ALL
    methods at R0 gives ONE shared reference set so weak_family_asr is comparable
    across methods (a fixed target, not each method's lucky family draw).

    Why this replaces the old hardcoded WEAK={profile_swap,3d_mask}: with briefs<9
    the adaptive family selector samples a different subset each round, so a fixed
    literal set is undefined for methods that never drew it (M2 drew neither in p7).
    Refs: 'Comparison requires valid measurement' (OpenReview d7hqAhLvWG) — ASR
    comparisons must control the per-category mix; per-category ASR is the fix.
    """
    fam = {}  # family -> [pass, n] at R0
    for method_rows in all_rows.values():
        if not method_rows:
            continue
        r0 = min(r[0] for r in method_rows)
        for R, f, sp, _ in method_rows:
            if R == r0:
                fam.setdefault(f, [0, 0])
                fam[f][0] += 1 if sp else 0
                fam[f][1] += 1
    weak = {f for f, (p, n) in fam.items() if n and p / n < thresh}
    return weak or FALLBACK_WEAK


def sig(family, full_json):
    """Distinct-attack signature: (family, ordered tools that actually ran)."""
    try:
        d = json.loads(full_json)
    except Exception:
        return (family, ())
    ex = d.get('execution', [])
    if isinstance(ex, dict):
        ex = ex.get('steps', [])
    tools = tuple(s.get('tool') for s in (ex or [])
                  if isinstance(s, dict) and s.get('error') is None)
    return (family, tools)


def load(mdir, db_name):
    """Return rows as (round_idx, family, sandbox_pass, full_json).

    NOTE: the DB stores round_id=0 for every trajectory because each round is a
    separate `--rounds 1` process invocation. The true round boundary is the
    run_id prefix of trajectory_id (run-TIMESTAMP_r0_f.._b.._g..). We segment by
    run_id ordered by timestamp to recover rounds 0..N.
    """
    db = mdir / db_name
    if not db.exists():
        return []
    c = sqlite3.connect(str(db))
    try:
        raw = list(c.execute(
            "SELECT trajectory_id, attack_family, sandbox_pass, full_json, timestamp "
            "FROM trajectories"))
    finally:
        c.close()
    # run_id -> min timestamp, to order rounds
    run_ts = {}
    for tid, fam, sp, fj, ts in raw:
        runid = re.split(r'_r\d+_', tid)[0]
        run_ts[runid] = min(run_ts.get(runid, ts), ts)
    order = {rid: i for i, rid in enumerate(sorted(run_ts, key=run_ts.get))}
    rows = []
    for tid, fam, sp, fj, ts in raw:
        runid = re.split(r'_r\d+_', tid)[0]
        rows.append((order[runid], fam, sp, fj))
    rows.sort(key=lambda r: r[0])
    return rows


def per_method(rows, weak_set):
    """Self-evolution curves that are valid under family-draw confound.

    Adds (vs the old version):
      * coverage_norm : coverage_cum / cumulative attempts (QDRT 2506.07121 —
                        normalize diversity by total probing attempts so a method
                        that simply ran more rollouts doesn't 'win' on coverage).
      * weak_cum_asr  : CUMULATIVE ASR on the shared data-driven weak_set, pooled
                        over rounds 0..R (stratified; measures improvement WHERE
                        there was headroom, on a fixed target set).
      * family_cum_asr: per-family cumulative (pass/attempts) up to round R — the
                        per-category ASR the red-team-measurement literature asks
                        for. This is the confound-free way to compare methods.
    """
    rounds = sorted({r[0] for r in rows})
    seen = set()                      # cumulative distinct successful signatures
    cov_cum, cov_norm, inst, best = [], [], [], []
    weak_inst, weak_cum = [], []
    attempts_cum = []
    best_v = 0.0
    n_seen_attempts = 0
    weak_p = weak_n = 0               # cumulative pass/attempts on weak_set
    fam_pn = {}                       # family -> [cum_pass, cum_n]
    fam_cum_asr = {}                  # family -> list over rounds (cum ASR or None)
    for R in rounds:
        rr = [r for r in rows if r[0] == R]
        n = len(rr)
        n_seen_attempts += n
        attempts_cum.append(n_seen_attempts)
        npass = sum(1 for r in rr if r[2])
        inst_v = npass / n if n else 0.0
        inst.append(inst_v)
        best_v = max(best_v, inst_v)
        best.append(best_v)
        for r in rr:
            if r[2]:
                seen.add(sig(r[1], r[3]))
            fam_pn.setdefault(r[1], [0, 0])
            fam_pn[r[1]][0] += 1 if r[2] else 0
            fam_pn[r[1]][1] += 1
        cov_cum.append(len(seen))
        cov_norm.append(len(seen) / n_seen_attempts if n_seen_attempts else 0.0)
        # weak-set ASR: instantaneous (this round) and cumulative (pooled 0..R)
        wk = [r for r in rr if r[1] in weak_set]
        weak_inst.append((sum(1 for r in wk if r[2]) / len(wk)) if wk else None)
        weak_p += sum(1 for r in wk if r[2])
        weak_n += len(wk)
        weak_cum.append((weak_p / weak_n) if weak_n else None)
        # per-family cumulative ASR snapshot at this round
        for f, (p, nn) in fam_pn.items():
            fam_cum_asr.setdefault(f, [None] * len(rounds))
        for f in fam_cum_asr:
            if f in fam_pn and fam_pn[f][1]:
                fam_cum_asr[f][rounds.index(R)] = fam_pn[f][0] / fam_pn[f][1]
    return {'rounds': rounds, 'inst': inst, 'best': best,
            'coverage_cum': cov_cum, 'coverage_norm': cov_norm,
            'attempts_cum': attempts_cum,
            'weak_inst_asr': weak_inst, 'weak_cum_asr': weak_cum,
            'family_cum_asr': fam_cum_asr,
            'total_unique': len(seen), 'total_attempts': n_seen_attempts}


def fmt_row(label, vals, pct=False):
    cells = []
    for v in vals:
        if v is None:
            cells.append('   -')
        elif pct:
            cells.append(f'{v:5.0%}')
        else:
            cells.append(f'{v:5.0f}')
    return f'  {label:14s} ' + ' '.join(cells)


def main():
    base = Path(sys.argv[1])
    # Pass 1: load every method's rows.
    raw = {}
    for m, db in DBS.items():
        rows = load(base / m, db)
        if rows:
            raw[m] = rows
    # Derive ONE shared, data-driven weak/hard family set from pooled R0.
    weak_set = compute_weak_set(raw)
    # Pass 2: per-method curves against the shared weak set.
    out = {m: per_method(rows, weak_set) for m, rows in raw.items()}

    print(f'\nSelf-evolution metrics (frozen detector) — {base.name}')
    print(f'shared weak/hard family set (pooled R0 ASR<0.5): {sorted(weak_set)}\n')
    for m, c in out.items():
        rs = c['rounds']
        hdr = '  ' + ' ' * 16 + ' '.join(f'  R{r}' for r in rs)
        print(f'[{m}]  unique_attacks={c["total_unique"]}  attempts={c["total_attempts"]}')
        print(hdr)
        print(fmt_row('coverage_cum', c['coverage_cum']))
        print(fmt_row('coverage_norm', c['coverage_norm'], pct=True))
        print(fmt_row('best_so_far', c['best'], pct=True))
        print(fmt_row('weak_cum_asr', c['weak_cum_asr'], pct=True))
        print(fmt_row('inst_bypass*', c['inst'], pct=True))  # *context only
        cov = c['coverage_cum']
        mono = all(cov[i + 1] >= cov[i] for i in range(len(cov) - 1))
        d_cov = cov[-1] - cov[0] if cov else 0
        print(f'  -> coverage monotone={mono}  Δcoverage(R0->Rn)=+{d_cov}\n')

    # Stratified, confound-free cross-method comparison: per-family CUMULATIVE
    # final ASR. Compare methods only WITHIN a family (controls the family draw).
    fams = sorted({f for c in out.values() for f in c['family_cum_asr']})
    methods = list(out)
    print('per-family cumulative ASR @ final round (stratified; pass/attempts):')
    print('  ' + 'family'.ljust(14) + ''.join(m.upper().rjust(8) for m in methods))
    for f in fams:
        cells = ''
        for m in methods:
            seq = out[m]['family_cum_asr'].get(f)
            v = next((x for x in reversed(seq) if x is not None), None) if seq else None
            cells += ('   -   ' if v is None else f'{v:6.0%} ').rjust(8)
        print('  ' + f.ljust(14) + cells)

    (base / 'p6_selfevo_metrics.json').write_text(
        json.dumps({'weak_set': sorted(weak_set), 'methods': out},
                   ensure_ascii=False, indent=2))
    print('\nsaved ->', base / 'p6_selfevo_metrics.json')


if __name__ == '__main__':
    main()
