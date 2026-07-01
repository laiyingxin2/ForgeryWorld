"""M2-P0-4: 6 evaluation metrics from internal article (图9 verbatim).

内部文章 §2.2.3 评测指标体系 列出 6 个指标:
  1. ASR (Attack Success Rate)              — bypass / total
  2. 覆盖率 (Coverage)                       — distinct families exercised / 9
  3. DSR (Defense Success Rate)             — 1 - ASR (detector's view)
  4. FPR (False Positive Rate)              — real flagged as fake / total real
  5. 进化率 (Evolution Rate)                 — new high-score chains per round / total
  6. N轮提升率 (N-round Improvement Rate)    — bypass_rate[R_n] - bypass_rate[R_0]
  7. 闭环有效性 (Loop Effectiveness)         — fraction of bypass cases that produced SFT data

之前 analyze_runs.py 只输出 bypass_rate; paper 需要 6 个全报.

Input: per-round trajectory list + (optional) FF++ benchmark predictions
Output: dict with 6 metrics.
"""
from __future__ import annotations
import json
import sqlite3
from pathlib import Path
from typing import Optional
from collections import defaultdict


# 9 families from DESIGN_V3
ALL_FAMILIES = [
    "frontal_swap", "profile_swap", "id_diff", "reenact", "morph",
    "3d_mask", "replay", "adv_patch", "audio_synth",
]


def compute_six_metrics(
    db_path: str,
    benchmark_report_path: Optional[str] = None,
    seed_library_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> dict:
    """Compute the 6 metrics from a run's outputs.

    Args:
        db_path: outputs/<run>/data_flow_v2.db (the trajectory SQLite)
        benchmark_report_path: outputs/p2_bench/ff++_n30.json (optional, gives FPR)
        seed_library_path: outputs/<run>/seed_library_v[12]/seeds.db (gives 进化率)
        output_dir: if set, write metrics.json there

    Returns:
        dict with 7 entries (6 metrics + meta).
    """
    if not Path(db_path).exists():
        return {"error": f"db not found: {db_path}"}

    # ── 1. ASR + DSR per round + per family ──
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT round_id, attack_family, sandbox_pass, data_route FROM trajectories"
    ).fetchall()
    n = len(rows)
    if n == 0:
        return {"error": "empty db"}

    per_round = defaultdict(lambda: {"n": 0, "bypass": 0})
    per_family = defaultdict(lambda: {"n": 0, "bypass": 0})
    n_route_sft = 0
    n_bypass = 0
    for round_id, fam, bp, route in rows:
        per_round[round_id]["n"] += 1
        per_round[round_id]["bypass"] += int(bp or 0)
        per_family[fam]["n"] += 1
        per_family[fam]["bypass"] += int(bp or 0)
        if bp: n_bypass += 1
        if route == "SFT": n_route_sft += 1
    asr = n_bypass / n
    dsr = 1.0 - asr

    # ── 2. 覆盖率 ──
    covered_families = set(fam for fam, st in per_family.items() if st["n"] > 0)
    coverage = len(covered_families) / len(ALL_FAMILIES)

    # ── 3. FPR (from FF++ benchmark if available) ──
    fpr = None
    fpr_source = "missing"
    if benchmark_report_path and Path(benchmark_report_path).exists():
        try:
            bench = json.loads(Path(benchmark_report_path).read_text())
            metrics = bench.get("metrics", {})
            # use defender_lora's specificity (TN / (TN+FP)) → FPR = 1 - specificity
            det = metrics.get("defender_lora") or metrics.get("fakevlm_base") or {}
            spec = det.get("specificity (real recall)")
            if spec is not None:
                fpr = round(1.0 - float(spec), 4)
                fpr_source = "defender_lora" if "defender_lora" in metrics else "fakevlm_base"
        except Exception:
            pass

    # ── 4. 进化率 ──
    evolution_rate = None
    n_seeds = None
    if seed_library_path and Path(seed_library_path).exists():
        try:
            sconn = sqlite3.connect(seed_library_path)
            n_seeds = sconn.execute("SELECT COUNT(*) FROM seed_chains WHERE status='active'").fetchone()[0]
            n_high = sconn.execute(
                "SELECT COUNT(*) FROM seed_chains WHERE weighted_score > 0.65 AND status='active'"
            ).fetchone()[0]
            sconn.close()
            evolution_rate = round(n_high / max(n_seeds, 1), 4) if n_seeds else 0.0
        except Exception:
            pass

    # ── 5. N 轮提升率 ──
    sorted_rounds = sorted(per_round.keys())
    n_round_improvement = None
    round_curve = []
    if len(sorted_rounds) >= 2:
        r0 = sorted_rounds[0]; rn = sorted_rounds[-1]
        rate_r0 = per_round[r0]["bypass"] / max(per_round[r0]["n"], 1)
        rate_rn = per_round[rn]["bypass"] / max(per_round[rn]["n"], 1)
        n_round_improvement = round(rate_rn - rate_r0, 4)
    for r in sorted_rounds:
        round_curve.append({
            "round": r, "n": per_round[r]["n"],
            "bypass_rate": round(per_round[r]["bypass"] / max(per_round[r]["n"], 1), 4),
        })

    # ── 6. 闭环有效性 ──
    loop_effectiveness = round(n_route_sft / max(n_bypass, 1), 4) if n_bypass else 0.0

    metrics_out = {
        "n_trajectories":         n,
        "ASR_overall":            round(asr, 4),
        "DSR_overall":            round(dsr, 4),
        "coverage_families":      round(coverage, 4),
        "covered_family_count":   f"{len(covered_families)}/{len(ALL_FAMILIES)}",
        "FPR":                    fpr,
        "FPR_source":             fpr_source,
        "evolution_rate":         evolution_rate,
        "seed_lib_size":          n_seeds,
        "N_round_improvement":    n_round_improvement,
        "round_curve":            round_curve,
        "loop_effectiveness":     loop_effectiveness,
        "per_family":             {f: {"n": s["n"], "bypass": s["bypass"],
                                        "rate": round(s["bypass"] / max(s["n"], 1), 4)}
                                    for f, s in per_family.items()},
    }
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(output_dir) / "six_metrics.json"
        out_path.write_text(json.dumps(metrics_out, ensure_ascii=False, indent=2))
        metrics_out["_written_to"] = str(out_path)
    return metrics_out


def pretty_print(m: dict, label: str = "") -> None:
    print(f"\n══════ 6-metric report {label} ══════")
    print(f"  n_traj          : {m.get('n_trajectories', '?')}")
    print(f"  ASR (黑召回)     : {m.get('ASR_overall', '?')}")
    print(f"  DSR (防御成功)   : {m.get('DSR_overall', '?')}")
    print(f"  覆盖率 family    : {m.get('coverage_families', '?')} ({m.get('covered_family_count', '?')})")
    print(f"  FPR              : {m.get('FPR', 'missing')} [{m.get('FPR_source', '?')}]")
    print(f"  进化率           : {m.get('evolution_rate', '?')} (seed_lib={m.get('seed_lib_size', '?')})")
    print(f"  N 轮提升率       : {m.get('N_round_improvement', '?')}")
    print(f"  闭环有效性       : {m.get('loop_effectiveness', '?')}")
    print(f"  round_curve      : {m.get('round_curve', [])}")
    print(f"  per_family       :")
    for fam, s in (m.get("per_family") or {}).items():
        print(f"    {fam:14s}: bypass {s['bypass']}/{s['n']} = {s['rate']:.2%}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--bench", default=None)
    ap.add_argument("--seed", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    m = compute_six_metrics(args.db, args.bench, args.seed, args.out)
    pretty_print(m, args.label)
