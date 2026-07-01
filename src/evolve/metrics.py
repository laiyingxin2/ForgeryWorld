"""Evaluation harness for the inner MAP-Elites red-team run.

Consumes the two artifacts the inner loop writes (iterations.jsonl + archive.json)
and reports the metrics that are *valid under a frozen detector*. As the project's
metric decision records (and the QD red-teaming literature) make explicit, the
instantaneous mean bypass-per-round is the WRONG curve here: the loop deliberately
shifts probability mass toward weak cells, so a mean over a non-stationary mix
mechanically declines even while the attacker is discovering new bypasses. We
instead report cumulative / coverage / best-so-far quantities, which are monotone
by construction and reward a better learner.

Metrics
-------
  archive_coverage     filled grid cells / reachable grid cells
  qd_score             sum over cells of the cell's max fitness (QD standard)
  coverage_cum[i]      # DISTINCT grid cells with >=1 bypass discovered by iter i
                       (monotone non-decreasing — the honest self-evolution curve)
  best_so_far[i]       running max single-candidate fitness up to iter i
  bypass_total         # elites the frozen detector judged real
  weak_family_asr      per forgery_family bypass rate; `weak` = families whose
                       EARLY (seed-phase) ASR had headroom (< thresh)
  per_cell_bypass      bypass count per grid cell (the diversity of the breach)
  face_type_counts     {face, low_id_face, non_face} label distribution

Usage:
    python -m evolve.metrics <RUN_DIR> [--weak-thresh 0.5] [--seed-iters 6]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def _load_iters(run_dir: Path) -> List[dict]:
    fp = run_dir / "iterations.jsonl"
    if not fp.exists():
        raise FileNotFoundError(f"no iterations.jsonl in {run_dir}")
    rows = []
    for line in fp.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    rows.sort(key=lambda r: r.get("i", 0))
    return rows


def _load_archive(run_dir: Path) -> dict:
    fp = run_dir / "archive.json"
    return json.loads(fp.read_text()) if fp.exists() else {}


def compute(run_dir: Path, weak_thresh: float = 0.5, seed_iters: int = 6) -> Dict[str, Any]:
    rows = _load_iters(run_dir)
    archive = _load_archive(run_dir)

    # ── cumulative coverage of BYPASSED cells + best-so-far fitness ──
    seen_bypass_cells = set()
    coverage_cum, best_so_far = [], []
    running_best = 0.0
    for r in rows:
        cell = tuple(r.get("cell", []))
        if r.get("bypass"):
            seen_bypass_cells.add(cell)
        coverage_cum.append(len(seen_bypass_cells))
        running_best = max(running_best, float(r.get("fitness", 0.0)))
        best_so_far.append(round(running_best, 4))

    # ── per-family bypass + weak set from the seed phase ──
    fam_pass: Dict[str, List[int]] = defaultdict(lambda: [0, 0])   # family -> [pass, n]
    fam_pass_seed: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        fam = (r.get("cell") or ["?"])[0]
        bypass = 1 if r.get("bypass") else 0
        fam_pass[fam][0] += bypass; fam_pass[fam][1] += 1
        if r.get("i", 1e9) <= seed_iters:
            fam_pass_seed[fam][0] += bypass; fam_pass_seed[fam][1] += 1

    family_asr = {f: round(p / n, 3) if n else 0.0 for f, (p, n) in fam_pass.items()}
    # weak = had headroom early (seed ASR < thresh, or never bypassed in seed phase)
    weak_families = sorted(
        f for f, (p, n) in fam_pass_seed.items() if (n == 0 or p / n < weak_thresh)
    )
    weak_pass = sum(fam_pass[f][0] for f in weak_families)
    weak_n = sum(fam_pass[f][1] for f in weak_families)
    weak_family_asr = round(weak_pass / weak_n, 3) if weak_n else 0.0

    # ── archive structural metrics ──
    cells = archive.get("cells", {})
    qd_score = 0.0
    per_cell_bypass: Dict[str, int] = {}
    for cell_key, bucket in cells.items():
        if bucket:
            qd_score += max(float(e.get("fitness", 0.0)) for e in bucket)
        per_cell_bypass[cell_key] = sum(1 for e in bucket if e.get("bypass"))

    n_iter = len(rows)
    out = {
        "schema_version": archive.get("schema_version", "1.0"),
        "run_dir": str(run_dir),
        "n_iterations": n_iter,
        "archive_cells_filled": archive.get("n_cells", len(cells)),
        "archive_elites": archive.get("n_elites", sum(len(b) for b in cells.values())),
        "qd_score": round(qd_score, 4),
        "bypass_total": archive.get("n_bypass",
                                    sum(per_cell_bypass.values())),
        "bypass_rate_overall": round(sum(1 for r in rows if r.get("bypass")) / n_iter, 3)
                               if n_iter else 0.0,
        "coverage_cum_final": coverage_cum[-1] if coverage_cum else 0,
        "best_so_far_final": best_so_far[-1] if best_so_far else 0.0,
        "weak_families": weak_families,
        "weak_family_asr": weak_family_asr,
        "family_asr": family_asr,
        "face_type_counts": archive.get("face_type_counts", {}),
        "curves": {
            "coverage_cum": coverage_cum,
            "best_so_far": best_so_far,
        },
        "per_cell_bypass": {k: v for k, v in sorted(
            per_cell_bypass.items(), key=lambda kv: kv[1], reverse=True) if v > 0},
    }
    return out


def _fmt(report: Dict[str, Any]) -> str:
    L = []
    L.append(f"run                  {report['run_dir']}")
    L.append(f"iterations           {report['n_iterations']}")
    L.append(f"archive cells filled {report['archive_cells_filled']}  "
             f"elites {report['archive_elites']}")
    L.append(f"QD-score             {report['qd_score']}")
    L.append(f"bypass total / rate  {report['bypass_total']}  "
             f"({report['bypass_rate_overall']})")
    L.append(f"coverage_cum final   {report['coverage_cum_final']} distinct bypassed cells")
    L.append(f"best_so_far final    {report['best_so_far_final']}")
    L.append(f"weak families        {report['weak_families']}")
    L.append(f"weak_family_asr      {report['weak_family_asr']}")
    L.append(f"family_asr           {report['family_asr']}")
    L.append(f"face_type_counts     {report['face_type_counts']}")
    if report["per_cell_bypass"]:
        L.append("top bypassed cells:")
        for k, v in list(report["per_cell_bypass"].items())[:10]:
            L.append(f"    {v:3d}  {k}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--weak-thresh", type=float, default=0.5)
    ap.add_argument("--seed-iters", type=int, default=6)
    ap.add_argument("--out", default=None, help="write JSON report here (default: <run>/metrics.json)")
    a = ap.parse_args()
    run_dir = Path(a.run_dir)
    report = compute(run_dir, weak_thresh=a.weak_thresh, seed_iters=a.seed_iters)
    out = Path(a.out) if a.out else run_dir / "metrics.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(_fmt(report))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
