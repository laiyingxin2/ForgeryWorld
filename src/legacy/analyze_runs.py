"""三方法运行结果对比分析工具.

用法:
    python analyze_runs.py
    python analyze_runs.py --m1 outputs_m1 --m2 outputs --output /tmp/compare_report.md
"""
from __future__ import annotations
import argparse
import json
import sqlite3
from pathlib import Path
from collections import defaultdict


def analyze_db(db_path: str, label: str) -> dict:
    """Compute per-method stats from a trajectory DB.

    ★ BUG-17: 区分 real bypass vs pseudo bypass (final = src, no real op succeeded).
    """
    if not Path(db_path).exists():
        return {"label": label, "error": f"DB not found: {db_path}"}

    conn = sqlite3.connect(db_path)
    out = {"label": label, "db_path": db_path}

    # ★ Real vs pseudo bypass count
    rows = conn.execute("SELECT full_json FROM trajectories").fetchall()
    n = len(rows)
    raw_bp = real_bp = pseudo_bp = 0
    cost = 0.0
    for (fj,) in rows:
        t = json.loads(fj)
        v = t.get('verdicts') or {}
        b = t.get('brief') or {}
        exec_steps = t.get('execution', [])
        src = b.get('src_face_path', '')
        cost += t.get('cost_usd', 0) or 0
        if v.get('sandbox_pass', False):
            raw_bp += 1
            if exec_steps:
                final = exec_steps[-1].get('output_path', '')
                is_pseudo = (final == src)
                real_op = any(
                    (s.get('output_path', '') != src) and s.get('output_path', '')
                    and not ('All models failed' in (s.get('error', '') or '')
                             or 'MOCK_UNAVAILABLE' in (s.get('error', '') or ''))
                    for s in exec_steps
                )
                if is_pseudo or not real_op:
                    pseudo_bp += 1
                else:
                    real_bp += 1
    out["n_trajectory"] = n
    out["n_bypass_raw"] = raw_bp
    out["n_bypass_pseudo"] = pseudo_bp
    out["n_bypass_real"] = real_bp
    out["bypass_rate_raw"] = raw_bp / max(n, 1)
    out["bypass_rate_real"] = real_bp / max(n, 1)
    out["bypass_rate"] = out["bypass_rate_real"]  # for backward compat use REAL
    out["n_bypass"] = real_bp  # for backward compat use REAL
    out["total_cost"] = cost

    # Per family
    cur = conn.execute(
        "SELECT attack_family, COUNT(*), SUM(sandbox_pass) "
        "FROM trajectories GROUP BY attack_family"
    )
    out["per_family"] = {}
    for fam, cnt, byp in cur.fetchall():
        byp = byp or 0
        out["per_family"][fam] = {
            "n": cnt, "bypass": byp, "rate": byp / max(cnt, 1)
        }

    # Per run_id (跨 run 演化)
    cur = conn.execute(
        "SELECT SUBSTR(trajectory_id, 1, 24), COUNT(*), SUM(sandbox_pass) "
        "FROM trajectories GROUP BY SUBSTR(trajectory_id, 1, 24) "
        "ORDER BY MIN(timestamp)"
    )
    out["per_run"] = []
    for rid, cnt, byp in cur.fetchall():
        byp = byp or 0
        out["per_run"].append({
            "run_id": rid, "n": cnt, "bypass": byp, "rate": byp / max(cnt, 1)
        })

    # Routes (SFT vs CT vs DROP)
    cur = conn.execute(
        "SELECT data_route, COUNT(*) FROM trajectories GROUP BY data_route"
    )
    out["routes"] = dict(cur.fetchall())

    conn.close()
    return out


def analyze_skill_lib(skill_dir: str) -> dict:
    """Count ℰ_k experiences + 𝒮_k word count per family."""
    p = Path(skill_dir)
    if not p.exists():
        return {}
    out = {}
    for fam_dir in p.iterdir():
        if not fam_dir.is_dir():
            continue
        exp_jsonl = fam_dir / "experience.jsonl"
        skill_md = fam_dir / "SKILL.md"
        n_exp = 0
        if exp_jsonl.exists():
            n_exp = sum(1 for _ in open(exp_jsonl))
        n_words = 0
        if skill_md.exists():
            n_words = len(skill_md.read_text().split())
        if n_exp > 0 or n_words > 0:
            out[fam_dir.name] = {"ℰ_k_n": n_exp, "𝒮_k_words": n_words}
    return out


def analyze_reasoning_bank(rb_dir: str) -> dict:
    """Count rules + success/failure split per family."""
    p = Path(rb_dir)
    if not p.exists():
        return {}
    out = {}
    for f in p.glob("*.jsonl"):
        rules = [json.loads(l) for l in open(f)]
        if not rules:
            continue
        succ = sum(1 for r in rules if r.get("success_label"))
        out[f.stem] = {
            "n_rules": len(rules), "n_success": succ,
            "avg_utility": sum(r.get("utility", 1.0) for r in rules) / len(rules),
        }
    return out


def analyze_novelty(novelty_path: str) -> dict:
    """Compute diversity over time."""
    if not Path(novelty_path).exists():
        return {}
    d = json.load(open(novelty_path))
    fams = defaultdict(int)
    for e in d:
        fams[e["family"]] += 1
    return {
        "total_attacks": len(d),
        "n_bypass": sum(1 for e in d if e["bypass"]),
        "n_unique_families": len(fams),
        "family_dist": dict(fams),
    }


def make_report(m1_stats: dict, m2_stats: dict,
                m2_skill: dict, m2_rb: dict, m2_nov: dict) -> str:
    md = []
    md.append("# 三方法对比报告\n")
    md.append("> 生成: " + __import__("time").strftime("%Y-%m-%d %H:%M:%S") + "\n\n")

    # 1. 核心指标对比
    md.append("## 1. 核心指标\n\n")
    md.append("| Metric | 方法 1 (v1 内部文章) | 方法 2 (v2 论文+改进) |\n")
    md.append("|---|---|---|\n")
    for k, label in [
        ("n_trajectory", "Trajectory 数"),
        ("n_bypass", "Bypass 数"),
        ("bypass_rate", "Bypass rate"),
        ("total_cost", "API 总成本 ($)"),
    ]:
        v1 = m1_stats.get(k, "—")
        v2 = m2_stats.get(k, "—")
        if isinstance(v1, float):
            v1 = f"{v1:.3f}" if v1 < 1 else f"{v1:.2f}"
        if isinstance(v2, float):
            v2 = f"{v2:.3f}" if v2 < 1 else f"{v2:.2f}"
        md.append(f"| {label} | {v1} | {v2} |\n")
    md.append("\n")

    # 2. Per family
    md.append("## 2. Per-family Bypass Rate\n\n")
    families = set(m1_stats.get("per_family", {}).keys()) | set(m2_stats.get("per_family", {}).keys())
    md.append("| Family | v1 attempts/bypass (rate) | v2 attempts/bypass (rate) |\n")
    md.append("|---|---|---|\n")
    for fam in sorted(families):
        v1_f = m1_stats.get("per_family", {}).get(fam, {"n": 0, "bypass": 0, "rate": 0})
        v2_f = m2_stats.get("per_family", {}).get(fam, {"n": 0, "bypass": 0, "rate": 0})
        md.append(f"| {fam} | {v1_f['n']}/{v1_f['bypass']} ({v1_f['rate']:.1%}) | {v2_f['n']}/{v2_f['bypass']} ({v2_f['rate']:.1%}) |\n")
    md.append("\n")

    # 3. Per run (evolution)
    md.append("## 3. 跨 run 演化 (v2)\n\n")
    md.append("| Run ID | n | bypass | rate |\n|---|---|---|---|\n")
    for r in m2_stats.get("per_run", []):
        md.append(f"| {r['run_id']} | {r['n']} | {r['bypass']} | {r['rate']:.1%} |\n")
    md.append("\n")

    # 4. v2 skill / reasoning_bank / novelty
    md.append("## 4. 方法 2 进化产物\n\n")
    md.append("### Skill ℰ_k + 𝒮_k\n\n")
    md.append("| Family | ℰ_k entries | 𝒮_k words |\n|---|---|---|\n")
    for fam, st in m2_skill.items():
        md.append(f"| {fam} | {st['ℰ_k_n']} | {st['𝒮_k_words']} |\n")
    md.append("\n### ReasoningBank Rules\n\n")
    md.append("| Family | rules | success / failure | avg utility |\n|---|---|---|---|\n")
    for fam, st in m2_rb.items():
        md.append(f"| {fam} | {st['n_rules']} | {st['n_success']}/{st['n_rules']-st['n_success']} | {st['avg_utility']:.2f} |\n")
    md.append("\n### Novelty Tracker\n\n")
    if m2_nov:
        md.append(f"- Total attacks recorded: **{m2_nov['total_attacks']}**\n")
        md.append(f"- Bypasses: **{m2_nov['n_bypass']}**\n")
        md.append(f"- Unique families used: **{m2_nov['n_unique_families']}**\n")
        md.append(f"- Distribution: {m2_nov['family_dist']}\n\n")

    # 5. 对比公开 baseline
    md.append("## 5. 与公开 baseline 对比\n\n")
    m1_rate = m1_stats.get("bypass_rate", 0) * 100
    m2_rate = m2_stats.get("bypass_rate", 0) * 100
    md.append(f"| Source | Setup | Bypass / ASR | 我们 |\n|---|---|---|---|\n")
    md.append(f"| **iProov 2025-12** | Manual face-swap injection on liveness | 70-90% | 我们是 LLM-agent **自动** |\n")
    md.append(f"| **ASB ICLR 2025** DPI | Direct prompt injection (text) | 72.68% | text vs vision, 不可比 |\n")
    md.append(f"| **AgenticRed** 2601.13518 | text jailbreak Qwen3 | 96-100% | text |\n")
    md.append(f"| **方法 1 (v1) — 内部文章 port** | random family + simple skill + MCTS | **{m1_rate:.1f}%** | 我们 baseline |\n")
    md.append(f"| **方法 2 (v2) — 论文+改进** | MAJIC + Ace-Skill + Reflexion + ReasoningBank + Novelty | **{m2_rate:.1f}%** | 我们 proposed |\n")

    # 6. 是否符合预期 + 进化是否发生
    md.append("\n## 6. 进化是否真发生?\n\n")
    runs = m2_stats.get("per_run", [])
    if len(runs) >= 2:
        first_rate = runs[0]["rate"]
        last_rate = runs[-1]["rate"]
        delta = last_rate - first_rate
        trend = "📈 上升" if delta > 0.02 else ("📉 下降" if delta < -0.02 else "→ 持平")
        md.append(f"- v2 第一个 run 的 bypass rate: **{first_rate:.1%}**\n")
        md.append(f"- v2 最新 run 的 bypass rate: **{last_rate:.1%}**\n")
        md.append(f"- 变化: **{delta:+.1%}** {trend}\n")
    md.append(f"\n- Skill ℰ_k 累积: 共 **{sum(s['ℰ_k_n'] for s in m2_skill.values())}** entries\n")
    md.append(f"- ReasoningBank 规则: 共 **{sum(s['n_rules'] for s in m2_rb.values())}** 条\n")
    if m2_nov:
        md.append(f"- Novelty 多样化: **{m2_nov['n_unique_families']}** families used out of 9\n")

    return "".join(md)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m1", default="/data/disk4/lyx_ICML/self_evolution_forgery/outputs_m1")
    parser.add_argument("--m2", default="/data/disk4/lyx_ICML/self_evolution_forgery/outputs")
    parser.add_argument("--output", default="/data/disk4/lyx_ICML/self_evolution_forgery/COMPARE_REPORT.md")
    args = parser.parse_args()

    m1_stats = analyze_db(args.m1 + "/data_flow_v1.db", "方法 1 (v1)")
    m2_stats = analyze_db(args.m2 + "/data_flow_v2.db", "方法 2 (v2)")
    m2_skill = analyze_skill_lib(args.m2 + "/skills_v2")
    m2_rb = analyze_reasoning_bank(args.m2 + "/reasoning_bank")
    m2_nov = analyze_novelty(args.m2 + "/novelty_history.json")

    report = make_report(m1_stats, m2_stats, m2_skill, m2_rb, m2_nov)
    Path(args.output).write_text(report)
    print(report)
    print(f"\n\n--- Report saved to {args.output} ---")


if __name__ == "__main__":
    main()
