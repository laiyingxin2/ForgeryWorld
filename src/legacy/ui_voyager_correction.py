"""UI-Voyager-style 失败归因 + 成功轨迹纠正 (arxiv 2603.24533, NeurIPS).

User survey 推荐 #1: "目前最贴近真正自我进化的多模态方案".

UI-Voyager 核心 pipeline (vs simple Reflexion):
  1. Agent 执行 → 成功/失败
  2. **失败归因**: 从失败轨迹中找出真正导致失败的关键分叉点 (key failure step)
  3. **成功轨迹检索**: 找同 task 类相似的成功轨迹
  4. **轨迹纠正**: 用成功轨迹的后续步骤修正失败轨迹
  5. (训练侧) Rejection Fine-Tuning + Group Relative Self-Distillation (GRSD)
     → 我们这里只做 inference-side correction, 训练侧 deferred 到 method 3

Reference: arxiv 2603.24533v1 (论文 GPT-4 GenAgent 81.0% Pass@1 on AndroidWorld)

Face-forgery 适配:
  - "失败" = sandbox.tier2 判定 fake (bypass=False)
  - "关键分叉点" = chain 里 ArcFace_id_sim / SSIM / NIQE 急剧 degrade 那一步
  - "相似成功轨迹" = data_flow.db 里同 family、相似 chain shape、sandbox_pass=1 的 traj
  - "纠正" = 截取失败 chain[:failure_step] + 拼接成功 chain[matched_step:]
  - 输出: corrected_chain 立即喂回 orchestrator 再执行 (R+1 training data)

vs STaSC: STaSC 让 model 改自己的答案 (LLM 自纠); UI-Voyager 用别人的成功轨迹纠
正失败 (cross-trajectory). 互补.
"""
from __future__ import annotations
import json
import sqlite3
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_log = logging.getLogger(__name__)


@dataclass
class FailurePoint:
    step_idx: int                # 0-indexed step where things went bad
    op_name: str                 # the operator that triggered degradation
    metric_evidence: dict        # which Tier-1 metric crossed threshold
    severity: float              # 0-1 importance score


@dataclass
class CorrectedChain:
    family: str
    original_failed_chain: list      # [{tool, params}, ...]
    failure_point: FailurePoint
    matched_success_traj_id: str     # which historical success was the donor
    matched_success_chain: list
    corrected_chain: list            # the new chain to re-execute
    rationale: str = ""


# ────────────────────────── 失败归因 ──────────────────────────

def attribute_failure(
    failed_chain: list,            # [{tool, params}, ...]
    per_step_metrics: list,        # list of dict per step: {arcface_id_sim, ssim, niqe, ...}
    final_verdict: dict,           # tier2 + tier3 verdict
) -> FailurePoint:
    """Identify the step where chain 'broke'.

    Heuristic per UI-Voyager: look for the step with biggest metric
    degradation. For face-forgery:
      - arcface_id_sim dropping (loss of identity) > 0.15 in one step = bad
      - ssim_vs_src dropping > 0.30 = obvious manipulation
      - niqe spiking > 5 = poor quality intro

    Returns FailurePoint indexing into failed_chain.
    """
    if not failed_chain or not per_step_metrics:
        # nothing to attribute; blame the last step generically
        return FailurePoint(
            step_idx=max(0, len(failed_chain) - 1),
            op_name=(failed_chain[-1].get("tool", "?") if failed_chain else "?"),
            metric_evidence={"reason": "no per-step metrics available"},
            severity=0.5,
        )
    n = min(len(failed_chain), len(per_step_metrics))
    if n < 2:
        return FailurePoint(
            step_idx=0, op_name=(failed_chain[0].get("tool", "?") if failed_chain else "?"),
            metric_evidence=per_step_metrics[0] if per_step_metrics else {},
            severity=0.5,
        )
    worst_step = 0
    worst_sev = 0.0
    worst_evi: dict = {}
    for i in range(1, n):
        prev = per_step_metrics[i - 1]
        cur = per_step_metrics[i]
        d_arc = abs(float(prev.get("arcface_id_sim", 0)) -
                    float(cur.get("arcface_id_sim", 0)))
        d_ssim = abs(float(prev.get("ssim_vs_src", 0)) -
                     float(cur.get("ssim_vs_src", 0)))
        d_niqe = abs(float(prev.get("niqe", 5)) - float(cur.get("niqe", 5)))
        # weighted severity: arcface drop dominates (it = identity transfer signal)
        sev = 0.5 * d_arc + 0.3 * d_ssim + 0.2 * min(d_niqe / 10, 1.0)
        if sev > worst_sev:
            worst_sev = sev
            worst_step = i
            worst_evi = {
                "Δarcface_id_sim": round(d_arc, 4),
                "Δssim_vs_src":    round(d_ssim, 4),
                "Δniqe":           round(d_niqe, 4),
            }
    return FailurePoint(
        step_idx=worst_step,
        op_name=failed_chain[worst_step].get("tool", "?"),
        metric_evidence=worst_evi,
        severity=round(worst_sev, 4),
    )


# ────────────────────────── 成功轨迹检索 ──────────────────────────

def retrieve_similar_success(
    db_path: str,
    family: str,
    failed_chain_tools: list[str],
    min_overlap: int = 2,
    limit: int = 5,
) -> list[dict]:
    """Find historical successful trajectories with same family + chain shape overlap.

    Returns list of {traj_id, chain, sandbox_pass} ordered by chain-overlap DESC.
    """
    if not Path(db_path).exists():
        return []
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT trajectory_id, full_json FROM trajectories "
            "WHERE attack_family=? AND sandbox_pass=1 ORDER BY rowid DESC LIMIT 200",
            (family,)
        ).fetchall()
    finally:
        conn.close()
    failed_set = set(failed_chain_tools)
    scored = []
    for tid, full_json in rows:
        try:
            traj = json.loads(full_json)
        except Exception:
            continue
        exec_steps = traj.get("execution") or []
        chain_tools = [s.get("tool", "?") for s in exec_steps]
        if not chain_tools: continue
        overlap = len(failed_set & set(chain_tools))
        if overlap < min_overlap: continue
        scored.append({
            "trajectory_id": tid,
            "chain_tools": chain_tools,
            "chain": [{"tool": s.get("tool", "?"),
                       "params": s.get("params", {})} for s in exec_steps],
            "overlap": overlap,
            "len_diff": abs(len(chain_tools) - len(failed_chain_tools)),
        })
    scored.sort(key=lambda x: (-x["overlap"], x["len_diff"]))
    return scored[:limit]


# ────────────────────────── 轨迹纠正 ──────────────────────────

def correct_chain(
    failed_chain: list,
    failure_point: FailurePoint,
    matched_success: dict,
    max_chain_len: int = 8,
) -> Optional[CorrectedChain]:
    """Build a corrected chain by:
      - keep failed_chain[:failure_step] (the part that was fine)
      - find an equivalent step in matched_success and graft success[match+1:]

    Strategy: find earliest step in matched_success whose op == failed[failure-1] op
              (i.e. last good step), then graft the rest.
    Fallback: just use entire matched_success chain.
    """
    if not matched_success:
        return None
    success_chain = matched_success["chain"]
    success_tools = matched_success["chain_tools"]
    failure_step = failure_point.step_idx

    # find equivalent point in success chain
    graft_idx = -1
    if failure_step > 0:
        last_good_op = failed_chain[failure_step - 1].get("tool", "?")
        for i, tool in enumerate(success_tools):
            if tool == last_good_op:
                graft_idx = i
                break
    if graft_idx >= 0 and graft_idx + 1 < len(success_chain):
        new_chain = (failed_chain[:failure_step]
                     + success_chain[graft_idx + 1:])
    else:
        # no graft point — just adopt entire success chain
        new_chain = list(success_chain)

    # cap to max length
    new_chain = new_chain[:max_chain_len]
    rationale = (
        f"Failure at step {failure_step} ({failure_point.op_name}) — "
        f"evidence {failure_point.metric_evidence}; "
        f"grafted from success traj "
        f"[{matched_success['trajectory_id'][:24]}] at op "
        f"{success_tools[graft_idx] if graft_idx >= 0 else 'start'}."
    )
    return CorrectedChain(
        family=matched_success.get("family", ""),
        original_failed_chain=failed_chain,
        failure_point=failure_point,
        matched_success_traj_id=matched_success["trajectory_id"],
        matched_success_chain=success_chain,
        corrected_chain=new_chain,
        rationale=rationale,
    )


# ────────────────────────── Top-level wrapper ──────────────────────

def ui_voyager_correct(
    failed_chain: list,
    per_step_metrics: list,
    final_verdict: dict,
    family: str,
    db_path: str,
) -> Optional[CorrectedChain]:
    """One-call: attribute failure → retrieve success → correct chain.
    Returns None if no matching success found (cold start)."""
    fp = attribute_failure(failed_chain, per_step_metrics, final_verdict)
    failed_tools = [s.get("tool", "?") for s in failed_chain]
    candidates = retrieve_similar_success(db_path, family, failed_tools, min_overlap=2)
    if not candidates:
        _log.info(f"[ui_voyager] no similar success found for family={family} "
                  f"(chain={' → '.join(failed_tools)})")
        return None
    chosen = candidates[0]
    chosen["family"] = family
    return correct_chain(failed_chain, fp, chosen)


# ────────────────────────── smoke ──────────────────────────────

if __name__ == "__main__":
    import tempfile
    logging.basicConfig(level=logging.INFO)
    with tempfile.TemporaryDirectory() as td:
        db = f"{td}/test.db"
        conn = sqlite3.connect(db)
        conn.executescript("""
        CREATE TABLE trajectories (
            trajectory_id TEXT PRIMARY KEY, attack_family TEXT,
            sandbox_pass INTEGER, full_json TEXT
        );""")
        # insert a successful traj
        success_traj = {
            "execution": [
                {"tool": "face_align", "params": {}},
                {"tool": "inswapper_128_local", "params": {"blend": 0.7}},
                {"tool": "gpt_image_two", "params": {"prompt": "natural"}},
                {"tool": "jpeg_85", "params": {"quality": 85}},
            ],
            "attack_family": "frontal_swap",
        }
        conn.execute("INSERT INTO trajectories VALUES (?, ?, ?, ?)",
                      ("traj_001", "frontal_swap", 1, json.dumps(success_traj)))
        conn.commit()
        conn.close()

        # simulate a failed traj
        failed = [
            {"tool": "face_align", "params": {}},
            {"tool": "inswapper_128_local", "params": {"blend": 0.4}},   # too low blend
            {"tool": "resize_bicubic", "params": {"scale": 0.7}},          # destroyed res
        ]
        per_step = [
            {"arcface_id_sim": 0.95, "ssim_vs_src": 0.99, "niqe": 5.0},
            {"arcface_id_sim": 0.72, "ssim_vs_src": 0.40, "niqe": 5.8},  # big drop
            {"arcface_id_sim": 0.35, "ssim_vs_src": 0.20, "niqe": 8.5},  # disaster
        ]
        result = ui_voyager_correct(failed, per_step, {"is_fake": True},
                                      "frontal_swap", db)
        print("=== UI-Voyager correction smoke ===")
        if result is None:
            print("  no correction found")
        else:
            print(f"  failure point: step {result.failure_point.step_idx} "
                  f"({result.failure_point.op_name}) severity={result.failure_point.severity}")
            print(f"  evidence: {result.failure_point.metric_evidence}")
            print(f"  matched donor: {result.matched_success_traj_id}")
            print(f"  original chain: {[s['tool'] for s in result.original_failed_chain]}")
            print(f"  corrected chain: {[s['tool'] for s in result.corrected_chain]}")
            print(f"  rationale: {result.rationale[:200]}")
        print("\nsmoke PASS" if result else "SMOKE FAIL — no donor found")
