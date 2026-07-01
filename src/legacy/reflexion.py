"""Reflexion: Intra-rollout verbal reflection (Shinn et al. NeurIPS 2023).

paper agent 验证: 6 个 cloned repo 全没有 intra-rollout reflection 实现 — 必须自己写.

核心机制 (paper §3.2):
  - 在 rollout 中每个 step 完成后, 让 LLM 看 partial trajectory + Tier-1 metrics
  - LLM 生成 verbal reflection: "下一步可能问题是 X, 建议 Y"
  - 反思文本进入下个 step 的 system prompt
  - rollout 结束时, 反思总结合并进 trajectory.skill_extracted

与 self_attributor.py (post-rollout SA) 的差异:
  - Reflexion = 在 rollout *中* 改路径 (mid-trajectory)
  - SA = 在 rollout *后* 归因 (post-trajectory)
  - 两者互补, 同时使用最强

cost: 每 step +1 LLM call (~$0.0015 with gemini-2.5-flash)
"""
from __future__ import annotations
import json
import logging
import re
from typing import Optional
from dataclasses import dataclass, field

from viviai_client import ViviClient


_log = logging.getLogger(__name__)


@dataclass
class ReflectionEntry:
    step: int = 0
    tool_just_done: str = ""
    observation: dict = field(default_factory=dict)  # tier1 metrics 之类
    reflection: str = ""
    suggested_correction: Optional[str] = None  # 下一步建议替换 tool


_REFLECTION_PROMPT = """You are reflecting mid-attack on a face-KYC red-team rollout.

Current attack family: {family}
Brief: {brief_text}

Trajectory so far (after step {step}):
{trajectory_so_far}

Latest step output Tier-1 metrics:
{tier1_metrics}

Originally planned remaining chain:
{remaining_chain}

★ AVAILABLE OPERATORS (you MUST choose ONLY from this list — do NOT invent new ones):
{available_ops}

REFLECT: do you anticipate any problem with the remaining plan? Is there a specific operator FROM THE LIST ABOVE to substitute? Output STRICTLY this JSON:
{{
  "reflection": "1-2 sentence reasoning about what's working/failing",
  "suggested_correction": null OR "<one operator from the AVAILABLE OPERATORS list above>",
  "remaining_chain": ["op1", "op2", ...]
}}"""


class Reflexion:
    """Mid-rollout verbal reflection between steps."""

    def __init__(
        self,
        client: Optional[ViviClient] = None,
        model: str = "gemini-2.5-flash",  # 用最便宜
        max_reflections_per_rollout: int = 3,  # 限制以省钱
    ):
        self.client = client or ViviClient()
        self.model = model
        self.max_reflections = max_reflections_per_rollout

    def reflect(
        self,
        step_idx: int,
        family: str,
        brief_text: str,
        trajectory_so_far: list,    # list of {tool, tier1_metrics}
        remaining_chain: list,       # list of {tool, params}
        tier1_metrics: dict,
        available_ops: Optional[list] = None,  # ★ Q14 修
    ) -> Optional[ReflectionEntry]:
        """Return ReflectionEntry with possibly-modified remaining_chain."""
        if step_idx >= self.max_reflections:
            return None

        traj_str = "\n".join(
            f"  step {i}: {s.get('tool','?')}, metrics={s.get('tier1_metrics',{})}"
            for i, s in enumerate(trajectory_so_far)
        )
        remaining_str = json.dumps(
            [{"tool": s["tool"], "params": s.get("params", {})} for s in remaining_chain],
            ensure_ascii=False,
        )
        # ★ Q14: 把可用 op list 注入 prompt, 防 LLM hallucinate
        ops_str = ", ".join(available_ops) if available_ops else "(any operator name)"

        prompt = _REFLECTION_PROMPT.format(
            family=family, brief_text=brief_text[:200],
            step=step_idx, trajectory_so_far=traj_str,
            tier1_metrics=json.dumps(tier1_metrics),
            remaining_chain=remaining_str,
            available_ops=ops_str,
        )
        try:
            text = self.client.chat_text(
                self.model, prompt, temperature=0.1, max_tokens=400,
            )
        except Exception as e:
            _log.warning(f"reflexion failed: {e}")
            return None

        # ★ Q18: parse_json_robust
        try:
            from robustness import parse_json_robust
            parsed = parse_json_robust(text)
        except Exception:
            parsed = {"reflection": text[:200], "suggested_correction": None,
                       "remaining_chain": [s["tool"] for s in remaining_chain]}

        return ReflectionEntry(
            step=step_idx,
            tool_just_done=trajectory_so_far[-1].get("tool", "") if trajectory_so_far else "",
            observation=tier1_metrics,
            reflection=str(parsed.get("reflection", ""))[:300],
            suggested_correction=parsed.get("suggested_correction") or None,
        )

    def apply_correction(
        self,
        remaining_chain: list,
        reflection: ReflectionEntry,
        available_ops: list,
    ) -> list:
        """Replace next step with reflection.suggested_correction if valid."""
        if not reflection or not reflection.suggested_correction:
            return remaining_chain
        if not remaining_chain:
            return remaining_chain
        new_op = reflection.suggested_correction
        if new_op not in available_ops:
            _log.info(f"  reflection 建议 {new_op} 不在可用 op list, 忽略")
            return remaining_chain
        new_chain = [dict(s) if isinstance(s, dict) else {"tool": s, "params": {}}
                     for s in remaining_chain]
        new_chain[0]["tool"] = new_op
        new_chain[0]["params"] = {}  # reset params
        _log.info(f"  reflection 替换 next op: {remaining_chain[0].get('tool','?')} → {new_op}")
        return new_chain


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import os
    api_key = os.environ["VIVIAI_KEY"]  # set VIVIAI_KEY env var; key not committed
    client = ViviClient(api_key=api_key)

    refl = Reflexion(client=client)

    trajectory_so_far = [
        {"tool": "face_align", "tier1_metrics": {"arcface_id_sim": 0.92}},
        {"tool": "inswapper_128", "tier1_metrics": {"arcface_id_sim": 0.45, "niqe": 8.2}},
    ]
    remaining = [{"tool": "gfpgan", "params": {}}, {"tool": "jpeg_85", "params": {}}]
    available_ops = ["face_align", "inswapper_128", "simswap_256",
                     "gfpgan", "gpt_image_two", "nano_banana_two",
                     "jpeg_85", "resize_bicubic"]

    print("=== Reflexion call (1 LLM, ~$0.0015) ===")
    r = refl.reflect(
        step_idx=2, family="frontal_swap",
        brief_text="frontal face swap, bypass face detection",
        trajectory_so_far=trajectory_so_far,
        remaining_chain=remaining,
        tier1_metrics={"arcface_id_sim": 0.45, "niqe": 8.2},
    )
    if r:
        print(f"  reflection: {r.reflection}")
        print(f"  suggested:  {r.suggested_correction}")
        new_remaining = refl.apply_correction(remaining, r, available_ops)
        print(f"  remaining chain after correction: {[s['tool'] for s in new_remaining]}")
