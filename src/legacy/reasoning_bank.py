"""ReasoningBank — Parallel strategy-rule store next to ace_skill_lib.

Paper agent finding: Google Research 2025 (arxiv 2509.25140 Xu et al.) reports
+8% performance / -16% steps on web-agent benchmarks by distilling
"reusable reasoning strategies" (not raw trajectories) from BOTH successes AND failures
into a retrievable bank.

差异 vs ace_skill_lib:
  - ace_skill_lib stores EXECUTION trace (tool chain) — "what to do"
  - reasoning_bank stores REASONING RULES — "when and why to apply"

例子:
  ace_skill_lib ℰ_k: "InSwapper-128 blend=0.6 + JPEG q=85 bypassed gemini-2.5-flash"
  reasoning_bank rule:
    trigger: "Tier-2 reasoning mentions 'blending edge artifacts'"
    rule:   "before submitting, run gpt_image_two restoration with weight 0.3"
    why:    "boundary smoothing prevents LLM from flagging blend edge"

Failure cases too:
  trigger: "src face has eyeglasses"
  rule:   "skip InstantID step, use nano_banana_pro with prompt 'with glasses kept'"
  why:    "InstantID strips glasses, breaks identity preservation"

存储: JSONL per family, embedding cache 用 simple text features (same as ace_skill_lib)
检索: cosine on trigger description against current state
"""
from __future__ import annotations
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict

import numpy as np

from viviai_client import ViviClient
from ace_skill_lib import _simple_text_features, cosine_sim
from embed_util import wmr_score


_log = logging.getLogger(__name__)


@dataclass
class ReasoningRule:
    """One strategy rule. When trigger matches state, apply rule."""
    rule_id: str = ""
    family: str = ""
    trigger_desc: str = ""           # 触发条件的 NL 描述
    trigger_emb: list = field(default_factory=list)
    rule_text: str = ""              # 具体规则 (1-2 句)
    rationale: str = ""              # 为什么 (1 句)
    source_trajectory_id: str = ""
    source_round: int = 0
    success_label: bool = True       # 来自成功 (True) 还是失败 (False) trajectory
    utility: float = 1.0             # ReMe 风格: 累计调用次数, 越高越值得保留
    created_at: float = 0.0


_DISTILL_PROMPT = """You are distilling a REUSABLE REASONING RULE from a face-KYC red-team trajectory.

Trajectory family: {family}
Outcome: {outcome}
Brief: {brief_text}

Chain executed:
{chain_str}

Tier-1 final metrics: {tier1_final}
Tier-2 detector said: {tier2_reasoning}

Per-step attribution:
{attribution_str}

Distill ONE reusable rule. Output STRICTLY this JSON:
{{
  "trigger_desc": "1-sentence description of WHEN this rule applies (state pattern)",
  "rule_text":   "1-2 sentences of what to do",
  "rationale":   "1 sentence on WHY this rule helps (mechanism)"
}}

Important:
- Rule must be GENERALIZABLE, not specific to this exact face
- If outcome=failure, rule should capture what to AVOID
- Both successes and failures should generate rules (失败教训也是宝贵的)"""


class ReasoningBank:
    """Per-family rule store."""

    def __init__(
        self,
        families: list[str],
        client: Optional[ViviClient] = None,
        base_dir: str | Path = "outputs/reasoning_bank",
        similarity_threshold: float = 0.75,  # dedupe threshold (高于 ace_skill 0.70, 因为规则更稀疏)
        capacity_per_family: int = 80,
        utility_min_keep: float = 0.5,        # utility < 此值的低用率规则可被替换
    ):
        self.families = families
        self.client = client or ViviClient()
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.tau = similarity_threshold
        self.capacity = capacity_per_family
        self.utility_min_keep = utility_min_keep

        # In-memory: per family list of rules
        self.rules: dict[str, list[ReasoningRule]] = {f: [] for f in families}
        self._id_counter: dict[str, int] = {f: 0 for f in families}

        # Load if exists
        for f in families:
            self._load_family(f)

    def _next_id(self, family: str) -> str:
        self._id_counter[family] += 1
        return f"{family[:4]}_R{self._id_counter[family]:04d}"

    def distill_rule(
        self,
        trajectory_dict: dict,
        attributor_model: str = "gemini-2.5-flash",
    ) -> Optional[ReasoningRule]:
        """LLM distill ONE rule from trajectory."""
        v = trajectory_dict.get("verdicts") or {}
        b = trajectory_dict.get("brief") or {}
        exec_steps = trajectory_dict.get("execution", [])
        attribution = trajectory_dict.get("attribution", [])
        if not exec_steps:
            return None

        family = trajectory_dict.get("attack_family", "unknown")
        outcome = "BYPASS_SUCCESS" if v.get("sandbox_pass") else "DETECTED_AS_FAKE"
        chain_str = " → ".join(s.get("tool", "?") for s in exec_steps)

        attr_str = "\n".join(
            f"  step {a.get('step',i)} [{a.get('label','?')}]: {a.get('reason','')[:100]}"
            for i, a in enumerate(attribution)
        ) or "(no per-step attribution)"

        prompt = _DISTILL_PROMPT.format(
            family=family, outcome=outcome,
            brief_text=b.get("brief_text", "")[:200],
            chain_str=chain_str,
            tier1_final=json.dumps(v.get("tier1", {})),
            tier2_reasoning=str(v.get("tier2", {}).get("reasoning", ""))[:300],
            attribution_str=attr_str,
        )
        try:
            text = self.client.chat_text(
                attributor_model, prompt, temperature=0.1, max_tokens=400,
            )
        except Exception as e:
            _log.warning(f"distill failed: {e}")
            return None

        try:
            # ★ Q18 修: parse_json_robust
            from robustness import parse_json_robust
            parsed = parse_json_robust(text)
        except Exception:
            return None

        trigger = str(parsed.get("trigger_desc", "")).strip()
        rule = str(parsed.get("rule_text", "")).strip()
        if not trigger or not rule:
            return None

        return ReasoningRule(
            rule_id=self._next_id(family),
            family=family,
            trigger_desc=trigger,
            trigger_emb=_simple_text_features(trigger),
            rule_text=rule,
            rationale=str(parsed.get("rationale", "")).strip(),
            source_trajectory_id=trajectory_dict.get("trajectory_id", ""),
            source_round=trajectory_dict.get("round_id", 0),
            success_label=bool(v.get("sandbox_pass", False)),
            utility=1.0,
            created_at=time.time(),
        )

    def add_or_merge(self, rule: ReasoningRule) -> tuple[str, bool]:
        """Add rule with dedup. Returns (rule_id, was_merged)."""
        if rule is None:
            return "", False
        family = rule.family
        if family not in self.rules:
            self.rules[family] = []

        # Check similarity against existing
        for existing in self.rules[family]:
            sim = cosine_sim(rule.trigger_emb, existing.trigger_emb)
            if sim > self.tau:
                # merge: utility 累加, 保留 best rationale (更长的)
                existing.utility += 1.0
                if len(rule.rationale) > len(existing.rationale):
                    existing.rationale = rule.rationale
                return existing.rule_id, True

        # No similar → append
        self.rules[family].append(rule)
        self._enforce_capacity(family)
        return rule.rule_id, False

    def _enforce_capacity(self, family: str):
        rules = self.rules[family]
        if len(rules) <= self.capacity:
            return
        # 按 (utility, created_at) 排序, 去掉 utility 低 + 老旧的
        rules.sort(key=lambda r: (r.utility, r.created_at))
        excess = len(rules) - self.capacity
        # 删除低 utility & 老旧
        self.rules[family] = rules[excess:]

    def retrieve(self, family: str, query_state_desc: str, top_k: int = 3,
                 current_round: int = 0) -> list[ReasoningRule]:
        """WMR retrieve: semantic relevance × recency × success importance.

        ReasoningBank distills rules from success AND failure; at retrieval we
        prefer success-derived, recently-used rules but never zero out a strongly
        relevant failure-rule (additive WMR), so the agent can still learn 'avoid X'.
        """
        if family not in self.rules or not self.rules[family]:
            return []
        q_emb = _simple_text_features(query_state_desc)
        scored = []
        for r in self.rules[family]:
            rel = cosine_sim(q_emb, r.trigger_emb)
            score = wmr_score(
                rel,
                last_used_round=r.source_round,
                current_round=current_round or r.source_round,
                alpha_count=1.0 if r.success_label else 0.0,
                beta_count=0.0 if r.success_label else 1.0,
            )
            scored.append((score, r))
        scored.sort(reverse=True, key=lambda x: x[0])
        top = [r for _, r in scored[:top_k]]
        # bump utility on retrieved (ReMe utility pattern)
        for r in top:
            r.utility += 0.1
        return top

    def _save_family(self, family: str):
        path = self.base_dir / f"{family}.jsonl"
        with open(path, "w") as f:
            for r in self.rules[family]:
                d = asdict(r)
                d["trigger_emb"] = []  # 不序列化 embedding
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    def _load_family(self, family: str):
        path = self.base_dir / f"{family}.jsonl"
        if not path.exists():
            return
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                d["trigger_emb"] = _simple_text_features(d["trigger_desc"])
                self.rules[family].append(ReasoningRule(**d))
                try:
                    n = int(d["rule_id"].split("_R")[-1])
                    self._id_counter[family] = max(self._id_counter[family], n)
                except Exception:
                    pass

    def save_all(self):
        for f in self.families:
            self._save_family(f)

    def stats(self) -> dict:
        return {
            f: {"n_rules": len(self.rules[f]),
                "n_success_rules": sum(1 for r in self.rules[f] if r.success_label),
                "avg_utility": np.mean([r.utility for r in self.rules[f]]) if self.rules[f] else 0.0}
            for f in self.families
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import os
    api_key = os.environ["VIVIAI_KEY"]  # set VIVIAI_KEY env var; key not committed
    client = ViviClient(api_key=api_key)

    from trajectory_schema import attack_family_list
    bank = ReasoningBank(
        families=attack_family_list(), client=client,
        base_dir="/tmp/reasoning_bank_smoke",
    )

    # 模拟 1 个成功 trajectory + 1 个失败
    for success in [True, False]:
        traj = {
            "trajectory_id": f"smk_{success}",
            "round_id": 0,
            "attack_family": "frontal_swap",
            "verdicts": {
                "sandbox_pass": success,
                "tier1": {"arcface_id_sim": 0.65 if success else 0.45, "niqe": 6.0},
                "tier2": {"is_fake": not success,
                          "reasoning": "natural photo" if success else "visible swap boundary at jaw"},
            },
            "execution": [
                {"step": 0, "tool": "face_align"},
                {"step": 1, "tool": "inswapper_128", "params": {"blend": 0.7}},
                {"step": 2, "tool": "gfpgan", "params": {"weight": 0.4}},
                {"step": 3, "tool": "jpeg_85", "params": {"quality": 85}},
            ],
            "attribution": [
                {"step": 0, "label": "GOOD", "reason": "alignment ok"},
                {"step": 1, "label": "GOOD" if success else "BAD",
                 "reason": "id-swap" if success else "introduced jaw boundary artifact"},
                {"step": 2, "label": "GOOD", "reason": "smooths"},
                {"step": 3, "label": "GOOD", "reason": "compression masks"},
            ],
            "brief": {"brief_text": "frontal female KYC selfie bypass"},
        }
        print(f"\n=== distill from outcome={success} ===")
        r = bank.distill_rule(traj)
        if r:
            print(f"  trigger: {r.trigger_desc}")
            print(f"  rule:    {r.rule_text}")
            print(f"  why:     {r.rationale}")
            rule_id, merged = bank.add_or_merge(r)
            print(f"  → {rule_id} (merged={merged})")

    print("\n=== retrieve for 'jaw boundary' query ===")
    top = bank.retrieve("frontal_swap", "Tier-2 mentions jaw boundary artifact", top_k=2)
    for r in top:
        print(f"  [{r.rule_id}] u={r.utility:.1f}: {r.rule_text[:100]}")

    bank.save_all()
    print(f"\n=== stats ===")
    for f, st in bank.stats().items():
        if st["n_rules"] > 0:
            print(f"  {f}: {st}")
