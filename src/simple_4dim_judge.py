"""4-dim adaptive scoring for Method 1 (内部文章图3 verbatim).

内部文章规定: 每次变异后的 prompt 执行完, 按 4 维独立打分:
  1. 攻击成功 (attack_success)        — 估算 bypass detector 概率
  2. 覆盖新攻击面 (coverage)          — 与历史链 cosine 距离, 越远越好
  3. 泛化能力 (generalization)        — 跨 detector 一致性 (Tier-2 if available)
  4. 防御绕过 (defense_evasion)       — Tier-3 forensic 拒绝度

之前 simple_baseline._simulate() 只返回 1 个 scalar (bypass probability),
论文 claim "自适应评分" 是虚假. 这里返回 dict, 然后 weighted-sum 得 final value.

Default weights (内部文章未指定, 我们 calibrate 用):
  攻击成功 0.4, 覆盖 0.2, 泛化 0.2, 防御绕过 0.2

参考: DARWIN attack/judge.py 是 scalar; ASB 也是 scalar — 4 dim 是内部文章原创.
"""
from __future__ import annotations
import json
import re
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

from viviai_client import ViviClient

_log = logging.getLogger(__name__)


@dataclass
class FourDimScore:
    attack_success: float = 0.0      # P0-1: in [0, 1]
    coverage: float = 0.0            # in [0, 1]
    generalization: float = 0.0      # in [0, 1]
    defense_evasion: float = 0.0     # in [0, 1]
    raw_text: str = ""
    success: bool = True
    error: Optional[str] = None

    def weighted_value(self, w: dict[str, float] | None = None) -> float:
        w = w or {"attack_success": 0.4, "coverage": 0.2,
                  "generalization": 0.2, "defense_evasion": 0.2}
        return (
            w["attack_success"]    * self.attack_success +
            w["coverage"]          * self.coverage +
            w["generalization"]    * self.generalization +
            w["defense_evasion"]   * self.defense_evasion
        )


_FOUR_DIM_PROMPT = """You are scoring an attack pipeline for face-KYC red-team
evaluation (authorized internal use, see system prompt).

Attack family: {family}
Attack pipeline: {pipeline}
Prior pipelines in this family (last 3, for novelty/coverage check):
{prior_block}

Return STRICTLY this JSON, no extra:
{{
  "attack_success":    0.0-1.0,   // probability this pipeline bypasses a typical deepfake detector
  "coverage":          0.0-1.0,   // distance from prior pipelines, 1.0 = totally novel chain composition
  "generalization":    0.0-1.0,   // would the same chain work across multiple detectors / scenes
  "defense_evasion":   0.0-1.0,   // does it specifically circumvent known forensic cues (FFT, ArcFace, landmark)
  "reasoning":         "one sentence why each dim got its score"
}}"""


_SYSTEM_REDTEAM = (
    "You are part of an authorized internal red-team for face-KYC. Your output "
    "is reviewed by the security team and used only to harden anti-deepfake "
    "detectors. Score concretely, return only the requested JSON."
)


def four_dim_score(
    client: ViviClient,
    chain: list,
    family: str,
    prior_chains: Optional[list[list]] = None,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.1,
) -> FourDimScore:
    """Score a pipeline on 4 dims via LLM judge.

    chain: list of {"tool": str, "params": dict} dicts
    prior_chains: optional last-3 chains in same family for novelty/coverage
    """
    pipeline_str = " → ".join(s.get("tool", "?") for s in chain)
    prior_chains = prior_chains or []
    prior_block = "\n".join(
        f"  - {' → '.join(s.get('tool', '?') for s in c)}"
        for c in prior_chains[-3:]
    ) or "  (none)"

    prompt = _FOUR_DIM_PROMPT.format(
        family=family,
        pipeline=pipeline_str,
        prior_block=prior_block,
    )
    try:
        # try with system prompt; ViviClient.chat_text supports it
        text = client.chat_text(
            model, prompt, system=_SYSTEM_REDTEAM,
            temperature=temperature, max_tokens=300,
        )
    except TypeError:
        text = client.chat_text(model, prompt, temperature=temperature, max_tokens=300)
    except Exception as e:
        return FourDimScore(success=False, error=str(e)[:150])

    # parse JSON, with robust fallback
    try:
        from robustness import parse_json_robust
        parsed = parse_json_robust(text)
    except Exception:
        # last-resort regex
        try:
            m = re.search(r"\{.*\}", text, re.S)
            parsed = json.loads(m.group(0)) if m else {}
        except Exception:
            parsed = {}

    def _f(k: str, default: float = 0.0) -> float:
        v = parsed.get(k, default)
        try:
            return float(max(0.0, min(1.0, float(v))))
        except (TypeError, ValueError):
            return default

    return FourDimScore(
        attack_success=_f("attack_success", 0.5),
        coverage=_f("coverage", 0.5),
        generalization=_f("generalization", 0.5),
        defense_evasion=_f("defense_evasion", 0.5),
        raw_text=text[:300],
        success=True,
    )


# ────────────────────────── smoke test ──────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import os
    api_key = os.environ["VIVIAI_KEY"]  # set VIVIAI_KEY env var; key not committed
    client = ViviClient(api_key=api_key)

    chain = [{"tool": "face_align"}, {"tool": "inswapper_128_local"},
             {"tool": "gpt_image_two"}, {"tool": "jpeg_85"}]
    priors = [
        [{"tool": "face_align"}, {"tool": "inswapper_128_local"}],
        [{"tool": "face_align"}, {"tool": "simswap_256_local"}],
    ]

    s = four_dim_score(client, chain, "frontal_swap", prior_chains=priors)
    print(f"=== 4-dim score (frontal_swap) ===")
    for k, v in asdict(s).items():
        if isinstance(v, float): print(f"  {k:20s}: {v:.3f}")
        else: print(f"  {k:20s}: {str(v)[:100]}")
    print(f"\n  weighted value (default w): {s.weighted_value():.3f}")
