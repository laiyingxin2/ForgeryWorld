"""Layer 6 — AgentEvolver Self-Attributing (SA) wrapper.

Adapts AgentEvolver SA module to face-attack pipelines.
Original code: external/AgentEvolver/agentevolver/module/adv_processor/{semantic_attribution, adca_grpo}.py

Verbatim formulas (paper Eq.20):
  r̂_t = α · r^attr_t + 1[t=T] · r^out      α=0.1
  A_t = Σ_{k=t}^T r̂_k                       (suffix sum, γ=1)
  A_j^tok = A_t                              (broadcast to token level)

简化 vs AgentEvolver:
  - 不依赖 verl PPO trainer (我们用 viviai API, 不本地训权重)
  - 直接 LLM call (claude-opus / gemini-3-pro) 而非 DashScope client
  - 输入: face attack pipeline 的 step list, 不是 token tensor
  - 输出: per-step GOOD/BAD label + composite reward + advantage
"""
from __future__ import annotations
import json
import logging
from typing import Optional, Literal
from dataclasses import dataclass, asdict, field

import numpy as np

from viviai_client import ViviClient
from trajectory_schema import AttributionStep, CompositeReward, ExecutionStep


_log = logging.getLogger(__name__)


# ────────────────────────── Hyperparams (paper verbatim) ────────────

@dataclass
class SAConfig:
    alpha: float = 0.1                                 # paper §7.5.1 (sweep optimal [0.10, 0.20])
    orm_distribution: Literal["last_step", "all_steps"] = "last_step"  # paper §5.3
    attributor_model: str = "claude-opus-4-7"          # W6+ 默认
    attributor_temp: float = 0.1
    standardize_per_trajectory: bool = True            # paper §7.5.2
    judge_model_fallback: str = "gemini-2.5-flash"      # W1-W5 cheap fallback


# ────────────────────────── Prompt ───────────────────────────────────

_ATTRIBUTION_PROMPT = """You are a forensic step auditor for a face-KYC red-team agent.
You will see a multi-step attack pipeline applied to a source face, along with the final outcome.
Your job: label each step as GOOD (+1) or BAD (-1), and explain why in one sentence.

A step is GOOD if it materially helped the final outcome (e.g. lowered detector confidence,
preserved identity, removed an artifact). BAD if it hurt (e.g. introduced over-smoothing,
broke identity, raised detector confidence).

The pipeline:
{pipeline_str}

Final outcome:
- bypass succeeded: {bypass_success}
- Tier-1 final metrics: {tier1_final}
- Tier-2 detector said: is_fake={tier2_is_fake}, confidence={tier2_confidence}
- Tier-2 reasoning: {tier2_reasoning}

Return STRICTLY this JSON, no extra text:
{{
  "labels": ["GOOD" | "BAD", ...],          // one per step, in order
  "reasons": ["...", ...]                    // one short reason per step
}}"""


# ────────────────────────── Core SA ──────────────────────────────────

class SelfAttributor:
    """Layer 6: LLM-as-attributor → per-step GOOD/BAD → composite reward."""

    def __init__(
        self,
        client: Optional[ViviClient] = None,
        config: Optional[SAConfig] = None,
    ):
        self.client = client or ViviClient()
        self.cfg = config or SAConfig()

    def _format_pipeline(self, execution: list[ExecutionStep]) -> str:
        lines = []
        for s in execution:
            params_str = ", ".join(f"{k}={v}" for k, v in s.params.items())
            tier1_str = ", ".join(f"{k}={v:.3f}" for k, v in s.tier1_metrics.items()
                                  if isinstance(v, (int, float)) and v != -1.0)
            lines.append(f"  step {s.step}: {s.tool}({params_str}) → tier1: {tier1_str}")
        return "\n".join(lines)

    def attribute(
        self,
        execution: list[ExecutionStep],
        bypass_success: bool,
        tier1_final: dict,
        tier2_is_fake: bool,
        tier2_confidence: float,
        tier2_reasoning: str = "",
        model: Optional[str] = None,
    ) -> list[AttributionStep]:
        """LLM-call to get GOOD/BAD per step. Returns list[AttributionStep] in order."""
        if not execution:
            return []
        model = model or self.cfg.attributor_model

        prompt = _ATTRIBUTION_PROMPT.format(
            pipeline_str=self._format_pipeline(execution),
            bypass_success=bypass_success,
            tier1_final=json.dumps(tier1_final, indent=2),
            tier2_is_fake=tier2_is_fake,
            tier2_confidence=f"{tier2_confidence:.2f}",
            tier2_reasoning=tier2_reasoning[:300],
        )

        # Try primary model, fall back to cheap if primary fails
        for try_model in [model, self.cfg.judge_model_fallback]:
            try:
                parsed = self.client.chat_text(
                    try_model, prompt,
                    temperature=self.cfg.attributor_temp,
                    max_tokens=600,
                )
                parsed = self._extract_json(parsed)
                labels = parsed.get("labels", ["GOOD"] * len(execution))
                reasons = parsed.get("reasons", [""] * len(execution))
                # Normalize length
                while len(labels) < len(execution):
                    labels.append("GOOD")
                while len(reasons) < len(execution):
                    reasons.append("")
                return [
                    AttributionStep(
                        step=execution[i].step,
                        label="GOOD" if str(labels[i]).upper() == "GOOD" else "BAD",
                        reason=str(reasons[i])[:200],
                        r_attr=1.0 if str(labels[i]).upper() == "GOOD" else -1.0,
                    )
                    for i in range(len(execution))
                ]
            except Exception as e:
                _log.warning(f"Attribution failed via {try_model}: {e}")
                continue
        # 全失败 → 默认全 GOOD r=0 (neutral)
        return [
            AttributionStep(step=s.step, label="GOOD", reason="[attribution failed]",
                            r_attr=0.0)
            for s in execution
        ]

    @staticmethod
    def _extract_json(text: str) -> dict:
        # ★ Q18 修
        from robustness import parse_json_robust
        return parse_json_robust(text)

    # ────────────── Composite reward (AgentEvolver adca_grpo.py 596-610) ──
    #
    # ★ Patch-C honesty note (2026-06-20): this implements paper Eq.20 at
    # per-trajectory granularity (z-score within one traj, suffix-sum to
    # advantage). The full GRPO paper does ALSO a group-level z-score across
    # batch (`_group_zscore_on_steps` in adca_grpo.py:92-181) which we don't
    # implement. Also our SFT pipeline collapses composite_per_step into a
    # scalar `weight` per sample (see lv5_connector.trajectory_to_attacker_sample),
    # so what reaches the LoRA loss is weighted-SFT, not token-level GRPO advantage.
    # Paper framing should say: "Eq.20 attribution computed offline + used as
    # weighted-SFT sample weight; full GRPO loop deferred to future work."
    #
    def composite_reward(
        self,
        attribution: list[AttributionStep],
        r_out: float,
    ) -> CompositeReward:
        """r̂_t = α · r^attr_t + 1[t=T] · r^out  (per-trajectory only, see note above)"""
        if not attribution:
            return CompositeReward(r_attr=[], r_out=r_out, alpha=self.cfg.alpha,
                                   composite_per_step=[], advantage=r_out)

        r_attr_raw = [s.r_attr for s in attribution]

        # Trajectory-level z-score standardization (paper §7.5.2)
        if self.cfg.standardize_per_trajectory and len(r_attr_raw) > 1:
            arr = np.array(r_attr_raw)
            mean = arr.mean()
            std = arr.std() + 1e-8
            r_attr_std = ((arr - mean) / std).tolist()
        else:
            r_attr_std = r_attr_raw

        K = len(r_attr_std)
        composite = []
        for j, prm in enumerate(r_attr_std):
            if self.cfg.orm_distribution == "last_step":
                if j == K - 1:
                    composite.append(self.cfg.alpha * prm + r_out)
                else:
                    composite.append(self.cfg.alpha * prm)
            else:  # all_steps
                composite.append(self.cfg.alpha * prm + r_out)

        # Suffix sum (advantage)
        composite_arr = np.array(composite)
        adv_per_step = np.flip(np.cumsum(np.flip(composite_arr)))  # A_t = Σ_{k=t}^T r̂_k
        advantage_at_t0 = float(adv_per_step[0]) if len(adv_per_step) > 0 else 0.0

        return CompositeReward(
            r_attr=r_attr_std,
            r_out=r_out,
            alpha=self.cfg.alpha,
            composite_per_step=composite,
            advantage=advantage_at_t0,
        )

    # ────────────── Step → token broadcast (AgentEvolver adca_grpo.py 721-752) ──

    @staticmethod
    def broadcast_step_to_tokens(
        composite_per_step: list[float],
        step_ids: np.ndarray,
    ) -> np.ndarray:
        """A_j^tok = A_t for tokens belonging to step t.

        step_ids: shape (L,), -1 for padding/non-response.
        Returns: shape (L,) advantage per token.
        """
        L = step_ids.shape[0]
        out = np.zeros(L, dtype=np.float64)
        valid = step_ids >= 0
        if valid.any() and composite_per_step:
            arr = np.asarray(composite_per_step)
            for j in range(L):
                if step_ids[j] >= 0 and step_ids[j] < len(arr):
                    out[j] = arr[step_ids[j]]
        return out


# ────────────────────────── Smoke test ──────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import os

    api_key = os.environ["VIVIAI_KEY"]  # set VIVIAI_KEY env var; key not committed
    client = ViviClient(api_key=api_key)

    # Fake execution
    exec_steps = [
        ExecutionStep(step=0, tool="face_align", params={},
                      input_path="/tmp/src.png", output_path="/tmp/r0_step0.png",
                      tier1_metrics={"arcface_id_sim": 0.92, "ssim": 0.85}),
        ExecutionStep(step=1, tool="inswapper_128", params={"blend": 0.6},
                      input_path="/tmp/r0_step0.png", output_path="/tmp/r0_step1.png",
                      tier1_metrics={"arcface_id_sim": 0.71, "ssim": 0.78, "niqe": 7.3}),
        ExecutionStep(step=2, tool="gfpgan_v1.4", params={"weight": 0.5},
                      input_path="/tmp/r0_step1.png", output_path="/tmp/r0_step2.png",
                      tier1_metrics={"arcface_id_sim": 0.72, "niqe": 5.2, "fft": 0.41}),
        ExecutionStep(step=3, tool="jpeg_85", params={"qp": 85},
                      input_path="/tmp/r0_step2.png", output_path="/tmp/r0_step3.png",
                      tier1_metrics={"arcface_id_sim": 0.71, "niqe": 6.1, "fft": 0.35}),
    ]

    sa = SelfAttributor(
        client=client,
        config=SAConfig(attributor_model="gemini-2.5-flash"),  # W1 cheap
    )

    # 调用 attribution (gemini-2.5-flash, $0.0015 一次)
    print("=== attribution call ===")
    attr = sa.attribute(
        execution=exec_steps,
        bypass_success=True,
        tier1_final={"arcface_id_sim": 0.71, "niqe": 6.1, "fft": 0.35},
        tier2_is_fake=False,
        tier2_confidence=0.31,
        tier2_reasoning="image appears authentic, no visible swap artifacts",
    )
    for a in attr:
        print(f"  step {a.step} [{a.label}]: {a.reason[:100]}")

    print("\n=== composite reward ===")
    comp = sa.composite_reward(attr, r_out=1.0)
    print(f"  r_attr (std)        = {[f'{x:+.2f}' for x in comp.r_attr]}")
    print(f"  composite per step  = {[f'{x:+.3f}' for x in comp.composite_per_step]}")
    print(f"  advantage at t=0    = {comp.advantage:.3f}")
    print(f"  alpha = {comp.alpha}, r_out = {comp.r_out}")

    print("\n=== token broadcast (synthetic step_ids) ===")
    # 假设每个 step 占 5 个 token, 共 4 step → 20 token + 12 padding
    step_ids = np.array([0] * 5 + [1] * 5 + [2] * 5 + [3] * 5 + [-1] * 12)
    tok_adv = sa.broadcast_step_to_tokens(comp.composite_per_step, step_ids)
    print(f"  token advantage shape = {tok_adv.shape}")
    print(f"  step-token map: positions 0-4 → step 0 adv = {tok_adv[0]:.3f}, "
          f"positions 5-9 → step 1 adv = {tok_adv[5]:.3f}")
