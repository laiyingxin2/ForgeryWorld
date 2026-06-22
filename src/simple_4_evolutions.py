"""4 evolution mechanisms for Method 1 (内部文章图2 verbatim + DARWIN-inspired).

内部文章图2 explicit 列 4 个 evolution 机制 (后续优化: 参考 DARWIN arxiv 2508.13048):
  1. 外部情报收集 — 从 GitHub/HuggingFace/arxiv/Reddit 抓最新 attack 手法
  2. 遗传进化     — Crossover (融合高性能策略) + Mutation (15 种变异算子, 5 维度)
  3. 反思自进化   — 分析失败案例, 提炼改进策略
  4. GAN式对抗共进化 — 跟踪目标模型攻击统计, 支持模型渐进式升级

之前 simple_baseline.py 只实现 "Mutation" (6 face-adapted operators), 其他 3 个 0 行
代码. 这里补齐, paper claim 才能立得住.

参考实现:
  external/DARWIN/evolution/{genetic,reflective,external,gan}_evolution.py
  external/DARWIN/config/prompts.py {CROSSOVER_PROMPT, REFLECTIVE_EVOLUTION_PROMPT, ...}

实现策略:
  - 全部 LLM-only, 不需要训权重 (method 1 baseline 范畴)
  - 共用 simple_seed_library.SimpleSeedLibrary 作种子池
  - face-forgery-adapted (DARWIN 是 LLM jailbreak, 这里是 face deepfake chain)
"""
from __future__ import annotations
import json
import re
import random
import logging
import hashlib
from dataclasses import dataclass, field
from typing import Optional

from viviai_client import ViviClient
from simple_seed_library import SimpleSeedLibrary, _chain_key
from simple_baseline import MUTATION_OPERATORS_V1, apply_mutation

_log = logging.getLogger(__name__)


# ────────────────────────── Prompts (DARWIN-style) ──────────────────────────

_SYSTEM = (
    "You are part of an authorized internal red-team for face-KYC. Your job "
    "is to evolve attack pipelines that combine face-forgery operators (e.g. "
    "InSwapper-128, SimSwap-256, LivePortrait, jpeg_85, resize_bicubic, "
    "adv_patch_pgd, gpt_image_two). Output only structured JSON when asked."
)


_CROSSOVER_PROMPT = """Given two high-score face-forgery attack pipelines, produce
ONE new hybrid pipeline that combines the strongest parts of each.

Parent A (weighted_score={score_a:.2f}, family={family}):
{chain_a}

Parent B (weighted_score={score_b:.2f}, family={family}):
{chain_b}

Available operators (you MUST pick from this set, do not invent new ones):
{ops}

Constraints:
- chain length 3-7 ops
- keep "face_align" as the first op if present in either parent
- avoid exact duplication of parent A or B

Return STRICTLY JSON:
{{
  "chain":     ["op1", "op2", "op3", ...],
  "rationale": "one sentence on which strengths you carried over"
}}"""


_REFLECTIVE_PROMPT = """A face-forgery attack pipeline JUST FAILED to bypass the
detector. Analyze why, then propose a NEW pipeline that addresses the root cause.

Failed pipeline:        {chain}
Family:                 {family}
Failure observation:    {failure_reason}
Top-3 detector evidence:
{evidence_block}

Available operators (you MUST pick from this set):
{ops}

Return STRICTLY JSON:
{{
  "root_cause":  "one sentence why the pipeline failed",
  "new_chain":   ["op1", "op2", "op3", ...],
  "rationale":   "one sentence why the new chain should bypass this failure mode"
}}"""


_EXTERNAL_INTEL_PROMPT = """You are an external intelligence collector for a face-KYC
red-team agent. Given this brief abstract of a public source (arxiv / GitHub /
HuggingFace), extract attack-chain "strategy cards" the agent could try.

Source title:    {title}
Source snippet:  {snippet}

Available operators (must pick from this set; do not invent):
{ops}

Return STRICTLY JSON (1-3 cards):
{{
  "cards": [
    {{
      "name":      "short cardname",
      "chain":     ["op1", "op2", ...],
      "rationale": "one sentence why this might work against a 2026 face-KYC detector",
      "family":    "frontal_swap | profile_swap | id_diff | reenact | morph | 3d_mask | replay | adv_patch | audio_synth"
    }}
  ]
}}"""


# ────────────────────────── Static intelligence cards ──────────────────────

# Cheap "external intelligence" without scraping live arxiv (would need network +
# parsing); these are real cards extracted from 2024-2026 face-forgery papers
# that we've already read for this project. The intelligence is from real public
# sources but pre-extracted so we don't burn API calls on scraping.
_INTEL_CARDS_SEED = [
    {
        "title": "InstantID: Zero-shot Identity-Preserving Generation (2024)",
        "snippet": "Identity-preserving image generation using InsightFace+IP-Adapter, "
                   "showing strong identity transfer across pose+lighting.",
    },
    {
        "title": "Face-Adapter for Pre-trained Diffusion (CVPR 2024)",
        "snippet": "Face-Adapter conditions Stable Diffusion on a single ID image; "
                   "produces high-fidelity reenactment with arbitrary expression+pose.",
    },
    {
        "title": "LivePortrait (KwaiVGI 2024)",
        "snippet": "Implicit-keypoint-based portrait reenactment, robust to extreme "
                   "head rotations; bypass many traditional liveness checks.",
    },
    {
        "title": "DiffFAS: Diffusion-based Face Anti-Spoofing Attack (NeurIPS 2024)",
        "snippet": "Latent-diffusion-based generation of high-quality spoof faces "
                   "specifically targeting face anti-spoofing CNNs.",
    },
    {
        "title": "AdvFace: PGD-style adversarial patches on FAS models (2023)",
        "snippet": "PGD-trained adversarial patches under print-then-photo "
                   "transformation, robust to camera + screen replay.",
    },
    {
        "title": "DF40: 40-method Deepfake Detection Benchmark (ICLR 2024)",
        "snippet": "Cross-method evaluation shows combinations of SimSwap+morph+"
                   "JPEG-compression families are the hardest to detect.",
    },
]


# ────────────────────────── 1. 遗传进化 (Genetic) ──────────────────────

class GeneticEvolution:
    """Crossover (融合高性能策略) + Mutation (利用 simple_baseline 6 operators)."""

    def __init__(self, client: ViviClient, seed_library: SimpleSeedLibrary,
                  available_ops: list[str],
                  model: str = "gemini-2.5-flash",
                  mutation_rate: float = 0.3,
                  top_k_for_crossover: int = 5):
        self.client = client
        self.seed_library = seed_library
        self.ops = available_ops
        self.model = model
        self.mutation_rate = mutation_rate
        self.top_k = top_k_for_crossover

    def evolve_one(self, family: str) -> Optional[dict]:
        """Run one offspring generation. Returns the new chain dict or None."""
        # 70% crossover, 30% mutation (DARWIN default)
        do_mutation = random.random() < self.mutation_rate
        top_seeds = self.seed_library.get_top_seeds(family, top_k=self.top_k)
        if len(top_seeds) < 2:
            # not enough parents → mutation path only
            do_mutation = True
        if do_mutation and top_seeds:
            return self._mutate(family, top_seeds[0])
        elif not do_mutation and len(top_seeds) >= 2:
            return self._crossover(family, top_seeds[0], top_seeds[1])
        return None

    def _mutate(self, family: str, parent_seed: dict) -> Optional[dict]:
        """Apply random face-adapted mutation operator."""
        mut_name = random.choice(list(MUTATION_OPERATORS_V1.keys()))
        new_chain = apply_mutation(parent_seed["chain"], mut_name, family, self.ops)
        if new_chain == parent_seed["chain"]:
            return None
        return {
            "family": family, "chain": new_chain, "source": "genetic_mutation",
            "mutation": mut_name, "parent_chain_id": parent_seed["chain_id"],
        }

    def _crossover(self, family: str, a: dict, b: dict) -> Optional[dict]:
        """LLM-based crossover (DARWIN CROSSOVER_PROMPT verbatim adaptation)."""
        prompt = _CROSSOVER_PROMPT.format(
            score_a=a["weighted_score"], family=family,
            chain_a=" → ".join(s.get("tool", "?") for s in a["chain"]),
            score_b=b["weighted_score"],
            chain_b=" → ".join(s.get("tool", "?") for s in b["chain"]),
            ops=", ".join(self.ops),
        )
        try:
            text = self.client.chat_text(self.model, prompt, system=_SYSTEM,
                                          temperature=0.7, max_tokens=300)
        except Exception as e:
            _log.warning(f"crossover LLM failed: {e}"); return None
        try:
            from robustness import parse_json_robust
            parsed = parse_json_robust(text)
        except Exception:
            return None
        chain_tools = parsed.get("chain", [])
        if not chain_tools: return None
        # filter to available ops
        chain_tools = [t for t in chain_tools if t in self.ops]
        if len(chain_tools) < 2: return None
        new_chain = [{"tool": t, "params": {}} for t in chain_tools]
        return {
            "family": family, "chain": new_chain, "source": "genetic_crossover",
            "rationale": parsed.get("rationale", "")[:200],
            "parent_chain_ids": [a["chain_id"], b["chain_id"]],
        }


# ────────────────────────── 2. 反思自进化 (Reflective) ──────────────────────

class ReflectiveEvolution:
    """Analyze failed pipeline + propose new one (DARWIN reflective_evolution.py)."""

    def __init__(self, client: ViviClient, available_ops: list[str],
                  model: str = "gemini-2.5-flash"):
        self.client = client
        self.ops = available_ops
        self.model = model

    def reflect_and_propose(self, failed_chain: list, family: str,
                              failure_reason: str = "",
                              tier1_evidence: Optional[dict] = None) -> Optional[dict]:
        """Returns a new chain proposal + root_cause analysis."""
        ev = tier1_evidence or {}
        evidence_block = "\n".join(
            f"  - {k}: {v:.4f}" for k, v in ev.items()
            if isinstance(v, (int, float)) and v != -1.0
        ) or "  (none)"
        prompt = _REFLECTIVE_PROMPT.format(
            chain=" → ".join(s.get("tool", "?") for s in failed_chain),
            family=family,
            failure_reason=failure_reason[:200] or "detector caught it",
            evidence_block=evidence_block,
            ops=", ".join(self.ops),
        )
        try:
            text = self.client.chat_text(self.model, prompt, system=_SYSTEM,
                                          temperature=0.5, max_tokens=400)
        except Exception as e:
            _log.warning(f"reflective LLM failed: {e}"); return None
        try:
            from robustness import parse_json_robust
            parsed = parse_json_robust(text)
        except Exception:
            return None
        chain_tools = parsed.get("new_chain", [])
        chain_tools = [t for t in chain_tools if t in self.ops]
        if len(chain_tools) < 2: return None
        new_chain = [{"tool": t, "params": {}} for t in chain_tools]
        return {
            "family": family, "chain": new_chain, "source": "reflective",
            "root_cause": parsed.get("root_cause", "")[:200],
            "rationale": parsed.get("rationale", "")[:200],
        }


# ────────────────────────── 3. 外部情报收集 (External Intelligence) ──────────

class ExternalIntelligence:
    """Extract attack-chain ideas from public sources (papers/GitHub/HF).

    Lightweight version: uses pre-curated intelligence cards from 2024-2026
    face-forgery papers (would need network for live scraping); LLM extracts
    chains using DARWIN-style EXTERNAL_STRATEGY_REVIEW_PROMPT.
    """

    def __init__(self, client: ViviClient, available_ops: list[str],
                  model: str = "gemini-2.5-flash"):
        self.client = client
        self.ops = available_ops
        self.model = model
        self._intel_pool = list(_INTEL_CARDS_SEED)

    def collect_one(self, force_card: Optional[dict] = None) -> list[dict]:
        """Returns list of {family, chain, source='external_intel', rationale}."""
        card = force_card or random.choice(self._intel_pool)
        prompt = _EXTERNAL_INTEL_PROMPT.format(
            title=card["title"], snippet=card["snippet"],
            ops=", ".join(self.ops),
        )
        try:
            text = self.client.chat_text(self.model, prompt, system=_SYSTEM,
                                          temperature=0.6, max_tokens=500)
        except Exception as e:
            _log.warning(f"external intel LLM failed: {e}"); return []
        try:
            from robustness import parse_json_robust
            parsed = parse_json_robust(text)
        except Exception:
            return []
        cards = parsed.get("cards", []) or []
        out = []
        for c in cards[:3]:
            chain_tools = [t for t in c.get("chain", []) if t in self.ops]
            if len(chain_tools) < 2: continue
            out.append({
                "family": c.get("family", "frontal_swap"),
                "chain": [{"tool": t, "params": {}} for t in chain_tools],
                "source": "external_intel",
                "name": c.get("name", "")[:80],
                "rationale": c.get("rationale", "")[:200],
                "intel_source": card["title"],
            })
        return out


# ────────────────────────── 4. GAN式对抗共进化 (GAN Co-Evolution) ──────────

@dataclass
class DetectorUpgradePlan:
    should_upgrade: bool
    reason: str
    next_detector_hint: str = ""
    recent_bypass_rate: float = 0.0
    threshold: float = 0.5


class GANCoEvolution:
    """Detector progression / model升级 trigger.

    DARWIN: 当 attack 长期成功率 > threshold → upgrade target model.
    For face-forgery: 当 global bypass rate > 0.5 N 轮持续 → 提议升级 detector.

    Our progression (weak → strong):
      gemini-2.5-flash → gemini-3-pro-preview → FakeVLM-base → FakeVLM+LoRA(R0) → R1 → R2 → R+1
    """
    DETECTOR_PROGRESSION = [
        "gemini-2.5-flash",
        "gemini-3-pro-preview",
        "fakevlm_base",
        "fakevlm_lora_r0",
        "fakevlm_lora_r1",
        "fakevlm_lora_r2",
        "fakevlm_lora_r3+",
    ]

    def __init__(self, success_threshold: float = 0.5, min_attacks: int = 10):
        self.success_threshold = success_threshold
        self.min_attacks = min_attacks

    def evaluate(self, current_detector: str,
                  recent_outcomes: list[dict]) -> DetectorUpgradePlan:
        """recent_outcomes: list of {bypass: bool, family: str}.
        If bypass rate > threshold over min_attacks → suggest upgrade."""
        n = len(recent_outcomes)
        if n < self.min_attacks:
            return DetectorUpgradePlan(
                should_upgrade=False,
                reason=f"too few attacks ({n} < {self.min_attacks})",
                recent_bypass_rate=0.0, threshold=self.success_threshold,
            )
        n_bypass = sum(1 for o in recent_outcomes if o.get("bypass"))
        rate = n_bypass / n
        if rate < self.success_threshold:
            return DetectorUpgradePlan(
                should_upgrade=False,
                reason=f"bypass rate {rate:.2%} < threshold {self.success_threshold:.0%}",
                recent_bypass_rate=rate, threshold=self.success_threshold,
            )
        # find current in progression
        try:
            cur_idx = self.DETECTOR_PROGRESSION.index(current_detector)
        except ValueError:
            cur_idx = -1
        if cur_idx + 1 >= len(self.DETECTOR_PROGRESSION):
            return DetectorUpgradePlan(
                should_upgrade=False,
                reason=f"already on strongest detector {current_detector}",
                recent_bypass_rate=rate, threshold=self.success_threshold,
            )
        next_det = self.DETECTOR_PROGRESSION[cur_idx + 1]
        return DetectorUpgradePlan(
            should_upgrade=True,
            reason=f"bypass rate {rate:.2%} ≥ {self.success_threshold:.0%} over {n} attacks",
            next_detector_hint=next_det,
            recent_bypass_rate=rate, threshold=self.success_threshold,
        )


# ────────────────────────── Top-level orchestrator ─────────────────────────

class FourEvolutionsOrchestrator:
    """Run all 4 evolution mechanisms in sequence for one round."""

    def __init__(self, client: ViviClient, seed_library: SimpleSeedLibrary,
                  available_ops: list[str]):
        self.client = client
        self.seed_library = seed_library
        self.ops = available_ops
        self.genetic = GeneticEvolution(client, seed_library, available_ops)
        self.reflective = ReflectiveEvolution(client, available_ops)
        self.external = ExternalIntelligence(client, available_ops)
        self.gan = GANCoEvolution()

    def run_round(self, families: list[str],
                   recent_failures: list[dict] | None = None,
                   recent_outcomes: list[dict] | None = None,
                   current_detector: str = "gemini-2.5-flash",
                   n_genetic_per_family: int = 1,
                   n_reflective: int = 1,
                   n_external: int = 1) -> dict:
        """Returns stats on what each mechanism produced."""
        stats = {"genetic_added": 0, "reflective_added": 0, "external_added": 0,
                 "gan_upgrade_suggested": False, "details": []}
        # 1. Genetic per family
        for fam in families[:5]:  # cap families to limit cost
            for _ in range(n_genetic_per_family):
                cand = self.genetic.evolve_one(fam)
                if cand:
                    cid = self.seed_library.promote_chain(
                        family=fam, chain=cand["chain"],
                        four_dim={"weighted": 0.5, "attack_success": 0.5,
                                  "coverage": 0.6, "generalization": 0.5,
                                  "defense_evasion": 0.5},
                        source=cand["source"],
                        parent_chain_id=cand.get("parent_chain_id", ""),
                    )
                    if cid: stats["genetic_added"] += 1
                    stats["details"].append({"mech": "genetic", "result": cand.get("source"), "added": bool(cid)})
        # 2. Reflective from recent failures
        for fail in (recent_failures or [])[:n_reflective]:
            cand = self.reflective.reflect_and_propose(
                fail.get("chain", []), fail.get("family", "frontal_swap"),
                failure_reason=fail.get("reason", ""),
                tier1_evidence=fail.get("tier1", {}),
            )
            if cand:
                cid = self.seed_library.promote_chain(
                    family=cand["family"], chain=cand["chain"],
                    four_dim={"weighted": 0.55, "attack_success": 0.5,
                              "coverage": 0.7, "generalization": 0.5,
                              "defense_evasion": 0.6},
                    source="reflective",
                )
                if cid: stats["reflective_added"] += 1
                stats["details"].append({"mech": "reflective", "root_cause": cand.get("root_cause"), "added": bool(cid)})
        # 3. External intelligence
        for _ in range(n_external):
            new_cards = self.external.collect_one()
            for c in new_cards:
                cid = self.seed_library.promote_chain(
                    family=c["family"], chain=c["chain"],
                    four_dim={"weighted": 0.5, "attack_success": 0.5,
                              "coverage": 0.8, "generalization": 0.6,
                              "defense_evasion": 0.5},
                    source="external_intel",
                )
                if cid: stats["external_added"] += 1
                stats["details"].append({"mech": "external", "intel_source": c.get("intel_source"), "added": bool(cid)})
        # 4. GAN co-evolution: suggest detector upgrade
        plan = self.gan.evaluate(current_detector, recent_outcomes or [])
        stats["gan_upgrade_suggested"] = plan.should_upgrade
        stats["gan_plan"] = {
            "should_upgrade": plan.should_upgrade, "reason": plan.reason,
            "next_detector_hint": plan.next_detector_hint,
            "recent_bypass_rate": plan.recent_bypass_rate,
        }
        return stats


# ────────────────────────── smoke ──────────────────────────────

if __name__ == "__main__":
    import tempfile, os
    logging.basicConfig(level=logging.INFO)
    api_key = os.environ["VIVIAI_KEY"]  # set VIVIAI_KEY env var; key not committed
    client = ViviClient(api_key=api_key)
    with tempfile.TemporaryDirectory() as td:
        lib = SimpleSeedLibrary(db_path=f"{td}/seeds.db")
        # bootstrap 3 chains
        lib.bootstrap_from_brief_chains("frontal_swap", [
            ["face_align", "inswapper_128_local"],
            ["face_align", "simswap_256_local", "jpeg_85"],
            ["face_align", "inswapper_128_local", "gpt_image_two", "resize_bicubic"],
        ])
        ops = ["face_align", "inswapper_128_local", "simswap_256_local",
               "liveportrait", "gpt_image_two", "jpeg_85", "resize_bicubic",
               "adv_patch_pgd"]

        orch = FourEvolutionsOrchestrator(client, lib, ops)
        # simulated recent failures + outcomes
        fails = [{"chain": [{"tool": "face_align"}, {"tool": "inswapper_128_local"}],
                  "family": "frontal_swap",
                  "reason": "ArcFace ID-sim too low (0.32)",
                  "tier1": {"arcface_id_sim": 0.32, "fft_artifact_score": 0.18}}]
        outcomes = [{"bypass": True} for _ in range(7)] + [{"bypass": False} for _ in range(3)]

        stats = orch.run_round(
            families=["frontal_swap", "reenact"],
            recent_failures=fails,
            recent_outcomes=outcomes,
            current_detector="gemini-2.5-flash",
            n_genetic_per_family=1, n_reflective=1, n_external=1,
        )
        print("\n=== 4-evolution round ===")
        print(f"  genetic_added:    {stats['genetic_added']}")
        print(f"  reflective_added: {stats['reflective_added']}")
        print(f"  external_added:   {stats['external_added']}")
        print(f"  gan plan: {stats['gan_plan']}")
        print("\n  details:")
        for d in stats["details"]:
            print(f"    {d}")
        print(f"\n  seed_library stats: {lib.stats()}")
