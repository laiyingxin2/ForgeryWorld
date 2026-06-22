"""9-Family Attack Agent Pool (内部文章图6 verbatim).

内部文章: "针对9类风险, 创建对应的9个攻击 agent. 定义中为了发挥 agent 的自主性,
只定义攻击的 agent 目标、攻击的风险和部分攻击手法, 而非具体对话内容".

之前实现: 1 个 setter + family selector — 9 个 family 都用同一个 system prompt.
现在改: 每个 family 独立 system prompt + 自己的 skill_doc + 经验池.

设计:
  - FamilyAttackAgent: 单个 family 的 agent state (goal + risks + partial methods)
  - FamilyAttackAgentPool: 持有 9 个 agent, route 按 family
  - route() 给 setter 调用前先拿到 family-specific prompt
  - update_experience() rollout 后回馈给特定 family agent
"""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

_log = logging.getLogger(__name__)


# 9 family (per DESIGN_V3 + 内部文章) face-adapted attack agent definitions
# 内部文章 ASB benchmark 是 LLM agent attack (TOOL_MISUSE/PROMPT_INJECTION 等);
# 我们 face-KYC 化, 9 个 attack family:
_AGENT_DEFINITIONS: dict[str, dict] = {
    "frontal_swap": {
        "goal": "Generate frontal face-swap attacks that bypass face-KYC verification on the most common (frontal portrait) scenario.",
        "risks": "ArcFace ID-cosine drift, JPEG-block visible seam, eye-region asymmetry, hairline halo.",
        "partial_methods": "InSwapper-128 / SimSwap-256 + jpeg_85 + optional Adv-patch on FAS CNN.",
    },
    "profile_swap": {
        "goal": "Generate side-angle (15°-60°) face-swap attacks that survive landmark-consistency checks.",
        "risks": "3D-warp seam at ear/jaw, depth inconsistency, lighting from off-axis.",
        "partial_methods": "SimSwap-256 (better for profile) + LivePortrait re-pose + resize_bicubic.",
    },
    "id_diff": {
        "goal": "Generate ID-preserved high-fidelity edits (e.g., glasses, makeup, age) that retain ArcFace but bypass texture-based detectors.",
        "risks": "Frequency artifacts from diffusion model, over-smoothed skin, glasses reflection inconsistency.",
        "partial_methods": "GPT-image-2 inpainting + InstantID-style identity reinforce + JPEG.",
    },
    "reenact": {
        "goal": "Generate driving-source-style reenactment (smile→neutral→talk) that fakes liveness motion.",
        "risks": "Inter-frame jitter, mouth-cavity teeth artifacts, eye-blink timing too regular.",
        "partial_methods": "LivePortrait + FaceVid2Vid + post-process jitter smoothing.",
    },
    "morph": {
        "goal": "Generate face-morph attacks (blend src+tgt identities) for ePassport-style verification.",
        "risks": "Inner-face landmark interpolation visible, eye-color blend, dual-identity ghost artifact.",
        "partial_methods": "StyleGAN-morph + manual mask blend + GFPGAN clean-up.",
    },
    "3d_mask": {
        "goal": "Render 3D-mask attacks (DECA / FLAME mesh + texture) that fake depth structure.",
        "risks": "Inner-mouth no shadow, ear cartilage flat, eye specular static.",
        "partial_methods": "DECA-FLAME 3D mesh + per-vertex texture transfer + screen-print simulation.",
    },
    "replay": {
        "goal": "Simulate screen-replay (display → camera) attacks with moiré + color shift to bypass liveness.",
        "risks": "Moiré pattern detection, glare from screen reflection, refresh-rate flicker on long capture.",
        "partial_methods": "screen-replay simulator + Moiré pattern injection + recompress + optional gpt_image_two cleanup.",
    },
    "adv_patch": {
        "goal": "Apply adversarial patches (PGD on FAS CNN) to flip detector decision while keeping image visually natural.",
        "risks": "Visible patch under inspection, fails when detector model unknown, gradient-leak transferability bound.",
        "partial_methods": "torchattacks PGD on a public FAS CNN + patch in cheek/forehead region.",
    },
    "audio_synth": {
        "goal": "Synthesize voice (XTTS) + lip-sync for multi-modal KYC (challenge-response) bypass.",
        "risks": "Phoneme-viseme mismatch, prosody flatness, audio-visual sync drift.",
        "partial_methods": "XTTS clone + LivePortrait lip-sync + audio post-process.",
    },
}


@dataclass
class FamilyAttackAgent:
    family: str
    goal: str
    risks: str
    partial_methods: str
    skill_doc: str = ""          # accumulated SOP (跨 round skill 库)
    experience_log: list = field(default_factory=list)  # list of {chain, bypass, reasoning}
    total_attempts: int = 0
    total_bypasses: int = 0

    @property
    def bypass_rate(self) -> float:
        return self.total_bypasses / max(self.total_attempts, 1)

    def to_system_prompt(self) -> str:
        """Per-family system prompt for the setter LLM (内部文章图6 verbatim)."""
        # last 3 experiences for in-context learning
        exp_block = ""
        if self.experience_log:
            exp_block = "\n\nRecent attempts (last 3):\n" + "\n".join(
                f"  - chain: {e.get('chain_str', '?')} → bypass={e.get('bypass', False)}"
                for e in self.experience_log[-3:]
            )
        return (
            f"You are the dedicated attack agent for family **{self.family}** "
            f"in an authorized internal red-team for face-KYC.\n\n"
            f"Goal:            {self.goal}\n"
            f"Known risks:     {self.risks}\n"
            f"Partial methods: {self.partial_methods}\n\n"
            f"Your bypass rate so far: {self.bypass_rate:.2%} "
            f"({self.total_bypasses}/{self.total_attempts})\n\n"
            f"Family-specific SOP (accumulated):\n{self.skill_doc or '(empty)'}\n"
            f"{exp_block}\n\n"
            f"Your job: produce attack briefs SPECIFIC to this family's risks. "
            f"Don't drift into other families' techniques. Output requested JSON only."
        )

    def update_experience(self, chain: list, bypass: bool, reasoning: str = "") -> None:
        chain_str = " → ".join(s.get("tool", "?") for s in chain) if chain else ""
        self.experience_log.append({
            "chain_str": chain_str, "bypass": bypass,
            "reasoning": reasoning[:200],
        })
        self.total_attempts += 1
        if bypass: self.total_bypasses += 1
        # cap experience_log to last 50 (memory bound)
        if len(self.experience_log) > 50:
            self.experience_log = self.experience_log[-50:]


class FamilyAttackAgentPool:
    """9-agent pool, route by family. Persistent across runs."""

    def __init__(self, base_dir: str | Path = "outputs/family_agents"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.agents: dict[str, FamilyAttackAgent] = {}
        for fam, defn in _AGENT_DEFINITIONS.items():
            self.agents[fam] = FamilyAttackAgent(family=fam, **defn)
        # auto-load persisted state on init
        self.load_all()

    def get(self, family: str) -> FamilyAttackAgent:
        return self.agents.get(family) or self.agents["frontal_swap"]

    def update_experience(self, family: str, chain: list, bypass: bool,
                           reasoning: str = "") -> None:
        ag = self.agents.get(family)
        if ag:
            ag.update_experience(chain, bypass, reasoning)

    def update_skill_doc(self, family: str, new_skill_text: str) -> None:
        ag = self.agents.get(family)
        if ag:
            ag.skill_doc = (ag.skill_doc + "\n\n" + new_skill_text).strip()
            # cap to ~2000 chars (W=1000 word ≈ 5-6k chars; we keep half)
            if len(ag.skill_doc) > 5000:
                ag.skill_doc = "...(truncated)\n" + ag.skill_doc[-5000:]

    def save_all(self) -> None:
        for fam, ag in self.agents.items():
            (self.base_dir / f"{fam}.json").write_text(
                json.dumps(asdict(ag), ensure_ascii=False, indent=2)
            )

    def load_all(self) -> int:
        loaded = 0
        for fam in self.agents:
            p = self.base_dir / f"{fam}.json"
            if not p.exists(): continue
            try:
                d = json.loads(p.read_text())
                ag = self.agents[fam]
                ag.skill_doc = d.get("skill_doc", "")
                ag.experience_log = list(d.get("experience_log", []))[-50:]
                ag.total_attempts = int(d.get("total_attempts", 0))
                ag.total_bypasses = int(d.get("total_bypasses", 0))
                loaded += 1
            except Exception:
                pass
        return loaded

    def stats(self) -> dict:
        return {fam: {"attempts": ag.total_attempts, "bypasses": ag.total_bypasses,
                       "bypass_rate": round(ag.bypass_rate, 3),
                       "skill_doc_len": len(ag.skill_doc),
                       "exp_log_len": len(ag.experience_log)}
                for fam, ag in self.agents.items()}


# ────────────────────────── smoke ──────────────────────────────

if __name__ == "__main__":
    import tempfile
    logging.basicConfig(level=logging.INFO)
    with tempfile.TemporaryDirectory() as td:
        pool = FamilyAttackAgentPool(base_dir=td)
        assert len(pool.agents) == 9
        # update a couple of agents
        pool.update_experience("frontal_swap",
            [{"tool": "face_align"}, {"tool": "inswapper_128_local"}],
            bypass=True, reasoning="JPEG-85 masked the seam.")
        pool.update_experience("frontal_swap",
            [{"tool": "face_align"}, {"tool": "simswap_256_local"}],
            bypass=False, reasoning="Ear cartilage flat detected.")
        pool.update_skill_doc("frontal_swap",
            "## R0\nBlend ≥ 0.7 + jpeg q=85 + face_align affine 512×512 reliable.")
        pool.save_all()
        # reload and verify
        pool2 = FamilyAttackAgentPool(base_dir=td)
        ag = pool2.get("frontal_swap")
        assert ag.total_attempts == 2 and ag.total_bypasses == 1
        assert len(ag.experience_log) == 2
        assert "R0" in ag.skill_doc
        print(f"[9-agent pool smoke] loaded {len(pool2.agents)} agents")
        print(f"  frontal_swap: attempts={ag.total_attempts} bypasses={ag.total_bypasses} "
              f"rate={ag.bypass_rate:.2%}")
        print(f"  system prompt sample (first 300c):")
        print("    " + ag.to_system_prompt()[:300].replace("\n", "\n    "))
        print(f"\n  stats: {pool2.stats()['frontal_swap']}")
        print("\nsmoke PASS")
