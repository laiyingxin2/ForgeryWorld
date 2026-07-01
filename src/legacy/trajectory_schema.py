"""Trajectory schema (pydantic) — shared by Baseline #1 (内部文章 port) and Baseline #2 (proposed).

DESIGN_V3.md §5 锁定. 字段命名按"未来能直接喂 Qwen2.5-VL SFT" 设计.

用法:
    from trajectory_schema import Trajectory, Brief, ExecutionStep, Verdicts
    t = Trajectory(
        trajectory_id="r0_f0_b1_g3",
        round_id=0,
        baseline="v2",
        attack_family="frontal_swap",
        ...
    )
    t.save_jsonl("outputs/trajectories/r0.jsonl")  # 自动 append

读取:
    for t in Trajectory.iter_jsonl("outputs/trajectories/r0.jsonl"):
        ...
"""
from __future__ import annotations
import json
import hashlib
import time
from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass, field, asdict


# ────────────────────────── Sub-schemas ──────────────────────────────

@dataclass
class Brief:
    """Layer 2 出题组输出的 forgery 任务卡."""
    src_face_path: str
    tgt_face_path: Optional[str] = None
    attack_class: str = ""              # frontal_id_replace / profile_swap / ...
    suggested_chain: list = field(default_factory=list)
    params_hints: dict = field(default_factory=dict)
    src_arcface_emb: Optional[list] = None
    tgt_arcface_emb: Optional[list] = None
    brief_text: str = ""                # natural-language 描述, 主管用
    generator_model: str = ""           # which LLM produced this brief


@dataclass
class SkillLookup:
    """Layer 3 retrieval 记录."""
    family_id: int = 0
    family_name: str = ""
    S_k_version: str = "v0"
    E_k_retrieved_ids: list = field(default_factory=list)
    prioritized_weight: float = 0.0     # Ace-Skill Eq.4 w_t(x_i)


@dataclass
class LookaheadCandidate:
    """Layer 4 k=3 sampling 每个候选."""
    pipeline: list = field(default_factory=list)   # list of {tool, params}
    proxy_score: float = 0.0
    selected: bool = False                          # whether this was executed


@dataclass
class ExecutionStep:
    """Layer 4 实际执行的每一步."""
    step: int = 0
    tool: str = ""                      # operator name
    params: dict = field(default_factory=dict)
    input_path: str = ""
    output_path: str = ""
    tier1_metrics: dict = field(default_factory=dict)  # ArcFace/SSIM/NIQE/FFT
    duration_sec: float = 0.0
    error: Optional[str] = None


@dataclass
class Verdicts:
    """Layer 5 3-tier sandbox 输出, 直接来自 sandbox.SandboxVerdict.asdict()."""
    sandbox_pass: bool = False
    bypass_confirmed_by: list = field(default_factory=list)
    tier1: dict = field(default_factory=dict)
    tier2: dict = field(default_factory=dict)
    tier3: Optional[dict] = None
    cost_usd: float = 0.0
    detector_signature: str = ""


@dataclass
class AttributionStep:
    """Layer 6 AgentEvolver SA 单步标签."""
    step: int = 0
    label: Literal["GOOD", "BAD"] = "GOOD"
    reason: str = ""
    r_attr: float = 0.0          # raw {+1, -1}


@dataclass
class CompositeReward:
    """Layer 6 复合 reward."""
    r_attr: list = field(default_factory=list)   # per-step
    r_out: float = 0.0                           # trajectory-level outcome
    alpha: float = 0.1                           # AgentEvolver paper
    composite_per_step: list = field(default_factory=list)
    advantage: float = 0.0                       # A_t = Σ r̂_k


@dataclass
class DefenderExport:
    """Layer 10 Lv5 模型更新接口."""
    image_path: str = ""
    label: dict = field(default_factory=dict)    # {is_fake: bool, family: str}
    forensic_cot: str = ""                       # Tier-3 critique → CoT label
    ready_for_sft: bool = False
    weight: float = 1.0                           # 难度加权 (hard negative 给更高)


# ────────────────────────── Main Trajectory ──────────────────────────

@dataclass
class Trajectory:
    """完整一次 attack rollout. SQLite / jsonl 直接序列化用."""
    # 标识
    trajectory_id: str = ""
    round_id: int = 0
    baseline: Literal["v1", "v2"] = "v2"
    attack_family: str = ""

    # 签名 (区分代际)
    policy_signature: str = ""               # e.g. "gemini-2.5-flash/T=0.7/v3"
    detector_signature: str = ""

    # 各 layer 数据
    brief: Optional[Brief] = None
    skill_lookup: Optional[SkillLookup] = None
    lookahead_candidates: list = field(default_factory=list)
    execution: list = field(default_factory=list)
    verdicts: Optional[Verdicts] = None
    attribution: list = field(default_factory=list)
    composite_reward: Optional[CompositeReward] = None

    # 数据流路由 (Layer 7 UI-TARS-2)
    data_route: Literal["SFT", "CT", "DROP"] = "DROP"
    jacquard_dedupe_key: str = ""

    # 主管抽取的 skill (Layer 5)
    skill_extracted: str = ""

    # Defender 输出接口 (Layer 10)
    defender_export: Optional[DefenderExport] = None

    # Meta
    cost_usd: float = 0.0
    timestamp: float = 0.0
    version_hash: str = ""

    # ───── 工具方法 ─────

    def to_dict(self) -> dict:
        """Recursive asdict — 处理嵌套 dataclass + None."""
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Trajectory":
        """Reconstruct from dict (e.g. loaded from jsonl)."""
        # 重建嵌套
        brief = Brief(**d["brief"]) if d.get("brief") else None
        skill_lookup = SkillLookup(**d["skill_lookup"]) if d.get("skill_lookup") else None
        lookahead = [LookaheadCandidate(**c) for c in d.get("lookahead_candidates", [])]
        execution = [ExecutionStep(**s) for s in d.get("execution", [])]
        verdicts = Verdicts(**d["verdicts"]) if d.get("verdicts") else None
        attribution = [AttributionStep(**s) for s in d.get("attribution", [])]
        composite = CompositeReward(**d["composite_reward"]) if d.get("composite_reward") else None
        defender = DefenderExport(**d["defender_export"]) if d.get("defender_export") else None

        meta = {k: v for k, v in d.items() if k not in {
            "brief", "skill_lookup", "lookahead_candidates", "execution",
            "verdicts", "attribution", "composite_reward", "defender_export"
        }}
        return cls(
            **meta,
            brief=brief,
            skill_lookup=skill_lookup,
            lookahead_candidates=lookahead,
            execution=execution,
            verdicts=verdicts,
            attribution=attribution,
            composite_reward=composite,
            defender_export=defender,
        )

    def save_jsonl(self, path: str | Path):
        """Append-only jsonl write. 一行一条 trajectory."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(self.to_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def iter_jsonl(cls, path: str | Path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield cls.from_dict(json.loads(line))

    def compute_hash(self) -> str:
        """Content hash for Jacquard dedupe."""
        key_str = (
            f"{self.attack_family}|"
            f"{[(s.tool, str(s.params)) for s in self.execution]}|"
            f"{self.verdicts.sandbox_pass if self.verdicts else None}"
        )
        return hashlib.md5(key_str.encode()).hexdigest()


# ────────────────────────── Helper factories ─────────────────────────

def new_trajectory_id(round_id: int, family_idx: int, brief_idx: int, rollout_idx: int) -> str:
    return f"r{round_id}_f{family_idx}_b{brief_idx}_g{rollout_idx}"


def attack_family_list() -> list[str]:
    """K=9 default attack families. Layer 1 Markov state space."""
    return [
        "frontal_swap",   # 0 — InSwapper/SimSwap on frontal faces
        "profile_swap",   # 1 — same but on profile
        "id_diff",        # 2 — InstantID/PuLID/nanobanana ID-preserve
        "reenact",        # 3 — LivePortrait/FaceVid2Vid head reenact
        "morph",          # 4 — StyleGAN morph
        "3d_mask",        # 5 — DECA/FLAME 3D mask synth
        "replay",         # 6 — screen replay + Moiré
        "adv_patch",      # 7 — PGD on FAS CNN
        "audio_synth",    # 8 — XTTS voice clone (for voice-prompted liveness)
    ]


# ────────────────────────── Smoke test ───────────────────────────────

if __name__ == "__main__":
    # round 0, family 0 (frontal_swap), brief 0, rollout 3
    t = Trajectory(
        trajectory_id=new_trajectory_id(0, 0, 0, 3),
        round_id=0,
        baseline="v2",
        attack_family="frontal_swap",
        policy_signature="gemini-2.5-flash/T=0.7/v3",
        detector_signature="tier1_func+tier2_gemini-2.5-flash",
    )
    t.brief = Brief(
        src_face_path="/tmp/src.png",
        attack_class="frontal_id_replace",
        suggested_chain=["face_align", "inswapper_128", "jpeg_85"],
        brief_text="Frontal female, EU lighting, replace with target ID",
        generator_model="gemini-2.5-flash",
    )
    t.skill_lookup = SkillLookup(family_id=0, family_name="frontal_swap",
                                  S_k_version="v0", prioritized_weight=0.83)
    t.execution = [
        ExecutionStep(step=0, tool="face_align", params={},
                      input_path="/tmp/src.png", output_path="/tmp/r0_step0.png",
                      tier1_metrics={"arcface_id_sim": 0.92}),
        ExecutionStep(step=1, tool="inswapper_128", params={"blend": 0.6},
                      input_path="/tmp/r0_step0.png", output_path="/tmp/r0_step1.png",
                      tier1_metrics={"arcface_id_sim": 0.71}),
    ]
    t.verdicts = Verdicts(
        sandbox_pass=True,
        bypass_confirmed_by=["tier2"],
        tier1={"arcface_id_sim": 0.71, "ssim_vs_src": 0.78, "niqe": 7.3},
        tier2={"model": "gemini-2.5-flash", "is_fake": False, "confidence": 0.31,
               "reasoning": "appears authentic..."},
        cost_usd=0.0015,
    )
    t.attribution = [
        AttributionStep(step=0, label="GOOD", reason="alignment prereq"),
        AttributionStep(step=1, label="GOOD", reason="identity gap created"),
    ]
    t.composite_reward = CompositeReward(
        r_attr=[1.0, 1.0], r_out=1.0, alpha=0.1,
        composite_per_step=[0.1 + 0, 0.1 + 1.0], advantage=1.1,
    )
    t.data_route = "SFT"
    t.jacquard_dedupe_key = t.compute_hash()
    t.defender_export = DefenderExport(
        image_path="/tmp/r0_step1.png",
        label={"is_fake": True, "family": "frontal_swap"},
        forensic_cot="The image bears subtle blending artifacts at the jaw...",
        ready_for_sft=True, weight=1.0,
    )
    t.timestamp = time.time()
    t.cost_usd = 0.0015

    # 写 + 读 round-trip
    out = "/tmp/sandbox_smoke_traj.jsonl"
    Path(out).unlink(missing_ok=True)
    t.save_jsonl(out)
    [t2] = list(Trajectory.iter_jsonl(out))
    assert t2.trajectory_id == t.trajectory_id
    assert t2.verdicts.sandbox_pass == t.verdicts.sandbox_pass
    assert len(t2.execution) == 2
    print("✓ trajectory_schema round-trip OK")
    print(f"  hash = {t2.jacquard_dedupe_key}")
    print(f"  attack_family list (K={len(attack_family_list())}) = {attack_family_list()}")
