"""Main orchestrator — Baseline #1 (mode='v1') / Baseline #2 (mode='v2').

把全部 10 layer 串起来. 参数化 baseline_mode, 共用所有 shared layer:
  Layer 0/2/5/7/8/9/10 都共用 (env / multi-agent / sandbox / data flow / memory / diagnose / export)
  Baseline #1 替换 Layer 1/3/4/6 为 simple_baseline.py 模块
  Baseline #2 用 Layer 1/3/4/6 的 paper-based 实现

Usage:
    from orchestrator import Orchestrator, OrchestratorConfig
    orc = Orchestrator(OrchestratorConfig(baseline_mode="v2", n_rounds=2, n_briefs_per_round=4))
    orc.run()

CLI: python orchestrator.py --mode v2 --rounds 2 --briefs 4 --rollouts 2
"""
from __future__ import annotations
import argparse
import json
import logging
import time
import uuid
import random
from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass, field

import numpy as np

from viviai_client import ViviClient
from sandbox import SandboxVerifier
from dataclasses import asdict
from trajectory_schema import (
    Trajectory, Brief, ExecutionStep, Verdicts, AttributionStep,
    LookaheadCandidate, SkillLookup, DefenderExport, CompositeReward,
    new_trajectory_id, attack_family_list,
)
from markov_family import MarkovFamilySelector, MarkovConfig
from ace_skill_lib import SkillLibrary, AceSkillConfig
from multi_agent_gen import MultiAgentBenchmarkGen, MultiAgentConfig
from pipeline_planner import PipelinePlanner, ToolGraph
from self_attributor import SelfAttributor, SAConfig
from data_flow import DataFlow
from diagnosis_and_export import Diagnoser, DefenderExporter
from simple_baseline import (
    SimpleFamilySelector, SimpleSkillBook,
    SimpleMCTSPlanner, MUTATION_OPERATORS_V1,
)
# ★ Q4 修: 集成 3 个新模块
from reflexion import Reflexion
from reasoning_bank import ReasoningBank
from novelty import NoveltyTracker, NoveltyConfig
# ★ Bug-18 修: OpHealthTracker
from op_health import OpHealthTracker
from operators.api_image import (
    NanoBananaOne, NanoBananaTwo, NanoBananaPro, GptImageTwo,
)
# ★ 2026-06-20 fix: unified real-op library (same one method4 uses). Previously
# M1/M2/M3 mocked every non-API op as identity-pass → forged==source (arcface=1.0)
# → PSEUDO_BYPASS_REJECTED → 0% bypass, no learning gradient. Wire it in below.
from operators import OPERATOR_REGISTRY, resolve_op
try:
    from operators.local_swap import LocalInSwapperOperator
except ImportError:
    LocalInSwapperOperator = None


_log = logging.getLogger(__name__)


def _soft_evade_score(tier1: dict) -> float:
    """连续 bypass-proximity ∈ [0,1]: 越像"骗过 FakeVLM"越高.

    FakeVLM 只给二值标签(无 confidence), 故无法直接拿检测器连续信号.
    这里用 tier1 反取证代理, 对准 FakeVLM 自述的失败模式
    ("skin too smooth / lacks texture / unnatural / misaligned features"):
      - fft_artifact_score 低  → 频域伪影少
      - landmark_consistency 高 → 五官对齐自然(非 misaligned mouths)
      - maniqa(TV proxy)高    → 纹理/细节多(非 overly smooth)
      - niqe(BRISQUE)低       → 自然度高
    纯启发式: 目的是在 bypass=0 时给搜索一个平滑梯度, 非校准指标.
    """
    if not isinstance(tier1, dict) or tier1.get("error"):
        return 0.0
    def clip01(x):
        try: return max(0.0, min(1.0, float(x)))
        except Exception: return 0.0
    fft  = clip01(1.0 - float(tier1.get("fft_artifact_score", 1.0)))
    lmk  = clip01(tier1.get("landmark_consistency", 0.0))
    tex  = clip01(float(tier1.get("maniqa", 0.0)) / 100.0)
    niqe = clip01(1.0 - float(tier1.get("niqe", 100.0)) / 100.0)
    return 0.35 * fft + 0.25 * lmk + 0.20 * tex + 0.20 * niqe


# ────────────────────────── Config ──────────────────────────────────

@dataclass
class OrchestratorConfig:
    baseline_mode: Literal["v1", "v2"] = "v2"
    n_rounds: int = 2
    n_briefs_per_round: int = 4
    n_rollouts_per_brief: int = 2
    output_dir: str = "outputs"
    # ★ Long-horizon (长程) persistence: a SHARED bank dir for cross-scenario skill/
    # markov/reasoning accumulation (Voyager fixed-ckpt pattern). Empty → falls back to
    # output_dir (legacy per-run behavior). When set, the evolutionary bank (markov,
    # skills_v1/v2, reasoning_bank, novelty, family_agents, op_health, seed libs,
    # videoweaver, meta-state, coevo snapshots) loads/saves here while trajectories /
    # reports / data_flow / memory stay per-scenario in output_dir.
    bank_dir: str = ""
    seed: int = 42

    # tier-2 model (sandbox.py)
    tier2_model: str = "gemini-2.5-flash"
    tier3_enabled: bool = False
    # P2: viviai claude-opus-4-7 is 503; gemini-3-pro-preview works for
    # forensic cross-check style prompts (sees pre-computed tier1+tier2 evidence
    # so Google safety classifier is much less likely to refuse than direct
    # attack-brief generation).
    tier3_model: str = "gemini-3-pro-preview"
    # ★ Lv5 switch
    tier2_backend: str = "viviai"   # 'viviai' | 'fakevlm_local'
    fakevlm_endpoint: str = "http://localhost:8000/v1"

    # multi-agent config preset
    multi_agent_preset: Literal["w1_cheap", "w6_full"] = "w1_cheap"
    # 是否每个 brief 后真跑 3-checker median (内部文章 verbatim, P0-B)
    enable_checkers: bool = True
    checker_min_overall: float = 0.0  # if >0, briefs with overall<thresh get logged warning

    # ★ TIER1-M1-3: method 1 也用 Markov+Q-Learning (内部文章图2 verbatim)
    # 之前 method 1 用 SimpleFamilySelector (均匀随机), 跟内部文章不符
    v1_use_markov: bool = True
    # ★ TIER1-M1-2: method 1 启用种子库 (高分回池)
    v1_use_seed_library: bool = True
    v1_seed_promote_threshold: float = 0.65   # weighted_score ≥ this → 入库
    # ★ TIER1-M1-4: method 1 自动从 checker issues 写约束
    v1_auto_constraint_from_checker: bool = True

    # ★ Tier 2-1: UI-Voyager 失败归因 + 成功轨迹纠正 (post-failure)
    enable_ui_voyager_correction: bool = True
    # ★ Tier 2-2: VideoWeaver Composition+Creator 2-layer skill
    enable_videoweaver_skills: bool = True
    # ★ Tier 2-3 (Method 3 挂载+一起更新): co-evolution loop coordinator
    enable_method3_coevolution: bool = True
    coevolution_coupling_strength: float = 1.0  # 0..1, how aggressively to push defender state to attacker

    # 是否真的执行 API attack op (False = mock 模式, 只算 metrics, 不调 nanobanana 烧钱)
    execute_api_ops: bool = True

    # **新加**: 仅暴露 API ops 给 setter (避免 setter 选择本地未装的 op)
    api_only_ops: bool = False

    # ★ Q4 修: 3 个新模块开关 (Baseline #2 改进版)
    enable_reflexion: bool = True       # intra-rollout 反思
    enable_reasoning_bank: bool = True  # 平行 strategy-rule 流
    enable_novelty: bool = True         # 反 mode collapse novelty term
    reflexion_max_per_rollout: int = 2  # 限制省钱
    # ★ 连续 bypass-proximity 奖励: 二值 bypass=0 时给搜索一个朝"反检测"爬的梯度
    # (FakeVLM 只给二值标签, 无连续 confidence; 用 tier1 反取证代理对准其失败模式)
    evade_shaping_weight: float = 0.3   # 0 = 关闭(回退旧二值行为)

    # ★ search finding #2 修: cost budget guard ($47K stall loop)
    cost_budget_usd: float = 5.0        # 单次 orchestrator.run 硬上限
    abort_on_budget_exceeded: bool = True

    # supervisor 抽 Δ𝒮_k 的频率 (每 N 个 rollout 一次)
    supervisor_every: int = 1  # 默认每 rollout 都抽; 跑大 batch 时可调高

    # 是否在 init 时 load 之前的 skill (cross-run 累积关键!)
    persist_skills_across_runs: bool = True

    # **关键**: run_id 是 trajectory_id 前缀, 避免跨 run 碰撞 (SQLite INSERT OR REPLACE silent overwrite)
    # 默认使用 timestamp; 也可以用户指定 (e.g. "run_2026-06-19_v1")
    run_id: str = ""  # 空表示用 timestamp

    # data pool: 真实 face image 路径 list (用作 src_face)
    src_face_paths: list = field(default_factory=list)


# ────────────────────────── Orchestrator ────────────────────────────

class Orchestrator:
    def __init__(self, cfg: OrchestratorConfig):
        self.cfg = cfg
        random.seed(cfg.seed); np.random.seed(cfg.seed)

        # 生成或采用 run_id (解决 BUG A: trajectory_id 跨 run 碰撞)
        if not cfg.run_id:
            cfg.run_id = time.strftime("run-%Y%m%d-%H%M%S")
        self.run_id = cfg.run_id

        # output dirs
        self.out_dir = Path(cfg.output_dir)
        # ★ Long-horizon bank dir (Voyager fixed-ckpt). Shared across outer scenarios so
        # the evolutionary bank accumulates instead of cold-starting each scenario.
        self.bank_dir = Path(cfg.bank_dir) if cfg.bank_dir else self.out_dir
        (self.out_dir / "trajectories").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "skills").mkdir(parents=True, exist_ok=True)
        (self.bank_dir / "markov").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "reports").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "face_attack_outputs").mkdir(parents=True, exist_ok=True)
        if self.bank_dir != self.out_dir:
            _log.info(f"[bank] long-horizon shared bank_dir = {self.bank_dir}")

        # viviai client (single shared)
        import os
        api_key = os.environ["VIVIAI_KEY"]  # set VIVIAI_KEY env var; key not committed
        self.client = ViviClient(api_key=api_key)

        # Shared layers
        self.families = attack_family_list()
        self.sandbox = SandboxVerifier(
            client=self.client,
            tier2_model=cfg.tier2_model,
            tier3_enabled=cfg.tier3_enabled,
            tier3_model=cfg.tier3_model,
            tier2_backend=cfg.tier2_backend,
            fakevlm_endpoint=cfg.fakevlm_endpoint,
        )

        ma_cfg = (MultiAgentConfig.w6_full() if cfg.multi_agent_preset == "w6_full"
                  else MultiAgentConfig.w1_cheap())
        self.multi_agent = MultiAgentBenchmarkGen(client=self.client, config=ma_cfg)

        # ★ M2-P0-1: 9-agent family pool (内部文章图6 verbatim)
        # 每个 family 独立 system prompt + 跨 round 经验日志
        from family_attack_agents import FamilyAttackAgentPool
        self.family_agents = FamilyAttackAgentPool(
            base_dir=str(self.bank_dir / "family_agents"),
        )
        agent_loaded_stats = self.family_agents.stats()
        loaded_count = sum(1 for s in agent_loaded_stats.values() if s["attempts"] > 0)
        if loaded_count > 0:
            _log.info(f"[family_agents] loaded {loaded_count}/9 with prior experience")

        # ★ Tier 2-2: VideoWeaver 2-layer skill (Composition + Creator) — applies to v1+v2
        self.videoweaver_skills = None
        if getattr(cfg, "enable_videoweaver_skills", True):
            try:
                from videoweaver_skills import VideoWeaverSkills
                self.videoweaver_skills = VideoWeaverSkills(
                    db_path=str(self.bank_dir / "videoweaver_skills" / "skills.db"),
                )
            except Exception as e:
                _log.warning(f"[videoweaver] init failed: {e}")

        # ★ Tier 2-3 (Method 3 挂载+一起更新): co-evolution loop coordinator
        # ★ BUG-5 fix: 用 load_latest 跨进程恢复 state (multi_round.sh 每 round 新进程)
        self.coevo = None
        if getattr(cfg, "enable_method3_coevolution", True):
            try:
                from method3_coevolution import CoEvolutionLoop
                coevo_dir = str(self.bank_dir / "coevo_snapshots")
                self.coevo = CoEvolutionLoop.load_latest(
                    out_dir=coevo_dir,
                    vllm_endpoint=cfg.fakevlm_endpoint,
                )
                self.coevo.coupling = getattr(cfg, "coevolution_coupling_strength", 1.0)
            except Exception as e:
                _log.warning(f"[coevo] init failed: {e}")

        self.data_flow = DataFlow(db_path=str(self.out_dir / f"data_flow_{cfg.baseline_mode}.db"))
        self.diagnoser = Diagnoser(self.data_flow)
        self.exporter = DefenderExporter(self.data_flow)

        # ── Layer 8: L1-L5 Memory Hierarchy (内部文章 verbatim) ──
        from memory_hierarchy import MemoryHierarchy, MetaSkill  # noqa: F401
        self.mem = MemoryHierarchy(
            root=str(self.out_dir),
            run_id=getattr(cfg, "run_id", "") or time.strftime("run-%Y%m%d-%H%M%S"),
        )

        # Baseline-specific layers
        if cfg.baseline_mode == "v2":
            # BUG B 修: Markov 矩阵跨 run 持久化
            markov_path = self.bank_dir / "markov" / "markov_state.json"
            if cfg.persist_skills_across_runs and markov_path.exists():
                self.family_selector = MarkovFamilySelector.load(str(markov_path))
                _log.info(f"[markov] loaded from {markov_path}, "
                          f"global_bypass={self.family_selector.global_bypass_rate():.3f}")
            else:
                self.family_selector = MarkovFamilySelector(families=self.families)

            self.skill_lib = SkillLibrary(
                families=self.families, config=AceSkillConfig(),
                client=self.client,
                base_dir=str(self.bank_dir / "skills_v2"),
            )
            # 关键: 跨 run 累积 skill, init 时 load
            if cfg.persist_skills_across_runs:
                self.skill_lib.load_all()
                total_exp = sum(len(p.experiences) for p in self.skill_lib.pools.values())
                _log.info(f"[skill_lib] loaded persistent state: "
                          f"{total_exp} total experiences across {len(self.families)} families")
            self.planner = PipelinePlanner(
                client=self.client, tool_graph=ToolGraph(),
                k_candidates=3, lookahead_depth=2,
                proxy_model=cfg.tier2_model,
            )
            self.attributor = SelfAttributor(
                client=self.client,
                config=SAConfig(attributor_model=cfg.tier2_model),
            )
            # ★ Q4: 3 个新模块 init
            self.reflexion = (
                Reflexion(client=self.client, model=cfg.tier2_model,
                          max_reflections_per_rollout=cfg.reflexion_max_per_rollout)
                if cfg.enable_reflexion else None
            )
            self.reasoning_bank = (
                ReasoningBank(families=self.families, client=self.client,
                              base_dir=str(self.bank_dir / "reasoning_bank"))
                if cfg.enable_reasoning_bank else None
            )
            self.novelty = (
                NoveltyTracker(
                    config=NoveltyConfig(K_history=64, novelty_weight=0.4),
                    persist_path=str(self.bank_dir / "novelty_history.json"),
                ) if cfg.enable_novelty else None
            )
        else:  # v1
            # ★ TIER1-M1-3: method 1 也用 Markov+Q-Learning (内部文章图2)
            if cfg.v1_use_markov:
                markov_path = self.bank_dir / "markov" / "markov_state_v1.json"
                if cfg.persist_skills_across_runs and markov_path.exists():
                    self.family_selector = MarkovFamilySelector.load(str(markov_path))
                    _log.info(f"[markov-v1] loaded from {markov_path}, "
                              f"global_bypass={self.family_selector.global_bypass_rate():.3f}")
                else:
                    self.family_selector = MarkovFamilySelector(families=self.families)
                    _log.info(f"[markov-v1] fresh init (v1 也用 Markov,内部文章图2 verbatim)")
            else:
                self.family_selector = SimpleFamilySelector(families=self.families)
            self.skill_book_v1 = SimpleSkillBook(
                families=self.families,
                base_dir=str(self.bank_dir / "skills_v1"),
            )
            # ★ BUG-V1A 修: v1 也跨 run 累积 skill (用户问 "方法 1 没问题吗")
            if cfg.persist_skills_across_runs:
                self.skill_book_v1.load_all()
                total_constraints = sum(self.skill_book_v1.append_count.values())
                _log.info(f"[skill_book_v1] loaded persistent state: "
                          f"{total_constraints} total constraints across families")
            # ★ TIER1-M1-2: 启用种子库 (DARWIN StrategyPool 模式, 高分回池)
            self.seed_library_v1 = None
            if cfg.v1_use_seed_library:
                from simple_seed_library import SimpleSeedLibrary
                self.seed_library_v1 = SimpleSeedLibrary(
                    db_path=str(self.bank_dir / "seed_library_v1" / "seeds.db"),
                )
                stats = self.seed_library_v1.stats()
                _log.info(f"[seed_library_v1] loaded: {stats}")
            self.mcts_planner = SimpleMCTSPlanner(
                client=self.client, n_iterations=4, proxy_model=cfg.tier2_model,
                seed_library=self.seed_library_v1,
                seed_promote_threshold=cfg.v1_seed_promote_threshold,
            )
            self.planner = None; self.attributor = None; self.skill_lib = None
            self.reflexion = None; self.reasoning_bank = None; self.novelty = None
        # ★ Bug-18: OpHealthTracker — 降 min_calls 到 1, 1 个 503 立即 blacklist
        if not hasattr(self, 'op_health'):
            self.op_health = OpHealthTracker(
                window_per_op=20, min_calls_for_stats=1,
                persist_path=str(self.bank_dir / "op_health.json"),
            )

        # Attack operators (API only for prototype — local ones need install)
        self.api_ops = {
            "nano_banana_one": NanoBananaOne(client=self.client,
                out_dir=str(self.out_dir / "face_attack_outputs")),
            "nano_banana_two": NanoBananaTwo(client=self.client,
                out_dir=str(self.out_dir / "face_attack_outputs")),
            "nano_banana_pro": NanoBananaPro(client=self.client,
                out_dir=str(self.out_dir / "face_attack_outputs")),
            "gpt_image_two": GptImageTwo(client=self.client,
                out_dir=str(self.out_dir / "face_attack_outputs")),
        }
        # ★ 加本地 InSwapper (替代不可用 nano_banana viviai endpoints)
        if LocalInSwapperOperator is not None:
            try:
                self.api_ops["inswapper_128_local"] = LocalInSwapperOperator(
                    out_dir=str(self.out_dir / "face_attack_outputs"),
                )
                _log.info("[orchestrator] LocalInSwapper enabled")
            except Exception as e:
                _log.warning(f"[orchestrator] LocalInSwapper init failed: {e}")

        # ★ P0-D: 2nd 本地 face-swap (SimSwap-256, profile-friendly)
        try:
            from operators.local_simswap import LocalSimSwapOperator as _SS
            self.api_ops["simswap_256_local"] = _SS(
                out_dir=str(self.out_dir / "face_attack_outputs"),
            )
            _log.info("[orchestrator] LocalSimSwap enabled")
        except Exception as e:
            _log.warning(f"[orchestrator] LocalSimSwap init failed: {e}")

        # operator listing for multi_agent prompts
        # --api-only-ops 模式: 只暴露 API ops + shared 给 setter
        if cfg.api_only_ops:
            self.operator_list = [
                "face_align",
                "inswapper_128_local",       # ★ 本地 face-swap #1 (InSwapper, ezioruan ONNX)
                "simswap_256_local",         # ★ P0-D: 本地 face-swap #2 (Chen 2020 SimSwap)
                # nano_banana_pro/two/one 全部 model_not_found 在 viviai
                # 保留 listed 但 OpHealthTracker 自然 blacklist
                "nano_banana_pro", "nano_banana_two", "nano_banana_one",
                "gpt_image_two",
                "jpeg_85", "resize_bicubic",
            ]
        else:
            # ★ 2026-06-20 fix: advertise ONLY ops that exist in OPERATOR_REGISTRY.
            # Old list named "inswapper_128"/"simswap_256"/"facevid2vid"/"replay_sim"
            # which are NOT registry keys (real keys are inswapper_128_local /
            # simswap_256_local / screen_replay_sim) → policy picked phantom names →
            # all mock-passed. Use registry keys so every advertised op is real.
            self.operator_list = sorted(OPERATOR_REGISTRY.keys())

        # ── L8 L3 view: register skill_lib + markov as project-level references ──
        if hasattr(self, "skill_lib") and self.skill_lib is not None:
            self.mem.register_l3_view("skill_lib", self.skill_lib)
        if hasattr(self, "skill_book_v1") and self.skill_book_v1 is not None:
            self.mem.register_l3_view("skill_book_v1", self.skill_book_v1)
        if hasattr(self, "family_selector") and self.family_selector is not None:
            self.mem.register_l3_view("markov", self.family_selector)
        if hasattr(self, "reasoning_bank") and self.reasoning_bank is not None:
            self.mem.register_l3_view("reasoning_bank", self.reasoning_bank)

        # BUG E 修: recent_briefs/outcomes 跨 run 持久化
        self._meta_state_path = self.bank_dir / "orchestrator_meta.json"
        self.recent_briefs: list[Brief] = []
        self.recent_outcomes: list[dict] = []
        self.total_cost = 0.0  # BUG F 修: 累计成本跨 run

        if cfg.persist_skills_across_runs and self._meta_state_path.exists():
            try:
                meta = json.loads(self._meta_state_path.read_text())
                self.recent_outcomes = list(meta.get("recent_outcomes", []))[-30:]
                self.total_cost = float(meta.get("total_cost_cumulative", 0.0))
                # recent_briefs 用简化 list[dict] → Brief
                self.recent_briefs = [
                    Brief(**b) for b in meta.get("recent_briefs", [])[-15:]
                    if isinstance(b, dict)
                ]
                _log.info(f"[meta] loaded: cum_cost=${self.total_cost:.4f}, "
                          f"recent_outcomes={len(self.recent_outcomes)}")
            except Exception as e:
                _log.warning(f"[meta] load failed: {e}, starting fresh")

    # ─────────────── Layer 4 execution (mocked for non-API ops) ──────

    def execute_pipeline(
        self,
        chain: list,
        brief: Brief,
        round_id: int,
        traj_id: str,
        family_name: str = "",  # ★ Q4: for reflexion
    ) -> list[ExecutionStep]:
        """Execute pipeline step-by-step. For non-API ops, mock as identity-pass.

        ★ Q4 加: Reflexion 在每步后触发, 可能改 remaining chain.
        Returns list[ExecutionStep].
        """
        current_img = brief.src_face_path
        steps = []
        mutable_chain = list(chain)  # 拷贝, 可在循环里改
        i = 0
        while i < len(mutable_chain):
            step = mutable_chain[i]
            tool = step.get("tool", "unknown") if isinstance(step, dict) else step
            tool = resolve_op(tool)  # ★ legacy/phantom name -> registry key
            params = step.get("params", {}) if isinstance(step, dict) else {}

            t0 = time.time()
            error = None
            is_mock = False
            if tool in self.api_ops and self.cfg.execute_api_ops:
                op = self.api_ops[tool]
            elif tool in OPERATOR_REGISTRY and self.cfg.execute_api_ops:
                # ★ 2026-06-20 fix: real local op from the unified registry (same
                # library method4 uses). Instantiate per-call like method4 does.
                try:
                    op = OPERATOR_REGISTRY[tool](
                        client=self.client,
                        out_dir=str(self.out_dir / "face_attack_outputs"),
                    )
                except Exception as e:
                    op = None
                    error = f"[OP_INIT_FAIL] {tool}: {str(e)[:120]}"
                    is_mock = True
            else:
                op = None

            if op is not None:
                op_params = {**params}
                if "instruction" not in op_params and brief.brief_text:
                    op_params["instruction"] = (
                        f"Generate a photorealistic face in scene: {brief.brief_text[:200]}"
                    )
                try:
                    result = op.run(src_face_path=current_img, params=op_params, size="1024x1024")
                except Exception as e:
                    result = None
                    error = f"[OP_RUN_FAIL] {tool}: {str(e)[:120]}"
                    is_mock = True
                if result is not None:
                    # ★ Bug-18: record op health
                    self.op_health.record(tool, result.success, error=result.error)
                    if result.success and result.output_path and Path(result.output_path).exists():
                        current_img = result.output_path
                        self.total_cost += getattr(result, "cost_usd", 0.0)
                    else:
                        error = result.error or "op failed"
                        is_mock = True
            elif not error:
                # truly unknown op (not in api_ops nor registry) → identity pass
                is_mock = True
                error = f"[MOCK_UNAVAILABLE_LOCAL_OP] {tool}"

            # Tier-1 metrics on current image (cheap, no API)
            tier1 = {}
            if current_img and Path(current_img).exists():
                try:
                    from sandbox import tier1_function_checks
                    tier1 = tier1_function_checks(current_img, src_face_path=brief.src_face_path)
                except Exception as e:
                    tier1 = {"error": str(e)}

            steps.append(ExecutionStep(
                step=i, tool=tool, params=params,
                input_path=brief.src_face_path if i == 0 else steps[-1].output_path,
                output_path=current_img,
                tier1_metrics=tier1,
                duration_sec=time.time() - t0,
                error=error,
            ))

            # ★ Q4: Reflexion 在每步后触发 (限 max_reflections_per_rollout 次)
            if (self.reflexion and i < len(mutable_chain) - 1
                    and i < self.cfg.reflexion_max_per_rollout):
                traj_so_far = [{"tool": s.tool, "tier1_metrics": s.tier1_metrics}
                               for s in steps]
                remaining = mutable_chain[i + 1:]
                # ★ Bug-18: 把 op health summary 拼到 family_name 让 Reflexion 看到
                op_health_str = self.op_health.get_health_summary(sort_by="rate")
                blacklist = self.op_health.blacklist_failing_ops(max_rate=0.2)
                # 从 available_ops 移除 blacklist
                healthy_ops = [o for o in self.operator_list if o not in blacklist]
                refl = self.reflexion.reflect(
                    step_idx=i + 1,
                    family=family_name + f"\n\nOP HEALTH (success rate, recent calls):\n{op_health_str}\n"
                                          f"BLACKLIST (success rate <20%, AVOID): {blacklist}",
                    brief_text=brief.brief_text,
                    trajectory_so_far=traj_so_far,
                    remaining_chain=remaining,
                    tier1_metrics=tier1,
                    available_ops=healthy_ops,  # ★ Bug-18: 不暴露 broken op 给 Reflexion
                )
                self.total_cost += 0.0015  # reflexion call cost
                if refl and refl.suggested_correction:
                    # 替换 next step (i+1)
                    new_remaining = self.reflexion.apply_correction(
                        remaining, refl, available_ops=self.operator_list,
                    )
                    mutable_chain = mutable_chain[: i + 1] + new_remaining
            i += 1
        return steps

    # ─────────────── One rollout ────────────────────────────────────

    def run_one_rollout(
        self,
        round_id: int,
        family_idx: int,
        brief_idx: int,
        rollout_idx: int,
    ) -> Trajectory:
        family_name = self.families[family_idx]
        # BUG A 修: run_id 前缀避免跨 run 碰撞 (SQLite INSERT OR REPLACE silent overwrite)
        traj_id = f"{self.run_id}_" + new_trajectory_id(round_id, family_idx, brief_idx, rollout_idx)
        _log.info(f"\n─── Rollout {traj_id} family={family_name} ───")

        # Pick src face randomly from pool
        src_face = (random.choice(self.cfg.src_face_paths) if self.cfg.src_face_paths
                    else "")

        # ── Layer 3: skill retrieve (v2) or get doc (v1) ──
        injected_meta_names: list[str] = []   # L4 meta-skills shown to setter this rollout
        if self.cfg.baseline_mode == "v2":
            skill_doc, top_exps = self.skill_lib.retrieve(
                family_name,
                query=f"how to bypass detector for {family_name}",
                top_k=5,
                current_round=round_id,
            )
            # ★ Q4: ReasoningBank retrieve top-3 rules, 注入 brief 上下文
            if self.reasoning_bank:
                top_rules = self.reasoning_bank.retrieve(
                    family_name,
                    query_state_desc=f"face KYC attack on {family_name}",
                    top_k=3,
                    current_round=round_id,
                )
                if top_rules:
                    # 把规则文本拼到 skill_doc 末尾, setter 会读
                    rules_section = "\n\n## ReasoningBank Rules (apply when trigger matches):\n"
                    for r in top_rules:
                        rules_section += (f"- [{r.rule_id}] trigger: {r.trigger_desc[:80]}\n"
                                          f"  → rule: {r.rule_text[:100]}\n")
                    skill_doc = skill_doc + rules_section
                    _log.info(f"  [reasoning_bank] injected {len(top_rules)} rules")

            sl = SkillLookup(
                family_id=family_idx, family_name=family_name,
                S_k_version=f"r{round_id}",
                E_k_retrieved_ids=[e.exp_id for e in top_exps],
                prioritized_weight=float(np.mean([e.applicability_score for e in top_exps])
                                          if top_exps else 0.5),
            )
        else:
            skill_doc = self.skill_book_v1.get_doc(family_name)
            top_exps = []
            sl = SkillLookup(family_id=family_idx, family_name=family_name,
                              S_k_version=f"r{round_id}_v1")

        # ── Layer 2: 出题组 → brief ──
        setter = "setter_a" if rollout_idx % 2 == 0 else "setter_b"
        # ★ Bug-18: 把 op health 注入 skill_doc 让 setter 看到
        op_health_summary = self.op_health.get_health_summary(sort_by="rate")
        blacklist = self.op_health.blacklist_failing_ops(max_rate=0.2)
        healthy_op_list = [o for o in self.operator_list if o not in blacklist]
        if op_health_summary and "no op call" not in op_health_summary:
            skill_doc = skill_doc + (
                f"\n\n## ⚡ OPERATOR HEALTH (current API reliability):\n"
                f"{op_health_summary}\n\n"
                f"★ AVOID these failing operators: {blacklist}\n"
                f"★ Prefer high-success-rate operators."
            )

        # ★ BUG-3 fix: VideoWeaver Composition+Creator 2-layer prior 注入 setter
        # 之前 VideoWeaver 库写入但 setter 不读 → 是 write-only telemetry
        # 现在 setter 看到 "top composition + best creator params" 作 reference
        if (hasattr(self, "videoweaver_skills") and self.videoweaver_skills is not None):
            try:
                vw_rec = self.videoweaver_skills.recommend_brief(family_name)
                if vw_rec:
                    chain_str = " → ".join(
                        f"{c['tool']}({c.get('params', {})})" for c in vw_rec["chain"]
                    )
                    skill_doc = skill_doc + (
                        f"\n\n## 📘 VideoWeaver 2-layer prior for {family_name}\n"
                        f"### Composition (best chain shape so far):\n"
                        f"  {chain_str}\n"
                        f"### Rationale: {vw_rec['rationale']}\n"
                        f"★ You may use this as starting structure but feel free to "
                        f"replace any op or params to explore new chains."
                    )
            except Exception as e:
                _log.warning(f"  videoweaver prior failed: {e}")

        # ★ L4 meta-skill 注入 (cross-family transferable patterns). 之前 promote 后
        # 只存不读 → applied_count 永远 0 (write-only). 现在: retrieve 本 family 适用的
        # meta-skill, 注入 setter, 并记下名字 → rollout 结束按真实 sandbox 结果回写
        # record_meta_skill_application, 闭合 ReasoningBank/ACE 的 helpful/harmful 计数环.
        if self.cfg.baseline_mode == "v2" and getattr(self, "mem", None) is not None:
            try:
                metas = self.mem.get_meta_skills(family=family_name)
                if metas:
                    meta_block = "\n\n## 🧠 Cross-family META-SKILLS (proven transferable):\n"
                    for m in metas[:3]:
                        meta_block += (f"- [{m.name}] (spans {', '.join(m.spans_families)}; "
                                       f"success {m.success_rate:.0%}/{m.applied_count} uses)\n"
                                       f"  {m.body[:200]}\n")
                        injected_meta_names.append(m.name)
                    skill_doc = skill_doc + meta_block
                    _log.info(f"  [L4] injected {len(injected_meta_names)} meta-skill(s)")
            except Exception as e:
                _log.warning(f"  meta-skill injection failed: {e}")

        # ★ F2 FIX (核心 self-evolution): 强制 setter 从 seed_library top-k chain 起步,
        # mutate ≤2 ops. 之前 setter 只是"参考" skill_doc → LLM creative override
        # 导致 explore 占主导, 学不到的 winning chain 无法 exploit.
        # 现在: prefix 一个 hard directive 让 setter MUST start from PROVEN chain.
        seed_lib_obj = (getattr(self, "seed_library_v2", None) if self.cfg.baseline_mode == "v2"
                        else getattr(self, "seed_library_v1", None))
        if seed_lib_obj is not None:
            try:
                top_seeds = seed_lib_obj.get_top_seeds(family_name, top_k=3)
                if top_seeds:
                    proven_block = "\n".join(
                        f"  [{s['weighted_score']:.2f}] "
                        f"{' → '.join(c.get('tool','?') for c in s['chain'])}"
                        for s in top_seeds
                    )
                    skill_doc = (
                        f"## 🔥 PROVEN WORKING CHAINS for {family_name} (HIGHEST PRIORITY)\n"
                        f"Below are chains that previously bypassed the detector. "
                        f"Your brief MUST start from the top chain and mutate AT MOST 2 ops. "
                        f"Do NOT invent a completely new chain — exploit what works.\n\n"
                        f"{proven_block}\n\n"
                        f"---\n\n" + skill_doc  # original skill_doc appended below
                    )
                    _log.info(f"  [F2] forced top-{len(top_seeds)} seed in setter prompt "
                              f"(top score={top_seeds[0]['weighted_score']:.2f})")
            except Exception as e:
                _log.warning(f"  [F2] seed injection failed: {e}")

        # ★ M2-P0-1: 9-agent family pool — 用 family-specific system prompt
        family_sys_prompt = None
        if hasattr(self, "family_agents"):
            family_sys_prompt = self.family_agents.get(family_name).to_system_prompt()
        try:
            brief, _raw = self.multi_agent.generate_brief(
                family=family_name, skill_doc=skill_doc,
                retrieved_experiences=top_exps,
                prior_briefs=self.recent_briefs[-5:],
                operator_list=healthy_op_list,  # ★ Bug-18: 不暴露 broken op
                setter_role=setter, src_face_path=src_face,
                family_system_prompt=family_sys_prompt,
            )
        except Exception as e:
            _log.error(f"setter failed: {e}; using fallback brief")
            brief = Brief(
                src_face_path=src_face, attack_class=family_name,
                suggested_chain=["face_align", "nano_banana_two", "jpeg_85"],
                generator_model=f"fallback/{setter}",
            )
        self.recent_briefs.append(brief)

        # ── Layer 2 cont.: 3-checker median (内部文章 verbatim 4 维评分) ──
        checker_overall = -1.0
        checker_issues: list = []
        if getattr(self.cfg, "enable_checkers", True):
            try:
                check_score = self.multi_agent.check_brief(
                    brief=brief,
                    operator_list=healthy_op_list,
                    recent_experiences=top_exps,
                )
                checker_overall = float(check_score.overall)
                checker_issues = list(check_score.issues)
                _log.info(
                    f"  [checker median] success={check_score.attack_success_potential:.0f} "
                    f"novelty={check_score.novelty_coverage:.0f} "
                    f"generalize={check_score.generalization:.0f} "
                    f"evasion={check_score.defense_evasion:.0f} "
                    f"OVERALL={check_score.overall:.0f}"
                )
                # Cost: 3 checkers × tokens. Track via op_health-like counter.
                self.total_cost += 0.003  # rough $0.001/checker
                # ★ TIER1-M1-4: method 1 自动从 checker issues 写入 skill 约束
                # (内部文章图11: 每轮质检发现的问题 → Skill 新约束 → 下次不重犯)
                if (self.cfg.baseline_mode == "v1"
                        and getattr(self.cfg, "v1_auto_constraint_from_checker", True)
                        and hasattr(self, "skill_book_v1")
                        and checker_issues):
                    # 拼成 1 条 constraint, 注明 round + issues
                    cons = (f"R{round_id} checker-flagged issues "
                            f"(overall={check_score.overall:.0f}): " +
                            "; ".join(str(x)[:120] for x in checker_issues[:5]))
                    self.skill_book_v1.append_constraint(family_name, cons)
                    _log.info(f"  [v1 auto-constraint] +1 written to {family_name} skill")
            except Exception as e:
                _log.warning(f"  3-checker scoring failed: {e}")

        # ── Layer 4: planner ──
        if self.cfg.baseline_mode == "v2":
            # ★ BUG-2 fix: 传 seed_library_v2 让 planner 真用高分回池
            # lazy-init seed_library_v2 (4-evolution + ui_voyager 会往里写)
            if not hasattr(self, "seed_library_v2"):
                try:
                    from simple_seed_library import SimpleSeedLibrary
                    self.seed_library_v2 = SimpleSeedLibrary(
                        db_path=str(self.bank_dir / "seed_library_v2" / "seeds.db"),
                    )
                except Exception:
                    self.seed_library_v2 = None
            # prior_pipelines = 最近 5 个 brief 的 chain (for coverage 维度)
            prior_pls = [b.suggested_chain for b in self.recent_briefs[-5:]
                          if b.suggested_chain]
            chain, cands = self.planner.plan(
                family=family_name, brief_hints=brief.suggested_chain,
                brief_text=brief.brief_text, max_steps=5,
                seed_library=self.seed_library_v2,
                prior_pipelines=[[{"tool": t} for t in p] for p in prior_pls],
            )
            lookahead = [
                LookaheadCandidate(
                    pipeline=c.pipeline, proxy_score=c.proxy_score,
                    selected=c.selected,
                ) for c in cands
            ]
        else:
            # Baseline #1: MCTS
            # ★ BUG-1 fix: 优先从 seed_library 拿高分回池 seed (内部文章图1/3 verbatim)
            # 之前总是 brief.suggested_chain → MCTS 探索完不复用任何累积经验
            # 现在: 50% 概率从 top-3 seed 抽 + 50% 从 brief 抽 (保多样性)
            seed_chain = None
            if (getattr(self, "seed_library_v1", None) is not None
                    and random.random() < 0.5):
                top_seeds = self.seed_library_v1.get_top_seeds(
                    family=family_name, top_k=3)
                if top_seeds:
                    chosen = random.choice(top_seeds)
                    seed_chain = chosen["chain"]
                    _log.info(f"  [seed_lib] MCTS seed from top-{len(top_seeds)} "
                              f"chain_id={chosen['chain_id'][:18]} "
                              f"score={chosen['weighted_score']:.3f}")
            if seed_chain is None:
                seed_chain = [{"tool": t, "params": {}}
                              for t in brief.suggested_chain]
            chain, history = self.mcts_planner.search(
                seed_chain=seed_chain, family=family_name,
                available_ops=self.operator_list,
                available_mutations=list(MUTATION_OPERATORS_V1.keys()),
            )
            lookahead = []  # no lookahead in v1

        # ── Layer 4: execute (★ Q4: 加 family_name 给 reflexion) ──
        exec_steps = self.execute_pipeline(chain, brief, round_id, traj_id,
                                            family_name=family_name)

        # ── Layer 5: sandbox verify (on FINAL image) ──
        final_img = exec_steps[-1].output_path if exec_steps else src_face
        if not final_img or not Path(final_img).exists():
            _log.warning(f"  final image missing, sandbox verify on src instead")
            final_img = src_face

        # ★ BUG-17 修: 检测 pseudo-bypass (chain 没真生成新图, final = src)
        # 防 nano_banana 503 全失败 → mock pass-through → sandbox 评原图 → 假 bypass
        pseudo_bypass = (final_img == src_face)
        real_op_succeeded = any(
            (s.output_path != src_face)
            and (s.output_path is not None)
            and not (s.error and ("All models failed" in s.error
                                  or "MOCK_UNAVAILABLE" in s.error))
            for s in exec_steps
        )

        try:
            v_obj = self.sandbox.verify(
                forged_path=final_img,
                src_face_path=src_face,
                attack_family=family_name,
            )
            self.total_cost += v_obj.cost_usd
            # ★ BUG-17: pseudo-bypass detection
            real_bypass = v_obj.sandbox_pass and real_op_succeeded and not pseudo_bypass
            if v_obj.sandbox_pass and not real_bypass:
                _log.warning(f"  🚨 PSEUDO-BYPASS detected: chain didn't generate new attack image "
                             f"(pseudo={pseudo_bypass}, real_op={real_op_succeeded}). Marking as FAIL.")
                bypass_confirmed = ["PSEUDO_BYPASS_REJECTED"]
            else:
                bypass_confirmed = v_obj.bypass_confirmed_by
            verdicts = Verdicts(
                sandbox_pass=real_bypass,
                bypass_confirmed_by=bypass_confirmed,
                tier1=v_obj.tier1, tier2=v_obj.tier2, tier3=v_obj.tier3,
                cost_usd=v_obj.cost_usd,
                detector_signature=v_obj.detector_signature,
            )
        except Exception as e:
            _log.error(f"  sandbox failed: {e}")
            verdicts = Verdicts(sandbox_pass=False, tier1={}, tier2={"error": str(e)})

        # ── Layer 6: attribution (v2 only) ──
        attribution = []; composite = None
        if self.cfg.baseline_mode == "v2" and exec_steps and verdicts.tier2:
            try:
                attr_list = self.attributor.attribute(
                    execution=exec_steps,
                    bypass_success=verdicts.sandbox_pass,
                    tier1_final=verdicts.tier1,
                    tier2_is_fake=bool(verdicts.tier2.get("is_fake", False)),
                    tier2_confidence=float(verdicts.tier2.get("confidence", 0.5)),
                    tier2_reasoning=str(verdicts.tier2.get("reasoning", "")),
                )
                self.total_cost += 0.002  # attribution cost
                attribution = attr_list
                # ★ 连续 r_out: 真突破=1.0; 否则=w·soft_evade(给搜索朝反检测爬的梯度)
                if verdicts.sandbox_pass:
                    r_out = 1.0
                else:
                    r_out = self.cfg.evade_shaping_weight * _soft_evade_score(verdicts.tier1)
                composite = self.attributor.composite_reward(attr_list, r_out=r_out)
            except Exception as e:
                _log.warning(f"  attribution failed: {e}")

        # ── Build Trajectory ──
        traj = Trajectory(
            trajectory_id=traj_id, round_id=round_id,
            baseline=self.cfg.baseline_mode,
            attack_family=family_name,
            policy_signature=f"{self.cfg.tier2_model}/T=0.7/{self.cfg.baseline_mode}",
            detector_signature=verdicts.detector_signature,
            brief=brief, skill_lookup=sl,
            lookahead_candidates=lookahead, execution=exec_steps,
            verdicts=verdicts, attribution=attribution,
            composite_reward=composite,
            data_route="DROP", jacquard_dedupe_key="",
            timestamp=time.time(), cost_usd=verdicts.cost_usd,
        )
        # Layer 10: defender export hint
        if verdicts.sandbox_pass and final_img and Path(final_img).exists():
            traj.defender_export = DefenderExport(
                image_path=final_img,
                label={"is_fake": True, "family": family_name},
                forensic_cot=(verdicts.tier3 or {}).get("reasoning", "")
                              or (verdicts.tier2 or {}).get("reasoning", ""),
                ready_for_sft=True,
            )

        # Track outcome
        self.recent_outcomes.append({
            "bypass": verdicts.sandbox_pass, "family": family_name,
            "summary": str((verdicts.tier2 or {}).get("reasoning", ""))[:200],
        })

        # ★ M2-P0-1: 9-agent family pool — record per-family experience
        if hasattr(self, "family_agents"):
            try:
                exec_chain_for_agent = [{"tool": s.tool} for s in (exec_steps or [])]
                self.family_agents.update_experience(
                    family_name, exec_chain_for_agent,
                    bypass=bool(verdicts.sandbox_pass),
                    reasoning=str((verdicts.tier2 or {}).get("reasoning", ""))[:300],
                )
            except Exception as e:
                _log.warning(f"family_agents update failed: {e}")

        # ★ Tier 2-2 fix: VideoWeaver Composition+Creator 2-layer skill record.
        # Composition records the chain SHAPE; Creator now records EVERY op
        # (incl. default/empty params) so per-op success stats accumulate.
        # success is gated on the real sandbox verdict (verdicts.sandbox_pass).
        if hasattr(self, "videoweaver_skills") and self.videoweaver_skills is not None:
            try:
                vw_chain = [{"tool": s.tool, "params": s.params or {}}
                            for s in (exec_steps or [])]
                if vw_chain:
                    self.videoweaver_skills.record_rollout(
                        family_name, vw_chain,
                        success=bool(verdicts.sandbox_pass),
                        reasoning=str((verdicts.tier2 or {}).get("reasoning", ""))[:200],
                    )
            except Exception as e:
                _log.warning(f"videoweaver skills update failed: {e}")

        # ★ Fix ③: wire REAL sandbox outcome back into seed library.
        # Before this, record_attempt was never called → sandbox_success_count
        # stayed 0 for every seed → seeds were "0 sandbox-verified" (proxy-only).
        # Now: on real bypass, promote the winning chain + record a true success;
        # on reused-seed failure, record the failure so prune() can retire it.
        active_seed_lib = (getattr(self, "seed_library_v2", None)
                           if self.cfg.baseline_mode == "v2"
                           else getattr(self, "seed_library_v1", None))
        if active_seed_lib is not None and exec_steps:
            try:
                from simple_seed_library import _chain_key
                exec_chain = [{"tool": s.tool, "params": getattr(s, "params", {}) or {}}
                              for s in exec_steps]
                chain_id = _chain_key(family_name, exec_chain)
                if verdicts.sandbox_pass:
                    # ensure a sandbox-verified winner is in the pool, then count it
                    active_seed_lib.promote_chain(
                        family_name, exec_chain,
                        four_dim={"weighted": 0.75, "attack_success": 1.0,
                                   "coverage": 0.6, "generalization": 0.6,
                                   "defense_evasion": 1.0},
                        source="sandbox_verified",
                    )
                    active_seed_lib.record_attempt(chain_id, success=True)
                else:
                    # only updates if this chain is a tracked seed (no-op otherwise)
                    active_seed_lib.record_attempt(chain_id, success=False)
            except Exception as e:
                _log.warning(f"seed_library sandbox write-back failed: {e}")

        # ★ L4 meta-skill applied loop: close the helpful/harmful counter using the
        # REAL sandbox verdict for every meta-skill injected into this rollout's brief.
        if injected_meta_names and getattr(self, "mem", None) is not None:
            for _mname in injected_meta_names:
                try:
                    self.mem.record_meta_skill_application(
                        _mname, success=bool(verdicts.sandbox_pass))
                except Exception as e:
                    _log.warning(f"meta-skill application record failed ({_mname}): {e}")

        # ★ Tier 2-3 (Method 3 挂载+一起更新): record per-rollout into co-evolution state
        if hasattr(self, "coevo") and self.coevo is not None:
            try:
                self.coevo.record_rollout(
                    family_name, bool(verdicts.sandbox_pass),
                    defender_confidence=float((verdicts.tier2 or {}).get("confidence", 0.5)),
                    defender_reasoning=str((verdicts.tier2 or {}).get("reasoning", ""))[:500],
                )
            except Exception as e:
                _log.warning(f"coevo record_rollout failed: {e}")

        # ★ Tier 2-1: UI-Voyager 失败归因 + 成功轨迹纠正 (post-failure analysis)
        # Only triggers on bypass=False; pulls from data_flow.db for donor success traj
        if (not verdicts.sandbox_pass and exec_steps and
                getattr(self.cfg, "enable_ui_voyager_correction", True)):
            try:
                from ui_voyager_correction import ui_voyager_correct
                failed_chain = [{"tool": s.tool, "params": getattr(s, "params", {})}
                                 for s in exec_steps]
                # build per-step metrics from exec_steps' tier1
                per_step_metrics = [getattr(s, "tier1_metrics", {}) or {}
                                     for s in exec_steps]
                # fallback: use verdicts.tier1 as the last step's metrics
                if not any(per_step_metrics) and verdicts.tier1:
                    per_step_metrics[-1] = verdicts.tier1
                cor = ui_voyager_correct(
                    failed_chain=failed_chain,
                    per_step_metrics=per_step_metrics,
                    final_verdict=dict(verdicts.tier2 or {}),
                    family=family_name,
                    db_path=str(self.out_dir / f"data_flow_{self.cfg.baseline_mode}.db"),
                )
                if cor is not None:
                    _log.info(f"  [ui_voyager] failure at step {cor.failure_point.step_idx} "
                              f"({cor.failure_point.op_name}); grafted from "
                              f"{cor.matched_success_traj_id[:18]}")
                    # save correction for next round's seed pool
                    cor_dir = self.out_dir / "ui_voyager_corrections"
                    cor_dir.mkdir(parents=True, exist_ok=True)
                    cor_path = cor_dir / f"{traj_id}_correction.json"
                    cor_path.write_text(json.dumps({
                        "traj_id": traj_id, "family": family_name,
                        "original_chain": [s["tool"] for s in cor.original_failed_chain],
                        "corrected_chain": [s["tool"] for s in cor.corrected_chain],
                        "failure_point": {"step": cor.failure_point.step_idx,
                                           "op": cor.failure_point.op_name,
                                           "severity": cor.failure_point.severity,
                                           "evidence": cor.failure_point.metric_evidence},
                        "donor_traj": cor.matched_success_traj_id,
                        "rationale": cor.rationale,
                    }, ensure_ascii=False, indent=2))
                    # also promote corrected chain into seed library (if v2 has one)
                    if hasattr(self, "seed_library_v2") and self.seed_library_v2:
                        self.seed_library_v2.promote_chain(
                            family_name, cor.corrected_chain,
                            four_dim={"weighted": 0.7, "attack_success": 0.7,
                                       "coverage": 0.6, "generalization": 0.6,
                                       "defense_evasion": 0.6},
                            source="ui_voyager_correction",
                        )
            except Exception as e:
                _log.warning(f"ui_voyager correction failed: {e}")

        # ★ Q4: Novelty 记录 + composite reward
        novelty_meta = None
        if self.novelty and exec_steps:
            chain_for_novelty = [{"tool": s.tool} for s in exec_steps]
            novelty_meta = self.novelty.composite_reward(
                bypass=float(verdicts.sandbox_pass),
                attack_family=family_name,
                chain=chain_for_novelty,
                src_face_path=src_face,
            )
            self.novelty.record(family_name, chain_for_novelty,
                                verdicts.sandbox_pass, src_face)
            _log.info(f"  [novelty] score={novelty_meta['novelty_score']:.3f}, "
                      f"composite={novelty_meta['composite_reward']:.3f}"
                      + (" (repeated!)" if novelty_meta['is_repeated'] else ""))

        # ★ Q4: ReasoningBank distill + add
        if self.reasoning_bank and exec_steps and attribution:
            rule = self.reasoning_bank.distill_rule(
                traj.to_dict(),
                attributor_model=self.cfg.tier2_model,
            )
            if rule:
                rule_id, merged = self.reasoning_bank.add_or_merge(rule)
                _log.info(f"  [reasoning_bank] +{rule_id} (merged={merged})")
                self.total_cost += 0.0015
                # ★ BUG-16 修: 每个 rollout 后 atomic-save, 防 timeout/crash 丢失 in-memory
                try:
                    self.reasoning_bank.save_all()
                except Exception as e:
                    _log.warning(f"  rb save failed: {e}")

        # ── Layer 7: data flow commit ──
        route = self.data_flow.commit_trajectory(traj.to_dict())
        _log.info(f"  → bypass={verdicts.sandbox_pass}, route={route}, "
                  f"cost=${verdicts.cost_usd:.4f}")

        # ★ BUG-4 fix support: attach novelty_meta so Markov update can see it
        traj._novelty_meta = novelty_meta
        return traj

    # ─────────────── One round ──────────────────────────────────────

    def run_round(self, round_id: int):
        _log.info(f"\n══════ ROUND {round_id} ({self.cfg.baseline_mode}) ══════")
        current_family = 0
        # L8: round-scoped session memory begin
        self.mem.start_round(round_id)

        for brief_i in range(self.cfg.n_briefs_per_round):
            # Layer 1: select family
            current_family = self.family_selector.select_next(current_family)

            for rollout_i in range(self.cfg.n_rollouts_per_brief):
                rollout_id = f"r{round_id}_b{brief_i}_g{rollout_i}"
                self.mem.start_rollout(rollout_id)
                try:
                    traj = self.run_one_rollout(round_id, current_family, brief_i, rollout_i)
                    # Layer 1 update: Markov reward from outcome
                    # ★ TIER1-M1-3: v1 也 update (前提是 family_selector 是 MarkovFamilySelector)
                    has_update = hasattr(self.family_selector, 'update') and (
                        self.cfg.baseline_mode == "v2" or self.cfg.v1_use_markov)
                    if has_update:
                        # ★ BUG-4 fix: Novelty composite_reward 也喂给 Markov update
                        # 之前: r = 0/1 bypass binary,完全忽视 novelty 项
                        # 现在: 如果 traj 携带 novelty_meta,用 0.7·bypass + 0.3·novelty
                        bypass_r = 1.0 if (traj.verdicts and traj.verdicts.sandbox_pass) else 0.0
                        novelty_r = 0.0
                        if (self.novelty is not None and
                                hasattr(traj, "_novelty_meta") and traj._novelty_meta):
                            novelty_r = float(traj._novelty_meta.get("novelty_score", 0.0))
                        r = 0.7 * bypass_r + 0.3 * novelty_r if novelty_r > 0 else bypass_r
                        # ★ 无突破时叠加连续 bypass-proximity 梯度(让家族选择器朝反检测爬,
                        #    而非仅靠 novelty 探索;有突破时 bypass 仍强主导,不加 evade)
                        if bypass_r == 0.0 and traj.verdicts:
                            r += self.cfg.evade_shaping_weight * _soft_evade_score(traj.verdicts.tier1)
                        prev_family = current_family
                        self.family_selector.update(prev_family, current_family, r)
                    # L8: end-of-rollout summary
                    self.mem.end_rollout(rollout_id, {
                        "bypass": bool(traj.verdicts and traj.verdicts.sandbox_pass),
                        "family": traj.attack_family,
                        "cost": float(traj.cost_usd or 0.0),
                    })
                except Exception as e:
                    _log.exception(f"rollout failed: {e}")
                    self.mem.end_rollout(rollout_id, {"error": str(e)[:200]})

                # Layer 3 update: write skill update from this trajectory
                if self.cfg.baseline_mode == "v2":
                    self._update_skill_v2(round_id)
                else:
                    self._update_skill_v1(round_id)

        # End-of-round
        self._end_of_round(round_id)
        # L8: persist round session memory + attempt L4 cross-family meta-skill promotion
        try:
            diag = self.diagnoser.diagnose(round_id) if hasattr(self, "diagnoser") else None
            diag_dict = (
                {"global_bypass_rate": getattr(diag, "global_bypass_rate", None),
                 "family_bypass_rates": getattr(diag, "family_bypass_rates", {})}
                if diag else None
            )
        except Exception:
            diag_dict = None
        self.mem.end_round(round_id, diagnosis=diag_dict)

        # try promote a cross-family meta-skill if ≥2 families share a pattern
        if self.cfg.baseline_mode == "v2" and self.skill_lib is not None:
            family_docs = {}
            for fam, doc in getattr(self.skill_lib, "docs", {}).items():
                txt = getattr(doc, "content", "")
                if txt:
                    family_docs[fam] = txt
            # Fix ③: only mine families with a REAL sandbox-verified bypass,
            # so L4 meta-skills reflect what actually worked (not proxy/log noise).
            verified_families = None
            seed_lib = (getattr(self, "seed_library_v2", None)
                        if self.cfg.baseline_mode == "v2"
                        else getattr(self, "seed_library_v1", None))
            if seed_lib is not None:
                try:
                    verified_families = {
                        r["family"] for r in seed_lib.get_all_active()
                        if int(r.get("sandbox_success_count", 0)) > 0
                    }
                except Exception as e:
                    _log.warning(f"  [L4] verified-family lookup failed: {e}")
            if len(family_docs) >= 2:
                ms = self.mem.promote_to_meta_skill(family_docs,
                                                    threshold_families=2,
                                                    round_id=round_id,
                                                    verified_families=verified_families)
                if ms:
                    _log.info(f"  [L4] promoted meta-skill {ms.name} spans {ms.spans_families}")

        # L8: crystallize at end of every round (cheap, JSON snapshot)
        snap = self.mem.crystallize(
            tag=f"R{round_id}",
            payload={"diagnosis": diag_dict, "total_cost": self.total_cost},
        )
        _log.info(f"  [L5] snapshot → {snap.name}")

    def _update_skill_v2(self, round_id: int):
        """Layer 3 v2: 把最新一个 trajectory 的 skill_extracted 写回 ace_skill_lib.

        2 个动作:
          (a) **每次** add_experience 到 ℰ_k (无 LLM call, 免费, 必做)
          (b) **每 N 次** supervisor_extract_delta_skill 抽 strategic Δ𝒮_k (LLM call, 受 cost gate)
        """
        if not self.recent_outcomes:
            return
        last = self.recent_outcomes[-1]
        family = last["family"]

        # ── (a) ℰ_k 累加: 总是做 ──
        try:
            exp_text = (f"{family} bypass={last['bypass']}: "
                        f"{last['summary'][:200]}")
            exp_id, merged = self.skill_lib.add_experience(
                family, exp_text, round_id=round_id,
                trajectory_id=f"r{round_id}", success=last["bypass"],
            )
            _log.info(f"  ℰ_k[{family}] += {exp_id} (merged={merged})")
        except Exception as e:
            _log.warning(f"  add_experience failed: {e}")

        # ── (b) 𝒮_k 主管 delta: 每 supervisor_every 次触发 ──
        if (len(self.recent_outcomes) % self.cfg.supervisor_every == 0):
            try:
                delta = self.multi_agent.supervisor_extract_delta_skill(
                    family=family,
                    recent_briefs=self.recent_briefs[-3:],
                    check_scores=[],
                    recent_outcomes=self.recent_outcomes[-5:],
                    current_skill_doc=self.skill_lib.docs[family].content,
                )
                self.skill_lib.update_skill(family, delta)
                doc = self.skill_lib.docs[family]
                _log.info(f"  𝒮_k[{family}] updated, word_count={doc.word_count()}")
                # persist version snapshot to data_flow.skill_versions (was never
                # wired → table stayed empty even when the doc evolved).
                try:
                    self.data_flow.snapshot_skill(family, doc.version, doc.content)
                except Exception as e:
                    _log.warning(f"  snapshot_skill failed: {e}")
            except Exception as e:
                _log.warning(f"  skill update v2 failed: {e}")

    def _update_skill_v1(self, round_id: int):
        """Layer 3 v1: 主管手动追加约束 to single-layer markdown."""
        if not self.recent_outcomes:
            return
        # gated by supervisor_every for cost
        if len(self.recent_outcomes) % max(self.cfg.supervisor_every, 1) != 0:
            return
        last = self.recent_outcomes[-1]
        fam = last["family"]
        constraint = f"R{round_id} outcome: bypass={last['bypass']}, summary={last['summary'][:200]}"
        self.skill_book_v1.append_constraint(fam, constraint)
        try:
            self.data_flow.snapshot_skill(
                fam, self.skill_book_v1.append_count.get(fam, 0),
                self.skill_book_v1.get_doc(fam))
        except Exception as e:
            _log.warning(f"  snapshot_skill (v1) failed: {e}")

    def _end_of_round(self, round_id: int):
        # Snapshot Markov state (v2)
        if self.cfg.baseline_mode == "v2":
            self.data_flow.snapshot_markov(round_id,
                self.family_selector.transition_summary(), run_id=self.run_id)
            self.family_selector.save(str(self.bank_dir / "markov" / "markov_state.json"))
        # ★ TIER1-M1-3: v1 也持久化 Markov state (内部文章图2)
        elif self.cfg.baseline_mode == "v1" and self.cfg.v1_use_markov \
                and isinstance(self.family_selector, MarkovFamilySelector):
            (self.bank_dir / "markov").mkdir(parents=True, exist_ok=True)
            self.family_selector.save(str(self.bank_dir / "markov" / "markov_state_v1.json"))
        # ★ M2-P0-1: 持久化 family_agents 经验日志 (v1 + v2 都用)
        if hasattr(self, "family_agents"):
            try:
                self.family_agents.save_all()
            except Exception as e:
                _log.warning(f"family_agents save failed: {e}")

        # ★ Tier 2-3 (Method 3 挂载+一起更新): broadcast defender state to attacker
        # 这是 "一起更新" 的核心 — 把 detector 现状 push 到 attacker 的 skill_lib / family_agents / Markov
        if hasattr(self, "coevo") and self.coevo is not None:
            try:
                if self.coevo.current_state is None:
                    # ensure state exists (in case start_round wasn't called)
                    self.coevo.start_round(round_id=round_id, lora_path="(in-orchestrator)")
                upd = self.coevo.broadcast_to_attacker(
                    attacker_skill_lib=getattr(self, "skill_lib", None),
                    family_agents=getattr(self, "family_agents", None),
                    markov=self.family_selector if hasattr(self, "family_selector") else None,
                )
                _log.info(f"  [co-evo] defender→attacker broadcast: channels={upd.get('channels', [])}")
                snap_path = self.coevo.snapshot(str(self.bank_dir / "coevo_snapshots"))
                if snap_path: _log.info(f"  [co-evo] state snapshot → {Path(snap_path).name}")
            except Exception as e:
                _log.warning(f"co-evo broadcast failed: {e}")
        # ★ M2-P0-2: v2 也跑 4 evolution mechanisms (复用 v1 同一个 module)
        # 为 v2 需要个 seed_library — lazy init
        if (self.cfg.baseline_mode == "v2"
                and getattr(self.cfg, "v1_use_seed_library", True)):
            if not hasattr(self, "seed_library_v2"):
                from simple_seed_library import SimpleSeedLibrary
                self.seed_library_v2 = SimpleSeedLibrary(
                    db_path=str(self.bank_dir / "seed_library_v2" / "seeds.db"),
                )
            try:
                from simple_4_evolutions import FourEvolutionsOrchestrator
                evo = FourEvolutionsOrchestrator(
                    client=self.client, seed_library=self.seed_library_v2,
                    available_ops=self.operator_list,
                )
                recent_fails = [
                    {"chain": [], "family": o["family"],
                     "reason": o.get("summary", "")[:200]}
                    for o in self.recent_outcomes[-5:] if not o.get("bypass")
                ]
                stats_v2 = evo.run_round(
                    families=self.families,
                    recent_failures=recent_fails,
                    recent_outcomes=self.recent_outcomes[-20:],
                    current_detector=self.cfg.tier2_model,
                    n_genetic_per_family=1, n_reflective=1, n_external=1,
                )
                _log.info(f"  [v2 4-evolution] genetic+={stats_v2['genetic_added']} "
                          f"reflective+={stats_v2['reflective_added']} "
                          f"external+={stats_v2['external_added']} "
                          f"gan_upgrade={stats_v2['gan_upgrade_suggested']}")
            except Exception as e:
                _log.warning(f"v2 4-evolution failed: {e}")
        # ★ TIER1-M1-5: v1 跑 4 evolution mechanisms (内部文章图2 verbatim 4 个)
        if (self.cfg.baseline_mode == "v1"
                and getattr(self, "seed_library_v1", None) is not None):
            try:
                from simple_4_evolutions import FourEvolutionsOrchestrator
                evo = FourEvolutionsOrchestrator(
                    client=self.client, seed_library=self.seed_library_v1,
                    available_ops=self.operator_list,
                )
                # gather recent failures for reflective mechanism
                recent_fails = [
                    {"chain": [{"tool": s.tool} for s in []],  # we don't have exec_steps here
                     "family": o["family"],
                     "reason": o.get("summary", "")[:200]}
                    for o in self.recent_outcomes[-5:] if not o.get("bypass")
                ]
                stats = evo.run_round(
                    families=self.families,
                    recent_failures=recent_fails,
                    recent_outcomes=self.recent_outcomes[-20:],
                    current_detector=self.cfg.tier2_model,
                    n_genetic_per_family=1, n_reflective=1, n_external=1,
                )
                _log.info(f"  [4-evolution] genetic+={stats['genetic_added']} "
                          f"reflective+={stats['reflective_added']} "
                          f"external+={stats['external_added']} "
                          f"gan_upgrade={stats['gan_upgrade_suggested']}")
                if stats["gan_upgrade_suggested"]:
                    _log.info(f"  [GAN] suggest upgrade detector → "
                              f"{stats['gan_plan']['next_detector_hint']} "
                              f"({stats['gan_plan']['reason']})")
                self.total_cost += 0.005 * (1 + sum(
                    [stats[k] for k in ("genetic_added", "reflective_added", "external_added")]
                ))
            except Exception as e:
                _log.warning(f"  4-evolution failed: {e}")
            # prune low-performance seeds at end of round
            try:
                pruned = self.seed_library_v1.prune()
                if pruned: _log.info(f"  [seed_library] pruned {len(pruned)} low-perf seeds")
            except Exception:
                pass
        # Diagnose (限定 current run 的 trajectory)
        stats = self.data_flow.round_bypass_stats(round_id, run_id=self.run_id)
        diag = self.diagnoser.diagnose(round_id)
        _log.info(f"\n[End of round {round_id}] global_bypass_rate={diag.global_bypass_rate:.3f}")
        for fam, st in diag.family_bypass_rates.items():
            _log.info(f"  {fam}: bypass_rate={st:.3f}")
        _log.info(f"  weak families: {diag.weak_families}")
        # Layer 9: Markov boost to weak (v2 only)
        if self.cfg.baseline_mode == "v2":
            for weak_name in diag.weak_families:
                widx = self.families.index(weak_name)
                self.family_selector.boost_weak_family(widx, boost=diag.boost_recommendation)
        # Save report
        (self.out_dir / "reports" / f"r{round_id}_{self.cfg.baseline_mode}.json").write_text(
            json.dumps({
                "round_id": round_id, "baseline": self.cfg.baseline_mode,
                "diagnosis": {
                    "global_bypass_rate": diag.global_bypass_rate,
                    "family_bypass_rates": diag.family_bypass_rates,
                    "weak_families": diag.weak_families,
                    "strong_families": diag.strong_families,
                    "next_round_family_weights": diag.next_round_family_weights,
                },
                "total_cost_usd": self.total_cost,
            }, indent=2, ensure_ascii=False)
        )
        # Save skill lib
        if self.cfg.baseline_mode == "v2":
            self.skill_lib.save_all()
        else:
            self.skill_book_v1.save_all()

        # BUG E + F 修: meta state (recent_briefs/outcomes/cum_cost) 跨 run 持久化
        try:
            meta = {
                "run_id": self.run_id,
                "total_cost_cumulative": self.total_cost,
                "recent_briefs": [
                    asdict(b) for b in self.recent_briefs[-15:] if b is not None
                ],
                "recent_outcomes": self.recent_outcomes[-30:],
            }
            self._meta_state_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        except Exception as e:
            _log.warning(f"meta save failed: {e}")

    # ─────────────── Main run ─────────────────────────────────────

    def run(self):
        for r in range(self.cfg.n_rounds):
            # ★ search finding #2: cost budget guard ($47K stall loop)
            if (self.cfg.abort_on_budget_exceeded
                    and self.total_cost > self.cfg.cost_budget_usd):
                _log.error(
                    f"\n!!! COST BUDGET EXCEEDED: ${self.total_cost:.4f} > "
                    f"${self.cfg.cost_budget_usd:.2f} — aborting after round {r-1}")
                break
            self.run_round(r)
        # ★ Q4: ReasoningBank save 在 run 末
        if self.reasoning_bank:
            self.reasoning_bank.save_all()

        # Layer 10: export defender SFT pool
        out_jsonl = self.out_dir / f"defender_sft_{self.cfg.baseline_mode}.jsonl"
        stats = self.exporter.export(str(out_jsonl))
        _log.info(f"\n══════ Defender SFT export ({self.cfg.baseline_mode}) ══════")
        _log.info(f"  {stats}")
        _log.info(f"\n══════ TOTAL COST: ${self.total_cost:.4f} ══════")

        # Trigger retrain (if conditions met)
        last_diag = self.diagnoser.diagnose(self.cfg.n_rounds - 1)
        if self.exporter.should_trigger_retrain(last_diag, threshold=0.4, min_attempts=4):
            from diagnosis_and_export import emit_retrain_script
            script = emit_retrain_script(
                round_id=self.cfg.n_rounds - 1,
                bypass_rate=last_diag.global_bypass_rate,
                sft_jsonl=str(out_jsonl), version=1,
                out_path=str(self.out_dir / "retrain_v1.sh"),
            )
            _log.info(f"  Retrain script emitted: {script}")

        self.data_flow.close()


# ────────────────────────── CLI ────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["v1", "v2"], default="v2")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--briefs", type=int, default=2)
    parser.add_argument("--rollouts", type=int, default=1)
    parser.add_argument("--tier2", default="gemini-2.5-flash")
    parser.add_argument("--tier3", action="store_true")
    parser.add_argument("--tier3-model", default="gemini-3-pro-preview",
                        help="Tier-3 forensic cross-check model (default gemini-3-pro-preview, claude-opus 503)")
    parser.add_argument("--no-api-ops", action="store_true",
                        help="跳过 nanobanana 等真实 attack op (省钱)")
    parser.add_argument("--api-only-ops", action="store_true",
                        help="仅暴露 API ops 给 setter (避免本地 op 未装)")
    parser.add_argument("--tier2-backend", choices=["viviai", "fakevlm_local"],
                        default="viviai", help="★ Lv5: 切 sandbox Tier-2 detector backend")
    parser.add_argument("--fakevlm-endpoint", default="http://localhost:8000/v1")
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--bank-dir", default="",
                        help="long-horizon (长程) shared bank dir for cross-scenario "
                             "skill/markov/reasoning accumulation; empty=use --out")
    parser.add_argument("--src-pool", nargs="+", default=None,
                        help="src face image paths (>=1)")
    parser.add_argument("--multi-agent-preset", choices=["w1_cheap", "w6_full"],
                        default="w1_cheap",
                        help="L2 fan-out: w1_cheap (all flash) or w6_full (gemini-3-pro mix)")
    parser.add_argument("--no-checkers", action="store_true",
                        help="跳过 3-checker median 评分 (省钱)")
    parser.add_argument("--evade-weight", type=float, default=0.3,
                        help="连续 bypass-proximity 奖励权重 (bypass=0 时给搜索朝反检测爬的梯度; "
                             "0=关闭, 回退旧二值行为)")
    args = parser.parse_args()

    src_pool = args.src_pool or [
        "/data/disk4/lyx_ICML/hf_models_lyx/04_id_preserving/InstantX__InstantID/examples/0.png",
    ]

    cfg = OrchestratorConfig(
        baseline_mode=args.mode,
        n_rounds=args.rounds,
        n_briefs_per_round=args.briefs,
        n_rollouts_per_brief=args.rollouts,
        tier2_model=args.tier2,
        tier3_enabled=args.tier3,
        tier3_model=args.tier3_model,
        execute_api_ops=not args.no_api_ops,
        api_only_ops=args.api_only_ops,
        tier2_backend=args.tier2_backend,
        fakevlm_endpoint=args.fakevlm_endpoint,
        output_dir=args.out,
        bank_dir=args.bank_dir,
        src_face_paths=src_pool,
        multi_agent_preset=args.multi_agent_preset,
        enable_checkers=not args.no_checkers,
        evade_shaping_weight=args.evade_weight,
    )

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    orc = Orchestrator(cfg)
    orc.run()


if __name__ == "__main__":
    main()
