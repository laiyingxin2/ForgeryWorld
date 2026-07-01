"""Method 4 — Face-type-conditioned attacker (Eevee-inspired, fresh orchestrator).

Architecture (1 traj):
  src_face → face_metadata_router → cluster_id (e.g. male_adult_frontal)
           → pareto_skill_pool.get_top_snippet(family, cluster) → snippet
           → multi_agent_gen.generate_brief(skill_doc = snippet, ...)
           → pipeline_planner.plan() → chain
           → execute_pipeline()
           → sandbox.verify()
           → if bypass: pool.promote_on_bypass(family, cluster, ...)
              else:     pool.record_use(snippet_id, src_face_id, success=False)
  + co-evolution: every N rollouts, regenerate snippet candidates from top-bypass
    chains using LLM mutator (Eevee-inspired but cheaper).

Does NOT touch existing orchestrator.py / pipeline_planner.py / etc — purely
additive. Imports them as libraries.
"""
from __future__ import annotations
import json
import time
import logging
import random
import os
import sys
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from viviai_client import ViviClient
from sandbox import SandboxVerifier
from multi_agent_gen import MultiAgentBenchmarkGen, MultiAgentConfig
from data_flow import DataFlow
from trajectory_schema import (
    Trajectory, Brief, Verdicts, ExecutionStep, attack_family_list,
    new_trajectory_id,
)
from markov_family import MarkovFamilySelector
from op_health import OpHealthTracker
from operators import OPERATOR_REGISTRY, resolve_op

from method4_face_metadata_router import extract_metadata, FaceMetadata
from method4_pareto_skill_pool import ParetoSkillPool

_log = logging.getLogger(__name__)


# ────────────────────────── Config ────────────────────────────────

@dataclass
class Method4Config:
    out_dir: str = "outputs/method4"
    src_face_paths: list = field(default_factory=list)
    families: list = field(default_factory=attack_family_list)
    n_rounds: int = 3
    n_briefs_per_round: int = 4
    n_rollouts_per_brief: int = 1
    tier2_model: str = "gemini-2.5-flash"
    tier2_backend: str = "viviai"
    fakevlm_endpoint: str = "http://localhost:8000/v1"
    execute_api_ops: bool = True
    multi_agent_preset: str = "w6_full"
    pareto_k_max: int = 5
    api_key: str = os.environ.get("VIVIAI_KEY", "")


# ────────────────────────── helpers ─────────────────────────────

def _face_id(src_path: str) -> str:
    """Stable short ID for a face image path."""
    return Path(src_path).stem


def _execute_pipeline_minimal(chain: list, brief: Brief, op_registry: dict,
                                 client: ViviClient,
                                 face_out_dir: str) -> tuple[list[ExecutionStep], str]:
    """Minimal pipeline executor: try each op, fallback to mock if not in registry.
    Returns (exec_steps, final_image_path)."""
    Path(face_out_dir).mkdir(parents=True, exist_ok=True)
    src = brief.src_face_path
    current_img = src
    exec_steps: list[ExecutionStep] = []
    for i, step in enumerate(chain):
        tool = resolve_op(step.get("tool", ""))  # ★ legacy/phantom name -> registry key
        params = step.get("params", {}) or {}
        t0 = time.time()
        out_path = current_img
        err = None
        success = False
        op_cls = op_registry.get(tool)
        if op_cls is None:
            # mock: keep image unchanged
            err = f"op '{tool}' not registered"
        else:
            try:
                op = op_cls(client=client, out_dir=face_out_dir)
                r = op.run(src_face_path=current_img, params=params)
                if r.success and r.output_path and Path(r.output_path).exists():
                    out_path = r.output_path
                    current_img = out_path
                    success = True
                else:
                    err = r.error or "op failed"
            except Exception as e:
                err = str(e)[:200]
        exec_steps.append(ExecutionStep(
            tool=tool, params=params, output_path=out_path, error=err,
            duration_sec=time.time() - t0,
        ))
    return exec_steps, current_img


# ────────────────────────── Main runner ──────────────────────────

class Method4Runner:
    """Method 4 orchestrator — face-type-conditioned, Pareto-front skill pool."""

    def __init__(self, cfg: Method4Config):
        self.cfg = cfg
        self.out_dir = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = "m4-" + time.strftime("%Y%m%d-%H%M%S")

        # API client + multi-agent
        self.client = ViviClient(api_key=cfg.api_key)
        ma_cfg = (MultiAgentConfig.w6_full() if cfg.multi_agent_preset == "w6_full"
                  else MultiAgentConfig.w1_cheap())
        self.multi_agent = MultiAgentBenchmarkGen(client=self.client, config=ma_cfg)

        # Sandbox (reuse Tier-1 7 real metrics + Tier-2)
        self.sandbox = SandboxVerifier(
            client=self.client, tier2_model=cfg.tier2_model,
            tier3_enabled=False, tier2_backend=cfg.tier2_backend,
            fakevlm_endpoint=cfg.fakevlm_endpoint,
        )

        # Data flow (separate DB so method 4 doesn't pollute methods 1/2/3)
        self.data_flow = DataFlow(
            db_path=str(self.out_dir / "data_flow_m4.db"),
        )

        # Markov family selector (per-family weighting, reused)
        self.markov = MarkovFamilySelector(families=cfg.families)
        markov_path = self.out_dir / "markov_m4_state.json"
        if markov_path.exists():
            self.markov = MarkovFamilySelector.load(str(markov_path))

        # Op health
        self.op_health = OpHealthTracker(
            persist_path=str(self.out_dir / "op_health_m4.json"),
        )

        # ★ NEW: Pareto skill pool + face metadata router
        self.pool = ParetoSkillPool(
            db_path=str(self.out_dir / "pareto.db"),
            k_max=cfg.pareto_k_max,
        )
        n_init = self.pool.bootstrap_init()
        if n_init > 0:
            _log.info(f"[m4] bootstrap: +{n_init} init snippets")

        # Operator list
        self.operator_list = sorted(OPERATOR_REGISTRY.keys()) + [
            "face_align", "jpeg_85", "resize_bicubic", "gfpgan",
            "liveportrait", "facevid2vid", "stylegan_morph",
            "deca_3dmask", "replay_sim", "adv_patch_pgd",
            "xtts_audio", "roop",
        ]
        # dedup
        seen = set(); self.operator_list = [
            o for o in self.operator_list if not (o in seen or seen.add(o))
        ]
        # Per-rollout metadata cache (avoid re-extracting for same face in same round)
        self._metadata_cache: dict[str, FaceMetadata] = {}

    # ──────── 1 rollout ────────
    def run_one_rollout(self, round_id: int, family_idx: int,
                          brief_idx: int, rollout_idx: int) -> dict:
        family = self.cfg.families[family_idx]
        traj_id = f"{self.run_id}_" + new_trajectory_id(round_id, family_idx, brief_idx, rollout_idx)
        src_face = random.choice(self.cfg.src_face_paths)
        face_id = _face_id(src_face)

        # ★ STEP 1: face metadata router → cluster_id
        if face_id not in self._metadata_cache:
            self._metadata_cache[face_id] = extract_metadata(src_face)
        meta = self._metadata_cache[face_id]
        cluster_id = meta.cluster_id if not meta.is_cold() else "_init"
        _log.info(f"\n─── M4 Rollout {traj_id} | family={family} | "
                  f"face={face_id} | cluster={cluster_id} ───")

        # ★ STEP 2: pool.get_top_snippet (Pareto top for this cluster, fallback init)
        top = self.pool.get_top_snippet(family, cluster_id)
        if top is None:
            snippet_text = "(no snippet — explore freely)"
            snippet_id = ""
        else:
            snippet_text = top["text"]
            snippet_id = top["snippet_id"]
            _log.info(f"  [pareto] using snippet [{snippet_id[:24]}] "
                      f"rate={top['success_rate']:.2%} "
                      + ("(FALLBACK from _init)" if top.get("is_fallback") else ""))

        # ★ STEP 3: build skill_doc with snippet + cluster context
        # ★ F2 FIX: HARD directive to use top Pareto chain, mutate ≤2 ops (not creative override)
        skill_doc = (
            f"# 🔥 PROVEN ATTACK CHAIN for {family} on cluster={cluster_id}\n"
            f"(src face: gender={meta.gender}, age_group={meta.age_group}, pose={meta.pose})\n\n"
            f"## TOP Pareto chain — MUST start from this (success_rate "
            f"{(top['success_rate'] if top else 0):.2%}):\n"
            f"{snippet_text}\n\n"
            f"## INSTRUCTIONS:\n"
            f"1. Use the above chain as your starting structure\n"
            f"2. Mutate AT MOST 2 ops (swap 1 op or tweak 1 param)\n"
            f"3. Do NOT invent a completely new chain — exploit what works\n"
            f"4. Adapt to this specific cluster's features (e.g., for {meta.age_group} "
            f"{meta.gender}, adjust accordingly)"
        )

        # ★ STEP 4: gen brief via multi_agent (reuse w6_full 6-LLM)
        healthy_ops = [o for o in self.operator_list
                       if o not in self.op_health.blacklist_failing_ops(max_rate=0.2)]
        setter = "setter_a" if rollout_idx % 2 == 0 else "setter_b"
        try:
            brief, _raw = self.multi_agent.generate_brief(
                family=family, skill_doc=skill_doc,
                retrieved_experiences=[], prior_briefs=[],
                operator_list=healthy_ops,
                setter_role=setter, src_face_path=src_face,
            )
        except Exception as e:
            _log.error(f"  setter failed: {e}; fallback chain")
            brief = Brief(
                src_face_path=src_face, attack_class=family,
                suggested_chain=["face_align", "inswapper_128_local", "jpeg_85"],
                generator_model=f"fallback/{setter}",
            )

        # ★ STEP 5: execute pipeline
        chain = [{"tool": t, "params": {}} for t in brief.suggested_chain[:6]]
        exec_steps, final_img = _execute_pipeline_minimal(
            chain, brief, OPERATOR_REGISTRY, self.client,
            face_out_dir=str(self.out_dir / "face_attack_outputs"),
        )
        # op health tracking
        for s in exec_steps:
            self.op_health.record(s.tool, succeeded=(s.error is None),
                                    error=s.error)

        # ★ STEP 6: sandbox verify
        try:
            v_obj = self.sandbox.verify(
                forged_path=final_img, src_face_path=src_face,
                attack_family=family,
            )
            verdicts = Verdicts(
                sandbox_pass=v_obj.sandbox_pass,
                bypass_confirmed_by=v_obj.bypass_confirmed_by,
                tier1=v_obj.tier1, tier2=v_obj.tier2, tier3=v_obj.tier3,
                cost_usd=v_obj.cost_usd,
                detector_signature=v_obj.detector_signature,
            )
            # pseudo-bypass check
            real_op_succeeded = any(s.error is None and s.output_path != src_face
                                    for s in exec_steps)
            if verdicts.sandbox_pass and not real_op_succeeded:
                _log.warning("  pseudo-bypass detected (no real op succeeded)")
                verdicts.sandbox_pass = False
        except Exception as e:
            _log.error(f"  sandbox failed: {e}")
            verdicts = Verdicts(sandbox_pass=False, tier1={}, tier2={"error": str(e)})

        _log.info(f"  → bypass={verdicts.sandbox_pass}, cost=${verdicts.cost_usd:.4f}")

        # ★ STEP 7: update Pareto pool
        if snippet_id:
            self.pool.record_use(snippet_id, face_id, verdicts.sandbox_pass)
        # promote on bypass: synthesize a refined snippet from this winning chain
        if verdicts.sandbox_pass:
            chain_str = " → ".join(s.tool for s in exec_steps)
            new_snippet = (
                f"[{cluster_id}, R{round_id}] {chain_str}; "
                f"derived from {('fallback init' if top and top.get('is_fallback') else snippet_id[:10])}; "
                f"tier1: arcface={verdicts.tier1.get('arcface_id_sim', -1):.2f}, "
                f"niqe={verdicts.tier1.get('niqe', -1):.1f}"
            )
            new_sid = self.pool.promote_on_bypass(
                family=family, cluster_id=cluster_id,
                new_snippet_text=new_snippet,
                src_face_id=face_id, parent_snippet_id=snippet_id,
            )
            if new_sid:
                _log.info(f"  [pareto] promoted [{new_sid[-12:]}] to ({family}, {cluster_id})")

        # ★ STEP 8: commit trajectory
        traj = Trajectory(
            trajectory_id=traj_id, round_id=round_id,
            baseline="m4", attack_family=family,
            policy_signature=f"m4/{self.cfg.tier2_model}/cluster={cluster_id}",
            detector_signature=verdicts.detector_signature,
            brief=brief, skill_lookup=None,
            lookahead_candidates=[], execution=exec_steps,
            verdicts=verdicts, attribution=None,
            composite_reward=None, data_route="", jacquard_dedupe_key="",
            timestamp=time.time(), cost_usd=verdicts.cost_usd,
        )
        # add cluster_id as a custom field on traj
        try:
            self.data_flow.commit_trajectory(traj.to_dict())
        except Exception as e:
            _log.warning(f"  commit failed: {e}")

        # ★ STEP 9: Markov reward (with cluster bonus)
        r = 1.0 if verdicts.sandbox_pass else 0.0
        try:
            self.markov.update(family_idx, family_idx, r)
        except Exception:
            pass

        return {
            "traj_id": traj_id, "family": family, "cluster_id": cluster_id,
            "bypass": verdicts.sandbox_pass,
            "snippet_id": snippet_id,
            "cost": verdicts.cost_usd,
            "chain": [s.tool for s in exec_steps],
            "tier1": {k: v for k, v in (verdicts.tier1 or {}).items() if isinstance(v, (int, float))},
        }

    # ──────── round ────────
    def run_round(self, round_id: int) -> dict:
        _log.info(f"\n══════ M4 ROUND {round_id} ══════")
        results = []
        current_family = 0
        for brief_i in range(self.cfg.n_briefs_per_round):
            current_family = self.markov.select_next(current_family)
            for rollout_i in range(self.cfg.n_rollouts_per_brief):
                try:
                    r = self.run_one_rollout(round_id, current_family,
                                              brief_i, rollout_i)
                    results.append(r)
                except Exception as e:
                    _log.exception(f"rollout failed: {e}")
                    results.append({"error": str(e)})

        # round summary
        n = len(results)
        n_bypass = sum(1 for r in results if r.get("bypass"))
        per_cluster_bypass = {}
        for r in results:
            cid = r.get("cluster_id", "?")
            per_cluster_bypass.setdefault(cid, {"n": 0, "byp": 0})
            per_cluster_bypass[cid]["n"] += 1
            if r.get("bypass"):
                per_cluster_bypass[cid]["byp"] += 1
        bp_rate = n_bypass / max(n, 1)
        _log.info(f"\n[m4 r{round_id}] global bypass = {bp_rate:.2%} ({n_bypass}/{n})")
        for cid, st in per_cluster_bypass.items():
            _log.info(f"  cluster={cid:30s}  {st['byp']}/{st['n']} = "
                      f"{st['byp']/max(st['n'],1):.0%}")
        # persist Markov
        try:
            self.markov.save(str(self.out_dir / "markov_m4_state.json"))
        except Exception: pass

        return {
            "round": round_id, "n": n, "bypass": n_bypass,
            "bypass_rate": bp_rate, "per_cluster": per_cluster_bypass,
            "results": results,
        }

    def run(self) -> dict:
        rounds = []
        for r in range(self.cfg.n_rounds):
            rounds.append(self.run_round(r))
        # final summary
        summary = {
            "run_id": self.run_id,
            "rounds": rounds,
            "pareto_stats": self.pool.stats(),
            "op_health": self.op_health.get_health_summary(sort_by="rate"),
        }
        out_path = self.out_dir / "method4_summary.json"
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        _log.info(f"\n══════ M4 SUMMARY → {out_path} ══════")
        rounds_disp = [(r['round'], round(r['bypass_rate'] * 100, 1)) for r in rounds]
        _log.info(f"  rounds: {rounds_disp}")
        _log.info(f"  pareto pool families: {len(summary['pareto_stats'])}")
        return summary


# ────────────────────────── CLI ──────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--briefs", type=int, default=4)
    parser.add_argument("--rollouts", type=int, default=1)
    parser.add_argument("--tier2", default="gemini-2.5-flash")
    parser.add_argument("--tier2-backend", choices=["viviai", "fakevlm_local"],
                        default="viviai")
    parser.add_argument("--src-pool", nargs="+", default=None)
    parser.add_argument("--out", default="outputs/method4")
    parser.add_argument("--preset", default="w6_full")
    parser.add_argument("--fakevlm-endpoint", default="http://localhost:8000/v1")
    args = parser.parse_args()

    src_pool = args.src_pool or sorted(
        str(p) for p in Path("/data/disk4/lyx_ICML/self_evolution_forgery/data/real_faces").glob("*.png")
    )
    cfg = Method4Config(
        out_dir=args.out, src_face_paths=src_pool,
        n_rounds=args.rounds, n_briefs_per_round=args.briefs,
        n_rollouts_per_brief=args.rollouts,
        tier2_model=args.tier2, tier2_backend=args.tier2_backend,
        fakevlm_endpoint=args.fakevlm_endpoint,
        multi_agent_preset=args.preset,
    )
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    runner = Method4Runner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
