"""Method 3 紧耦合 co-evolution loop — 用户明确要求 "挂载+一起更新".

之前架构 (P1-A multi_round.sh):
  Round 0:  生成 N attack → defender 判 → 收集 SFT → 训 LoRA → reload vLLM
  Round 1:  ...同上
  Round 2:  ...同上
  ↑ 问题: attacker 跟 defender LoRA reload 没"耦合" — attacker 不知道 defender 变强了

User 要求 "挂载+一起更新" 的精确含义:
  1. **挂载** = LoRA 始终 mounted in vLLM (已经做到 via --lora-modules)
  2. **一起更新** = 每次 defender LoRA reload 后, attacker 也要 *同步* 知道
     (a) defender 现在 "看" 哪些 cue (false positive 模式)
     (b) defender 在哪个 family 最强 / 最弱
     (c) 上轮哪些 chain 被识破, 哪些 slip through
     → attacker 下一轮 skill_doc + Markov + 种子库 受这些 info 影响

实现:
  CoEvolutionLoop:
    .start_round(round_id)            -- attacker + defender 双侧 init
    .run_one_batch(N=5)               -- N rollouts
    .train_defender_incrementally()   -- mini-batch SFT
    .reload_lora()                    -- hot-reload via vllm
    .broadcast_to_attacker()          -- ★ 关键: 把 defender 新状态 push 给 attacker
    .end_round()                      -- 落盘 + 算指标

关键改动: 之前 attacker 拿的 skill_doc 只有"过去攻击经验",现在加 "current
defender behavioral summary" 让 setter 立刻看到 detector 变强方向。
"""
from __future__ import annotations
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable

import requests

_log = logging.getLogger(__name__)


@dataclass
class DefenderState:
    """A snapshot of detector's current behavior — broadcast to attacker."""
    round_id: int
    lora_path: str
    lora_loaded: bool = False
    n_traj_seen: int = 0
    n_caught: int = 0
    catch_rate: float = 0.0          # 1 - bypass_rate
    # what kinds of cues defender currently catches well
    family_catch_rates: dict = field(default_factory=dict)
    avg_confidence: float = 0.0
    # patterns of false positives (real flagged as fake) — gives attacker info
    # on what the detector "fears" the most
    cue_summary: str = ""
    last_updated: float = field(default_factory=time.time)


class CoEvolutionLoop:
    """挂载+一起更新 single-class orchestrator.

    Compose: defender_lora_trainer + vllm_reloader + attacker_skill_broadcaster.
    """

    def __init__(
        self,
        vllm_endpoint: str = "http://localhost:8000/v1",
        defender_lora_name: str = "defender",
        coupling_strength: float = 1.0,  # 0..1: how aggressively to push state to attacker
    ):
        self.endpoint = vllm_endpoint
        self.defender_name = defender_lora_name
        self.coupling = coupling_strength
        self.current_state: Optional[DefenderState] = None
        self.history: list[DefenderState] = []
        # 7-line summary template (kept short to not blow setter prompt budget)
        self._summary_template = (
            "Defender state at start of round {round}:\n"
            "  • LoRA path:        {lora_path}\n"
            "  • Overall catch rate (last round): {catch_rate:.2%} "
            "(caught {caught}/{seen})\n"
            "  • Per-family catch rates:          {family_rates}\n"
            "  • Defender confidence avg:         {conf:.2f}\n"
            "  • Likely cues defender catches:    {cues}\n"
            "  ★ ATTACKER directive: avoid the cues above; "
            "favor under-tested chain shapes."
        )

    # ──────── lifecycle ────────
    def start_round(self, round_id: int, lora_path: str) -> DefenderState:
        """Initialize round state; LoRA assumed already mounted via vLLM."""
        st = DefenderState(round_id=round_id, lora_path=lora_path, lora_loaded=True)
        self.current_state = st
        return st

    def record_rollout(self, family: str, bypass: bool,
                         defender_confidence: float = 0.5,
                         defender_reasoning: str = "") -> None:
        """Per-rollout update — keep running tallies of defender state."""
        if self.current_state is None:
            self.current_state = DefenderState(round_id=0, lora_path="?")
        st = self.current_state
        st.n_traj_seen += 1
        if not bypass: st.n_caught += 1
        st.catch_rate = st.n_caught / max(st.n_traj_seen, 1)
        st.avg_confidence = (st.avg_confidence * (st.n_traj_seen - 1)
                             + defender_confidence) / max(st.n_traj_seen, 1)
        # accumulate per-family
        fam_row = st.family_catch_rates.setdefault(
            family, {"n": 0, "caught": 0, "rate": 0.0}
        )
        fam_row["n"] += 1
        if not bypass: fam_row["caught"] += 1
        fam_row["rate"] = fam_row["caught"] / max(fam_row["n"], 1)
        # gather distinctive cues defender mentions (cheap heuristic — top phrases)
        if defender_reasoning:
            r = defender_reasoning.lower()
            cues = []
            for keyword in ("frequency", "artifact", "asymmetry", "jpeg", "blur",
                              "lighting", "texture", "specular", "seam", "swap",
                              "warp", "halo", "edge", "skin", "noise"):
                if keyword in r:
                    cues.append(keyword)
            if cues:
                existing = (st.cue_summary or "").split(",")
                existing = [x.strip() for x in existing if x.strip()]
                merged = sorted(set(existing + cues))[:8]
                st.cue_summary = ", ".join(merged)

    # ──────── reload LoRA ────────
    def reload_lora(self, new_lora_path: str) -> bool:
        """Hot-reload LoRA via vLLM (when supported).
        Returns True on success, False on failure (requires full restart)."""
        try:
            # 1. unload old
            requests.post(f"{self.endpoint}/unload_lora_adapter",
                          json={"lora_name": self.defender_name}, timeout=10)
        except Exception:
            pass
        try:
            r = requests.post(
                f"{self.endpoint}/load_lora_adapter",
                json={"lora_name": self.defender_name, "lora_path": new_lora_path},
                timeout=30,
            )
            if r.status_code == 200:
                if self.current_state:
                    self.current_state.lora_path = new_lora_path
                    self.current_state.lora_loaded = True
                _log.info(f"[coevo] hot-reloaded LoRA → {new_lora_path}")
                return True
        except Exception as e:
            _log.warning(f"[coevo] hot-reload failed: {e}")
        # fallback: external script restart
        try:
            import subprocess
            ret = subprocess.run(
                ["bash",
                 "/data/disk4/lyx_ICML/self_evolution_forgery/scripts/"
                 "p1a_multi_round/restart_vllm_with_lora.sh",
                 new_lora_path],
                capture_output=True, text=True, timeout=180,
            )
            if ret.returncode == 0:
                if self.current_state:
                    self.current_state.lora_path = new_lora_path
                    self.current_state.lora_loaded = True
                return True
            _log.warning(f"[coevo] restart_vllm failed: {ret.stderr[:200]}")
        except Exception as e:
            _log.warning(f"[coevo] subprocess restart failed: {e}")
        return False

    # ──────── 一起更新 = broadcast defender state to attacker ────────
    def broadcast_to_attacker(
        self,
        attacker_skill_lib: object,                   # ace_skill_lib.SkillLibrary
        family_agents: Optional[object] = None,       # family_attack_agents.FamilyAttackAgentPool
        markov: Optional[object] = None,              # markov_family.MarkovFamilySelector
    ) -> dict:
        """Push current DefenderState into attacker's brain.

        Returns dict of what was updated.
        """
        if self.current_state is None:
            return {"updated": False, "reason": "no state"}
        st = self.current_state
        # 1. Compose human-readable summary
        fam_rates_str = ", ".join(
            f"{fam}={v['rate']:.0%}" for fam, v in sorted(
                st.family_catch_rates.items(), key=lambda kv: -kv[1]['rate'])[:5]
        ) or "(none yet)"
        summary = self._summary_template.format(
            round=st.round_id, lora_path=Path(st.lora_path).name,
            catch_rate=st.catch_rate, caught=st.n_caught, seen=st.n_traj_seen,
            family_rates=fam_rates_str, conf=st.avg_confidence,
            cues=st.cue_summary or "(unspecified)",
        )
        updates: dict = {"summary": summary, "channels": []}

        # 2. Push into Ace-Skill skill_lib (v2) — append to every family's doc
        if attacker_skill_lib is not None and hasattr(attacker_skill_lib, "docs"):
            try:
                for fam, doc in attacker_skill_lib.docs.items():
                    # the SkillDoc class has a `.content` attribute & update method
                    new_content = (doc.content + "\n\n## Defender feedback (round "
                                    f"{st.round_id})\n{summary}")
                    # cap at ~5000 chars
                    if len(new_content) > 5000:
                        new_content = new_content[-5000:]
                    doc.content = new_content
                updates["channels"].append(f"skill_lib_v2 (×{len(attacker_skill_lib.docs)} families)")
            except Exception as e:
                _log.warning(f"[coevo] skill_lib broadcast failed: {e}")

        # 3. Push into 9-agent family pool — update each family's skill_doc with
        # the per-family catch rate (highly targeted feedback)
        if family_agents is not None and hasattr(family_agents, "agents"):
            try:
                for fam, ag in family_agents.agents.items():
                    catch = st.family_catch_rates.get(fam, {})
                    if catch.get("n", 0) == 0: continue
                    snippet = (f"\n\n## R{st.round_id} defender feedback for {fam}\n"
                                f"- defender caught {catch['caught']}/{catch['n']} = {catch['rate']:.0%}\n"
                                f"- cues defender uses: {st.cue_summary or 'unspecified'}\n"
                                f"- avoid those cues; explore chain shapes not seen this round.")
                    ag.skill_doc = (ag.skill_doc + snippet)[-5000:]
                updates["channels"].append("family_agents (per-family targeted)")
            except Exception as e:
                _log.warning(f"[coevo] family_agents broadcast failed: {e}")

        # 4. Markov — boost families with LOW catch_rate (attacker hasn't broken them yet)
        # → re-direct attacker exploration toward easier wins
        if markov is not None and hasattr(markov, "boost_weak_family"):
            try:
                weak_for_attacker = []
                for fam, v in st.family_catch_rates.items():
                    # high defender catch = attacker weak there — boost OPPOSITE (low catch families)
                    if v["rate"] >= 0.7:
                        weak_for_attacker.append(fam)   # families where attacker losing
                if weak_for_attacker and hasattr(markov, "families"):
                    # boost the *other* families (where attacker has chance)
                    for fam in markov.families:
                        if fam in weak_for_attacker: continue
                        if fam in markov.families:
                            idx = markov.families.index(fam)
                            markov.boost_weak_family(idx, boost=0.1 * self.coupling)
                    updates["channels"].append(f"markov boost (×{len(markov.families)-len(weak_for_attacker)})")
            except Exception as e:
                _log.warning(f"[coevo] markov broadcast failed: {e}")

        return updates

    # ──────── snapshot ────────
    def snapshot(self, out_dir: str) -> str:
        """Persist current DefenderState as JSON + 同时写 latest.json (cross-process load)."""
        if self.current_state is None: return ""
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(out_dir) / f"defender_state_r{self.current_state.round_id}.json"
        from dataclasses import asdict
        out_path.write_text(json.dumps(asdict(self.current_state),
                                         ensure_ascii=False, indent=2))
        # ★ BUG-5 fix: also write latest.json so next process can resume state
        latest_path = Path(out_dir) / "latest.json"
        latest_path.write_text(json.dumps(asdict(self.current_state),
                                            ensure_ascii=False, indent=2))
        self.history.append(self.current_state)
        return str(out_path)

    @classmethod
    def load_latest(cls, out_dir: str,
                     vllm_endpoint: str = "http://localhost:8000/v1") -> "CoEvolutionLoop":
        """★ BUG-5 fix: cross-process resume — next orchestrator process loads last state.
        If no prior state, returns fresh instance."""
        co = cls(vllm_endpoint=vllm_endpoint)
        latest = Path(out_dir) / "latest.json"
        if latest.exists():
            try:
                d = json.loads(latest.read_text())
                co.current_state = DefenderState(
                    round_id=int(d.get("round_id", 0)),
                    lora_path=d.get("lora_path", ""),
                    lora_loaded=bool(d.get("lora_loaded", False)),
                    n_traj_seen=int(d.get("n_traj_seen", 0)),
                    n_caught=int(d.get("n_caught", 0)),
                    catch_rate=float(d.get("catch_rate", 0.0)),
                    family_catch_rates=d.get("family_catch_rates", {}),
                    avg_confidence=float(d.get("avg_confidence", 0.0)),
                    cue_summary=d.get("cue_summary", ""),
                )
                _log.info(f"[coevo] loaded prior state from {latest}: "
                          f"R{co.current_state.round_id}, "
                          f"catch_rate={co.current_state.catch_rate:.2%}")
            except Exception as e:
                _log.warning(f"[coevo] load_latest failed: {e}; starting fresh")
        return co


# ────────────────────────── smoke ──────────────────────────────

if __name__ == "__main__":
    import tempfile
    logging.basicConfig(level=logging.INFO)
    co = CoEvolutionLoop(coupling_strength=1.0)
    st = co.start_round(round_id=0, lora_path="/tmp/fake_lora_r0")
    # simulate 10 rollouts
    rolls = [
        ("frontal_swap", True,  0.30, "Frequency cues normal; texture looks natural; no obvious seam."),
        ("frontal_swap", False, 0.82, "Visible swap seam at the temple; jpeg artifact on hairline."),
        ("reenact",      False, 0.78, "Slight asymmetry in eye movement; lighting inconsistent."),
        ("morph",        True,  0.40, "Image seems authentic; no morph artifact detected."),
        ("morph",        False, 0.85, "Eye-color blend; dual-identity ghost artifact in irises."),
        ("frontal_swap", False, 0.91, "JPEG block visible; ArcFace identity drift."),
        ("reenact",      False, 0.80, "Inter-frame jitter; mouth-cavity teeth artifact."),
        ("reenact",      True,  0.35, "Smooth motion; natural skin texture preserved."),
        ("audio_synth",  False, 0.77, "Phoneme-viseme mismatch; lip-sync drift."),
        ("audio_synth",  False, 0.83, "Prosody flatness; audio-visual sync issue."),
    ]
    for fam, byp, conf, reason in rolls:
        co.record_rollout(fam, byp, conf, reason)
    print("=== DefenderState after 10 rollouts ===")
    print(f"  catch_rate: {st.catch_rate:.2%}  ({st.n_caught}/{st.n_traj_seen})")
    print(f"  per-family: {st.family_catch_rates}")
    print(f"  avg conf:   {st.avg_confidence:.3f}")
    print(f"  cues:       {st.cue_summary}")

    # mock attacker components
    class MockDoc:
        def __init__(self): self.content = "## frontal_swap initial SOP\n- blend=0.7 + jpeg q=85"
    class MockSkillLib:
        def __init__(self): self.docs = {"frontal_swap": MockDoc(),
                                          "reenact": MockDoc(),
                                          "morph": MockDoc(),
                                          "audio_synth": MockDoc()}
    class MockMarkov:
        families = ["frontal_swap", "reenact", "morph", "audio_synth"]
        def boost_weak_family(self, idx, boost): pass

    sl = MockSkillLib()
    mk = MockMarkov()
    updates = co.broadcast_to_attacker(attacker_skill_lib=sl, markov=mk)
    print("\n=== broadcast_to_attacker ===")
    print(f"  channels: {updates['channels']}")
    print(f"\n  summary pushed to skill_doc:")
    print("  " + updates["summary"].replace("\n", "\n  "))
    print(f"\n  frontal_swap doc after broadcast (tail 600c):")
    print("  " + sl.docs["frontal_swap"].content[-600:].replace("\n", "\n  "))
    print("\nsmoke PASS")
