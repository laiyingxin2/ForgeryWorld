"""Layer 8 — L1-L5 Memory Hierarchy (内部文章 verbatim).

5 scopes with distinct lifetimes:

  L1 Working      Single rollout. In-prompt only. Cleared per rollout.
                  Backed by an in-memory dict; no persistence.

  L2 Session      Single round (briefs × rollouts). Chat-history-equivalent.
                  Persisted as outputs/<run>/memory_l2/round_<r>.jsonl
                  Cleared at end of round.

  L3 Project      Cross-round persistent state: skill lib + Markov matrix +
                  experience pool. Backed by existing ace_skill_lib +
                  markov_family JSON files (already persistent). This module
                  is a *read-only view* on those, plus a registry.

  L4 Workspace    Cross-attack-family meta-skills (Creator skill from
                  AgentEvolver). E.g. "any swap+jpeg combo with q∈[80,90]
                  bypasses gemini-2.5-flash 60% of the time".
                  Persisted as outputs/<run>/memory_l4/meta_skills.json.

  L5 Crystallized Frozen snapshot of L3+L4 at major checkpoints
                  (release-ready, W12 paper deliverable).
                  Persisted as outputs/<run>/memory_l5/snapshot_<tag>.json.

Interface designed so orchestrator can call cleanly per lifecycle:
  mh.start_rollout(rid)        / mh.end_rollout(rid, summary)
  mh.start_round(rid)          / mh.end_round(rid, diagnosis)
  mh.add_meta_skill(...)       / mh.get_meta_skills()
  mh.crystallize(tag)
"""
from __future__ import annotations
import json
import time
import shutil
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional


# ────────────────────────── data classes ──────────────────────────

@dataclass
class MetaSkill:
    """L4 cross-family Creator skill. Discovered when N families converge
    on a similar pattern; promoted from L3 𝒮_k by supervisor or by
    pattern-mining."""
    name: str                        # short identifier, e.g. "swap+jpeg85"
    description: str                 # one-line summary
    body: str                        # markdown
    spans_families: list[str]        # families this applies to
    discovered_round: int
    applied_count: int = 0
    success_count: int = 0
    timestamp: float = field(default_factory=time.time)

    @property
    def success_rate(self) -> float:
        return self.success_count / max(self.applied_count, 1)


# ────────────────────────── main hierarchy ────────────────────────

class MemoryHierarchy:
    """Layer 8 unified facade for L1-L5 memory."""

    def __init__(self, root: str | Path, run_id: str = ""):
        self.root = Path(root)
        self.run_id = run_id or time.strftime("run-%Y%m%d-%H%M%S")
        # L1: in-memory only (cleared per rollout)
        self._l1_working: dict[str, Any] = {}
        # L2: round-scoped session log; persisted but cleared per round
        self._l2_session: list[dict] = []
        self._l2_dir = self.root / "memory_l2"
        self._l2_dir.mkdir(parents=True, exist_ok=True)
        # L3: read-through to ace_skill_lib + markov (we don't own state, just expose)
        self._l3_view: dict[str, Any] = {}
        # L4: cross-family meta-skills
        self._l4_dir = self.root / "memory_l4"
        self._l4_dir.mkdir(parents=True, exist_ok=True)
        self._l4_path = self._l4_dir / "meta_skills.json"
        self._l4_skills: dict[str, MetaSkill] = self._load_l4()
        # L5: crystallized snapshots
        self._l5_dir = self.root / "memory_l5"
        self._l5_dir.mkdir(parents=True, exist_ok=True)
        self._current_round_id: Optional[int] = None
        self._current_rollout_id: Optional[str] = None

    # ──────── L1 working memory (per rollout) ────────
    def start_rollout(self, rollout_id: str) -> None:
        self._l1_working = {"rollout_id": rollout_id, "started_at": time.time()}
        self._current_rollout_id = rollout_id

    def set_working(self, key: str, value: Any) -> None:
        self._l1_working[key] = value

    def get_working(self, key: str, default: Any = None) -> Any:
        return self._l1_working.get(key, default)

    def end_rollout(self, rollout_id: str, summary: dict) -> None:
        # spill condensed summary into L2 then clear L1
        rec = {"rollout_id": rollout_id, "summary": summary,
               "working_snapshot_keys": list(self._l1_working.keys())}
        self._l2_session.append(rec)
        self._l1_working = {}
        self._current_rollout_id = None

    # ──────── L2 session memory (per round) ────────
    def start_round(self, round_id: int) -> None:
        self._l2_session = []
        self._current_round_id = round_id

    def add_session_record(self, kind: str, payload: dict) -> None:
        """kind ∈ {brief, checker, attribution, sandbox, outcome, supervisor_delta}"""
        self._l2_session.append({
            "kind": kind, "payload": payload,
            "round_id": self._current_round_id,
            "rollout_id": self._current_rollout_id,
            "ts": time.time(),
        })

    def end_round(self, round_id: int, diagnosis: Optional[dict] = None) -> Path:
        """Persist this round's session log and clear L2 buffer."""
        out_path = self._l2_dir / f"round_{round_id:03d}.jsonl"
        with open(out_path, "w") as f:
            if diagnosis is not None:
                f.write(json.dumps({"kind": "diagnosis", "payload": diagnosis,
                                    "round_id": round_id, "ts": time.time()},
                                   ensure_ascii=False) + "\n")
            for r in self._l2_session:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        self._l2_session = []
        self._current_round_id = None
        return out_path

    def get_recent_session(self, n: int = 10) -> list[dict]:
        return self._l2_session[-n:]

    # ──────── L3 project memory (view onto skill_lib + markov) ────────
    def register_l3_view(self, name: str, ref: Any) -> None:
        """Orchestrator hands ace_skill_lib / markov refs in here so L8
        can introspect them. We only hold weakref-like; we don't own."""
        self._l3_view[name] = ref

    def get_l3_view(self, name: str) -> Any:
        return self._l3_view.get(name)

    # ──────── L4 workspace meta-skills (cross-family) ────────
    def _load_l4(self) -> dict[str, MetaSkill]:
        if not self._l4_path.exists():
            return {}
        try:
            raw = json.loads(self._l4_path.read_text())
            return {k: MetaSkill(**v) for k, v in raw.items()}
        except Exception:
            return {}

    def _save_l4(self) -> None:
        out = {k: asdict(v) for k, v in self._l4_skills.items()}
        self._l4_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))

    def add_meta_skill(self, skill: MetaSkill) -> None:
        # dedupe by name
        existing = self._l4_skills.get(skill.name)
        if existing:
            # merge: extend spans_families, keep older discovered_round
            existing.spans_families = sorted(set(existing.spans_families) |
                                              set(skill.spans_families))
            existing.body = skill.body  # last write wins on body
        else:
            self._l4_skills[skill.name] = skill
        self._save_l4()

    def record_meta_skill_application(self, name: str, success: bool) -> None:
        if name not in self._l4_skills:
            return
        s = self._l4_skills[name]
        s.applied_count += 1
        if success: s.success_count += 1
        self._save_l4()

    def get_meta_skills(self, family: Optional[str] = None) -> list[MetaSkill]:
        if family is None:
            return list(self._l4_skills.values())
        return [s for s in self._l4_skills.values() if family in s.spans_families]

    def promote_to_meta_skill(self, family_skills: dict[str, str],
                               threshold_families: int = 2,
                               round_id: int = 0) -> Optional[MetaSkill]:
        """Given {family: skill_doc_text}, detect if N≥threshold families share
        a common pattern. Heuristic: shared 4-gram across docs. If yes, promote.
        Returns the new MetaSkill or None."""
        from collections import Counter
        if len(family_skills) < threshold_families:
            return None
        family_ngrams = {}
        for fam, doc in family_skills.items():
            words = doc.lower().split()
            family_ngrams[fam] = set(
                " ".join(words[i:i+4]) for i in range(max(0, len(words)-3))
            )
        # find ngrams shared by ≥ threshold families
        counter = Counter()
        for fam, ngs in family_ngrams.items():
            for n in ngs:
                counter[n] += 1
        candidates = [(n, c) for n, c in counter.items() if c >= threshold_families]
        if not candidates:
            return None
        # pick the longest shared phrase (most informative)
        candidates.sort(key=lambda x: (-x[1], -len(x[0])))
        top_ngram, top_count = candidates[0]
        # use as meta-skill seed
        name = "meta-" + "-".join(top_ngram.split()[:3])[:60]
        spans = [fam for fam, ngs in family_ngrams.items() if top_ngram in ngs]
        ms = MetaSkill(
            name=name,
            description=f"cross-family pattern seen in {len(spans)} families: '{top_ngram}'",
            body=f"# {name}\n\n## 强制规则\n- pattern: `{top_ngram}`\n- 覆盖 family: {spans}\n",
            spans_families=spans, discovered_round=round_id,
        )
        self.add_meta_skill(ms)
        return ms

    # ──────── L5 crystallized snapshot ────────
    def crystallize(self, tag: str, payload: Optional[dict] = None,
                     extra_files: Optional[list[Path]] = None) -> Path:
        """Freeze current L3+L4 state into a snapshot file (release-ready)."""
        snap = {
            "tag": tag, "ts": time.time(),
            "run_id": self.run_id,
            "l3_view_keys": list(self._l3_view.keys()),
            "l4_meta_skills": {k: asdict(v) for k, v in self._l4_skills.items()},
            "payload": payload or {},
        }
        out_path = self._l5_dir / f"snapshot_{tag}.json"
        out_path.write_text(json.dumps(snap, ensure_ascii=False, indent=2))
        # copy extra files (e.g. LoRA adapter_config, current 𝒮_k) into snapshot dir
        if extra_files:
            extras_dir = self._l5_dir / f"snapshot_{tag}_extras"
            extras_dir.mkdir(exist_ok=True)
            for f in extra_files:
                if Path(f).exists():
                    shutil.copy2(f, extras_dir / Path(f).name)
        return out_path

    # ──────── observability ────────
    def summary(self) -> dict:
        return {
            "run_id": self.run_id,
            "l1_keys": list(self._l1_working.keys()),
            "l2_session_records_in_buffer": len(self._l2_session),
            "l2_persisted_rounds": sorted(p.stem for p in self._l2_dir.glob("round_*.jsonl")),
            "l3_view_refs": list(self._l3_view.keys()),
            "l4_meta_skills": list(self._l4_skills.keys()),
            "l5_snapshots": sorted(p.stem for p in self._l5_dir.glob("snapshot_*.json")),
        }


# ────────────────────────── smoke test ──────────────────────────

if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        mh = MemoryHierarchy(tmp, run_id="smoke")

        # L1
        mh.start_rollout("r0_b0_g0")
        mh.set_working("brief_chain", ["face_align", "inswapper_128_local"])
        assert mh.get_working("brief_chain") == ["face_align", "inswapper_128_local"]
        mh.end_rollout("r0_b0_g0", {"bypass": True, "family": "frontal_swap"})
        assert mh.get_working("brief_chain") is None  # cleared
        print("[L1] ok — working memory cleared per rollout")

        # L2
        mh.start_round(0)
        mh.add_session_record("brief", {"family": "reenact", "chain": ["a", "b"]})
        mh.add_session_record("outcome", {"bypass": True})
        out = mh.end_round(0, diagnosis={"global_bypass_rate": 0.5})
        assert out.exists()
        assert len(out.read_text().splitlines()) == 3   # diagnosis + 2 records
        print(f"[L2] ok — {out} has {len(out.read_text().splitlines())} records")

        # L3 view
        class FakeSkillLib: pass
        mh.register_l3_view("skill_lib", FakeSkillLib())
        assert mh.get_l3_view("skill_lib") is not None
        print("[L3] ok — view registered")

        # L4 meta-skills
        mh.add_meta_skill(MetaSkill(
            name="meta-jpeg-q85", description="jpeg q=85 mask works across swap families",
            body="...", spans_families=["frontal_swap", "profile_swap"], discovered_round=0))
        assert len(mh.get_meta_skills()) == 1
        assert len(mh.get_meta_skills(family="frontal_swap")) == 1
        assert len(mh.get_meta_skills(family="reenact")) == 0
        print(f"[L4] ok — {len(mh.get_meta_skills())} meta-skill(s) stored")

        # promote_to_meta_skill
        ms = mh.promote_to_meta_skill({
            "frontal_swap": "use inswapper 128 then jpeg q 85 to mask the seam",
            "profile_swap": "for profile angles use inswapper 128 then jpeg q 85 boundary blend",
            "reenact": "liveportrait reenactment then add gaussian noise sigma 1.5",
        }, threshold_families=2, round_id=0)
        if ms:
            print(f"[L4] promotion: {ms.name} spans {ms.spans_families}")

        # L5
        snap = mh.crystallize("R0", payload={"global_bypass_rate": 0.5,
                                              "weak_families": ["audio_synth"]})
        assert snap.exists()
        print(f"[L5] ok — snapshot at {snap}")

        # summary
        s = mh.summary()
        print(f"[summary] {json.dumps(s, indent=2, ensure_ascii=False)}")
        print("\nL1-L5 memory hierarchy smoke PASS")
