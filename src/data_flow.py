"""Layer 7 — Iterative Data Flow (UI-TARS-2 V=1→SFT, V=0→CT) + Jacquard dedupe.

SQLite for structured state, ChromaDB optional (we use in-memory dict fallback when chromadb 不可用).
内部文章 verbatim:
  J_word(resp, target) = |W_resp ∩ W_target| / |W_resp ∪ W_target|
"""
from __future__ import annotations
import json
import logging
import re
import sqlite3
import hashlib
from pathlib import Path
from typing import Optional, Iterable
from dataclasses import dataclass

import numpy as np


_log = logging.getLogger(__name__)


# ────────────────────────── Jacquard dedupe (内部文章 verbatim) ─────

def tokenize_words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def jacquard_word(a: str, b: str) -> float:
    """J_word(resp, target) = |W_a ∩ W_b| / |W_a ∪ W_b|"""
    sa, sb = tokenize_words(a), tokenize_words(b)
    if not sa and not sb:
        return 0.0
    inter = sa & sb
    union = sa | sb
    return len(inter) / max(len(union), 1)


# ────────────────────────── SQLite store ────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trajectories (
    trajectory_id     TEXT PRIMARY KEY,
    round_id          INTEGER,
    baseline          TEXT,
    attack_family     TEXT,
    sandbox_pass      INTEGER,
    data_route        TEXT,
    cost_usd          REAL,
    timestamp         REAL,
    jacquard_key      TEXT,
    full_json         TEXT,
    detector_signature TEXT,
    policy_signature  TEXT
);
CREATE INDEX IF NOT EXISTS idx_round_family ON trajectories(round_id, attack_family);
CREATE INDEX IF NOT EXISTS idx_route ON trajectories(data_route);
CREATE INDEX IF NOT EXISTS idx_pass ON trajectories(sandbox_pass);

-- BUG D 修: PK 加 run_id 前缀避免跨 run 覆盖
CREATE TABLE IF NOT EXISTS markov_snapshots (
    run_id            TEXT DEFAULT 'legacy',
    round_id          INTEGER,
    matrix_json       TEXT,
    family_rewards    TEXT,
    family_attempts   TEXT,
    timestamp         REAL,
    PRIMARY KEY (run_id, round_id)
);

CREATE TABLE IF NOT EXISTS skill_versions (
    family           TEXT,
    version          INTEGER,
    skill_doc        TEXT,
    timestamp        REAL,
    PRIMARY KEY (family, version)
);

CREATE TABLE IF NOT EXISTS bypass_stats (
    run_id           TEXT DEFAULT 'legacy',
    round_id         INTEGER,
    family           TEXT,
    attempts         INTEGER,
    bypass_count     INTEGER,
    bypass_rate      REAL,
    PRIMARY KEY (run_id, round_id, family)
);
"""


class DataFlow:
    """Layer 7 main entry. Wraps SQLite + Jacquard dedupe + ChromaDB semantic dedupe + routing."""

    def __init__(self, db_path: str = "outputs/data_flow.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.executescript(_SCHEMA_SQL)
        self.conn.commit()
        # ── P2: ChromaDB semantic dedupe (内部文章 verbatim 0.95 sem) ──
        self.chroma_client = None
        self.chroma_col = None
        try:
            import chromadb
            chroma_dir = str(Path(db_path).parent / "chroma_persist")
            Path(chroma_dir).mkdir(parents=True, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=chroma_dir)
            # Use unique collection per db to avoid cross-run contamination
            col_name = f"trajectories_{Path(db_path).stem}"[:60]
            self.chroma_col = self.chroma_client.get_or_create_collection(name=col_name)
        except Exception as e:
            import logging
            logging.getLogger(__name__).info(f"[data_flow] chromadb unavailable: {e}; sem-dedupe disabled")
            self.chroma_client = None
            self.chroma_col = None

    # ────── routing (UI-TARS-2 V=1→SFT, V=0→CT, low-quality→DROP) ─────

    def route(self, trajectory_dict: dict, quality_thresh: float = 60.0) -> str:
        """Return 'SFT' / 'CT' / 'DROP' for a trajectory dict.

        SFT: bypass succeeded AND quality high
        CT:  bypass failed OR moderate quality (no waste, becomes continual training)
        DROP: very low quality / error
        """
        v = trajectory_dict.get("verdicts") or {}
        sandbox_pass = bool(v.get("sandbox_pass", False))
        tier1 = v.get("tier1") or {}
        niqe = tier1.get("niqe", 10.0)
        arcface = tier1.get("arcface_id_sim", -1.0)

        if not sandbox_pass:
            return "CT"

        # SFT 条件: bypass + 质量过关
        # NIQE 越低越好 (<8), ArcFace ID-sim 中等 (避免身份完全 unchanged)
        if niqe < 9.0 and (arcface == -1.0 or 0.3 < arcface < 0.95):
            return "SFT"
        if niqe > 12.0:
            return "DROP"
        return "CT"

    # ────── dedupe via Jacquard (内部文章 verbatim) ──────

    def jacquard_signature(self, trajectory_dict: dict) -> str:
        """Build dedup signature string used for J_word comparison."""
        parts = [
            trajectory_dict.get("attack_family", ""),
            " ".join(s.get("tool", "") for s in trajectory_dict.get("execution", [])),
        ]
        brief = trajectory_dict.get("brief")
        if brief:
            parts.append(brief.get("attack_class", ""))
            parts.append(brief.get("brief_text", "")[:200])
        return " ".join(parts)

    def is_duplicate(
        self,
        new_sig: str,
        threshold: float = 0.8,
        check_last_n: int = 200,
        semantic_threshold: float = 0.92,
    ) -> tuple[bool, Optional[str]]:
        """Two-stage dedupe: Jacquard word (lex, 0.8) → ChromaDB sem (0.95).
        Returns (is_dup, matched_trajectory_id_or_None)."""
        # stage 1: Jacquard lexical
        cur = self.conn.execute(
            "SELECT trajectory_id, jacquard_key FROM trajectories "
            "ORDER BY rowid DESC LIMIT ?", (check_last_n,)
        )
        for tid, jk in cur.fetchall():
            if not jk: continue
            j = jacquard_word(new_sig, jk)
            if j > threshold:
                return True, tid
        # stage 2: ChromaDB semantic (if available)
        if self.chroma_col is not None:
            try:
                count = self.chroma_col.count()
                if count > 0:
                    res = self.chroma_col.query(query_texts=[new_sig], n_results=1)
                    dists = res.get("distances", [[]])[0] if res.get("distances") else []
                    ids = res.get("ids", [[]])[0] if res.get("ids") else []
                    if dists and ids:
                        # chromadb default embedder (all-MiniLM-L6-v2) returns
                        # L2 distance on L2-normalized vectors, so
                        # cos_sim = 1 - dist²/2 (verified empirically: d≈0 → sim≈1,
                        # d≈0.32 → sim≈0.949).
                        sim = max(0.0, 1.0 - (dists[0] ** 2) / 2.0)
                        if sim >= semantic_threshold:
                            return True, ids[0]
            except Exception:
                pass
        return False, None

    def _chroma_add(self, traj_id: str, sig: str) -> None:
        if self.chroma_col is None: return
        try:
            self.chroma_col.add(documents=[sig], ids=[traj_id])
        except Exception:
            pass  # duplicate id is fine

    # ────── store ──────

    def commit_trajectory(self, trajectory_dict: dict, allow_duplicate: bool = False) -> str:
        """Insert with dedupe + routing. Returns 'SFT' / 'CT' / 'DROP' / 'DUPLICATE'."""
        sig = self.jacquard_signature(trajectory_dict)
        if not allow_duplicate:
            dup, matched = self.is_duplicate(sig)
            if dup:
                trajectory_dict["data_route"] = "DUPLICATE"
                trajectory_dict["jacquard_dedupe_key"] = sig
                return "DUPLICATE"

        route = self.route(trajectory_dict)
        trajectory_dict["data_route"] = route
        trajectory_dict["jacquard_dedupe_key"] = sig

        v = trajectory_dict.get("verdicts") or {}
        self.conn.execute(
            "INSERT OR REPLACE INTO trajectories "
            "(trajectory_id, round_id, baseline, attack_family, sandbox_pass, "
            "data_route, cost_usd, timestamp, jacquard_key, full_json, "
            "detector_signature, policy_signature) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                trajectory_dict.get("trajectory_id", ""),
                trajectory_dict.get("round_id", 0),
                trajectory_dict.get("baseline", "v2"),
                trajectory_dict.get("attack_family", ""),
                int(bool(v.get("sandbox_pass", False))),
                route,
                trajectory_dict.get("cost_usd", 0.0),
                trajectory_dict.get("timestamp", 0.0),
                sig,
                json.dumps(trajectory_dict, ensure_ascii=False),
                trajectory_dict.get("detector_signature", ""),
                trajectory_dict.get("policy_signature", ""),
            ),
        )
        self.conn.commit()
        # also add to ChromaDB for future semantic dedup
        self._chroma_add(trajectory_dict.get("trajectory_id", ""), sig)
        return route

    # ────── stats ──────

    def round_bypass_stats(self, round_id: int, run_id: str = "legacy") -> dict:
        # 限定当前 run_id 的 trajectory (BUG D 修)
        cur = self.conn.execute(
            "SELECT attack_family, COUNT(*), SUM(sandbox_pass) "
            "FROM trajectories WHERE round_id=? AND trajectory_id LIKE ? "
            "GROUP BY attack_family",
            (round_id, f"{run_id}%" if run_id != "legacy" else "%")
        )
        rows = cur.fetchall()
        stats = {}
        for fam, cnt, byp in rows:
            byp = byp or 0
            stats[fam] = {
                "attempts": cnt,
                "bypass_count": byp,
                "bypass_rate": byp / max(cnt, 1),
            }
            self.conn.execute(
                "INSERT OR REPLACE INTO bypass_stats "
                "(run_id, round_id, family, attempts, bypass_count, bypass_rate) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, round_id, fam, cnt, byp, byp / max(cnt, 1)),
            )
        self.conn.commit()
        return stats

    def snapshot_markov(self, round_id: int, markov_dict: dict, run_id: str = "legacy"):
        import time as _t
        self.conn.execute(
            "INSERT OR REPLACE INTO markov_snapshots VALUES (?, ?, ?, ?, ?, ?)",
            (
                run_id,
                round_id,
                json.dumps(markov_dict.get("matrix", [])),
                json.dumps(markov_dict.get("family_bypass_rate", [])),
                json.dumps(markov_dict.get("family_attempts", [])),
                _t.time(),
            ),
        )
        self.conn.commit()

    def snapshot_skill(self, family: str, version: int, skill_doc: str):
        import time as _t
        self.conn.execute(
            "INSERT OR REPLACE INTO skill_versions VALUES (?, ?, ?, ?)",
            (family, version, skill_doc, _t.time()),
        )
        self.conn.commit()

    def export_sft_pool(self, out_jsonl: str, round_id: Optional[int] = None) -> int:
        """Layer 10 接口: export SFT trajectories for defender training."""
        Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)
        if round_id is None:
            cur = self.conn.execute(
                "SELECT full_json FROM trajectories WHERE data_route='SFT'"
            )
        else:
            cur = self.conn.execute(
                "SELECT full_json FROM trajectories WHERE data_route='SFT' AND round_id <= ?",
                (round_id,)
            )
        with open(out_jsonl, "w") as f:
            n = 0
            for (j,) in cur:
                f.write(j + "\n")
                n += 1
        return n

    def stats_summary(self) -> dict:
        cur = self.conn.execute(
            "SELECT data_route, COUNT(*) FROM trajectories GROUP BY data_route"
        )
        return dict(cur.fetchall())

    def close(self):
        self.conn.close()


# ────────────────────────── Smoke test ──────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import time as _t

    df = DataFlow(db_path="/tmp/data_flow_smoke.db")
    print("=== Jacquard test ===")
    a = "frontal_swap face_align inswapper_128 gfpgan jpeg_85"
    b = "frontal_swap face_align inswapper_128 gfpgan jpeg_75"
    c = "id_diff nano_banana_pro gpt_image_two jpeg_85"
    print(f"  J(a, b) = {jacquard_word(a, b):.3f}  (should be high, just QP differ)")
    print(f"  J(a, c) = {jacquard_word(a, c):.3f}  (should be low)")

    print("\n=== Commit 3 trajectories ===")
    for i, (sandbox_pass, niqe, sig_text) in enumerate([
        (True,  6.2, "frontal_swap inswapper jpeg"),
        (True,  7.1, "frontal_swap inswapper jpeg"),  # near-duplicate of #0
        (False, 8.5, "id_diff nano_banana_two gpt_image_two"),
    ]):
        traj = {
            "trajectory_id": f"r0_b{i}",
            "round_id": 0,
            "baseline": "v2",
            "attack_family": sig_text.split()[0],
            "verdicts": {
                "sandbox_pass": sandbox_pass,
                "tier1": {"niqe": niqe, "arcface_id_sim": 0.7},
            },
            "execution": [{"tool": t} for t in sig_text.split()],
            "brief": {"attack_class": sig_text.split()[0], "brief_text": sig_text},
            "cost_usd": 0.005,
            "timestamp": _t.time(),
        }
        route = df.commit_trajectory(traj)
        print(f"  trajectory r0_b{i}: route={route}, sandbox_pass={sandbox_pass}, niqe={niqe}")

    print(f"\n=== Routes summary ===")
    print(f"  {df.stats_summary()}")

    print(f"\n=== Round 0 bypass stats ===")
    for fam, st in df.round_bypass_stats(0).items():
        print(f"  {fam}: {st}")

    # SFT export
    n = df.export_sft_pool("/tmp/sft_export.jsonl")
    print(f"\n=== Exported {n} SFT trajectories to /tmp/sft_export.jsonl ===")
    df.close()
