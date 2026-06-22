"""Seed library for Method 1 (内部文章图1/图3/图9 verbatim 高分回池).

参考 DARWIN/strategy/strategy_pool.py (SQLite + ChromaDB) — 这里 face-forgery 化:
  - SQLite: chain_id, family, chain_json, 4dim scores, sandbox_success, attempts, generation
  - ChromaDB: 语义去重 (chain_str 作 doc)
  - bootstrap_from_brief_pool: 初始种子从 multi_agent 历史 brief 抽
  - promote_chain: MCTS 跑出的高分 chain 自动入库 (P0-3)
  - get_top_seeds: 下一轮 MCTS seed_chain 优先从 top_k 抽 (高分回池)
  - prune: consecutive_failures >= 5 的种子 silent

内部文章图1: "建立智能体评测的风险/攻击手法矩阵体系...形成数据生成体系"
内部文章图3: "高分变体回池成为新种子, 低分变体淘汰 — 实现了攻击集的自动进化"

不依赖 LLM, 纯本地数据结构.
"""
from __future__ import annotations
import json
import sqlite3
import time
import hashlib
import logging
from pathlib import Path
from typing import Optional

_log = logging.getLogger(__name__)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS seed_chains (
    chain_id TEXT PRIMARY KEY,
    family   TEXT NOT NULL,
    chain_json TEXT NOT NULL,          -- list[dict] serialized
    chain_str TEXT NOT NULL,           -- " → ".join(tools), for dedup
    attack_success REAL DEFAULT 0,
    coverage REAL DEFAULT 0,
    generalization REAL DEFAULT 0,
    defense_evasion REAL DEFAULT 0,
    weighted_score REAL DEFAULT 0,
    sandbox_success_count INTEGER DEFAULT 0,
    sandbox_trial_count INTEGER DEFAULT 0,
    consecutive_failures INTEGER DEFAULT 0,
    generation INTEGER DEFAULT 0,
    parent_chain_id TEXT DEFAULT '',
    status TEXT DEFAULT 'active',      -- active | silent
    source TEXT DEFAULT 'mcts',        -- mcts | brief | bootstrap
    created_at REAL,
    updated_at REAL
);
CREATE INDEX IF NOT EXISTS idx_seed_family_status ON seed_chains (family, status);
CREATE INDEX IF NOT EXISTS idx_seed_score ON seed_chains (weighted_score DESC);
"""


def _chain_key(family: str, chain: list) -> str:
    """Stable hash for a (family, chain) pair."""
    chain_str = " → ".join(s.get("tool", "?") for s in chain)
    h = hashlib.md5(f"{family}|{chain_str}".encode()).hexdigest()[:12]
    return f"{family}_{h}"


class SimpleSeedLibrary:
    """Method 1 种子库 (内部文章 verbatim 自动进化机制).

    Two backends:
      - SQLite: structured stats + retrieval
      - ChromaDB (optional): semantic dedup (skips if chromadb missing)
    """

    def __init__(self,
                 db_path: str = "outputs/seed_library/seeds.db",
                 chroma_persist_dir: Optional[str] = None,
                 semantic_dedup_threshold: float = 0.92,
                 prune_max_failures: int = 5):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.executescript(_SCHEMA)
        self.conn.commit()
        self.prune_max_failures = prune_max_failures
        # ChromaDB optional
        self.chroma_col = None
        self.semantic_dedup_threshold = semantic_dedup_threshold
        try:
            import chromadb
            persist = chroma_persist_dir or str(Path(db_path).parent / "chroma")
            Path(persist).mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=persist)
            col_name = f"seeds_{Path(db_path).stem}"[:60]
            self.chroma_col = client.get_or_create_collection(name=col_name)
        except Exception as e:
            _log.info(f"[seed_library] chromadb unavailable: {e}; sem-dedup disabled")

    # ──────── 入库 ────────
    def _is_semantic_duplicate(self, chain_str: str) -> tuple[bool, Optional[str]]:
        if self.chroma_col is None or self.chroma_col.count() == 0:
            return False, None
        try:
            res = self.chroma_col.query(query_texts=[chain_str], n_results=1)
            dists = (res.get("distances") or [[]])[0]
            ids = (res.get("ids") or [[]])[0]
            if dists and ids:
                # cos_sim = 1 - dist²/2  (chromadb L2 on normalized)
                sim = max(0.0, 1.0 - (dists[0] ** 2) / 2.0)
                if sim >= self.semantic_dedup_threshold:
                    return True, ids[0]
        except Exception:
            pass
        return False, None

    def promote_chain(self, family: str, chain: list,
                       four_dim: dict, source: str = "mcts",
                       parent_chain_id: str = "") -> Optional[str]:
        """High-score chain auto-入库 (P0-3).
        Returns chain_id if added, None if dedup'd."""
        chain_str = " → ".join(s.get("tool", "?") for s in chain)
        chain_id = _chain_key(family, chain)
        # exact-key existing?
        cur = self.conn.execute(
            "SELECT chain_id, weighted_score, sandbox_trial_count FROM seed_chains WHERE chain_id=?",
            (chain_id,))
        row = cur.fetchone()
        if row is not None:
            # update score (EMA)
            old_score = float(row[1]); old_trials = int(row[2])
            new_score = (old_score * old_trials + four_dim.get("weighted", 0.5)) / (old_trials + 1)
            self.conn.execute(
                "UPDATE seed_chains SET weighted_score=?, sandbox_trial_count=?, updated_at=? "
                "WHERE chain_id=?",
                (new_score, old_trials + 1, time.time(), chain_id))
            self.conn.commit()
            return None
        # semantic dedup
        dup, matched = self._is_semantic_duplicate(chain_str)
        if dup:
            return None
        now = time.time()
        gen = 0
        if parent_chain_id:
            pcur = self.conn.execute(
                "SELECT generation FROM seed_chains WHERE chain_id=?", (parent_chain_id,))
            prow = pcur.fetchone()
            if prow: gen = int(prow[0]) + 1
        self.conn.execute(
            "INSERT INTO seed_chains (chain_id, family, chain_json, chain_str, "
            "attack_success, coverage, generalization, defense_evasion, weighted_score, "
            "generation, parent_chain_id, source, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)",
            (chain_id, family, json.dumps(chain, ensure_ascii=False), chain_str,
             float(four_dim.get("attack_success", 0)),
             float(four_dim.get("coverage", 0)),
             float(four_dim.get("generalization", 0)),
             float(four_dim.get("defense_evasion", 0)),
             float(four_dim.get("weighted", 0)),
             gen, parent_chain_id, source, now, now))
        self.conn.commit()
        if self.chroma_col is not None:
            try:
                self.chroma_col.add(documents=[chain_str], ids=[chain_id],
                                     metadatas=[{"family": family, "source": source}])
            except Exception:
                pass
        return chain_id

    # ──────── 出库 ────────
    def get_top_seeds(self, family: str, top_k: int = 5) -> list[dict]:
        """高分回池: 下一轮 MCTS seed 优先抽 top-k."""
        cur = self.conn.execute(
            "SELECT chain_id, chain_json, weighted_score, attack_success, coverage, "
            "generalization, defense_evasion, generation FROM seed_chains "
            "WHERE family=? AND status='active' ORDER BY weighted_score DESC LIMIT ?",
            (family, top_k))
        out = []
        for row in cur.fetchall():
            out.append({
                "chain_id": row[0],
                "chain": json.loads(row[1]),
                "weighted_score": row[2],
                "attack_success": row[3], "coverage": row[4],
                "generalization": row[5], "defense_evasion": row[6],
                "generation": row[7],
            })
        return out

    def get_all_active(self, family: Optional[str] = None) -> list[dict]:
        if family:
            cur = self.conn.execute(
                "SELECT chain_id, family, chain_json, weighted_score, sandbox_trial_count, "
                "sandbox_success_count, consecutive_failures, generation FROM seed_chains "
                "WHERE family=? AND status='active'", (family,))
        else:
            cur = self.conn.execute(
                "SELECT chain_id, family, chain_json, weighted_score, sandbox_trial_count, "
                "sandbox_success_count, consecutive_failures, generation FROM seed_chains "
                "WHERE status='active'")
        return [{"chain_id": r[0], "family": r[1], "chain": json.loads(r[2]),
                 "weighted_score": r[3], "sandbox_trial_count": r[4],
                 "sandbox_success_count": r[5], "consecutive_failures": r[6],
                 "generation": r[7]} for r in cur.fetchall()]

    # ──────── 记录尝试 + 淘汰 ────────
    def record_attempt(self, chain_id: str, success: bool, score: float = 0.0) -> None:
        cur = self.conn.execute(
            "SELECT sandbox_trial_count, sandbox_success_count, consecutive_failures "
            "FROM seed_chains WHERE chain_id=?", (chain_id,))
        row = cur.fetchone()
        if row is None:
            return
        trials = int(row[0]) + 1
        successes = int(row[1]) + (1 if success else 0)
        cons_fail = 0 if success else int(row[2]) + 1
        self.conn.execute(
            "UPDATE seed_chains SET sandbox_trial_count=?, sandbox_success_count=?, "
            "consecutive_failures=?, updated_at=? WHERE chain_id=?",
            (trials, successes, cons_fail, time.time(), chain_id))
        self.conn.commit()

    def prune(self) -> list[str]:
        """淘汰低分: consecutive_failures ≥ threshold → status='silent'.
        Returns list of newly silent chain_ids."""
        cur = self.conn.execute(
            "SELECT chain_id FROM seed_chains WHERE status='active' AND consecutive_failures >= ?",
            (self.prune_max_failures,))
        silent_ids = [row[0] for row in cur.fetchall()]
        if silent_ids:
            placeholders = ",".join("?" for _ in silent_ids)
            self.conn.execute(
                f"UPDATE seed_chains SET status='silent', updated_at=? "
                f"WHERE chain_id IN ({placeholders})",
                [time.time()] + silent_ids)
            self.conn.commit()
        return silent_ids

    # ──────── 初始化种子 ────────
    def bootstrap_from_brief_chains(self, family: str, brief_chains: list[list],
                                      source: str = "bootstrap") -> int:
        """初始种子从 multi_agent brief 的 suggested_chain 抽."""
        added = 0
        for chain in brief_chains:
            chain_dicts = [s if isinstance(s, dict) else {"tool": s, "params": {}}
                           for s in chain]
            cid = self.promote_chain(
                family=family, chain=chain_dicts,
                four_dim={"weighted": 0.5, "attack_success": 0.5,
                          "coverage": 0.5, "generalization": 0.5,
                          "defense_evasion": 0.5},
                source=source,
            )
            if cid is not None: added += 1
        return added

    # ──────── stats ────────
    def stats(self) -> dict:
        cur = self.conn.execute(
            "SELECT family, status, COUNT(*), AVG(weighted_score) FROM seed_chains "
            "GROUP BY family, status")
        rows = cur.fetchall()
        out = {}
        for fam, st, n, avg_s in rows:
            out.setdefault(fam, {})[st] = {"count": n, "avg_weighted_score": round(avg_s or 0, 3)}
        return out

    def close(self):
        self.conn.close()


# ────────────────────────── smoke ──────────────────────────────

if __name__ == "__main__":
    import tempfile
    logging.basicConfig(level=logging.INFO)
    with tempfile.TemporaryDirectory() as td:
        lib = SimpleSeedLibrary(db_path=f"{td}/seeds.db")
        chains_added = lib.bootstrap_from_brief_chains(
            "frontal_swap",
            [["face_align", "inswapper_128_local"],
             ["face_align", "simswap_256_local", "jpeg_85"]])
        print(f"bootstrap: +{chains_added}")
        # promote a high-score MCTS finding
        cid = lib.promote_chain("frontal_swap",
            [{"tool": "face_align"}, {"tool": "inswapper_128_local"},
             {"tool": "gpt_image_two"}, {"tool": "jpeg_85"}],
            {"weighted": 0.82, "attack_success": 0.85, "coverage": 0.75,
             "generalization": 0.80, "defense_evasion": 0.85})
        print(f"promoted: {cid}")
        # try to re-promote same chain — should dedup
        cid2 = lib.promote_chain("frontal_swap",
            [{"tool": "face_align"}, {"tool": "inswapper_128_local"},
             {"tool": "gpt_image_two"}, {"tool": "jpeg_85"}],
            {"weighted": 0.85, "attack_success": 0.88, "coverage": 0.70,
             "generalization": 0.85, "defense_evasion": 0.88})
        print(f"re-promote (should be None): {cid2}")
        # top seeds
        top = lib.get_top_seeds("frontal_swap", top_k=3)
        print(f"top-3 seeds for frontal_swap:")
        for s in top:
            print(f"  [{s['chain_id'][:18]}] score={s['weighted_score']:.3f} "
                  f"chain={[c['tool'] for c in s['chain']]}")
        # record failure for one chain
        lib.record_attempt(cid, success=False, score=0.0)
        lib.record_attempt(cid, success=False, score=0.0)
        # prune (won't fire yet, max=5)
        print(f"stats: {lib.stats()}")
        print(f"prune (max_fail=5): {lib.prune()}")
        print("smoke PASS")
