"""VideoWeaver 2-layer skill split (arxiv 2606.08091, ByteDance + Zhejiang U).

User survey 推荐 #3: 非常新的多模态生成技能进化方案 (2026-06).

VideoWeaver 把视频生成技能拆 2 类:
  - **Composition Skill** (合成层): 任务拆解、镜头安排、工具编排、工作流规划
                                   → 在 face-forgery 就是 "chain shape" (which ops in which order)
  - **Creator Skill**     (创作层): 具体提示词、视觉效果、转场、音频、参数
                                   → 在 face-forgery 就是 "op-specific params" (blend, JPEG q, prompt)

Judge Agent 同时检查 整个工具调用 + 最终视频, 评分 + 入库新技能.

之前 Ace-Skill (ace_skill_lib.py) 是 single-layer skill_doc per family — Composition
和 Creator 混在一起. VideoWeaver 的 split 让两种 skill 各自进化、各自检索, 互不干扰.

实现:
  - CompositionSkill: family + chain_shape (op-name-only) + 成功率 + LLM rationale
  - CreatorSkill: family + op_name + recommended params + 成功率
  - SkillManager: 出题时分别 retrieve 2 类 top-k → 拼成 brief
  - update: 成功 rollout 后, chain shape 入 Composition 库, 每个 op 的 params 入 Creator 库
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
CREATE TABLE IF NOT EXISTS composition_skills (
    comp_id TEXT PRIMARY KEY,
    family TEXT NOT NULL,
    chain_shape TEXT NOT NULL,         -- json list of op names
    chain_str TEXT NOT NULL,           -- " → ".join(ops) for human view + dedup
    n_uses INTEGER DEFAULT 0,
    n_success INTEGER DEFAULT 0,
    success_rate REAL DEFAULT 0,
    rationale TEXT DEFAULT '',
    last_updated REAL
);
CREATE INDEX IF NOT EXISTS idx_comp_family_rate
    ON composition_skills (family, success_rate DESC);

CREATE TABLE IF NOT EXISTS creator_skills (
    creator_id TEXT PRIMARY KEY,
    family TEXT NOT NULL,
    op_name TEXT NOT NULL,
    recommended_params TEXT NOT NULL,  -- json dict
    params_str TEXT NOT NULL,          -- for human + dedup
    n_uses INTEGER DEFAULT 0,
    n_success INTEGER DEFAULT 0,
    success_rate REAL DEFAULT 0,
    rationale TEXT DEFAULT '',
    last_updated REAL
);
CREATE INDEX IF NOT EXISTS idx_creator_family_op_rate
    ON creator_skills (family, op_name, success_rate DESC);
"""


def _hash_key(prefix: str, *parts: str) -> str:
    h = hashlib.md5("|".join(parts).encode()).hexdigest()[:12]
    return f"{prefix}_{h}"


class VideoWeaverSkills:
    """2-layer skill library (Composition + Creator)."""

    def __init__(self, db_path: str = "outputs/videoweaver_skills/skills.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # ──────── Composition ────────
    def update_composition(self, family: str, chain: list, success: bool,
                            rationale: str = "") -> str:
        """Record a chain-shape usage. Returns comp_id."""
        ops = [s.get("tool", "?") for s in chain]
        chain_str = " → ".join(ops)
        comp_id = _hash_key("comp", family, chain_str)
        row = self.conn.execute(
            "SELECT n_uses, n_success FROM composition_skills WHERE comp_id=?",
            (comp_id,)
        ).fetchone()
        if row is None:
            self.conn.execute(
                "INSERT INTO composition_skills "
                "(comp_id, family, chain_shape, chain_str, n_uses, n_success, "
                "success_rate, rationale, last_updated) "
                "VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?)",
                (comp_id, family, json.dumps(ops, ensure_ascii=False), chain_str,
                 1 if success else 0, 1.0 if success else 0.0,
                 rationale[:300], time.time()))
        else:
            n_uses = int(row[0]) + 1
            n_success = int(row[1]) + (1 if success else 0)
            rate = n_success / max(n_uses, 1)
            self.conn.execute(
                "UPDATE composition_skills SET n_uses=?, n_success=?, success_rate=?, "
                "rationale=?, last_updated=? WHERE comp_id=?",
                (n_uses, n_success, rate,
                 (rationale or "")[:300], time.time(), comp_id))
        self.conn.commit()
        return comp_id

    def get_top_compositions(self, family: str, top_k: int = 3,
                              min_uses: int = 1) -> list[dict]:
        cur = self.conn.execute(
            "SELECT comp_id, chain_shape, n_uses, n_success, success_rate, rationale "
            "FROM composition_skills WHERE family=? AND n_uses>=? "
            "ORDER BY success_rate DESC, n_uses DESC LIMIT ?",
            (family, min_uses, top_k))
        return [{"comp_id": r[0], "chain_shape": json.loads(r[1]),
                 "n_uses": r[2], "n_success": r[3],
                 "success_rate": round(r[4], 3),
                 "rationale": r[5]} for r in cur.fetchall()]

    # ──────── Creator ────────
    def update_creator(self, family: str, op_name: str, params: dict,
                        success: bool, rationale: str = "") -> str:
        """Record a (op, params) usage for a family. Returns creator_id."""
        params_str = json.dumps(params, ensure_ascii=False, sort_keys=True)
        creator_id = _hash_key("crt", family, op_name, params_str)
        row = self.conn.execute(
            "SELECT n_uses, n_success FROM creator_skills WHERE creator_id=?",
            (creator_id,)
        ).fetchone()
        if row is None:
            self.conn.execute(
                "INSERT INTO creator_skills "
                "(creator_id, family, op_name, recommended_params, params_str, "
                "n_uses, n_success, success_rate, rationale, last_updated) "
                "VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, ?)",
                (creator_id, family, op_name, params_str, params_str,
                 1 if success else 0, 1.0 if success else 0.0,
                 rationale[:300], time.time()))
        else:
            n_uses = int(row[0]) + 1
            n_success = int(row[1]) + (1 if success else 0)
            rate = n_success / max(n_uses, 1)
            self.conn.execute(
                "UPDATE creator_skills SET n_uses=?, n_success=?, success_rate=?, "
                "rationale=?, last_updated=? WHERE creator_id=?",
                (n_uses, n_success, rate,
                 (rationale or "")[:300], time.time(), creator_id))
        self.conn.commit()
        return creator_id

    def get_top_creators(self, family: str, op_name: str, top_k: int = 3,
                          min_uses: int = 1) -> list[dict]:
        cur = self.conn.execute(
            "SELECT creator_id, recommended_params, n_uses, n_success, "
            "success_rate, rationale FROM creator_skills "
            "WHERE family=? AND op_name=? AND n_uses>=? "
            "ORDER BY success_rate DESC, n_uses DESC LIMIT ?",
            (family, op_name, min_uses, top_k))
        return [{"creator_id": r[0], "params": json.loads(r[1]),
                 "n_uses": r[2], "n_success": r[3],
                 "success_rate": round(r[4], 3),
                 "rationale": r[5]} for r in cur.fetchall()]

    # ──────── Update on rollout outcome ────────
    def record_rollout(self, family: str, chain: list, success: bool,
                        rationale: str = "", reasoning: str = "") -> dict:
        """One call after each rollout: update both Composition + Creator.

        `reasoning` is an alias for `rationale` (orchestrator passes either name).
        """
        if reasoning and not rationale:
            rationale = reasoning
        comp_id = self.update_composition(family, chain, success, rationale)
        creator_ids = []
        for step in chain:
            op = step.get("tool", "?")
            params = step.get("params", {}) or {}
            if not params: continue   # skip ops with no params (face_align etc.)
            cid = self.update_creator(family, op, params, success, rationale)
            creator_ids.append(cid)
        return {"comp_id": comp_id, "creator_ids": creator_ids}

    # ──────── Compose recommendation for next brief ────────
    def recommend_brief(self, family: str,
                         top_comp_k: int = 1,
                         top_creator_k: int = 1) -> Optional[dict]:
        """Build a brief recommendation using best Composition + best Creator per op.
        Returns dict {chain: [{tool, params}, ...], rationale: str} or None."""
        comps = self.get_top_compositions(family, top_k=top_comp_k, min_uses=1)
        if not comps: return None
        comp = comps[0]
        chain_shape = comp["chain_shape"]
        recommended_chain = []
        for op in chain_shape:
            creators = self.get_top_creators(family, op, top_k=top_creator_k, min_uses=1)
            params = creators[0]["params"] if creators else {}
            recommended_chain.append({"tool": op, "params": params})
        return {
            "chain": recommended_chain,
            "rationale": (f"Composition top-1 (success_rate={comp['success_rate']:.2%} "
                           f"over {comp['n_uses']} uses); Creator params per-op from best fits."),
            "source_comp_id": comp["comp_id"],
        }

    def stats(self) -> dict:
        ncomp = self.conn.execute("SELECT COUNT(*) FROM composition_skills").fetchone()[0]
        ncreator = self.conn.execute("SELECT COUNT(*) FROM creator_skills").fetchone()[0]
        avg_comp_rate = self.conn.execute(
            "SELECT AVG(success_rate) FROM composition_skills"
        ).fetchone()[0] or 0.0
        avg_creator_rate = self.conn.execute(
            "SELECT AVG(success_rate) FROM creator_skills"
        ).fetchone()[0] or 0.0
        return {
            "n_compositions": ncomp,
            "n_creators": ncreator,
            "avg_composition_success_rate": round(avg_comp_rate, 3),
            "avg_creator_success_rate": round(avg_creator_rate, 3),
        }

    def close(self): self.conn.close()


# ────────────────────────── smoke ──────────────────────────────

if __name__ == "__main__":
    import tempfile
    logging.basicConfig(level=logging.INFO)
    with tempfile.TemporaryDirectory() as td:
        sk = VideoWeaverSkills(db_path=f"{td}/skills.db")
        # simulate 4 rollouts, 3 success + 1 failure
        sk.record_rollout("frontal_swap",
            [{"tool": "face_align"}, {"tool": "inswapper_128_local", "params": {"blend": 0.7}},
             {"tool": "jpeg_85", "params": {"quality": 85}}], success=True)
        sk.record_rollout("frontal_swap",
            [{"tool": "face_align"}, {"tool": "inswapper_128_local", "params": {"blend": 0.7}},
             {"tool": "jpeg_85", "params": {"quality": 85}}], success=True)
        sk.record_rollout("frontal_swap",
            [{"tool": "face_align"}, {"tool": "inswapper_128_local", "params": {"blend": 0.7}},
             {"tool": "gpt_image_two", "params": {"prompt": "natural"}}], success=True)
        sk.record_rollout("frontal_swap",
            [{"tool": "face_align"}, {"tool": "simswap_256_local", "params": {"blend": 0.5}},
             {"tool": "resize_bicubic", "params": {"scale": 0.5}}], success=False)
        # query
        print("=== top composition for frontal_swap ===")
        for c in sk.get_top_compositions("frontal_swap"):
            print(f"  [{c['comp_id'][:14]}] {' → '.join(c['chain_shape']):60s} "
                  f"{c['n_success']}/{c['n_uses']} = {c['success_rate']}")
        print("\n=== top creator (inswapper_128_local) for frontal_swap ===")
        for c in sk.get_top_creators("frontal_swap", "inswapper_128_local"):
            print(f"  [{c['creator_id'][:14]}] params={c['params']} "
                  f"{c['n_success']}/{c['n_uses']} = {c['success_rate']}")
        print("\n=== recommended brief ===")
        rec = sk.recommend_brief("frontal_swap")
        print(f"  {rec}")
        print(f"\n  stats: {sk.stats()}")
        print("smoke PASS")
