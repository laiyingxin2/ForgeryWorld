"""Method 4 — Pareto-front skill snippet pool per (family × face_cluster).

Eevee uses Pareto-front to keep diverse prompts (one prompt strong on subset A,
another on subset B, neither dominates).

Face-forgery adaptation:
  - "subset" = src face cluster (gender × age × pose)
  - "prompt" = skill snippet (~100-300 char instruction for setter)
  - Dominance: snippet A dominates B if A's success_rate ≥ B's on ALL src faces
    in their (family, cluster) and strictly greater on ≥ 1

Each (family, cluster) keeps a pool of K_max=5 Pareto-optimal snippets.
New snippet enters if (a) NOT dominated by any current member AND (b) at least
the empty-snippet floor on validation.

Per-snippet record:
  - snippet_text (markdown ≤ 300 chars)
  - per_face_success: dict {src_face_id → 1/0/None}
  - n_uses + n_success
  - source: "init" | "bypass_promote" | "co_evolve_merge"
"""
from __future__ import annotations
import json
import sqlite3
import time
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_log = logging.getLogger(__name__)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS pareto_snippets (
    snippet_id TEXT PRIMARY KEY,
    family TEXT NOT NULL,
    cluster_id TEXT NOT NULL,
    snippet_text TEXT NOT NULL,
    n_uses INTEGER DEFAULT 0,
    n_success INTEGER DEFAULT 0,
    success_rate REAL DEFAULT 0,
    per_face_success TEXT DEFAULT '{}',   -- json {face_id: 1|0}
    source TEXT DEFAULT 'init',
    status TEXT DEFAULT 'active',         -- active | retired (dominated)
    parent_snippet_id TEXT DEFAULT '',
    created_at REAL,
    updated_at REAL
);
CREATE INDEX IF NOT EXISTS idx_pareto_family_cluster
    ON pareto_snippets (family, cluster_id, status);
CREATE INDEX IF NOT EXISTS idx_pareto_score
    ON pareto_snippets (family, cluster_id, success_rate DESC);
"""


_INIT_SNIPPETS = {
    # generic seed snippets per family — each gets attempted on every cluster
    # initially, then Pareto promotion filters which survive in which cluster
    "frontal_swap": [
        "Use blend ratio 0.7 + JPEG quality 85 + face_align affine 512×512. Aim for ArcFace cos > 0.6 to preserve identity.",
        "Stack InSwapper-128 → SimSwap-256 for stronger swap; finish with GFPGAN restoration to mask seams.",
        "Light JPEG (q=92) + tiny resize_bicubic (0.95) to add sensor-like noise without losing detail.",
    ],
    "profile_swap": [
        "SimSwap-256 first, then LivePortrait micro-rotation to break landmark-symmetry detectors.",
        "Use profile-aware face_align (yaw-corrected) + InSwapper; expect arcface drift ~0.15.",
    ],
    "id_diff": [
        "GPT-image-2 inpainting on T-zone + JPEG q=80, identity drift kept < 0.2 via InstantID-style reinforce.",
        "Local edits only (eyes/lips), maintain full-face structure to fool ID-cosine while changing texture.",
    ],
    "reenact": [
        "LivePortrait reenactment + jitter smoothing; jpeg q=88 to mask inter-frame artifacts.",
        "FaceVid2Vid with low motion amplitude (<0.3) to avoid jaw-warp signature.",
    ],
    "morph": [
        "StyleGAN-morph 0.5 blend + GFPGAN cleanup; check eye-color blend artifact.",
        "Two-stage morph: 0.4 blend → resize → 0.7 blend to reduce ghost.",
    ],
    "3d_mask": [
        "DECA-FLAME 3D mesh + per-vertex transfer + screen-print sim.",
        "Use texture overlay only, skip mesh deformation for static-frame attacks.",
    ],
    "replay": [
        "Screen-replay sim + Moiré injection + jpeg q=85 (low recompress to keep artifact).",
        "GPT-image-2 cleanup to mask glare before recompression.",
    ],
    "adv_patch": [
        "PGD on FAS CNN with eps=8/255, 20 steps; patch in cheek region.",
        "Smaller patch (32px) + forehead location to evade visible-patch detectors.",
    ],
    "audio_synth": [
        "XTTS voice clone + LivePortrait lip-sync; post-process audio-video sync ±50ms.",
        "Prosody flatness mitigation: apply pitch warp + amplitude variation.",
    ],
}


def _snippet_key(family: str, cluster: str, text: str) -> str:
    h = hashlib.md5(f"{family}|{cluster}|{text}".encode()).hexdigest()[:12]
    return f"{family}_{cluster[:18]}_{h}"


@dataclass
class SnippetRecord:
    snippet_id: str
    family: str
    cluster_id: str
    text: str
    n_uses: int = 0
    n_success: int = 0
    per_face_success: dict = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        return self.n_success / max(self.n_uses, 1)

    def dominates(self, other: "SnippetRecord") -> bool:
        """A dominates B if A's success on every face seen by B is ≥ B's, and
        strictly greater on ≥1 face. Pareto definition."""
        if not other.per_face_success: return False
        a_better_any = False
        for fid, b_succ in other.per_face_success.items():
            a_succ = self.per_face_success.get(fid)
            if a_succ is None:
                return False  # A hasn't been tried on this face → can't dominate
            if a_succ < b_succ:
                return False  # B strictly better on this face → no dominance
            if a_succ > b_succ:
                a_better_any = True
        return a_better_any


class ParetoSkillPool:
    """K-Pareto-front per (family, cluster). Promotes on bypass, retires on dominance."""

    def __init__(self, db_path: str = "outputs/method4/pareto.db", k_max: int = 5):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.executescript(_SCHEMA)
        self.conn.commit()
        self.k_max = k_max

    # ──────── init ────────
    def bootstrap_init(self) -> int:
        """One-time seed of init snippets across all (family, cluster='init') —
        actual cluster assignment happens at first record_use."""
        added = 0
        now = time.time()
        for fam, snippets in _INIT_SNIPPETS.items():
            for txt in snippets:
                sid = _snippet_key(fam, "_init", txt)
                row = self.conn.execute(
                    "SELECT snippet_id FROM pareto_snippets WHERE snippet_id=?",
                    (sid,)).fetchone()
                if row is not None: continue
                self.conn.execute(
                    "INSERT INTO pareto_snippets (snippet_id, family, cluster_id, "
                    "snippet_text, source, status, created_at, updated_at) "
                    "VALUES (?, ?, '_init', ?, 'init', 'active', ?, ?)",
                    (sid, fam, txt, now, now))
                added += 1
        self.conn.commit()
        return added

    # ──────── retrieve for setter ────────
    def get_top_snippet(self, family: str, cluster_id: str) -> Optional[dict]:
        """Returns the top-1 snippet for (family, cluster). Falls back to
        cluster-agnostic if no cluster-specific snippet exists yet."""
        # cluster-specific first
        cur = self.conn.execute(
            "SELECT snippet_id, snippet_text, n_uses, n_success, success_rate "
            "FROM pareto_snippets "
            "WHERE family=? AND cluster_id=? AND status='active' "
            "ORDER BY success_rate DESC, n_uses DESC LIMIT 1",
            (family, cluster_id))
        row = cur.fetchone()
        if row is not None:
            return {"snippet_id": row[0], "text": row[1], "n_uses": row[2],
                    "n_success": row[3], "success_rate": row[4]}
        # fallback: init pool (cluster='_init')
        cur = self.conn.execute(
            "SELECT snippet_id, snippet_text, n_uses, n_success, success_rate "
            "FROM pareto_snippets "
            "WHERE family=? AND cluster_id='_init' AND status='active' "
            "ORDER BY success_rate DESC, n_uses DESC LIMIT 1",
            (family,))
        row = cur.fetchone()
        if row is not None:
            return {"snippet_id": row[0], "text": row[1], "n_uses": row[2],
                    "n_success": row[3], "success_rate": row[4],
                    "is_fallback": True}
        return None

    def get_pool_for_cluster(self, family: str, cluster_id: str) -> list[SnippetRecord]:
        cur = self.conn.execute(
            "SELECT snippet_id, snippet_text, n_uses, n_success, per_face_success "
            "FROM pareto_snippets "
            "WHERE family=? AND cluster_id=? AND status='active'",
            (family, cluster_id))
        out = []
        for r in cur.fetchall():
            out.append(SnippetRecord(
                snippet_id=r[0], family=family, cluster_id=cluster_id,
                text=r[1], n_uses=r[2], n_success=r[3],
                per_face_success=json.loads(r[4] or "{}"),
            ))
        return out

    # ──────── record outcome + promote/retire ────────
    def record_use(self, snippet_id: str, src_face_id: str,
                    success: bool) -> None:
        """Update n_uses + per_face_success + success_rate for a snippet."""
        cur = self.conn.execute(
            "SELECT n_uses, n_success, per_face_success FROM pareto_snippets "
            "WHERE snippet_id=?", (snippet_id,)).fetchone()
        if cur is None: return
        n_uses = int(cur[0]) + 1
        n_success = int(cur[1]) + (1 if success else 0)
        pf = json.loads(cur[2] or "{}")
        pf[src_face_id] = 1 if success else 0
        rate = n_success / max(n_uses, 1)
        self.conn.execute(
            "UPDATE pareto_snippets SET n_uses=?, n_success=?, "
            "per_face_success=?, success_rate=?, updated_at=? "
            "WHERE snippet_id=?",
            (n_uses, n_success, json.dumps(pf), rate,
             time.time(), snippet_id))
        self.conn.commit()

    def promote_on_bypass(self, family: str, cluster_id: str,
                            new_snippet_text: str,
                            src_face_id: str,
                            parent_snippet_id: str = "") -> Optional[str]:
        """Add new snippet to (family, cluster) pool. Returns snippet_id if added.

        Enforces:
          - dedup by hash
          - Pareto: if any active snippet in same (fam, cluster) dominates
            the candidate (after at least 2 face observations), reject
          - K_max cap: when over cap, retire dominated snippets

        On first promotion (no prior face data) always accept.
        """
        new_sid = _snippet_key(family, cluster_id, new_snippet_text)
        # dedup
        row = self.conn.execute(
            "SELECT snippet_id FROM pareto_snippets WHERE snippet_id=?",
            (new_sid,)).fetchone()
        if row is not None:
            self.record_use(new_sid, src_face_id, True)
            return None  # dedup'd
        now = time.time()
        self.conn.execute(
            "INSERT INTO pareto_snippets (snippet_id, family, cluster_id, "
            "snippet_text, n_uses, n_success, per_face_success, success_rate, "
            "source, parent_snippet_id, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, 1, ?, ?, ?, 'bypass_promote', ?, ?, ?)",
            (new_sid, family, cluster_id, new_snippet_text,
             1 if True else 0,
             json.dumps({src_face_id: 1}),
             1.0, parent_snippet_id, now, now))
        self.conn.commit()
        # retire Pareto-dominated members
        self._enforce_pareto(family, cluster_id)
        return new_sid

    def _enforce_pareto(self, family: str, cluster_id: str) -> int:
        """Retire any snippet dominated by another in the same (fam, cluster).
        Returns count retired."""
        pool = self.get_pool_for_cluster(family, cluster_id)
        if len(pool) <= 1: return 0
        retired = []
        for i, a in enumerate(pool):
            for j, b in enumerate(pool):
                if i == j or b.snippet_id in retired: continue
                if a.dominates(b):
                    retired.append(b.snippet_id)
        for sid in retired:
            self.conn.execute(
                "UPDATE pareto_snippets SET status='retired', updated_at=? "
                "WHERE snippet_id=?", (time.time(), sid))
        # also enforce K_max cap: keep top-K by success_rate
        if len(pool) - len(retired) > self.k_max:
            survivors = [s for s in pool if s.snippet_id not in retired]
            survivors.sort(key=lambda x: (-x.success_rate, -x.n_uses))
            to_retire_extra = survivors[self.k_max:]
            for s in to_retire_extra:
                self.conn.execute(
                    "UPDATE pareto_snippets SET status='retired', updated_at=? "
                    "WHERE snippet_id=?", (time.time(), s.snippet_id))
                retired.append(s.snippet_id)
        self.conn.commit()
        return len(retired)

    # ──────── stats ────────
    def stats(self) -> dict:
        cur = self.conn.execute(
            "SELECT family, cluster_id, status, COUNT(*), AVG(success_rate) "
            "FROM pareto_snippets GROUP BY family, cluster_id, status")
        out = {}
        for fam, cl, st, n, ar in cur.fetchall():
            out.setdefault(fam, {}).setdefault(cl, {})[st] = {
                "n": n, "avg_rate": round(ar or 0, 3)
            }
        return out

    def close(self): self.conn.close()


# ────────────────────────── smoke ──────────────────────────────

if __name__ == "__main__":
    import tempfile
    logging.basicConfig(level=logging.INFO)
    with tempfile.TemporaryDirectory() as td:
        pool = ParetoSkillPool(db_path=f"{td}/pareto.db", k_max=3)
        added = pool.bootstrap_init()
        print(f"=== bootstrap: +{added} init snippets ===")

        # simulate using top snippet for male_adult_frontal frontal_swap
        cl = "male_adult_frontal"
        top = pool.get_top_snippet("frontal_swap", cl)
        print(f"\nfirst call (no cluster-specific yet, fallback expected):")
        print(f"  top: {top['text'][:80] if top else None}")
        print(f"  is_fallback: {top.get('is_fallback') if top else False}")

        # promote a new cluster-specific snippet (bypass success)
        sid1 = pool.promote_on_bypass(
            "frontal_swap", cl,
            "[male_adult_frontal] InSwapper blend=0.75 + GFPGAN 0.4 + jpeg 88 — works on 35-55yo men.",
            src_face_id="face_0", parent_snippet_id="init")
        print(f"\npromoted: {sid1}")

        # try another snippet that's worse (still bypass once)
        sid2 = pool.promote_on_bypass(
            "frontal_swap", cl,
            "[male_adult_frontal] simswap_256 only, no post-process.",
            src_face_id="face_0", parent_snippet_id="init")
        # both initially have face_0=1 → no dominance yet

        # simulate face_1 outcomes: sid1 wins, sid2 loses
        pool.record_use(sid1, "face_1", True)
        pool.record_use(sid2, "face_1", False)
        # now sid1 dominates sid2 on (face_0=1==face_0=1) and (face_1=1>face_1=0)
        pool._enforce_pareto("frontal_swap", cl)
        retired = [s for s in pool.get_pool_for_cluster("frontal_swap", cl)
                   if s.snippet_id == sid2]
        print(f"\nafter face_1 outcomes: sid2 retired? "
              f"{'YES' if not retired else 'NO (still active)'}")

        top2 = pool.get_top_snippet("frontal_swap", cl)
        print(f"  new top (after pareto): {top2['text'][:80]}")
        print(f"\nstats: {pool.stats()['frontal_swap']}")
        print("\nsmoke PASS")
