"""Novelty term — Anti mode-collapse for Markov family selection.

Paper agent (Lv6) 给的建议: AdvEvo-MARL + Multi-Agent Evolve 都用 novelty 项防止
attacker mode-collapse 到单一 high-ROI family.

机制:
  r_novel = success × (1 − max_cosine_sim_to_past_K_attacks)
  → 同一 attack pattern 反复成功也只算 1 次
  → 鼓励 attacker 探索新 family / 新 chain combo

集成点:
  - markov_family.py update() 之前用 reward_with_novelty()
  - Ace-Skill sampling: 对 ℰ_k 的 v_i 加 novelty penalty

cost: 每 trajectory +1 embedding (~$0.0001), or local features 完全免费
"""
from __future__ import annotations
import json
import logging
import time
from pathlib import Path
from typing import Optional, Deque
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ace_skill_lib import _simple_text_features, cosine_sim


_log = logging.getLogger(__name__)


@dataclass
class NoveltyConfig:
    K_history: int = 64                  # 记忆最近 K 个 attack
    novelty_weight: float = 0.4           # r_total = (1-w)·r_bypass + w·r_novel
    drop_below_novelty: float = 0.15      # 如果 attack 新颖度 < 此值, 视为重复


class NoveltyTracker:
    """Maintains rolling embedding bank of recent attacks; computes novelty per new attack."""

    def __init__(self, config: Optional[NoveltyConfig] = None,
                 persist_path: Optional[str] = None):
        self.cfg = config or NoveltyConfig()
        self.history: Deque[dict] = deque(maxlen=self.cfg.K_history)
        self.persist_path = persist_path
        if persist_path:
            self._load()

    def _attack_signature_text(self, attack_family: str, chain: list,
                                src_face_path: str = "") -> str:
        """Build a NL signature for the attack."""
        chain_str = " → ".join(
            s["tool"] if isinstance(s, dict) else str(s) for s in chain
        )
        # 不要把 src face hash 加进去 — 不同 face 同 chain 应该算同一种 attack pattern
        return f"family={attack_family} chain={chain_str}"

    def novelty_score(self, attack_family: str, chain: list,
                       src_face_path: str = "") -> float:
        """Return novelty in [0, 1]. 1 = totally new, 0 = identical to past."""
        if not self.history:
            return 1.0
        sig = self._attack_signature_text(attack_family, chain, src_face_path)
        emb = _simple_text_features(sig)
        max_sim = 0.0
        for h in self.history:
            s = cosine_sim(emb, h["emb"])
            if s > max_sim:
                max_sim = s
        return float(1.0 - max_sim)

    def record(self, attack_family: str, chain: list,
               bypass: bool, src_face_path: str = ""):
        """Add to history. Use bypass to weight (failed attacks contribute less to history)."""
        sig = self._attack_signature_text(attack_family, chain, src_face_path)
        self.history.append({
            "family": attack_family,
            "sig": sig,
            "emb": _simple_text_features(sig),
            "bypass": bypass,
            "timestamp": time.time(),
        })
        if self.persist_path:
            self._save()

    def composite_reward(self, bypass: float, attack_family: str,
                          chain: list, src_face_path: str = "") -> dict:
        """r_total = (1-w)·bypass + w·novelty.

        Returns dict with detail for logging.
        """
        novelty = self.novelty_score(attack_family, chain, src_face_path)
        w = self.cfg.novelty_weight
        r_total = (1.0 - w) * float(bypass) + w * novelty
        return {
            "bypass_reward": float(bypass),
            "novelty_score": novelty,
            "novelty_weight": w,
            "composite_reward": r_total,
            "is_repeated": novelty < self.cfg.drop_below_novelty,
        }

    def diversity_index(self) -> float:
        """Average pairwise distance among recent history. Higher = more diverse."""
        if len(self.history) < 2:
            return 1.0
        embs = [h["emb"] for h in self.history]
        total_dist = 0.0
        cnt = 0
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                total_dist += 1.0 - cosine_sim(embs[i], embs[j])
                cnt += 1
        return float(total_dist / max(cnt, 1))

    def family_distribution(self) -> dict:
        """How many of recent K attacks per family. Tells mode-collapse situation."""
        dist = {}
        for h in self.history:
            f = h["family"]
            dist[f] = dist.get(f, 0) + 1
        return dist

    def _save(self):
        if not self.persist_path:
            return
        Path(self.persist_path).parent.mkdir(parents=True, exist_ok=True)
        data = []
        for h in self.history:
            d = dict(h)
            d.pop("emb", None)
            data.append(d)
        Path(self.persist_path).write_text(json.dumps(data, ensure_ascii=False))

    def _load(self):
        p = Path(self.persist_path)
        if not p.exists():
            return
        try:
            data = json.loads(p.read_text())
            for d in data[-self.cfg.K_history:]:
                d["emb"] = _simple_text_features(d.get("sig", ""))
                self.history.append(d)
        except Exception as e:
            _log.warning(f"novelty load failed: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    tr = NoveltyTracker(config=NoveltyConfig(K_history=10, novelty_weight=0.4))

    # 模拟一系列 attack
    attacks = [
        ("frontal_swap", [{"tool": "face_align"}, {"tool": "inswapper_128"},
                          {"tool": "jpeg_85"}], True),
        ("frontal_swap", [{"tool": "face_align"}, {"tool": "inswapper_128"},
                          {"tool": "jpeg_85"}], True),   # 完全重复
        ("frontal_swap", [{"tool": "face_align"}, {"tool": "simswap_256"},
                          {"tool": "jpeg_85"}], True),    # 类似但 op 换
        ("id_diff", [{"tool": "face_align"}, {"tool": "nano_banana_pro"},
                     {"tool": "resize_bicubic"}], False),  # 完全不同 family
    ]
    print("=== Sequence of attacks ===")
    for fam, chain, bypass in attacks:
        r = tr.composite_reward(bypass, fam, chain)
        print(f"  {fam}: novelty={r['novelty_score']:.3f}, bypass={bypass}, "
              f"r_total={r['composite_reward']:.3f}, "
              f"repeated={r['is_repeated']}")
        tr.record(fam, chain, bypass)

    print(f"\n=== diversity_index = {tr.diversity_index():.3f}")
    print(f"=== family_dist = {tr.family_distribution()}")
