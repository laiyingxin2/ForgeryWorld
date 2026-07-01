"""Layer 1 — Markov Attack Family Selector (MAJIC verbatim).

Wraps DARWIN's MarkovSelector with face-attack 9 families and simplified API.
Original DARWIN code: external/DARWIN/attack/markov_selector.py lines 60-98.

Verbatim formulas (from MAJIC paper, confirmed in DARWIN code):
  M_ij^new = M_ij + α[r + γ·max_k(M_jk) − M_ij]    α=0.1, γ=0.5
  softmax(M_i,:) at T=0.15                          (DARWIN default)
  smoothing: M^reset = (1-β)M + β/K                 β=0.1

不直接 import DARWIN sqlite/chroma 重依赖, 在我们这个项目里 transition matrix 用 numpy + jsonl persist.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict, field

import numpy as np

from trajectory_schema import attack_family_list


_log = logging.getLogger(__name__)


# ────────────────────────── Hyperparams (verbatim) ──────────────────

@dataclass
class MarkovConfig:
    alpha: float = 0.1           # MAJIC update lr (DARWIN ALPHA)
    gamma: float = 0.5           # MAJIC discount (DARWIN GAMMA)
    beta_uniform_mix: float = 0.1  # smoothing toward uniform (DARWIN BETA)
    temperature: float = 0.15    # softmax T (DARWIN TEMPERATURE)
    init_uniform: bool = True    # init M with 1/K
    relative_reward: bool = True  # r_t = bypass_rate_family - bypass_rate_global


# ────────────────────────── Core ────────────────────────────────────

class MarkovFamilySelector:
    """K=9 face-attack families, MAJIC-style transition matrix.

    Usage:
        sel = MarkovFamilySelector()
        # round-by-round loop:
        next_family = sel.select_next(current_family_idx=0)
        # after running an attack and observing reward:
        sel.update(from_idx=0, to_idx=next_family, reward=0.7)
        sel.save("outputs/markov/round_3.json")
    """

    def __init__(
        self,
        families: Optional[list[str]] = None,
        config: Optional[MarkovConfig] = None,
    ):
        self.families = families or attack_family_list()
        self.K = len(self.families)
        self.cfg = config or MarkovConfig()

        if self.cfg.init_uniform:
            self.M = np.ones((self.K, self.K), dtype=np.float64) / self.K
        else:
            self.M = np.eye(self.K) * 0.5 + np.ones((self.K, self.K)) / (2 * self.K)
            self._normalize_rows()

        # 累计 reward 统计 — 用于 relative reward 计算 + diagnosis
        self.family_rewards = np.zeros(self.K)
        self.family_attempts = np.zeros(self.K, dtype=np.int64)

    def _normalize_rows(self):
        row_sums = self.M.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        self.M = self.M / row_sums

    def index_of(self, family_name: str) -> int:
        return self.families.index(family_name)

    def name_of(self, idx: int) -> str:
        return self.families[idx]

    def global_bypass_rate(self) -> float:
        total = self.family_attempts.sum()
        if total == 0:
            return 0.0
        return float(self.family_rewards.sum() / total)

    def family_bypass_rate(self, idx: int) -> float:
        if self.family_attempts[idx] == 0:
            return 0.0
        return float(self.family_rewards[idx] / self.family_attempts[idx])

    # ───── DARWIN markov_selector.py lines 60-78 (verbatim adapted) ─────
    def select_next(self, current_family_idx: int) -> int:
        """Softmax sample next family from M[current, :] at T=0.15."""
        if not (0 <= current_family_idx < self.K):
            return int(np.random.choice(self.K))

        row = self.M[current_family_idx].copy()
        # DARWIN: mix matrix probs with softmax Q + uniform
        # 我们简化: 直接 softmax 行 + uniform smoothing
        q_shifted = row - row.max()
        q_probs = np.exp(q_shifted / max(self.cfg.temperature, 1e-8))
        q_probs = q_probs / q_probs.sum()
        uniform = np.ones(self.K) / self.K
        probs = (1 - self.cfg.beta_uniform_mix) * q_probs + self.cfg.beta_uniform_mix * uniform
        probs = probs / probs.sum()

        return int(np.random.choice(self.K, p=probs))

    # ───── DARWIN markov_selector.py lines 80-98 (verbatim adapted) ─────
    def update(
        self,
        from_idx: int,
        to_idx: int,
        reward: float,
    ):
        """M_ij^new = M_ij + α·(r + γ·max_k M_jk - M_ij).

        reward: bypass_rate in [0,1] for this family in this round.
        If config.relative_reward, will use (reward - global_bypass_rate) instead.
        """
        if not (0 <= from_idx < self.K and 0 <= to_idx < self.K):
            return

        # 累积 stat
        self.family_rewards[to_idx] += reward
        self.family_attempts[to_idx] += 1

        # Relative reward (推荐, paper 也用相对的)
        if self.cfg.relative_reward:
            r = reward - self.global_bypass_rate()
        else:
            r = reward

        max_future = float(self.M[to_idx].max())
        delta = self.cfg.alpha * (r + self.cfg.gamma * max_future - self.M[from_idx, to_idx])
        self.M[from_idx, to_idx] = max(self.M[from_idx, to_idx] + delta, 0.0)
        self._normalize_rows()

    def boost_weak_family(self, weak_idx: int, boost: float = 0.15):
        """Layer 9 diagnosis-driven exploration boost.

        增加 weak_idx 这一列的所有 transition prob, normalize.
        """
        self.M[:, weak_idx] = self.M[:, weak_idx] + boost
        self._normalize_rows()

    def weak_families(self, top_n: int = 2) -> list[int]:
        """返回 bypass_rate 最低的 top_n family idx. Layer 9 用."""
        rates = np.array([self.family_bypass_rate(i) for i in range(self.K)])
        # 优先选 attempts > 0 的;尚未尝试的算 0
        return list(np.argsort(rates)[:top_n])

    def transition_summary(self) -> dict:
        """Stats for logging / diagnosis."""
        return {
            "families": self.families,
            "matrix": self.M.tolist(),
            "family_bypass_rate": [self.family_bypass_rate(i) for i in range(self.K)],
            "family_attempts": self.family_attempts.tolist(),
            "global_bypass_rate": self.global_bypass_rate(),
        }

    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "config": asdict(self.cfg),
            "families": self.families,
            "M": self.M.tolist(),
            "family_rewards": self.family_rewards.tolist(),
            "family_attempts": self.family_attempts.tolist(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "MarkovFamilySelector":
        with open(path) as f:
            data = json.load(f)
        sel = cls(
            families=data["families"],
            config=MarkovConfig(**data["config"]),
        )
        sel.M = np.array(data["M"])
        sel.family_rewards = np.array(data["family_rewards"])
        sel.family_attempts = np.array(data["family_attempts"], dtype=np.int64)
        return sel


# ────────────────────────── Smoke test ──────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.random.seed(42)

    sel = MarkovFamilySelector()
    print(f"K = {sel.K} families: {sel.families}")
    print(f"Initial M[0, :5] = {sel.M[0][:5]}")  # uniform 1/9 ≈ 0.111

    # 模拟 50 step: 从 family 0 开始, 随机奖励
    current = 0
    history = [current]
    for step in range(50):
        nxt = sel.select_next(current)
        # 模拟: frontal_swap (0) 和 id_diff (2) 成功率高, others 低
        if nxt in (0, 2):
            reward = 0.8 + np.random.rand() * 0.15
        else:
            reward = 0.1 + np.random.rand() * 0.2
        sel.update(current, nxt, reward)
        current = nxt
        history.append(current)

    print(f"\nAfter 50 steps:")
    print(f"  global bypass rate = {sel.global_bypass_rate():.3f}")
    summary = sel.transition_summary()
    for i, fam in enumerate(sel.families):
        print(f"  {fam:15s}: attempts={summary['family_attempts'][i]:3d}, "
              f"rate={summary['family_bypass_rate'][i]:.3f}")
    print(f"  weak top-2 = {[sel.families[i] for i in sel.weak_families(2)]}")

    # 测保存 / 加载
    sel.save("/tmp/markov_smoke.json")
    sel2 = MarkovFamilySelector.load("/tmp/markov_smoke.json")
    assert np.allclose(sel.M, sel2.M)
    print("\n✓ save/load round-trip OK")
