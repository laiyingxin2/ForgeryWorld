"""Baseline #1 — Internal article port (simplified, 3 classes in one file).

代替 Baseline #2 Layer 1/3/4/6 的简化版:
  - SimpleFamilySelector:    均匀随机 (替 Markov)
  - SimpleSkillBook:          单层 Markdown append-only (替 Ace-Skill 双流)
  - SimpleMCTSPlanner:        UCB1 4-step (替 lookahead k=3 d=2)
  - 6 mutation operators:     face-adapted from 内部文章 verbatim 6 个

Layer 2/5/7/9/10 复用 Baseline #2 的实现 (multi_agent_gen, sandbox, data_flow, ...).

Reward 在 Baseline #1: trajectory-level pass/fail, 无 step-level attribution.
"""
from __future__ import annotations
import json
import logging
import math
import random
from typing import Optional
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from viviai_client import ViviClient
from trajectory_schema import attack_family_list


_log = logging.getLogger(__name__)


# ────────────────────────── Simple family selector ───────────────────

class SimpleFamilySelector:
    """Baseline #1: 均匀随机选攻击 family (no Markov, no learning)."""
    def __init__(self, families: Optional[list[str]] = None):
        self.families = families or attack_family_list()
        self.K = len(self.families)

    def select_next(self, current_idx: int) -> int:
        return random.randrange(self.K)

    def update(self, *args, **kwargs):
        pass  # no learning


# ────────────────────────── Simple skill book ────────────────────────

class SimpleSkillBook:
    """Baseline #1: 每 family 一个 Markdown, append-only, 主管手动写约束.

    No Eq.4 prioritized sample, no Eq.7 merge, no Eq.8 compress — 只是 append.
    """
    def __init__(self, families: list[str], base_dir: str | Path = "outputs/skills_v1"):
        self.families = families
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.docs: dict[str, str] = {
            f: f"# {f} (Baseline #1 simplified)\n\n_No constraints yet._\n"
            for f in families
        }
        self.append_count: dict[str, int] = {f: 0 for f in families}

    def append_constraint(self, family: str, constraint_text: str):
        self.append_count[family] += 1
        self.docs[family] += f"\n## Constraint {self.append_count[family]}\n{constraint_text}\n"

    def get_doc(self, family: str) -> str:
        return self.docs.get(family, "")

    def save_all(self):
        for f in self.families:
            (self.base_dir / f"{f}.md").write_text(self.docs[f])

    def load_all(self):
        for f in self.families:
            path = self.base_dir / f"{f}.md"
            if path.exists():
                self.docs[f] = path.read_text()
                # ★ recover append_count from existing doc
                self.append_count[f] = self.docs[f].count("## Constraint")


# ────────────────────────── 6 mutation operators (内部文章 verbatim, face-adapted) ─

MUTATION_OPERATORS_V1 = {
    # 内部文章 6 个 mutation, face-adapted
    "synonym_rewrite":    "替换同 family 不同 operator (e.g. InSwapper → SimSwap)",
    "context_fusion":     "把当前 attack 叠加另一 family 的 op (e.g. swap + nano_banana_two)",
    "instruction_nest":   "pipeline 深度增加 (5 → 8 步)",
    "language_variant":   "改 prompt 描述方式 (e.g. nanobanana 的 prompt 改写)",
    "format_transform":   "改输出编解码 (JPEG 85 → PNG → WebP → recompress)",
    "semantic_reorder":   "跨 family 重组 step 顺序 (swap → degrade → ID-diff → swap)",
}


def apply_mutation(
    original_chain: list,
    mutation_name: str,
    family: str,
    available_ops: list[str],
) -> list:
    """Apply one of 6 mutations to an existing pipeline. Returns new chain."""
    chain = [dict(s) if isinstance(s, dict) else {"tool": s, "params": {}} for s in original_chain]

    if mutation_name == "synonym_rewrite":
        # 在 chain 里找一个 swap op, 换成同类别另一个
        synonyms = {
            "inswapper_128": ["simswap_256", "roop"],
            "simswap_256": ["inswapper_128", "roop"],
            "roop": ["inswapper_128", "simswap_256"],
            "gfpgan": ["gpt_image_two"],
            "jpeg_85": ["resize_bicubic"],
        }
        for s in chain:
            if s["tool"] in synonyms:
                alt = random.choice(synonyms[s["tool"]])
                if alt in available_ops:
                    s["tool"] = alt
                    return chain
        return chain

    if mutation_name == "context_fusion":
        # 在 swap 后插入一个 nano_banana
        for i, s in enumerate(chain):
            if "swap" in s["tool"]:
                chain.insert(i + 1, {"tool": "nano_banana_two", "params": {}})
                return chain
        chain.append({"tool": "nano_banana_two", "params": {}})
        return chain

    if mutation_name == "instruction_nest":
        # 延长 pipeline
        new_ops = ["gfpgan", "resize_bicubic", "jpeg_85"]
        for op in new_ops:
            if op not in [s["tool"] for s in chain]:
                chain.append({"tool": op, "params": {}})
        return chain

    if mutation_name == "language_variant":
        # 改 nano_banana 的 instruction (在 params 里)
        for s in chain:
            if "nano_banana" in s["tool"]:
                s["params"]["instruction"] = random.choice([
                    "Render as authentic smartphone selfie",
                    "Photorealistic portrait with natural skin texture",
                    "Magazine-style headshot, no AI artifacts",
                ])
                return chain
        # 没有 nano_banana, 插一个
        chain.insert(1, {"tool": "nano_banana_one",
                          "params": {"instruction": "Generate selfie-style natural photo"}})
        return chain

    if mutation_name == "format_transform":
        # 改 jpeg 的 quality 或换格式
        for s in chain:
            if s["tool"] == "jpeg_85":
                s["params"]["quality"] = random.choice([70, 75, 80, 85, 90])
                return chain
        chain.append({"tool": "jpeg_85", "params": {"quality": random.choice([70, 85, 90])}})
        return chain

    if mutation_name == "semantic_reorder":
        if len(chain) <= 3:
            return chain
        # 把最后一步插到中间
        last = chain.pop()
        idx = random.randint(1, len(chain) - 1)
        chain.insert(idx, last)
        return chain

    return chain


# ────────────────────────── MCTS UCB1 4-step (Baseline #1 Layer 4) ───

class MCTSNode:
    def __init__(self, chain: list, parent: Optional["MCTSNode"] = None, mutation: str = ""):
        self.chain = chain
        self.parent = parent
        self.children: list[MCTSNode] = []
        self.mutation = mutation
        self.visits = 0
        self.value_sum = 0.0

    @property
    def value(self) -> float:
        return self.value_sum / max(self.visits, 1)

    def ucb1(self, exploration: float = 1.41) -> float:
        if self.visits == 0:
            return float("inf")
        if self.parent is None or self.parent.visits == 0:
            return self.value
        return self.value + exploration * math.sqrt(math.log(self.parent.visits) / self.visits)


class SimpleMCTSPlanner:
    """Baseline #1 Layer 4: MCTS UCB1 4-step + 4-dim adaptive scoring (内部文章 verbatim).

    4 steps verbatim 内部文章:
      Selection (UCB1) → Expansion → Simulation (4-dim score) → Backpropagation

    Simulation 调 simple_4dim_judge.four_dim_score() 拿 (攻击成功 / 覆盖 / 泛化 / 防御绕过)
    4 维独立分,再 weighted_value() 聚合为标量喂 UCB1 backprop。
    高分变体 (score > seed_promote_threshold) 自动回池 (P0-3 seed library).
    """
    def __init__(
        self,
        client: ViviClient,
        n_iterations: int = 6,        # 浅 MCTS, 节省成本
        exploration: float = 1.41,
        proxy_model: str = "gemini-2.5-flash",
        max_chain_len: int = 8,
        score_weights: Optional[dict] = None,
        seed_library: Optional[object] = None,
        seed_promote_threshold: float = 0.65,
    ):
        self.client = client
        self.n_iter = n_iterations
        self.exp = exploration
        self.proxy_model = proxy_model
        self.max_chain_len = max_chain_len
        self.score_weights = score_weights
        self.seed_library = seed_library
        self.seed_promote_threshold = seed_promote_threshold
        # 记录 4-dim breakdown (paper telemetry)
        self.last_dim_scores: dict[str, list[float]] = {
            "attack_success": [], "coverage": [],
            "generalization": [], "defense_evasion": [], "weighted": [],
        }

    def _simulate(self, chain: list, family: str,
                   prior_chains: Optional[list] = None) -> float:
        """4-dim adaptive scoring (P0-1, 内部文章图3 verbatim).

        Returns the weighted scalar in [0,1] for MCTS backprop, BUT also
        stores the 4 sub-dimensions in self.last_dim_scores for paper telemetry.
        """
        from simple_4dim_judge import four_dim_score
        s = four_dim_score(self.client, chain, family,
                            prior_chains=prior_chains, model=self.proxy_model)
        if not s.success:
            # fallback: scalar 0.5 if LLM call failed
            for k in ("attack_success", "coverage", "generalization",
                      "defense_evasion"):
                self.last_dim_scores[k].append(0.5)
            self.last_dim_scores["weighted"].append(0.5)
            return 0.5
        self.last_dim_scores["attack_success"].append(s.attack_success)
        self.last_dim_scores["coverage"].append(s.coverage)
        self.last_dim_scores["generalization"].append(s.generalization)
        self.last_dim_scores["defense_evasion"].append(s.defense_evasion)
        v = s.weighted_value(self.score_weights)
        self.last_dim_scores["weighted"].append(v)
        # P0-3: 高分变体回池 (seed promotion)
        if self.seed_library is not None and v >= self.seed_promote_threshold:
            try:
                self.seed_library.promote_chain(
                    family=family, chain=chain,
                    four_dim={
                        "attack_success": s.attack_success,
                        "coverage": s.coverage,
                        "generalization": s.generalization,
                        "defense_evasion": s.defense_evasion,
                        "weighted": v,
                    },
                )
            except Exception as e:
                _log.warning(f"seed promote failed: {e}")
        return v

    def search(
        self,
        seed_chain: list,
        family: str,
        available_ops: list[str],
        available_mutations: Optional[list[str]] = None,
    ) -> tuple[list, list]:
        """Returns (best_chain, history_of_all_chains_explored)."""
        if available_mutations is None:
            available_mutations = list(MUTATION_OPERATORS_V1.keys())

        root = MCTSNode(seed_chain)
        history = [seed_chain]

        for it in range(self.n_iter):
            # 1. Selection: walk down tree by UCB1
            node = root
            depth = 0
            while node.children and depth < 3:
                node = max(node.children, key=lambda c: c.ucb1(self.exp))
                depth += 1

            # 2. Expansion: if node has been visited, expand new children
            if node.visits > 0 and len(node.chain) < self.max_chain_len:
                mut = random.choice(available_mutations)
                new_chain = apply_mutation(node.chain, mut, family, available_ops)
                if new_chain != node.chain:
                    child = MCTSNode(new_chain, parent=node, mutation=mut)
                    node.children.append(child)
                    node = child
                    history.append(new_chain)

            # 3. Simulation: 4-dim adaptive judge (内部文章图3)
            score = self._simulate(node.chain, family, prior_chains=history[:-1])

            # 4. Backpropagation
            cur = node
            while cur is not None:
                cur.visits += 1
                cur.value_sum += score
                cur = cur.parent

        # Best = highest value node among all visited
        def collect(n: MCTSNode, acc: list):
            acc.append(n)
            for c in n.children:
                collect(c, acc)
        all_nodes = []
        collect(root, all_nodes)
        best = max(all_nodes, key=lambda n: n.value if n.visits > 0 else -1)
        return best.chain, history


# ────────────────────────── Smoke test ──────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import os
    np.random.seed(7); random.seed(7)

    api_key = os.environ["VIVIAI_KEY"]  # set VIVIAI_KEY env var; key not committed
    client = ViviClient(api_key=api_key)

    # SimpleFamilySelector
    sel = SimpleFamilySelector()
    print(f"=== SimpleFamilySelector (random) ===")
    seq = [sel.families[sel.select_next(0)] for _ in range(5)]
    print(f"  5 picks: {seq}")

    # SimpleSkillBook
    book = SimpleSkillBook(attack_family_list(), base_dir="/tmp/skill_v1_test")
    book.append_constraint("frontal_swap", "Use blend in [0.55, 0.75], avoid > 0.85.")
    book.append_constraint("frontal_swap", "Always add jpeg_85 after gfpgan.")
    print(f"\n=== SimpleSkillBook (single-layer markdown) ===")
    print(book.get_doc("frontal_swap")[:400])

    # Mutation ops
    print(f"\n=== 6 mutation ops on a seed chain ===")
    seed = [{"tool": "face_align", "params": {}},
            {"tool": "inswapper_128", "params": {"blend": 0.7}},
            {"tool": "gfpgan", "params": {"weight": 0.4}},
            {"tool": "jpeg_85", "params": {"quality": 85}}]
    available = ["face_align", "inswapper_128", "simswap_256", "roop",
                  "gfpgan", "nano_banana_two", "nano_banana_one",
                  "jpeg_85", "resize_bicubic"]
    for mut in MUTATION_OPERATORS_V1:
        random.seed(hash(mut) % 100)
        new_chain = apply_mutation(seed, mut, "frontal_swap", available)
        print(f"  {mut:20s}: {' → '.join(s['tool'] for s in new_chain)}")

    # MCTS planner (6 iter, ~6 LLM calls, ~$0.01)
    print(f"\n=== MCTS UCB1 4-step planning (6 iter) ===")
    planner = SimpleMCTSPlanner(client=client, n_iterations=6)
    best, history = planner.search(seed_chain=seed, family="frontal_swap",
                                    available_ops=available)
    print(f"  explored {len(history)} chains")
    print(f"  BEST: {' → '.join(s['tool'] for s in best)}")
