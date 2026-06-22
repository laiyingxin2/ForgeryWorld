"""Layer 4 — Pipeline Planner with Lookahead.

WebEvolver-style k=3 candidate sampling + d=2 depth lookahead via proxy judge.
Also implements weighted random walk on Tool Graph (Agent-World).

NOT MCTS — that's Baseline #1 (mcts_planner.py).
"""
from __future__ import annotations
import json
import logging
import re
import random
from typing import Optional, Callable
from dataclasses import dataclass, field

import numpy as np

from viviai_client import ViviClient
from trajectory_schema import LookaheadCandidate


_log = logging.getLogger(__name__)


# ────────────────────────── Tool Graph ──────────────────────────────

@dataclass
class ToolNode:
    name: str
    family: str = ""               # which attack family it belongs to (or "shared")
    is_source: bool = False        # can it be a starting node?
    cost_usd: float = 0.0
    params_schema: dict = field(default_factory=dict)


class ToolGraph:
    """G=(V, E) with weights w_ij = call-dependency strength.

    Default 15 ops mapping per design doc §3.
    """

    DEFAULT_NODES = [
        # source / preprocessing
        ToolNode("face_align", family="shared", is_source=True, cost_usd=0.0),
        # local swap (★ 2026-06-20 fix: registry keys, not phantom names. Old
        # inswapper_128/simswap_256/roop/facevid2vid/replay_sim are NOT in
        # OPERATOR_REGISTRY → they errored at execution, dropping the identity
        # step and leaving only generative gpt_image_two → faceless forgeries.
        # roop/facevid2vid had no registry equivalent and are covered by
        # inswapper_128_local/liveportrait, so they are removed.)
        ToolNode("inswapper_128_local", family="frontal_swap"),
        ToolNode("simswap_256_local", family="profile_swap"),
        # reenact
        ToolNode("liveportrait", family="reenact"),
        # morph
        ToolNode("stylegan_morph", family="morph"),
        # 3d mask
        ToolNode("deca_3dmask", family="3d_mask"),
        # replay
        ToolNode("screen_replay_sim", family="replay"),
        # adv patch
        ToolNode("adv_patch_pgd", family="adv_patch"),
        # audio
        ToolNode("xtts_audio", family="audio_synth"),
        # API (id_diff / restoration)
        ToolNode("nano_banana_pro", family="id_diff", cost_usd=0.06),
        ToolNode("nano_banana_two", family="id_diff", cost_usd=0.04),
        ToolNode("nano_banana_one", family="morph", cost_usd=0.02),
        ToolNode("gpt_image_two", family="restoration", cost_usd=0.04),
        # post-process / degradation
        ToolNode("gfpgan", family="restoration"),
        ToolNode("jpeg_85", family="shared"),
        ToolNode("resize_bicubic", family="shared"),
    ]

    def __init__(self, nodes: Optional[list[ToolNode]] = None):
        self.nodes = nodes or list(self.DEFAULT_NODES)
        self.name_to_idx = {n.name: i for i, n in enumerate(self.nodes)}
        n = len(self.nodes)
        # Init edge weights heuristically
        # source → swap/id_diff/morph/reenact → restoration → degradation
        self.W = np.zeros((n, n), dtype=np.float64)
        for i, a in enumerate(self.nodes):
            for j, b in enumerate(self.nodes):
                if i == j:
                    continue
                w = self._heuristic_weight(a, b)
                self.W[i, j] = w
        # normalize rows
        row_sums = self.W.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        self.W = self.W / row_sums

    @staticmethod
    def _heuristic_weight(a: ToolNode, b: ToolNode) -> float:
        # face_align → almost anything
        if a.name == "face_align":
            if b.name in {"inswapper_128_local", "simswap_256_local", "liveportrait",
                          "stylegan_morph", "deca_3dmask",
                          "adv_patch_pgd", "nano_banana_two", "nano_banana_pro"}:
                return 1.0
            return 0.3

        # swap → restoration or degradation
        if a.family in {"frontal_swap", "profile_swap"}:
            if b.name in {"gfpgan", "gpt_image_two", "nano_banana_two", "jpeg_85"}:
                return 1.0
            if b.name == "resize_bicubic":
                return 0.6

        # id_diff (nano_banana) → degradation
        if a.family == "id_diff":
            if b.name in {"jpeg_85", "resize_bicubic", "gfpgan"}:
                return 0.8

        # restoration → degradation
        if a.family == "restoration":
            if b.name in {"jpeg_85", "resize_bicubic"}:
                return 1.0

        # 3d mask / replay / adv patch — terminal-ish
        if a.family in {"3d_mask", "replay", "adv_patch"}:
            if b.name in {"jpeg_85", "resize_bicubic"}:
                return 0.5
            return 0.1

        # default
        return 0.1

    def source_nodes(self) -> list[int]:
        return [i for i, n in enumerate(self.nodes) if n.is_source]

    def random_walk(
        self,
        max_steps: int = 6,
        start: Optional[int] = None,
        family_filter: Optional[str] = None,
    ) -> list[str]:
        """Weighted random walk on tool graph from a source node.

        family_filter: if set, prefer ops matching this family (boost +1 in transition).
        """
        if start is None:
            start = self.source_nodes()[0]  # face_align
        path = [start]
        for _ in range(max_steps - 1):
            row = self.W[path[-1]].copy()
            if family_filter:
                for j, node in enumerate(self.nodes):
                    if node.family == family_filter or node.family == "shared":
                        row[j] *= 2.0
            row_sum = row.sum()
            if row_sum == 0:
                break
            row = row / row_sum
            nxt = int(np.random.choice(len(self.nodes), p=row))
            if nxt in path:
                # 避免环路, 选一个新的
                unseen = [j for j in range(len(self.nodes)) if j not in path]
                if not unseen:
                    break
                nxt = unseen[np.argmax(row[unseen])]
            path.append(nxt)
        return [self.nodes[i].name for i in path]


# ────────────────────────── Pipeline Planner ────────────────────────

@dataclass
class PipelineCandidate:
    pipeline: list = field(default_factory=list)   # [{tool, params}, ...]
    proxy_score: float = 0.0
    selected: bool = False
    # M2-P0-3: 4 维 sub-scores (内部文章图3 verbatim), populated by proxy_score_candidate
    four_dim: dict = field(default_factory=dict)


class PipelinePlanner:
    """Layer 4 main: k=3 candidate sample + d=2 lookahead + best select.

    proxy_score_fn: callable taking partial pipeline + brief and returning float.
                    Default = LLM-as-judge cheap predictor.
    """

    def __init__(
        self,
        client: Optional[ViviClient] = None,
        tool_graph: Optional[ToolGraph] = None,
        k_candidates: int = 3,
        lookahead_depth: int = 2,
        proxy_model: str = "gemini-2.5-flash",
    ):
        self.client = client or ViviClient()
        self.tool_graph = tool_graph or ToolGraph()
        self.k = k_candidates
        self.d = lookahead_depth
        self.proxy_model = proxy_model

    def sample_candidates(
        self,
        family: str,
        brief_hints: list,            # suggested_chain
        max_steps: int = 6,
        seed_library: Optional[object] = None,  # ★ BUG-2 fix
    ) -> list[PipelineCandidate]:
        """Sample k=3 candidates: 1 from brief hints + (optional 1 from seed_library) + rest random walk.

        ★ BUG-2 fix: 优先把 seed_library 最高分 chain 作 1 候选 → 真正利用高分回池
        之前: 1 brief + (k-1) random walk → ui_voyager / 4-evolution 写入的 chain 全死在库里
        现在: 1 brief + 1 top-seed + (k-2) random walk
        """
        candidates: list[PipelineCandidate] = []

        # Candidate 1: brief's suggested_chain
        if brief_hints:
            chain = [{"tool": op, "params": {}} for op in brief_hints[:max_steps]]
            candidates.append(PipelineCandidate(pipeline=chain))

        # Candidate 2: top seed from seed_library (高分回池)
        if seed_library is not None:
            try:
                top_seeds = seed_library.get_top_seeds(family, top_k=3)
                if top_seeds:
                    s = top_seeds[0]
                    candidates.append(PipelineCandidate(
                        pipeline=s["chain"][:max_steps]
                    ))
                    _log.info(f"  [planner] +seed_lib candidate "
                              f"chain_id={s['chain_id'][:18]} score={s['weighted_score']:.3f}")
            except Exception as e:
                _log.warning(f"  seed_library candidate failed: {e}")

        # Remaining candidates: random walks biased to family
        n_remaining = max(self.k - len(candidates), 1)
        for _ in range(n_remaining):
            walk = self.tool_graph.random_walk(
                max_steps=max_steps, family_filter=family,
            )
            chain = [{"tool": op, "params": {}} for op in walk]
            candidates.append(PipelineCandidate(pipeline=chain))

        return candidates[:self.k]

    def proxy_score_candidate(
        self,
        candidate: PipelineCandidate,
        family: str,
        brief_text: str = "",
        prior_pipelines: Optional[list] = None,
    ) -> float:
        """Layer 4 d=2 partial lookahead — now uses 4-dim adaptive scoring
        (M2-P0-3: 内部文章图3 verbatim, parity with method 1 after Tier 1).

        Returns weighted scalar from (攻击成功 / 覆盖 / 泛化 / 防御绕过) 4 dims.
        Sub-dims also recorded on candidate for paper telemetry.
        """
        try:
            from simple_4dim_judge import four_dim_score
            s = four_dim_score(
                self.client, candidate.pipeline, family,
                prior_chains=prior_pipelines or [],
                model=self.proxy_model,
            )
            if s.success:
                # stash 4 dims on candidate (paper-grade observability)
                candidate.four_dim = {
                    "attack_success": s.attack_success,
                    "coverage": s.coverage,
                    "generalization": s.generalization,
                    "defense_evasion": s.defense_evasion,
                }
                return s.weighted_value()
        except Exception as e:
            _log.warning(f"4dim score failed, fallback to scalar: {e}")

        # fallback: legacy scalar prompt (kept for safety)
        d = min(self.d, len(candidate.pipeline))
        partial_str = " → ".join(
            f"{p['tool']}({json.dumps(p.get('params',{}))})"
            for p in candidate.pipeline[:d]
        )
        full_str = " → ".join(p["tool"] for p in candidate.pipeline)
        prompt = (
            f"Estimate bypass probability for attack pipeline.\n"
            f"Family: {family}\nBrief: {brief_text[:200]}\n"
            f"First {d} steps: {partial_str}\nFull: {full_str}\n"
            f"Return ONE number in [0.0, 1.0]."
        )
        try:
            text = self.client.chat_text(
                self.proxy_model, prompt, temperature=0.1, max_tokens=20,
            ).strip()
            m = re.search(r"\b([01]?\.\d+|[01]\.\d+|[01](?!\d))\b", text)
            if m: return max(0.0, min(1.0, float(m.group(1))))
            m = re.search(r"(\d{1,3})%", text)
            if m: return max(0.0, min(1.0, float(m.group(1)) / 100.0))
            return 0.5
        except Exception as e:
            _log.warning(f"proxy_score failed: {e}")
            return 0.5

    def plan(
        self,
        family: str,
        brief_hints: list,
        brief_text: str = "",
        max_steps: int = 6,
        seed_library: Optional[object] = None,  # ★ BUG-2 fix
        prior_pipelines: Optional[list] = None,  # for coverage dim of 4-dim score
    ) -> tuple[list, list[PipelineCandidate]]:
        """Main entry: returns (selected_pipeline, all_candidates_with_scores)."""
        cands = self.sample_candidates(family, brief_hints, max_steps,
                                         seed_library=seed_library)
        for c in cands:
            c.proxy_score = self.proxy_score_candidate(
                c, family, brief_text, prior_pipelines=prior_pipelines)
        best = max(cands, key=lambda c: c.proxy_score)
        best.selected = True
        return best.pipeline, cands


# ────────────────────────── Smoke test ──────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import os
    np.random.seed(42); random.seed(42)

    api_key = os.environ["VIVIAI_KEY"]  # set VIVIAI_KEY env var; key not committed
    client = ViviClient(api_key=api_key)

    graph = ToolGraph()
    print(f"Tool Graph: {len(graph.nodes)} nodes")
    print(f"  Source nodes: {[graph.nodes[i].name for i in graph.source_nodes()]}")

    # random walk
    print("\n=== Random walks for frontal_swap ===")
    for i in range(3):
        walk = graph.random_walk(max_steps=5, family_filter="frontal_swap")
        print(f"  walk {i}: {' → '.join(walk)}")

    # Planner
    planner = PipelinePlanner(client=client, tool_graph=graph, k_candidates=3, lookahead_depth=2)
    print("\n=== Plan for frontal_swap (k=3 lookahead) ===")
    selected, cands = planner.plan(
        family="frontal_swap",
        brief_hints=["face_align", "inswapper_128", "gfpgan", "jpeg_85"],
        brief_text="frontal female, indoor lighting, want bypass gemini judge",
        max_steps=5,
    )
    for i, c in enumerate(cands):
        marker = "★" if c.selected else " "
        chain = " → ".join(p["tool"] for p in c.pipeline)
        print(f"  {marker} cand {i}: score={c.proxy_score:.3f}: {chain}")
    print(f"\nSelected: {' → '.join(p['tool'] for p in selected)}")
