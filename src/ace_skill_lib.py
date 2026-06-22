"""Layer 3 — Ace-Skill dual-stream skill library.

Wraps Ace-Skill repo's SqrtBiasSampler + ExperienceManager + skill_builder.
Source: external/Ace-Skill/eval/ace_skill/
Confirms verbatim:
  - Eq.4 prioritized sampling: w_t ∝ √(v(1-v)) + γ(1-v) + ε
  - Eq.7 experience merge with τ=0.70 cosine
  - Eq.8 skill compress (LLM)
Override:
  - Ace-Skill default γ=0.2; we override to γ=0.4 per paper claim

简化 vs Ace-Skill:
  - 不直接复用他们的 OpenAI client (env var 链复杂), 我们用 viviai
  - K=9 attack family, 每个 family 单独 ℰ_k + 𝒮_k
  - 存储: jsonl for ℰ_k, .md for 𝒮_k (与 Ace-Skill 兼容)

成本估算 (per round):
  - text-embedding-3-small: $0.00002 per 1K tokens, embedding 一条 brief ~$0.0001
  - LLM merge (Eq.7) gemini-2.5-flash: $0.0015 per merge call
  - LLM compress (Eq.8) gemini-3-pro: $0.005 per compress
"""
from __future__ import annotations
import json
import logging
import re
from pathlib import Path
from typing import Optional, Iterable
from dataclasses import dataclass, field, asdict

import numpy as np

from viviai_client import ViviClient


_log = logging.getLogger(__name__)


# ────────────────────────── Hyperparams (verbatim Ace-Skill) ─────────

@dataclass
class AceSkillConfig:
    # Eq.4
    gamma: float = 0.4           # **override** Ace-Skill default 0.2 → paper 0.4
    epsilon: float = 0.1
    rho: float = 0.95            # lazy decay (Ace-Skill 0.9, paper 0.95)

    # Eq.7
    similarity_threshold: float = 0.70   # τ
    experience_pool_capacity: int = 120  # L

    # Eq.8
    skill_word_budget: int = 1000        # W
    skill_compress_trigger: int = 1100   # 11% slack

    # K-means clustering (here K is fixed = 9 attack families, so no clustering)
    K_families: int = 9

    # Embedding
    embedder_model: str = "text-embedding-3-small"  # OpenAI; via viviai
    # Note: viviai OpenAI 兼容 endpoint /v1/embeddings 应该 work, 但暂未在 viviai_client 实现.
    # Fallback: 用 LLM 摘要后 用 SHA hash 或者 simple text features.
    fallback_to_simple_features: bool = True


# ────────────────────────── Experience ──────────────────────────────

@dataclass
class Experience:
    """单条 tactical experience (ℰ_k entry)."""
    exp_id: str = ""
    text: str = ""               # natural language description
    embedding: list = field(default_factory=list)
    source_round: int = 0
    source_trajectory_id: str = ""
    applicability_score: float = 0.5      # v_i, Beta posterior mean
    alpha_count: float = 1.0              # successes
    beta_count: float = 1.0               # failures
    last_used_round: int = 0
    merged_from: list = field(default_factory=list)

    def update_v(self, decay_rho: float, current_round: int) -> float:
        """Beta posterior with lazy decay (Ace-Skill weighted_sampler.py lines 160-171)."""
        dt = max(current_round - self.last_used_round, 0)
        decay = decay_rho ** dt
        ca = self.alpha_count * decay
        cb = self.beta_count * decay
        self.applicability_score = (1.0 + ca) / (2.0 + ca + cb)
        return self.applicability_score


# ────────────────────────── Simple embedding fallback ───────────────

def _simple_text_features(text: str, dim: int = 256) -> list[float]:
    """Fallback embedding when text-embedding-3-small not available.

    Uses character-level n-gram hashing for stable + cheap embeddings.
    Cosine on these is meaningful but not as good as real embeddings — sufficient for prototyping.
    """
    import hashlib
    vec = np.zeros(dim, dtype=np.float64)
    # 3-gram hashing
    text_lower = text.lower()
    for i in range(len(text_lower) - 2):
        gram = text_lower[i:i + 3]
        h = int(hashlib.md5(gram.encode()).hexdigest()[:8], 16)
        vec[h % dim] += 1.0
    # word features
    for word in re.findall(r"\w+", text_lower):
        h = int(hashlib.md5(word.encode()).hexdigest()[:8], 16)
        vec[h % dim] += 2.0
    # normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def cosine_sim(a: list, b: list) -> float:
    a_v = np.asarray(a)
    b_v = np.asarray(b)
    if a_v.size == 0 or b_v.size == 0:
        return 0.0
    denom = (np.linalg.norm(a_v) * np.linalg.norm(b_v))
    if denom == 0:
        return 0.0
    return float(a_v.dot(b_v) / denom)


# ────────────────────────── ExperiencePool (ℰ_k) ────────────────────

class ExperiencePool:
    """ℰ_k for one attack family. L=120 capacity, Eq.7 merge."""

    def __init__(self, family_name: str, config: AceSkillConfig, client: ViviClient):
        self.family_name = family_name
        self.cfg = config
        self.client = client
        self.experiences: list[Experience] = []
        self._next_id_counter = 0

    def _next_id(self) -> str:
        self._next_id_counter += 1
        return f"{self.family_name[:4]}_E{self._next_id_counter:04d}"

    def embed(self, text: str) -> list:
        if self.cfg.fallback_to_simple_features:
            return _simple_text_features(text)
        # TODO: real embedding via viviai when supported
        return _simple_text_features(text)

    def add_or_merge(
        self,
        new_text: str,
        source_round: int,
        source_trajectory_id: str,
        success: bool,
    ) -> tuple[str, bool]:
        """Eq.7 merge logic: if cosine > τ with any existing, LLM-merge; else append.

        Returns (exp_id, was_merged).
        """
        new_emb = self.embed(new_text)
        # Find similar
        similar = []
        for e in self.experiences:
            sim = cosine_sim(new_emb, e.embedding)
            if sim > self.cfg.similarity_threshold:
                similar.append((sim, e))

        if not similar:
            # No similar → append
            new_exp = Experience(
                exp_id=self._next_id(),
                text=new_text,
                embedding=new_emb,
                source_round=source_round,
                source_trajectory_id=source_trajectory_id,
                alpha_count=1.0 if success else 0.0,
                beta_count=0.0 if success else 1.0,
                last_used_round=source_round,
            )
            self.experiences.append(new_exp)
            self._enforce_capacity()
            return new_exp.exp_id, False

        # LLM-merge
        similar.sort(reverse=True)
        merged_text = self._llm_merge(new_text, [e.text for _, e in similar])
        merged_ids = [e.exp_id for _, e in similar]
        # Remove old
        keep_ids = {e.exp_id for e in self.experiences} - set(merged_ids)
        self.experiences = [e for e in self.experiences if e.exp_id in keep_ids]
        # Add merged
        new_exp = Experience(
            exp_id=self._next_id(),
            text=merged_text,
            embedding=self.embed(merged_text),
            source_round=source_round,
            source_trajectory_id=source_trajectory_id,
            alpha_count=sum(e.alpha_count for _, e in similar) + (1.0 if success else 0.0),
            beta_count=sum(e.beta_count for _, e in similar) + (0.0 if success else 1.0),
            last_used_round=source_round,
            merged_from=merged_ids,
        )
        self.experiences.append(new_exp)
        self._enforce_capacity()
        return new_exp.exp_id, True

    def _llm_merge(self, new_text: str, similar_texts: list[str]) -> str:
        """Eq.7 LLM-driven merge into single coherent experience."""
        joined = "\n".join(f"- {t}" for t in [new_text] + similar_texts)
        prompt = (
            f"You are curating tactical attack-experience memory for a face-KYC red-team agent.\n"
            f"Merge the following similar entries into ONE concise, actionable experience "
            f"(under 80 words). Keep the most specific actionable details and remove duplication.\n\n"
            f"Entries to merge:\n{joined}\n\n"
            f"Output ONLY the merged text, no preamble."
        )
        try:
            return self.client.chat_text(
                "gemini-2.5-flash", prompt, temperature=0.1, max_tokens=200
            ).strip()
        except Exception as e:
            _log.warning(f"LLM merge failed, falling back to concat: {e}")
            return " | ".join([new_text] + similar_texts)[:500]

    def _enforce_capacity(self):
        """If |ℰ_k| > L, iteratively merge most similar pair."""
        while len(self.experiences) > self.cfg.experience_pool_capacity:
            # Find most similar pair
            best_pair = (0, 1)
            best_sim = -1.0
            for i in range(len(self.experiences)):
                for j in range(i + 1, len(self.experiences)):
                    s = cosine_sim(self.experiences[i].embedding, self.experiences[j].embedding)
                    if s > best_sim:
                        best_sim = s
                        best_pair = (i, j)
            i, j = best_pair
            a, b = self.experiences[i], self.experiences[j]
            merged = self._llm_merge(a.text, [b.text])
            new_exp = Experience(
                exp_id=self._next_id(),
                text=merged, embedding=self.embed(merged),
                source_round=max(a.source_round, b.source_round),
                source_trajectory_id=f"merge({a.exp_id},{b.exp_id})",
                alpha_count=a.alpha_count + b.alpha_count,
                beta_count=a.beta_count + b.beta_count,
                last_used_round=max(a.last_used_round, b.last_used_round),
                merged_from=[a.exp_id, b.exp_id],
            )
            keep = [e for k, e in enumerate(self.experiences) if k not in {i, j}]
            self.experiences = keep + [new_exp]

    def retrieve(self, query_text: str, top_k: int = 5) -> list[Experience]:
        """Cosine top-k."""
        if not self.experiences:
            return []
        q_emb = self.embed(query_text)
        scored = [(cosine_sim(q_emb, e.embedding), e) for e in self.experiences]
        scored.sort(reverse=True)
        return [e for _, e in scored[:top_k]]

    # ───── Eq.4 prioritized sampling (Ace-Skill weighted_sampler.py 136-138) ─────
    def prioritized_sample(self, n: int, current_round: int) -> list[Experience]:
        """w_t(x_i) ∝ √(v(1-v)) + γ(1-v) + ε"""
        if not self.experiences:
            return []
        # update v_i for all
        v = np.array([e.update_v(self.cfg.rho, current_round) for e in self.experiences])
        w = np.sqrt(v * (1.0 - v)) + self.cfg.gamma * (1.0 - v) + self.cfg.epsilon
        probs = w / w.sum()
        idx = np.random.choice(len(self.experiences), size=min(n, len(self.experiences)),
                               replace=False, p=probs)
        return [self.experiences[i] for i in idx]

    def save_jsonl(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for e in self.experiences:
                # 不序列化 embedding (太大), 加载时重算
                d = asdict(e)
                d["embedding"] = []
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    def load_jsonl(self, path: str | Path):
        self.experiences = []
        if not Path(path).exists():
            return
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                e = Experience(**d)
                e.embedding = self.embed(e.text)  # 重算 embedding
                self.experiences.append(e)
                # bump id counter
                try:
                    n = int(e.exp_id.split("_E")[-1])
                    self._next_id_counter = max(self._next_id_counter, n)
                except Exception:
                    pass


# ────────────────────────── SkillDoc (𝒮_k) ──────────────────────────

class SkillDoc:
    """Strategic skill markdown SOP for one attack family. Eq.8 compress."""

    def __init__(self, family_name: str, config: AceSkillConfig, client: ViviClient):
        self.family_name = family_name
        self.cfg = config
        self.client = client
        self.content: str = f"# {family_name} SOP\n\n_No experience yet._\n"
        self.version: int = 0

    def word_count(self) -> int:
        return len(re.findall(r"\w+", self.content))

    def needs_compress(self) -> bool:
        return self.word_count() > self.cfg.skill_compress_trigger

    def update(self, delta_text: str, force_compress: bool = False):
        """Append Δ𝒮_k, optionally compress if over W."""
        self.content += f"\n\n## Round-update\n{delta_text}\n"
        self.version += 1
        if force_compress or self.needs_compress():
            self.compress()

    def compress(self):
        """Eq.8: 𝒮_k ← Compress(Merge(𝒮_k, Δ𝒮_i), W=1000)."""
        prompt = (
            f"You are curating the strategic playbook for a face-KYC red-team agent.\n"
            f"Below is the current skill document for the '{self.family_name}' attack family.\n"
            f"Rewrite it to be more concise (<= {self.cfg.skill_word_budget} words), "
            f"keeping all actionable rules, parameter ranges, and failure-mode warnings. "
            f"Merge redundant sections. Use markdown headers.\n\n"
            f"Current doc:\n{self.content}\n\n"
            f"Output the rewritten markdown ONLY, no preamble."
        )
        try:
            compressed = self.client.chat_text(
                "gemini-3-pro-preview", prompt, temperature=0.1, max_tokens=2000
            ).strip()
            # Clean ```markdown fences
            compressed = re.sub(r"^```(markdown)?\n", "", compressed)
            compressed = re.sub(r"\n```$", "", compressed)
            self.content = compressed
            _log.info(f"  [{self.family_name}] compressed to {self.word_count()} words")
        except Exception as e:
            _log.warning(f"Compress failed for {self.family_name}: {e}")

    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.content)

    def load(self, path: str | Path):
        if Path(path).exists():
            self.content = Path(path).read_text()


# ────────────────────────── SkillLibrary (top-level) ─────────────────

class SkillLibrary:
    """K families, each with (ℰ_k ExperiencePool + 𝒮_k SkillDoc)."""

    def __init__(
        self,
        families: list[str],
        config: Optional[AceSkillConfig] = None,
        client: Optional[ViviClient] = None,
        base_dir: str | Path = "outputs/skills",
    ):
        self.families = families
        self.cfg = config or AceSkillConfig()
        self.client = client or ViviClient()
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.pools: dict[str, ExperiencePool] = {
            f: ExperiencePool(f, self.cfg, self.client) for f in families
        }
        self.docs: dict[str, SkillDoc] = {
            f: SkillDoc(f, self.cfg, self.client) for f in families
        }

    def add_experience(
        self,
        family: str,
        new_text: str,
        round_id: int,
        trajectory_id: str,
        success: bool,
    ):
        return self.pools[family].add_or_merge(new_text, round_id, trajectory_id, success)

    def update_skill(self, family: str, delta_text: str, force_compress: bool = False):
        self.docs[family].update(delta_text, force_compress=force_compress)

    def retrieve(self, family: str, query: str, top_k: int = 5) -> tuple[str, list[Experience]]:
        """Return (𝒮_k content, top-k ℰ_k entries) for use by Layer 2 出题组."""
        doc = self.docs[family].content
        exps = self.pools[family].retrieve(query, top_k=top_k)
        return doc, exps

    def save_all(self):
        for f in self.families:
            self.pools[f].save_jsonl(self.base_dir / f / "experience.jsonl")
            self.docs[f].save(self.base_dir / f / "SKILL.md")

    def load_all(self):
        for f in self.families:
            self.pools[f].load_jsonl(self.base_dir / f / "experience.jsonl")
            self.docs[f].load(self.base_dir / f / "SKILL.md")


# ────────────────────────── Smoke test ──────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.random.seed(0)

    import os
    api_key = os.environ["VIVIAI_KEY"]  # set VIVIAI_KEY env var; key not committed
    client = ViviClient(api_key=api_key)

    from trajectory_schema import attack_family_list
    lib = SkillLibrary(
        families=attack_family_list(),
        client=client,
        base_dir="/tmp/skill_lib_test",
    )

    # 加 5 条 frontal_swap 经验
    print("=== add 5 experiences to frontal_swap ===")
    for i in range(5):
        text = f"InSwapper-128 blend=0.{6+i} on frontal face dim=1024 → tier2 conf 0.{30-i*3}"
        exp_id, merged = lib.add_experience(
            "frontal_swap", text, round_id=0, trajectory_id=f"r0_g{i}", success=True
        )
        print(f"  added {exp_id}, merged={merged}")

    # retrieve top-3
    print("\n=== retrieve top-3 for query ===")
    doc, exps = lib.retrieve(
        "frontal_swap",
        "What blend ratio of InSwapper bypasses gemini judge?",
        top_k=3,
    )
    for e in exps:
        print(f"  {e.exp_id}: v={e.applicability_score:.2f}: {e.text[:80]}")

    # prioritized sample (Eq.4)
    print("\n=== Eq.4 prioritized sample 3 ===")
    sampled = lib.pools["frontal_swap"].prioritized_sample(3, current_round=1)
    for e in sampled:
        print(f"  {e.exp_id}: v={e.applicability_score:.2f}")

    # save / load round-trip
    lib.save_all()
    lib2 = SkillLibrary(families=attack_family_list(), client=client, base_dir="/tmp/skill_lib_test")
    lib2.load_all()
    assert len(lib2.pools["frontal_swap"].experiences) == len(lib.pools["frontal_swap"].experiences)
    print(f"\n✓ save/load OK, frontal_swap has {len(lib2.pools['frontal_swap'].experiences)} exp")
