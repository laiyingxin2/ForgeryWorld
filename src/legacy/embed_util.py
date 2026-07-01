"""Shared semantic-embedding utility.

Real semantic embeddings via ChromaDB's bundled DefaultEmbeddingFunction
(all-MiniLM-L6-v2, 384-dim, L2-normalized, ONNX runtime — local, no API cost,
deterministic). Falls back to a cheap lexical character-n-gram hash if the
ONNX embedder cannot be loaded (e.g. offline first-run before model download).

The viviai gateway exposes only chat + image models — no /v1/embeddings channel
(text-embedding-3-* return 404 model_not_found) — so a local embedder is the
honest choice here.

Public API:
    embed_text(text)             -> list[float]   (semantic if available)
    cosine_sim(a, b)             -> float
    wmr_score(relevance, ...)    -> float         (Generative-Agents / Mem0 retrieval)
    is_semantic()                -> bool          (True once real embedder is live)
"""
from __future__ import annotations

import hashlib
import logging
import re
import threading

import numpy as np

_log = logging.getLogger(__name__)

_LEX_DIM = 256
_lock = threading.Lock()
_embed_fn = None          # chromadb DefaultEmbeddingFunction singleton
_embed_fn_tried = False
_semantic = False
_cache: dict[str, list] = {}
_CACHE_MAX = 4096


def _get_embed_fn():
    """Lazily build (once) the ChromaDB DefaultEmbeddingFunction singleton."""
    global _embed_fn, _embed_fn_tried, _semantic
    if _embed_fn_tried:
        return _embed_fn
    with _lock:
        if _embed_fn_tried:
            return _embed_fn
        _embed_fn_tried = True
        try:
            from chromadb.utils import embedding_functions as ef
            fn = ef.DefaultEmbeddingFunction()
            # warm-up / sanity check
            v = fn(["warm up"])
            if v and len(v[0]) > 0:
                _embed_fn = fn
                _semantic = True
                _log.info("embed_util: semantic embedder active (dim=%d)", len(v[0]))
            else:
                _log.warning("embed_util: DefaultEmbeddingFunction returned empty; using lexical fallback")
        except Exception as e:  # noqa: BLE001
            _log.warning("embed_util: semantic embedder unavailable (%s); using lexical fallback", e)
        return _embed_fn


def is_semantic() -> bool:
    _get_embed_fn()
    return _semantic


def _lexical_features(text: str, dim: int = _LEX_DIM) -> list[float]:
    """Cheap deterministic fallback: char-3-gram + word hashing, L2-normalized."""
    vec = np.zeros(dim, dtype=np.float64)
    low = text.lower()
    for i in range(len(low) - 2):
        h = int(hashlib.md5(low[i:i + 3].encode()).hexdigest()[:8], 16)
        vec[h % dim] += 1.0
    for word in re.findall(r"\w+", low):
        h = int(hashlib.md5(word.encode()).hexdigest()[:8], 16)
        vec[h % dim] += 2.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def embed_text(text: str) -> list:
    """Return an embedding for `text` (semantic when available, else lexical).

    Result is cached by exact-string to avoid recomputation on reload/dedup loops.
    """
    if not text:
        return []
    cached = _cache.get(text)
    if cached is not None:
        return cached
    fn = _get_embed_fn()
    if fn is not None:
        try:
            vec = list(map(float, fn([text])[0]))
        except Exception as e:  # noqa: BLE001
            _log.warning("embed_util: embed failed (%s); lexical fallback for this call", e)
            vec = _lexical_features(text)
    else:
        vec = _lexical_features(text)
    if len(_cache) < _CACHE_MAX:
        _cache[text] = vec
    return vec


def cosine_sim(a: list, b: list) -> float:
    a_v = np.asarray(a, dtype=np.float64)
    b_v = np.asarray(b, dtype=np.float64)
    if a_v.size == 0 or b_v.size == 0 or a_v.size != b_v.size:
        return 0.0
    denom = np.linalg.norm(a_v) * np.linalg.norm(b_v)
    if denom == 0:
        return 0.0
    return float(a_v.dot(b_v) / denom)


def wmr_score(
    relevance: float,
    last_used_round: int = 0,
    current_round: int = 0,
    alpha_count: float = 1.0,
    beta_count: float = 1.0,
    recency_decay: float = 0.9,
    w_rel: float = 1.0,
    w_rec: float = 0.5,
    w_imp: float = 0.5,
) -> float:
    """Generative-Agents / Mem0 weighted memory retrieval (WMR).

    score = w_rel*relevance + w_rec*recency + w_imp*importance

    - relevance  : cosine similarity in [-1,1] (clamped to [0,1])
    - recency    : recency_decay ** (current_round - last_used_round)
    - importance : Beta posterior mean alpha/(alpha+beta) = empirical success rate

    Additive (not multiplicative) so a single zero factor doesn't annihilate an
    otherwise-strong memory — matches the Generative Agents formulation.
    """
    rel = max(0.0, min(1.0, relevance))
    dt = max(current_round - last_used_round, 0)
    recency = recency_decay ** dt
    total = alpha_count + beta_count
    importance = (alpha_count / total) if total > 0 else 0.5
    return w_rel * rel + w_rec * recency + w_imp * importance
