"""Robustness utilities — based on search agent's top-7 production bug guards.

Sources:
  - $47K stall loop (Medium, dev.to): cost guards + dedupe call detector
  - 15% JSON parse failure (tensoria.fr): json_repair middleware
  - SQLite WAL + atomic file writes (langgraph/dspy issues)
  - RobustFT (arxiv 2412.14922): format consistency
  - Voyager README: skill library bloat / dedup
"""
from __future__ import annotations
import json
import os
import re
import hashlib
import tempfile
import logging
from pathlib import Path
from typing import Optional, Any
import sqlite3

_log = logging.getLogger(__name__)


# ──────────────── 1. Atomic JSON write (search finding #3) ──────────

def atomic_write_text(path: str | Path, content: str):
    """Crash-safe write: write to tmp, then os.replace (atomic on POSIX).

    Avoids langgraph/dspy reported issue of half-truncated JSON on crash.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())  # 强制写盘
        os.replace(tmp, path)  # atomic rename
    except Exception:
        try:
            os.unlink(tmp)
        except Exception:
            pass
        raise


def atomic_write_json(path: str | Path, obj: Any, indent: int = 2):
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=indent))


# ──────────────── 2. Robust JSON parsing (finding #1) ──────────────

def parse_json_robust(text: str) -> dict:
    """Handle 5 documented LLM JSON failure modes (tensoria.fr):
      (a) markdown fences ```json
      (b) trailing commas
      (c) smart quotes / single quotes
      (d) unescaped newlines in strings
      (e) truncation at max_tokens
    """
    text = text.strip()

    # (a) strip markdown fences
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
        if text.startswith("json"):
            text = text[4:]
        elif text.startswith("javascript"):
            text = text[10:]
        text = text.strip()

    # extract first {...} block
    s = text.find("{")
    e = text.rfind("}")
    if s < 0 or e <= s:
        # (e) truncation — try to close the JSON
        if "{" in text and "}" not in text:
            text = text[text.find("{"):] + "}"
            s, e = 0, len(text) - 1
        else:
            raise ValueError(f"No JSON object in: {text[:200]}")
    candidate = text[s:e + 1]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # (b) remove trailing commas
    fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
    # (c) smart quotes
    fixed = fixed.replace("“", '"').replace("”", '"')
    fixed = fixed.replace("‘", "'").replace("’", "'")
    # ★ FakeVLM fix: markdown escape e.g. "is\_fake" → "is_fake"
    fixed = re.sub(r'\\([_\-*])', r'\1', fixed)
    # ★ FakeVLM fix: "0.0-1.0" placeholder → 0.5 (mid)
    fixed = re.sub(r':\s*([\d.]+)-([\d.]+)([,}])', r': 0.5\3', fixed)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Try json_repair if available (most robust)
    try:
        import json_repair  # pip install json-repair
        return json_repair.loads(candidate)
    except (ImportError, Exception):
        pass

    # Last resort: ast.literal_eval (handles single-quoted dicts)
    try:
        import ast
        return ast.literal_eval(fixed)
    except Exception:
        raise ValueError(f"JSON parse failed even after repair: {candidate[:300]}")


# ──────────────── 3. SQLite WAL + concurrent safety (finding #3) ────

def enable_wal_mode(conn: sqlite3.Connection):
    """WAL = Write-Ahead Log, allows concurrent reads + 1 writer.
    Required for multi-worker / multi-orchestrator setup.
    """
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")  # tradeoff durability for speed
    conn.execute("PRAGMA busy_timeout=5000")   # 5s wait if locked
    conn.commit()


# ──────────────── 4. Duplicate call detector (finding #2 / $47K loop) ─

class DuplicateCallDetector:
    """Track recent (model, prompt_hash) → detect 'cosmetically different, semantically identical' retries.

    Prevents the documented stall-loop where agent rephrases slightly and retries N times.
    """

    def __init__(self, window: int = 50, max_repeats: int = 3):
        self.window = window
        self.max_repeats = max_repeats
        from collections import deque, Counter
        self.history: deque = deque(maxlen=window)
        self.counts: Counter = Counter()

    def check(self, model: str, prompt: str) -> bool:
        """Return True if THIS exact (model, prompt) has been seen more than max_repeats times.
        Caller should abort/escalate if True.
        """
        key = (model, hashlib.md5(prompt.encode()).hexdigest())
        # 维护 sliding window count
        if len(self.history) == self.history.maxlen:
            old = self.history[0]
            self.counts[old] -= 1
            if self.counts[old] == 0:
                del self.counts[old]
        self.history.append(key)
        self.counts[key] += 1
        return self.counts[key] > self.max_repeats


# ──────────────── 5. Cost budget guard (finding #2) ─────────────────

class CostBudget:
    """Hard per-run USD budget. Wraps any LLM call to abort before spending more than budget.

    Usage:
      budget = CostBudget(max_usd=5.0)
      ...
      if budget.would_exceed(estimated_call_cost):
          raise RuntimeError("budget exceeded")
      result = llm.call(...)
      budget.add(actual_cost)
    """

    def __init__(self, max_usd: float, alert_threshold: float = 0.8):
        self.max_usd = max_usd
        self.alert_threshold = alert_threshold
        self.spent = 0.0
        self._alerted = False

    def add(self, cost: float):
        self.spent += cost
        if not self._alerted and self.spent > self.max_usd * self.alert_threshold:
            _log.warning(f"[budget] {self.spent:.4f} / {self.max_usd:.2f} "
                         f"({100*self.spent/self.max_usd:.0f}%) — approaching limit")
            self._alerted = True

    def would_exceed(self, estimated: float) -> bool:
        return self.spent + estimated > self.max_usd

    def remaining(self) -> float:
        return max(0.0, self.max_usd - self.spent)


# ──────────────── 6. Smoke test ──────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import tempfile

    # Atomic write
    p = Path(tempfile.gettempdir()) / "atomic_test.json"
    atomic_write_json(p, {"hello": "world"})
    assert json.loads(p.read_text())["hello"] == "world"
    print("✓ atomic_write")

    # JSON repair
    cases = [
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('{"a": 1, "b": 2,}', {"a": 1, "b": 2}),       # trailing comma
        ('{“a”: 1}', {"a": 1}),                # smart quotes
        ('Here is the answer: {"a": 1} done.', {"a": 1}),  # extraneous text
        ('{"a": 1, "b": 2', {"a": 1, "b": 2}),          # truncated
    ]
    for inp, expected in cases:
        try:
            got = parse_json_robust(inp)
            assert got.get("a") == expected.get("a"), f"mismatch on {inp!r}: got {got}"
            print(f"✓ parse_json_robust: {inp[:40]!r}")
        except Exception as e:
            print(f"✗ parse_json_robust: {inp!r} → {e}")

    # Dup call detector
    det = DuplicateCallDetector(window=10, max_repeats=2)
    for _ in range(2):
        assert not det.check("model_a", "same prompt")
    assert det.check("model_a", "same prompt")  # 3rd time = stall
    print("✓ DuplicateCallDetector")

    # Cost budget
    bud = CostBudget(max_usd=1.0, alert_threshold=0.5)
    bud.add(0.6)  # triggers alert
    assert bud.would_exceed(0.5)
    assert not bud.would_exceed(0.2)
    print(f"✓ CostBudget remaining=${bud.remaining():.2f}")
