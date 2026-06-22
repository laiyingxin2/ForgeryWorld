"""Bug-18 fix: OpHealthTracker — 追踪每个 op 最近 success rate.

问题: Reflexion + setter LLM 不知道当前 op 状态(e.g. nano_banana 503),
导致主动建议失败 op,陷入 pseudo-bypass 循环.

解决: 用 rolling window 记录每个 op 最近 N 次 success/fail,
把统计塞进 Reflexion + setter 的 prompt, 让 LLM 选高 reliability op.

成本: 0 LLM call, 仅本地 numpy.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from collections import defaultdict, deque
from typing import Optional


_log = logging.getLogger(__name__)


class OpHealthTracker:
    """Track per-op success/fail history. Persistent across runs."""

    def __init__(
        self,
        window_per_op: int = 20,        # rolling window size
        min_calls_for_stats: int = 3,   # 至少这么多 call 才报 stat
        unknown_default: float = 0.5,    # 无数据时假设 50% (中性)
        persist_path: Optional[str] = None,
    ):
        self.window = window_per_op
        self.min_calls = min_calls_for_stats
        self.unknown_default = unknown_default
        self.persist_path = persist_path

        # op_name → deque of 1.0/0.0 (success/fail)
        self.history: dict[str, deque] = defaultdict(lambda: deque(maxlen=window_per_op))
        # op_name → cumulative stats (跨 run 累计, 不被 window 截断)
        self.cumulative: dict[str, dict] = defaultdict(
            lambda: {"total_calls": 0, "total_success": 0, "total_503": 0, "total_other_fail": 0}
        )

        if persist_path:
            self._load()

    def record(self, op_name: str, succeeded: bool, error: Optional[str] = None):
        """Record one op call result."""
        if not op_name:
            return
        v = 1.0 if succeeded else 0.0
        self.history[op_name].append(v)
        c = self.cumulative[op_name]
        c["total_calls"] += 1
        if succeeded:
            c["total_success"] += 1
        elif error and ("503" in error or "Server" in error or "All models failed" in error):
            c["total_503"] += 1
        else:
            c["total_other_fail"] += 1
        if self.persist_path:
            self._save()

    def success_rate(self, op_name: str) -> float:
        """Recent rolling success rate; unknown → 0.5."""
        h = self.history.get(op_name)
        if not h or len(h) < self.min_calls:
            return self.unknown_default
        return sum(h) / len(h)

    def get_health_summary(
        self,
        sort_by: str = "rate",       # "rate" or "name"
        include_unknown: bool = False,
    ) -> str:
        """Plain-text summary for LLM prompt injection."""
        rows = []
        for op, hist in self.history.items():
            if len(hist) < self.min_calls and not include_unknown:
                continue
            rate = sum(hist) / max(len(hist), 1)
            c = self.cumulative[op]
            mark = "✓" if rate >= 0.7 else ("⚠️" if rate >= 0.4 else "❌")
            note = ""
            if c["total_503"] > 0 and rate < 0.5:
                note = f" (mostly 503: {c['total_503']}/{c['total_calls']})"
            rows.append((rate, op,
                          f"  {mark} {op}: {rate:.0%} success (last {len(hist)} calls){note}"))

        if not rows:
            return "(no op call data yet — try diverse ops)"

        if sort_by == "rate":
            rows.sort(key=lambda r: -r[0])  # 高的在前

        return "\n".join(r[2] for r in rows)

    def recommend_reliable_ops(self, min_rate: float = 0.4, top_n: int = 5) -> list[str]:
        """Return ops with success rate >= min_rate, sorted desc."""
        scored = []
        for op, hist in self.history.items():
            if len(hist) >= self.min_calls:
                rate = sum(hist) / len(hist)
                if rate >= min_rate:
                    scored.append((rate, op))
        scored.sort(reverse=True)
        return [op for _, op in scored[:top_n]]

    def blacklist_failing_ops(self, max_rate: float = 0.2) -> list[str]:
        """Return ops with success rate < max_rate (don't suggest these)."""
        out = []
        for op, hist in self.history.items():
            if len(hist) >= self.min_calls:
                rate = sum(hist) / len(hist)
                if rate < max_rate:
                    out.append(op)
        return out

    def _save(self):
        if not self.persist_path:
            return
        Path(self.persist_path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "history": {op: list(h) for op, h in self.history.items()},
            "cumulative": dict(self.cumulative),
        }
        Path(self.persist_path).write_text(json.dumps(data, indent=2))

    def _load(self):
        p = Path(self.persist_path) if self.persist_path else None
        if not p or not p.exists():
            return
        try:
            data = json.loads(p.read_text())
            for op, lst in data.get("history", {}).items():
                self.history[op] = deque(lst, maxlen=self.window)
            for op, c in data.get("cumulative", {}).items():
                self.cumulative[op] = dict(c)
            _log.info(f"[op_health] loaded {len(self.history)} ops from {p}")
        except Exception as e:
            _log.warning(f"[op_health] load failed: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    tr = OpHealthTracker(window_per_op=10, persist_path="/tmp/op_health_smoke.json")

    # 模拟一组 op call
    for _ in range(8):
        tr.record("nano_banana_pro", False, error="All models failed: 503")
    for _ in range(7):
        tr.record("gpt_image_two", True)
    for _ in range(3):
        tr.record("gpt_image_two", False, error="content policy refusal")
    for _ in range(5):
        tr.record("nano_banana_two", False, error="server 503")
    tr.record("face_align", True)   # only 1 call → unknown

    print("=== get_health_summary (LLM prompt) ===")
    print(tr.get_health_summary())

    print(f"\nRecommended reliable ops: {tr.recommend_reliable_ops()}")
    print(f"Blacklist failing ops:    {tr.blacklist_failing_ops()}")
    print(f"\nsuccess_rate(nano_banana_pro) = {tr.success_rate('nano_banana_pro')}")
    print(f"success_rate(gpt_image_two)    = {tr.success_rate('gpt_image_two')}")
    print(f"success_rate(face_align unknown) = {tr.success_rate('face_align')}")

    # round-trip persist
    tr2 = OpHealthTracker(window_per_op=10, persist_path="/tmp/op_health_smoke.json")
    print(f"\nAfter reload: {tr2.success_rate('nano_banana_pro'):.3f} (expect 0.000)")
