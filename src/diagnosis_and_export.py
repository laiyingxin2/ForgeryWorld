"""Layer 9 (diagnosis) + Layer 10 (defender export) combined.

Layer 9 (Agent-World style):
  - bypass_rate_k per family
  - weak_set = argmin_k bypass_rate
  - 给 Markov 矩阵 +ε exploration boost
  - Brief 池下一轮 30% 多采 weak

Layer 10 (Lv5 prep):
  - SFT pool → Qwen2.5-VL-7B / InternVL-3-8B training jsonl
  - Each entry: {messages: [..., image_path, label, forensic_cot]}
  - Trigger: bypass_rate > 60% sustained 2 round → 发 retrain signal
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from data_flow import DataFlow


_log = logging.getLogger(__name__)


# ────────────────────────── Layer 9: Diagnosis ──────────────────────

@dataclass
class DiagnosisResult:
    round_id: int
    global_bypass_rate: float
    family_bypass_rates: dict[str, float]
    weak_families: list[str]
    strong_families: list[str]
    boost_recommendation: float = 0.15
    next_round_family_weights: dict[str, float] = None


class Diagnoser:
    """Layer 9 diagnosis. Operates on data_flow stats."""

    def __init__(self, data_flow: DataFlow):
        self.df = data_flow

    def diagnose(
        self,
        round_id: int,
        top_n_weak: int = 2,
        top_n_strong: int = 2,
        boost: float = 0.15,
    ) -> DiagnosisResult:
        stats = self.df.round_bypass_stats(round_id)
        if not stats:
            return DiagnosisResult(
                round_id=round_id, global_bypass_rate=0.0,
                family_bypass_rates={}, weak_families=[], strong_families=[],
                boost_recommendation=boost, next_round_family_weights={},
            )

        rates = {fam: st["bypass_rate"] for fam, st in stats.items()}
        attempts = {fam: st["attempts"] for fam, st in stats.items()}
        total_attempts = sum(attempts.values())
        total_byp = sum(st["bypass_count"] for st in stats.values())
        global_rate = total_byp / max(total_attempts, 1)

        sorted_by_rate = sorted(rates.items(), key=lambda kv: kv[1])
        weak = [f for f, _ in sorted_by_rate[:top_n_weak]]
        strong = [f for f, _ in sorted_by_rate[-top_n_strong:]]

        # Next-round brief weights: 70% uniform + 30% biased to weak
        K = len(rates)
        weights = {f: 0.7 / K for f in rates}
        weak_bonus = 0.3 / max(len(weak), 1)
        for f in weak:
            weights[f] = weights[f] + weak_bonus

        return DiagnosisResult(
            round_id=round_id,
            global_bypass_rate=global_rate,
            family_bypass_rates=rates,
            weak_families=weak,
            strong_families=strong,
            boost_recommendation=boost,
            next_round_family_weights=weights,
        )

    def cross_round_trend(self, last_n_rounds: int = 3) -> dict:
        """Across rounds: which families are improving / regressing."""
        cur = self.df.conn.execute(
            "SELECT round_id, family, bypass_rate FROM bypass_stats "
            "ORDER BY round_id DESC, family"
        )
        by_family = {}
        for round_id, family, rate in cur.fetchall():
            by_family.setdefault(family, []).append((round_id, rate))
        trends = {}
        for family, series in by_family.items():
            series.sort()
            series = series[-last_n_rounds:]
            if len(series) >= 2:
                delta = series[-1][1] - series[0][1]
                trends[family] = {
                    "from_round": series[0][0], "to_round": series[-1][0],
                    "delta": delta,
                    "current_rate": series[-1][1],
                }
        return trends


# ────────────────────────── Layer 10: Defender Export ───────────────

@dataclass
class DefenderExportConfig:
    output_format: str = "qwen_vl_chat"  # 'qwen_vl_chat' / 'llava' / 'internvl'
    include_real_samples: bool = True     # 加入 real face (negative) 样本
    cot_field: str = "forensic_cot"        # Tier-3 critique as CoT
    weight_by_route: dict = None           # SFT=1.0, CT=0.3


class DefenderExporter:
    """Layer 10 接口: SFT pool → Qwen2.5-VL training format."""

    def __init__(self, data_flow: DataFlow, config: Optional[DefenderExportConfig] = None):
        self.df = data_flow
        self.cfg = config or DefenderExportConfig(
            weight_by_route={"SFT": 1.0, "CT": 0.3}
        )

    def _format_qwen_vl(self, traj_dict: dict) -> Optional[dict]:
        """Format a single trajectory as Qwen2.5-VL conversational SFT sample.

        ★ BUG-17: skip pseudo-bypass (final = src, no real op succeeded).
        """
        v = traj_dict.get("verdicts") or {}
        exec_steps = traj_dict.get("execution", [])
        b = traj_dict.get("brief") or {}
        if not exec_steps:
            return None

        # Use the FINAL image as the example
        final_img = exec_steps[-1].get("output_path", "")
        if not final_img:
            return None

        # ★ BUG-17 retroactive filter
        src = b.get("src_face_path", "")
        if final_img == src:
            return None  # pseudo-bypass, skip
        real_op = any(
            (s.get("output_path", "") != src) and s.get("output_path", "")
            and not ("All models failed" in (s.get("error", "") or "")
                     or "MOCK_UNAVAILABLE" in (s.get("error", "") or ""))
            for s in exec_steps
        )
        if not real_op:
            return None  # all mocked, skip

        family = traj_dict.get("attack_family", "unknown")
        sandbox_pass = bool(v.get("sandbox_pass", False))

        # Build CoT from Tier-3 reasoning (if present) or Tier-2
        tier3 = v.get("tier3") or {}
        tier2 = v.get("tier2") or {}
        forensic_cot = tier3.get("reasoning", "") or tier2.get("reasoning", "")

        # 1. Image is FAKE (we generated it) — label MUST be fake regardless of detector
        label = "fake"
        gt_explanation = (
            f"This image is a {family} forgery generated by the pipeline "
            f"{' → '.join(s.get('tool','') for s in exec_steps)}. "
        )

        # Add CoT if available
        if forensic_cot:
            gt_explanation += f" Forensic cues: {forensic_cot[:400]}"

        return {
            "id": traj_dict.get("trajectory_id", ""),
            "image": final_img,
            "conversations": [
                {
                    "from": "human",
                    "value": ("<image>\nIs this face image real or fake? "
                              "Reason step by step and conclude with <answer>real|fake</answer>.")
                },
                {
                    "from": "gpt",
                    "value": (
                        f"<think>\n{gt_explanation}\n</think>\n"
                        f"<answer>{label}</answer>"
                    )
                }
            ],
            "weight": self.cfg.weight_by_route.get(
                traj_dict.get("data_route", "SFT"), 1.0
            ),
            "meta": {
                "family": family,
                "bypass_succeeded": sandbox_pass,
                "detector_signature": traj_dict.get("detector_signature", ""),
                "round_id": traj_dict.get("round_id", 0),
            },
        }

    def export(
        self,
        output_jsonl: str | Path,
        round_id_range: Optional[tuple[int, int]] = None,
        include_routes: tuple = ("SFT", "CT"),
    ) -> dict:
        """Export to Qwen2.5-VL training jsonl. Returns stats."""
        Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)
        sql = "SELECT full_json FROM trajectories WHERE data_route IN ({})".format(
            ",".join("?" * len(include_routes))
        )
        params = list(include_routes)
        if round_id_range:
            sql += " AND round_id >= ? AND round_id <= ?"
            params.extend(round_id_range)

        n_written = 0; n_skipped = 0; by_family = {}
        # BUG C 修: append mode, 跨 run 累积 SFT pool
        mode = "a" if Path(output_jsonl).exists() else "w"
        with open(output_jsonl, mode) as f:
            for (full_json,) in self.df.conn.execute(sql, params):
                traj = json.loads(full_json)
                sample = self._format_qwen_vl(traj)
                if sample is None:
                    n_skipped += 1
                    continue
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                n_written += 1
                fam = traj.get("attack_family", "unknown")
                by_family[fam] = by_family.get(fam, 0) + 1

        return {
            "output_jsonl": str(output_jsonl),
            "n_written": n_written,
            "n_skipped": n_skipped,
            "by_family": by_family,
        }

    def should_trigger_retrain(
        self,
        diag: DiagnosisResult,
        threshold: float = 0.60,
        min_attempts: int = 100,
    ) -> bool:
        """Trigger defender retrain when bypass_rate sustained > threshold."""
        if diag.global_bypass_rate <= threshold:
            return False
        # 还要 confirm 至少有 min_attempts 样本
        total = sum(
            int(self.df.conn.execute(
                "SELECT COUNT(*) FROM trajectories WHERE round_id=?",
                (diag.round_id,)
            ).fetchone()[0])
            for _ in range(1)
        )
        return total >= min_attempts


# ────────────────────────── Lv5 retrain script template ─────────────

RETRAIN_SCRIPT_TEMPLATE = """#!/bin/bash
# Auto-generated defender retrain script
# Triggered after round {round_id} with global_bypass_rate = {bypass_rate:.3f}
# SFT pool: {sft_jsonl}

set -e

BASE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
OUTPUT_DIR="./defender_v{version}"
DATA_JSONL="{sft_jsonl}"
NUM_EPOCH=3
LR=1e-5
LORA_R=16

# Using swift / llamafactory / lit-llama style command
# Customize per your training stack
python -m swift sft \\
    --model "$BASE_MODEL" \\
    --dataset "$DATA_JSONL" \\
    --output_dir "$OUTPUT_DIR" \\
    --num_train_epochs $NUM_EPOCH \\
    --learning_rate $LR \\
    --lora_rank $LORA_R \\
    --train_type lora \\
    --max_length 4096 \\
    --gradient_accumulation_steps 4 \\
    --eval_strategy steps --eval_steps 200 \\
    --save_strategy steps --save_steps 500 \\
    --logging_steps 10

echo "Retrained defender_v{version} → $OUTPUT_DIR"
echo "Next: swap into sandbox.py Tier-2 + restart attack loop"
"""


def emit_retrain_script(round_id: int, bypass_rate: float, sft_jsonl: str,
                        version: int, out_path: str | Path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    script = RETRAIN_SCRIPT_TEMPLATE.format(
        round_id=round_id, bypass_rate=bypass_rate,
        sft_jsonl=sft_jsonl, version=version,
    )
    Path(out_path).write_text(script)
    import os
    os.chmod(out_path, 0o755)
    _log.info(f"Retrain script written: {out_path}")
    return str(out_path)


# ────────────────────────── Smoke test ──────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import time as _t

    df = DataFlow(db_path="/tmp/test_diag_export.db")

    # Commit some fake trajectories across 2 rounds, 3 families
    for round_id in [0, 1]:
        for fam, n_attempts, byp_rate in [
            ("frontal_swap", 8, 0.75),
            ("id_diff", 6, 0.50),
            ("reenact", 4, 0.10),     # weak
        ]:
            for i in range(n_attempts):
                sandbox_pass = i < int(n_attempts * byp_rate)
                traj = {
                    "trajectory_id": f"r{round_id}_{fam}_{i}",
                    "round_id": round_id,
                    "baseline": "v2",
                    "attack_family": fam,
                    "verdicts": {
                        "sandbox_pass": sandbox_pass,
                        "tier1": {"niqe": 7.0, "arcface_id_sim": 0.7},
                        "tier2": {"reasoning": f"Sample reasoning for {fam}"},
                        "tier3": {"reasoning": f"Forensic: visible {fam} artifacts at jaw"},
                    },
                    "execution": [
                        {"tool": "face_align", "output_path": ""},
                        {"tool": "inswapper_128",
                         "output_path": f"/tmp/r{round_id}_{fam}_{i}.png"},
                    ],
                    "brief": {"attack_class": fam, "brief_text": f"test {fam}"},
                    "cost_usd": 0.005,
                    "timestamp": _t.time(),
                    "detector_signature": "tier1_func+tier2_gemini-2.5-flash",
                }
                df.commit_trajectory(traj, allow_duplicate=True)

    print("=== Diagnose round 0 ===")
    diag = Diagnoser(df)
    r0 = diag.diagnose(round_id=0, top_n_weak=2)
    print(f"  global_bypass_rate = {r0.global_bypass_rate:.3f}")
    print(f"  family rates       = {r0.family_bypass_rates}")
    print(f"  weak families      = {r0.weak_families}")
    print(f"  strong families    = {r0.strong_families}")
    print(f"  next-round weights = {r0.next_round_family_weights}")

    print(f"\n=== Cross-round trend ===")
    for fam, t in diag.cross_round_trend().items():
        print(f"  {fam}: delta = {t['delta']:+.3f}, current = {t['current_rate']:.3f}")

    print(f"\n=== Defender export (Qwen2.5-VL format) ===")
    exporter = DefenderExporter(df)
    stats = exporter.export("/tmp/defender_v0_sft.jsonl")
    print(f"  {stats}")

    print(f"\n=== Retrain trigger check ===")
    should = exporter.should_trigger_retrain(r0, threshold=0.4, min_attempts=10)
    print(f"  should_trigger_retrain = {should}")
    if should:
        script_path = emit_retrain_script(
            round_id=0, bypass_rate=r0.global_bypass_rate,
            sft_jsonl="/tmp/defender_v0_sft.jsonl",
            version=1, out_path="/tmp/retrain_v1.sh",
        )
        print(f"  Retrain script emitted: {script_path}")

    df.close()
