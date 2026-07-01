"""Lv5 Weight-Level Self-Evolution Connector.

基于 4 个 Lv1-Lv6 paper-reading agent 综合后的最终设计:

  ★ 主体: SEAL (arxiv 2506.10943, MIT Pari/Kumar/Andreas 2025)
    - Qwen2.5-7B/VL-7B + LoRA (r=32, α=128)
    - ReST-EM 外层 RL (binary reward = bypass downstream perf improved)
    - Self-edit format: model 生成 synthetic finetune data + hyperparams

  ★ 数据规模: STaSC (arxiv 2503.08681, ACL 2025 wksp) small-data recipe
    - 500-5k 样本足够 (vs SEAL 1k-2k SQuAD)
    - LR 7e-6, batch 8, 1 epoch/iter, Ninit=5, Ncorr=5

  ★ Per-step credit: RISE (arxiv 2407.18219, NeurIPS 2024)
    - 每个 attack step 作 turn, sandbox per-step attribution 作 turn reward
    - Reward-weighted SFT or DPO

  ★ Composite reward (修正 SEAL 默认):
      r_total = α · sandbox_bypass (verifiable, 0/1)
              + β · skill_GOOD_count / steps (LLM-judge, [0,1])
      α=0.7, β=0.3 初期; β 可调

  ★ 防灾难性遗忘: STaR-style replay (anchor set from W1-W3 traj)

  ★ Defender Lv5 (独立): defender_export.py 已写, 触发 retrain_v1.sh

支持 2 个分离训练目标:
  (1) ATTACKER policy LLM: 训 Qwen2.5-VL-7B 替代 viviai API setter
  (2) DEFENDER detector LLM: 训 Qwen2.5-VL-7B 替代 sandbox Tier-2 (已在 defender_export.py)

本文件只负责生成训练 jsonl + 启动脚本; 实际 SFT 用 swift / LLaMA-Factory.
"""
from __future__ import annotations
import json
import logging
import re
from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass, field, asdict

from data_flow import DataFlow


_log = logging.getLogger(__name__)


# ────────────────────────── Config (verbatim from papers) ──────────

@dataclass
class Lv5Config:
    """Lv5 attacker/defender SFT config — SEAL + STaSC hybrid.

    ★ 用户 confirm: 用现成的 FakeVLM (llava-1.5-7b-fakevlm) 作 base, 而不是从零的 Qwen2.5-VL.
    FakeVLM 权重在: /cpfs01/bob_workspace/students/lyx/ICML/FakeVLM/Origin/FakeVLM/checkpoints/
    备选: Qwen3-VL-8B-instruct (LoRA adapter 已有)
    """
    # 默认: 改用现成 FakeVLM (LLaVA-1.5-7B 已经 fine-tune 到 fakevlm 任务上)
    # 仅做 SFT continuation, 而非从零训
    base_model: str = "/cpfs01/bob_workspace/students/lyx/ICML/FakeVLM/Origin/FakeVLM/checkpoints/multi_20260329_132526_llava-1.5-7b"
    # 备选 Qwen3-VL adapter (更大上下文, thinking mode):
    # base_model: str = "/cpfs01/bob_workspace/students/lyx/ICML/FakeVLM/Origin/FakeVLM/checkpoints/full_qwen_seq_highbs_20260329_165443_qwen3-vl-8b-thinking"
    lora_rank: int = 32             # SEAL: r=64; we halve for VL since VL embeddings bigger
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

    # STaSC small-data recipe
    learning_rate: float = 7e-6
    batch_size: int = 8
    grad_accumulation: int = 4
    n_epoch_per_iter: int = 1
    max_seq_length: int = 4096

    # Composite reward
    alpha_bypass: float = 0.7        # sandbox verifiable signal weight
    beta_skill: float = 0.3          # LLM-judge attribution signal weight

    # ReST-EM outer loop (SEAL)
    n_self_edits_per_iter: int = 4   # M_n in SEAL Algorithm 1
    n_test_per_edit: int = 5         # for binary reward eval
    n_outer_iter: int = 2            # B in SEAL Algorithm 1; OOM if 3+ on H100

    # Replay anchor set against catastrophic forgetting (STaR-style)
    replay_anchor_round_max: int = 0    # use round-0 trajectories as anchors
    replay_anchor_ratio: float = 0.2     # 20% of training mini-batch is replay anchors

    # Output format
    output_format: Literal["qwen_vl_chat", "llava"] = "qwen_vl_chat"


# ────────────────────────── Trajectory → training sample ────────────

def trajectory_to_attacker_sample(traj: dict, cfg: Lv5Config) -> Optional[dict]:
    """Convert one trajectory → Qwen2.5-VL attacker training sample.

    Attacker target: given (brief, src_image, family), generate a chain that bypasses detector.

    Format (Qwen2.5-VL multimodal chat):
      user: <image> + brief.brief_text + family + skill_lookup
      assistant: <think>strategy reasoning</think><chain>[op1, op2, ...]</chain>

    Sample weight:
      r_total = α · bypass + β · (good_step_count / total_steps)
    """
    v = traj.get("verdicts") or {}
    b = traj.get("brief") or {}
    exec_steps = traj.get("execution", [])
    attribution = traj.get("attribution", [])
    if not exec_steps or not b.get("src_face_path"):
        return None

    # ★ BUG Q9 修: 训练图必须是最后 step 的 output_path (= 实际攻击产物),
    # 而非 src_face_path (= 原图). 用 src_face 训会让 detector 学错.
    final_attack_image = exec_steps[-1].get("output_path") or b.get("src_face_path")
    from pathlib import Path as _P
    if not _P(final_attack_image).exists():
        for s in reversed(exec_steps):
            op = s.get("output_path", "")
            if op and _P(op).exists():
                final_attack_image = op
                break

    # ★ BUG-17: 过滤 pseudo-bypass (没真生成新图)
    src = b.get("src_face_path", "")
    if final_attack_image == src:
        return None
    real_op = any(
        (s.get("output_path", "") != src) and s.get("output_path", "")
        and not ("All models failed" in (s.get("error", "") or "")
                 or "MOCK_UNAVAILABLE" in (s.get("error", "") or ""))
        for s in exec_steps
    )
    if not real_op:
        return None

    bypass = float(bool(v.get("sandbox_pass", False)))
    # Per-step GOOD/BAD ratio
    if attribution:
        good_count = sum(1 for a in attribution if a.get("label") == "GOOD")
        skill_score = good_count / len(attribution)
    else:
        skill_score = bypass  # 没 attribution 时退化为 outcome

    r_total = cfg.alpha_bypass * bypass + cfg.beta_skill * skill_score

    # Skip if very low reward (don't pollute training set)
    if r_total < 0.2:
        return None

    # Build chain string
    chain_repr = " → ".join(
        f"{s['tool']}({json.dumps(s.get('params',{}), separators=(',',':'))})"
        for s in exec_steps
    )

    # Strategy reasoning from skill_extracted or attribution
    strategy = traj.get("skill_extracted") or ""
    if not strategy and attribution:
        strategy = "Strategy summary from attribution: " + "; ".join(
            f"step {a['step']} ({a['label']}): {a.get('reason','')[:80]}"
            for a in attribution
        )

    family = traj.get("attack_family", "unknown")
    brief_text = b.get("brief_text", "")[:400]
    skill_doc_hint = ""
    sl = traj.get("skill_lookup") or {}
    if sl.get("E_k_retrieved_ids"):
        skill_doc_hint = f"\n[Retrieved tactics: {','.join(sl['E_k_retrieved_ids'][:3])}]"

    user_msg = (
        f"<image>\nYou are a face-KYC red-team attacker. Generate an attack chain "
        f"that bypasses the deepfake detector.\n"
        f"Attack family: {family}\n"
        f"Brief: {brief_text}\n"
        f"{skill_doc_hint}\n"
        f"Output as: <think>STRATEGY</think><chain>OP1 → OP2 → ...</chain>"
    )
    assistant_msg = f"<think>{strategy[:600]}</think><chain>{chain_repr}</chain>"

    return {
        "id": traj.get("trajectory_id", ""),
        "image": final_attack_image,  # ★ 修后用攻击产物图, 不是 src
        "src_image": b["src_face_path"],  # 保留 src 路径作 metadata 备用
        "conversations": [
            {"from": "human", "value": user_msg},
            {"from": "gpt", "value": assistant_msg},
        ],
        "weight": r_total,
        "meta": {
            "family": family,
            "bypass": bool(bypass),
            "skill_score": skill_score,
            "round_id": traj.get("round_id", 0),
            "detector_signature": traj.get("detector_signature", ""),
        },
    }


# ────────────────────────── Per-step (RISE-style) ──────────────────

def trajectory_to_per_step_samples(traj: dict, cfg: Lv5Config) -> list[dict]:
    """RISE-style: each step is a separate "turn", reward by per-step attribution.

    Useful for fine-grained credit assignment when bypass=False but some steps GOOD.
    """
    b = traj.get("brief") or {}
    exec_steps = traj.get("execution", [])
    attribution = traj.get("attribution", [])
    if not exec_steps or not attribution or len(exec_steps) != len(attribution):
        return []

    samples = []
    family = traj.get("attack_family", "unknown")
    for i, (s, a) in enumerate(zip(exec_steps, attribution)):
        prefix_chain = " → ".join(es["tool"] for es in exec_steps[:i])
        next_op = s["tool"]
        params_str = json.dumps(s.get("params", {}), separators=(",", ":"))
        label = a.get("label", "GOOD")
        if label != "GOOD":
            continue   # 只在 GOOD step 上 train (positive-only)
        intermediate_img = s.get("input_path") if i > 0 else b.get("src_face_path")
        if not intermediate_img:
            continue

        user_msg = (
            f"<image>\nFace-KYC red-team agent. Current attack family: {family}.\n"
            f"Chain so far: {prefix_chain or '(empty, just started)'}.\n"
            f"What is the next best operator to apply?"
        )
        assistant_msg = f"<think>{a.get('reason','')[:200]}</think><op>{next_op}({params_str})</op>"

        samples.append({
            "id": f"{traj.get('trajectory_id','')}_step{i}",
            "image": intermediate_img,
            "conversations": [
                {"from": "human", "value": user_msg},
                {"from": "gpt", "value": assistant_msg},
            ],
            "weight": 1.0,
            "meta": {"family": family, "step": i, "label": label,
                      "round_id": traj.get("round_id", 0)},
        })
    return samples


# ────────────────────────── Lv5 Exporter ───────────────────────────

class Lv5AttackerExporter:
    """SEAL + STaSC + RISE hybrid attacker SFT exporter.

    生成两种 jsonl:
      - <prefix>_trajectory.jsonl: trajectory-level (Qwen2.5-VL chat, SEAL style)
      - <prefix>_per_step.jsonl  : per-step (RISE style)

    + 生成 retrain script (swift sft 命令).
    """

    def __init__(self, data_flow: DataFlow, config: Optional[Lv5Config] = None):
        self.df = data_flow
        self.cfg = config or Lv5Config()

    def _iter_trajectories(self, round_id_range: Optional[tuple] = None):
        sql = "SELECT full_json FROM trajectories"
        params = []
        if round_id_range:
            sql += " WHERE round_id >= ? AND round_id <= ?"
            params.extend(round_id_range)
        for (fj,) in self.df.conn.execute(sql, params):
            yield json.loads(fj)

    def export(
        self,
        out_dir: str | Path,
        round_id_range: Optional[tuple] = None,
        include_per_step: bool = True,
    ) -> dict:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        traj_jsonl = out_dir / "attacker_trajectory.jsonl"
        step_jsonl = out_dir / "attacker_per_step.jsonl"
        anchor_jsonl = out_dir / "attacker_anchor_replay.jsonl"

        n_traj = n_step = n_anchor = 0
        family_dist = {}

        with open(traj_jsonl, "w") as ft, open(step_jsonl, "w") as fs, open(anchor_jsonl, "w") as fa:
            for traj in self._iter_trajectories(round_id_range):
                sample = trajectory_to_attacker_sample(traj, self.cfg)
                if sample:
                    ft.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    n_traj += 1
                    fam = sample["meta"]["family"]
                    family_dist[fam] = family_dist.get(fam, 0) + 1
                    # anchor: round_id <= replay_anchor_round_max
                    if traj.get("round_id", 0) <= self.cfg.replay_anchor_round_max:
                        fa.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        n_anchor += 1
                if include_per_step:
                    for s in trajectory_to_per_step_samples(traj, self.cfg):
                        fs.write(json.dumps(s, ensure_ascii=False) + "\n")
                        n_step += 1

        # Write training script
        script_path = out_dir / "lv5_attacker_train.sh"
        script_path.write_text(self._build_train_script(out_dir, traj_jsonl, step_jsonl,
                                                         anchor_jsonl, n_traj, n_step))
        import os
        os.chmod(script_path, 0o755)

        # Write config snapshot
        (out_dir / "lv5_config.json").write_text(json.dumps(asdict(self.cfg), indent=2))

        return {
            "trajectory_jsonl": str(traj_jsonl),
            "per_step_jsonl": str(step_jsonl) if include_per_step else None,
            "anchor_replay_jsonl": str(anchor_jsonl),
            "train_script": str(script_path),
            "n_trajectory_samples": n_traj,
            "n_per_step_samples": n_step,
            "n_anchor_samples": n_anchor,
            "family_distribution": family_dist,
        }

    def _build_train_script(
        self, out_dir: Path, traj_jsonl: Path, step_jsonl: Path,
        anchor_jsonl: Path, n_traj: int, n_step: int,
    ) -> str:
        return f"""#!/bin/bash
# Lv5 ATTACKER SFT — SEAL + STaSC + RISE hybrid recipe
# Trajectory samples: {n_traj}, Per-step samples: {n_step}
# Auto-generated by lv5_connector.py

set -e
cd $(dirname $0)

BASE_MODEL="{self.cfg.base_model}"
OUTPUT_DIR="./attacker_v$(date +%Y%m%d_%H%M)"
TRAJ_JSONL="{traj_jsonl}"
STEP_JSONL="{step_jsonl}"
ANCHOR_JSONL="{anchor_jsonl}"

# 1. Merge trajectory + per-step + anchor (replay 20% from anchors against forgetting)
COMBINED_JSONL="$OUTPUT_DIR/combined_train.jsonl"
mkdir -p "$OUTPUT_DIR"
cat "$TRAJ_JSONL" "$STEP_JSONL" > "$COMBINED_JSONL"
# Replicate anchors {self.cfg.replay_anchor_ratio:.0%} of total to mitigate catastrophic forgetting
for i in $(seq 1 5); do cat "$ANCHOR_JSONL" >> "$COMBINED_JSONL"; done

# 2. swift sft (ModelScope SWIFT framework)
# install: pip install ms-swift>=2.5
swift sft \\
    --model "$BASE_MODEL" \\
    --dataset "$COMBINED_JSONL" \\
    --train_type lora \\
    --lora_rank {self.cfg.lora_rank} \\
    --lora_alpha {self.cfg.lora_alpha} \\
    --lora_dropout {self.cfg.lora_dropout} \\
    --target_modules {self.cfg.target_modules} \\
    --learning_rate {self.cfg.learning_rate} \\
    --num_train_epochs {self.cfg.n_epoch_per_iter} \\
    --per_device_train_batch_size {self.cfg.batch_size} \\
    --gradient_accumulation_steps {self.cfg.grad_accumulation} \\
    --max_length {self.cfg.max_seq_length} \\
    --output_dir "$OUTPUT_DIR" \\
    --eval_strategy steps --eval_steps 200 \\
    --save_strategy steps --save_steps 500 \\
    --logging_steps 10 \\
    --warmup_ratio 0.03 \\
    --weight_decay 0.0 \\
    --lr_scheduler_type cosine

echo ""
echo "════════════════════════════════════════════"
echo "Lv5 Attacker SFT completed: $OUTPUT_DIR"
echo "Next steps:"
echo "  1. Swap orchestrator setter to use this LoRA"
echo "  2. Run sandbox eval on held-out 500 trajectories"
echo "  3. If bypass_rate improved → keep; else revert (ReST-EM gate)"
echo "════════════════════════════════════════════"
"""


# ────────────────────────── Smoke test ─────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import time as _t

    # Test with a synthetic data_flow
    df = DataFlow(db_path="/tmp/lv5_test.db")

    # Commit 5 synthetic trajectories
    for i in range(5):
        bypass = i >= 3  # last 2 bypass
        traj = {
            "trajectory_id": f"test_t{i}",
            "round_id": 0,
            "baseline": "v2",
            "attack_family": "id_diff",
            "verdicts": {
                "sandbox_pass": bypass,
                "tier1": {"niqe": 6.0, "arcface_id_sim": 0.6},
                "tier2": {"reasoning": "looks natural"},
            },
            "execution": [
                {"step": 0, "tool": "face_align", "params": {},
                 "input_path": "/tmp/src.png", "output_path": "/tmp/r0_s0.png",
                 "tier1_metrics": {}},
                {"step": 1, "tool": "nano_banana_pro", "params": {"identity_hint": "person"},
                 "input_path": "/tmp/r0_s0.png", "output_path": "/tmp/r0_s1.png",
                 "tier1_metrics": {}},
                {"step": 2, "tool": "jpeg_85", "params": {"quality": 85},
                 "input_path": "/tmp/r0_s1.png", "output_path": "/tmp/r0_s2.png",
                 "tier1_metrics": {}},
            ],
            "attribution": [
                {"step": 0, "label": "GOOD", "reason": "alignment ok"},
                {"step": 1, "label": "GOOD" if bypass else "BAD",
                 "reason": "id-preserve worked" if bypass else "introduced artifact"},
                {"step": 2, "label": "GOOD" if bypass else "GOOD", "reason": "compression masked"},
            ],
            "brief": {"attack_class": "id_diff", "src_face_path": "/tmp/src.png",
                      "brief_text": "high-quality portrait of person X"},
            "skill_lookup": {"E_k_retrieved_ids": ["e1", "e2"]},
            "cost_usd": 0.05, "timestamp": _t.time(),
        }
        df.commit_trajectory(traj, allow_duplicate=True)

    exporter = Lv5AttackerExporter(df)
    stats = exporter.export("/tmp/lv5_export_test")
    print("=== Lv5 attacker export ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Show first sample
    print("\n=== First trajectory-level sample ===")
    with open(stats["trajectory_jsonl"]) as f:
        for line in f:
            sample = json.loads(line)
            print(f"  id={sample['id']} weight={sample['weight']:.3f}")
            print(f"  user (truncated): {sample['conversations'][0]['value'][:300]}")
            print(f"  gpt  (truncated): {sample['conversations'][1]['value'][:300]}")
            break

    print(f"\n✓ Lv5 connector working. Train script: {stats['train_script']}")
    df.close()
