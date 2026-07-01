"""Lv5 启动脚本 — 一站式从攻击数据到模型自演化.

3 个阶段:
  (A) Data prep: 把 outputs/data_flow_v2.db 里的 trajectory 转 Qwen2.5-VL/FakeVLM SFT 格式
  (B) vLLM deploy: 启动 FakeVLM as Tier-2 detector (需 GPU)
  (C) SFT loop: 训 FakeVLM 在新攻击数据上 → 验证 bypass rate 下降

Usage:
    python launch_lv5.py --phase prep      # 数据预处理
    python launch_lv5.py --phase deploy    # 启动 vLLM (GPU required)
    python launch_lv5.py --phase train     # 启动 SFT
    python launch_lv5.py --phase eval      # 训练后用新模型 evaluate
    python launch_lv5.py --phase all       # 三步依次
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from data_flow import DataFlow
from lv5_connector import Lv5AttackerExporter, Lv5Config
from diagnosis_and_export import DefenderExporter, DefenderExportConfig


def phase_prep(args):
    """阶段 A: 从 outputs/data_flow_v2.db 导出 attacker SFT + defender SFT pool."""
    print(f"\n═══ Phase A: Data Preparation ═══")
    df = DataFlow(db_path=args.db)

    # 1. Attacker SFT (SEAL-style)
    print(f"\n[A-1] Attacker SFT export (SEAL + STaSC + RISE 混合)")
    lv5_cfg = Lv5Config(base_model=args.fakevlm_ckpt)
    attacker_exporter = Lv5AttackerExporter(df, config=lv5_cfg)
    attacker_out = Path(args.out_dir) / "attacker_lv5"
    stats = attacker_exporter.export(str(attacker_out))
    print(f"  → attacker_trajectory: {stats['n_trajectory_samples']} samples")
    print(f"  → per_step:            {stats['n_per_step_samples']} samples")
    print(f"  → anchor_replay:       {stats['n_anchor_samples']} samples")
    print(f"  → script:              {stats['train_script']}")

    # 2. Defender SFT (Q9 修后 — 用 attack image, 不是 src)
    print(f"\n[A-2] Defender SFT export")
    def_cfg = DefenderExportConfig(weight_by_route={"SFT": 1.0, "CT": 0.3})
    def_exporter = DefenderExporter(df, config=def_cfg)
    def_jsonl = str(Path(args.out_dir) / "defender_sft_v2.jsonl")
    def_stats = def_exporter.export(def_jsonl)
    print(f"  → defender_sft.jsonl: {def_stats['n_written']} samples")
    print(f"  → family dist:        {def_stats['by_family']}")

    df.close()
    return {"attacker": stats, "defender": def_stats}


def phase_deploy(args):
    """阶段 B: 启动 FakeVLM vLLM server."""
    print(f"\n═══ Phase B: Deploy FakeVLM vLLM Server ═══")
    print(f"\nCheckpoint: {args.fakevlm_ckpt}")
    p = Path(args.fakevlm_ckpt)
    if not p.exists():
        print(f"  ✗ ckpt not found")
        return False
    sftens = list(p.glob("*.safetensors"))
    total_mb = sum(f.stat().st_size for f in sftens) / 1024 / 1024
    print(f"  ✓ {len(sftens)} safetensors, {total_mb:.0f} MB")

    # check GPU
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.free",
                                        "--format=csv,noheader"], text=True).strip()
        print(f"\nGPU:\n{out}")
    except Exception:
        print("\n⚠️ nvidia-smi 失败 — 没 GPU?")
        return False

    cmd = [
        "vllm", "serve", args.fakevlm_ckpt,
        "--port", str(args.vllm_port),
        "--tensor-parallel-size", "1",
        "--gpu-memory-utilization", "0.85",
        "--max-model-len", "4096",
        "--enforce-eager",
    ]
    print(f"\nCommand:\n  {' '.join(cmd)}")
    print(f"\n*** Server 启动可能需要 30-60 秒. 你可以 background 它. ***\n")
    print(f"To launch: {' '.join(cmd)}")

    deploy_script = Path(args.out_dir) / "lv5_deploy_vllm.sh"
    deploy_script.write_text(
        f"#!/bin/bash\nset -e\n{' '.join(cmd)} > /tmp/vllm_fakevlm.log 2>&1 &\n"
        f"echo \"vLLM PID=$!\"\nsleep 60\n"
        f"echo \"Check: curl http://localhost:{args.vllm_port}/health\"\n"
    )
    os.chmod(deploy_script, 0o755)
    print(f"  Script: {deploy_script}")
    return True


def phase_train(args):
    """阶段 C: SFT training — 用 attacker pool 训 FakeVLM."""
    print(f"\n═══ Phase C: SFT Training ═══")

    attacker_dir = Path(args.out_dir) / "attacker_lv5"
    train_script = attacker_dir / "lv5_attacker_train.sh"
    if not train_script.exists():
        print(f"  ✗ train script not found, run --phase prep first")
        return False

    print(f"\nTrain script: {train_script}")
    print(f"\nManual launch (background, GPU required):")
    print(f"  bash {train_script}")
    print(f"\nExpected time: 1-6 hours on H100 (3 epochs, LoRA r=32)")
    print(f"Output: ./attacker_v{{timestamp}}/")
    return True


def phase_eval(args):
    """阶段 D: 评估 — 训前 vs 训后 bypass rate 对比."""
    print(f"\n═══ Phase D: Evaluation ═══")
    print(f"\n用 sandbox 跑 30 fresh samples, Tier-2 切换:")
    print(f"  1. Tier-2 = viviai gemini-2.5-flash (baseline)")
    print(f"  2. Tier-2 = trained FakeVLM ({args.trained_lora})")
    print(f"\n命令:")
    print(f"  python orchestrator.py --mode v2 --rounds 1 --briefs 15 --rollouts 2 \\")
    print(f"      --tier2 gemini-2.5-flash --out outputs_pre_lv5")
    print(f"  # 然后切换 tier-2 → FakeVLM")
    print(f"  python orchestrator.py --mode v2 --rounds 1 --briefs 15 --rollouts 2 \\")
    print(f"      --tier2 fakevlm_local --out outputs_post_lv5")
    print(f"\n对比 bypass rate. 期望: post < pre (detector 被训得更强)")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["prep", "deploy", "train", "eval", "all"],
                        default="prep")
    parser.add_argument("--db",
        default="/data/disk4/lyx_ICML/self_evolution_forgery/outputs/data_flow_v2.db")
    parser.add_argument("--out_dir",
        default="/data/disk4/lyx_ICML/self_evolution_forgery/outputs/lv5")
    parser.add_argument("--fakevlm_ckpt",
        default="/cpfs01/bob_workspace/students/lyx/ICML/FakeVLM/Origin/FakeVLM/checkpoints/multi_20260329_132526_llava-1.5-7b")
    parser.add_argument("--vllm_port", type=int, default=8000)
    parser.add_argument("--trained_lora", default="")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if args.phase in ("prep", "all"):
        phase_prep(args)
    if args.phase in ("deploy", "all"):
        phase_deploy(args)
    if args.phase in ("train", "all"):
        phase_train(args)
    if args.phase in ("eval", "all"):
        phase_eval(args)

    print(f"\n\n═══ DONE ═══")


if __name__ == "__main__":
    main()
