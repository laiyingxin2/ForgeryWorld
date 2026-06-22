# Lv5 启动手册 — 一键四步

## Phase A: 数据准备 (已完成)
```
python launch_lv5.py --phase prep
# Output:
#   attacker_trajectory: 6+ samples
#   per_step:            35+ samples
#   anchor_replay:       6+ samples
#   defender_sft:        19+ samples
```

## Phase B: 启动 FakeVLM vLLM server (需要 GPU)
```
nvidia-smi  # 看 GPU
bash outputs/lv5/lv5_deploy_vllm.sh
# 等 60 秒
curl http://localhost:8000/health
# 期望: {"status":"ok"}
```

## Phase C: 跑 pre-train baseline (gemini Tier-2 vs FakeVLM Tier-2)
```
# 1. baseline (gemini)
python orchestrator.py --mode v2 --rounds 1 --briefs 8 --rollouts 2 \
    --api-only-ops --tier2-backend viviai \
    --out outputs_lv5_pre_baseline \
    --src-pool data/real_faces/*.png

# 2. FakeVLM untrained 当 Tier-2
python orchestrator.py --mode v2 --rounds 1 --briefs 8 --rollouts 2 \
    --api-only-ops --tier2-backend fakevlm_local \
    --fakevlm-endpoint http://localhost:8000/v1 \
    --out outputs_lv5_pre_fakevlm \
    --src-pool data/real_faces/*.png

# 期望: bypass rate 不同 (FakeVLM 是 face-specific = 更严格)
```

## Phase D: SFT 训练 (1-6h on H100)
```
bash outputs/lv5/attacker_lv5/lv5_attacker_train.sh
# 出 ./attacker_v{ts}/ LoRA
```

## Phase E: Eval 训后
```
# 1. 替换 vLLM 用新 LoRA
# 2. 重跑 orchestrator
# 3. 对比 pre vs post bypass rate
# 期望: post < pre (detector 更强了)
```

## 一键脚本
```
python launch_lv5.py --phase all
```
