#!/bin/bash
# P1-A multi-round train-eval loop (Method 3 paper-grade learning curve).
#
# For round R in 0..N_ROUNDS-1:
#   1. balance SFT pool (cumulative fake-positives + 50 real-positives)
#   2. train defender LoRA (warm-start from previous round when R>0)
#   3. restart vLLM with new LoRA
#   4. run K trajectories with new detector → measure bypass rate
#   5. append new bypass cases to cumulative fake-positive pool
# Output: per-round bypass_rate.json showing the learning curve.

set -e

cd /data/disk4/lyx_ICML/self_evolution_forgery
PROJ=/data/disk4/lyx_ICML/self_evolution_forgery
SCRIPTS=$PROJ/scripts/p1a_multi_round
PY=/cpfs01/bob_workspace/miniconda3/envs/fakevlm/bin/python
RUN_TAG="p1a_$(date +%Y%m%d_%H%M)"
OUT_BASE=$PROJ/outputs/p1a_multi_round/$RUN_TAG
mkdir -p $OUT_BASE/{rounds,pools,loras}

N_ROUNDS=${N_ROUNDS:-3}
N_TRAJ_PER_ROUND=${N_TRAJ_PER_ROUND:-5}   # briefs × rollouts inside orchestrator
N_REAL_PER_POOL=${N_REAL_PER_POOL:-50}
SRC_FACES=(
  $PROJ/data/real_faces/0_row0_real.png
  $PROJ/data/real_faces/0_row2_real.png
  $PROJ/data/real_faces/1_row1_real.png
)

# starting cumulative pool = original 113 fake-positives from v1
CUM_FAKE_POOL=$OUT_BASE/pools/cum_fake.jsonl
cp $PROJ/outputs/lv5/defender_sft_v2.jsonl $CUM_FAKE_POOL

PREV_LORA=""
SUMMARY=$OUT_BASE/learning_curve.json
echo "{" > $SUMMARY

for R in $(seq 0 $((N_ROUNDS-1))); do
  echo ""
  echo "═══════════════════════════════════════════════════════════════"
  echo "═══  ROUND $R / $((N_ROUNDS-1))                                ═══"
  echo "═══════════════════════════════════════════════════════════════"
  ROUND_DIR=$OUT_BASE/rounds/r${R}
  mkdir -p $ROUND_DIR

  # ── 1. balance SFT pool ──
  BALANCED=$OUT_BASE/pools/balanced_r${R}.jsonl
  echo ""
  echo "[R$R] step 1: balance SFT pool → $BALANCED"
  $PY $SCRIPTS/balance_sft_pool.py \
    --in-pool $CUM_FAKE_POOL \
    --real-faces-dir $PROJ/data/real_faces \
    --n-real $N_REAL_PER_POOL \
    --out $BALANCED

  # ── 2. train LoRA ──
  LORA_DIR=$OUT_BASE/loras/defender_r${R}
  echo ""
  echo "[R$R] step 2: train LoRA → $LORA_DIR"
  if [ -n "$PREV_LORA" ]; then
    echo "  warm-start from $PREV_LORA"
    WARM="--prev-lora $PREV_LORA"
  else
    WARM=""
  fi
  $PY $SCRIPTS/train_one_round.py \
    --data $BALANCED --out $LORA_DIR --epochs 3 $WARM 2>&1 | tee $ROUND_DIR/train.log | grep -E "loaded|epoch|saving|train_meta|loss"
  if [ ! -f $LORA_DIR/adapter_model.safetensors ]; then
    echo "[R$R] FATAL: training failed, no adapter_model.safetensors"; exit 1
  fi

  # ── 3. restart vLLM ──
  echo ""
  echo "[R$R] step 3: restart vLLM with new LoRA"
  bash $SCRIPTS/restart_vllm_with_lora.sh $LORA_DIR 2>&1 | tee $ROUND_DIR/vllm_restart.log | tail -10
  if ! curl -sf http://localhost:8000/health >/dev/null 2>&1; then
    echo "[R$R] FATAL: vLLM not up"; exit 1
  fi

  # ── 4. run K trajectories with new detector ──
  echo ""
  echo "[R$R] step 4: run $N_TRAJ_PER_ROUND traj with new detector"
  EVAL_OUT=$ROUND_DIR/orch_eval
  SRC_ARG="${SRC_FACES[$((R % ${#SRC_FACES[@]}))]}"
  cd $PROJ/src
  $PY orchestrator.py \
    --mode v2 --rounds 1 --briefs $N_TRAJ_PER_ROUND --rollouts 1 \
    --multi-agent-preset w6_full \
    --tier2-backend fakevlm_local \
    --src-pool $SRC_ARG \
    --out $EVAL_OUT 2>&1 | tee $ROUND_DIR/orch.log | grep -E "bypass|brief|round|checker|setter|cost|reasoning_bank|family|fakevlm" | tail -50
  cd $PROJ

  # extract bypass rate from this round's reports
  RPT=$(ls $EVAL_OUT/reports/r0_v2.json 2>/dev/null | head -1)
  if [ -n "$RPT" ] && [ -f "$RPT" ]; then
    BYPASS=$(python3 -c "import json; d=json.load(open('$RPT')); print(d['diagnosis']['global_bypass_rate'])")
    COST=$(python3 -c "import json; d=json.load(open('$RPT')); print(d.get('total_cost_usd', 0))")
  else
    BYPASS="null"; COST="null"
  fi
  echo "[R$R] bypass_rate=$BYPASS  cost=\$$COST"

  # ── 5. append new fake-positives to cumulative pool ──
  NEW_SFT=$EVAL_OUT/defender_sft_v2.jsonl
  N_NEW=0
  if [ -f $NEW_SFT ]; then
    N_NEW=$(wc -l < $NEW_SFT)
    cat $NEW_SFT >> $CUM_FAKE_POOL
    N_CUM=$(wc -l < $CUM_FAKE_POOL)
    echo "[R$R] cumulative fake pool from bypass cases: +$N_NEW → $N_CUM total"
  fi

  # ★ BUG-6 fix: if pool didn't grow (bypass=0), use STaSC to FORCE new SFT
  # from attack images that defender mistakenly marked as 'real'.
  # This breaks the "R0 catches all → pool stuck" stagnation loop.
  if [ "$N_NEW" = "0" ] || [ -z "$N_NEW" ]; then
    echo "[R$R] ★ pool didn't grow — invoking STaSC correction to synthesize new SFT"
    ATTACK_IMGS=$EVAL_OUT/face_attack_outputs
    if [ -d "$ATTACK_IMGS" ] && [ "$(ls -A $ATTACK_IMGS 2>/dev/null | wc -l)" -gt 0 ]; then
      # build metadata jsonl from attack images
      META=$ROUND_DIR/stasc_metadata.jsonl
      > $META
      for img in $ATTACK_IMGS/*.png; do
        echo "{\"image_path\":\"$img\",\"family\":\"frontal_swap\",\"pipeline_hint\":\"$(basename $img .png)\",\"prev_verdict\":\"real\"}" >> $META
      done
      N_IMG=$(wc -l < $META)
      echo "  $N_IMG attack images to correct"
      STASC_OUT=$ROUND_DIR/stasc_corrected_sft.jsonl
      $PY $SCRIPTS/inject_stasc_corrections.py \
        --image-dir $ATTACK_IMGS \
        --metadata $META \
        --out $STASC_OUT \
        --vllm-endpoint http://localhost:8000/v1 \
        --lora-model defender \
        --n-per-image 2 \
        --mode improving 2>&1 | tee $ROUND_DIR/stasc.log | tail -10
      if [ -f $STASC_OUT ]; then
        N_STASC=$(wc -l < $STASC_OUT)
        cat $STASC_OUT >> $CUM_FAKE_POOL
        N_CUM=$(wc -l < $CUM_FAKE_POOL)
        echo "[R$R] STaSC +$N_STASC → $N_CUM total (breaks stagnation)"
      fi
    else
      echo "  no attack images found, STaSC skipped"
    fi
  fi

  # append to summary json
  echo "  \"round_$R\": {\"bypass_rate\": $BYPASS, \"cost_usd\": $COST, \"lora\": \"$LORA_DIR\"}," >> $SUMMARY

  PREV_LORA=$LORA_DIR
done

# close summary json
sed -i '$ s/,$//' $SUMMARY
echo "}" >> $SUMMARY

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "═══  P1-A MULTI-ROUND DONE                                     "
echo "═══════════════════════════════════════════════════════════════"
echo "  summary: $SUMMARY"
cat $SUMMARY
