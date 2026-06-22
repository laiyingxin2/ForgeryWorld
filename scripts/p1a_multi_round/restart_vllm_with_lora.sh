#!/bin/bash
# Kill current vLLM (if any) and restart with a fresh LoRA mounted as "defender".
# Args: $1 = absolute path to LoRA adapter dir.
set -e
LORA_PATH="$1"
if [ -z "$LORA_PATH" ] || [ ! -d "$LORA_PATH" ]; then
  echo "[restart_vllm] usage: $0 <lora_adapter_dir>"; exit 1
fi
if [ ! -f "$LORA_PATH/adapter_config.json" ]; then
  echo "[restart_vllm] missing adapter_config.json in $LORA_PATH"; exit 1
fi

BASE="/cpfs01/bob_workspace/students/lyx/ICML/FakeVLM/Origin/FakeVLM/checkpoints/multi_20260329_132526_llava-1.5-7b"
PORT=8000
VLLM_PY="/cpfs01/bob_workspace/miniconda3/envs/gca-vllm/bin/python3.11"
VLLM_BIN="/cpfs01/bob_workspace/miniconda3/envs/gca-vllm/bin/vllm"
LOG="/tmp/vllm_p1a.log"

echo "[restart_vllm] killing any existing vLLM on port $PORT"
PID=$(ps aux | grep "vllm serve" | grep -v grep | awk '{print $2}' | head -1)
if [ -n "$PID" ]; then
  echo "  found PID $PID; sending TERM"
  kill -TERM $PID 2>/dev/null || true
  for i in $(seq 1 30); do
    if ! ps -p $PID >/dev/null 2>&1; then break; fi
    sleep 1
  done
  if ps -p $PID >/dev/null 2>&1; then
    echo "  still alive after 30s; sending KILL"
    kill -KILL $PID 2>/dev/null || true
    sleep 3
  fi
fi
# also wait for port to be released
for i in $(seq 1 30); do
  if ! ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":$PORT$"; then break; fi
  sleep 1
done

echo "[restart_vllm] launching with LoRA: $LORA_PATH"
nohup $VLLM_BIN serve "$BASE" \
  --port $PORT --enable-lora \
  --lora-modules defender="$LORA_PATH" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.40 \
  --max-model-len 4096 \
  --enforce-eager \
  > $LOG 2>&1 &
NEW_PID=$!
disown
echo "  new PID $NEW_PID; log $LOG"

echo "[restart_vllm] waiting for /health up to 180s"
for i in $(seq 1 180); do
  if curl -sf http://localhost:$PORT/health >/dev/null 2>&1; then
    echo "  /health OK after ${i}s"
    # confirm defender LoRA registered
    MODELS=$(curl -s http://localhost:$PORT/v1/models 2>/dev/null | python3 -c "import sys, json; d=json.load(sys.stdin); print([m['id'] for m in d['data']])" 2>/dev/null)
    echo "  /v1/models = $MODELS"
    if echo "$MODELS" | grep -q "defender"; then
      echo "[restart_vllm] OK"
      exit 0
    else
      echo "  WARN: 'defender' not in models list"
    fi
  fi
  sleep 1
done
echo "[restart_vllm] FAIL: /health not up after 180s"
tail -40 $LOG
exit 1
