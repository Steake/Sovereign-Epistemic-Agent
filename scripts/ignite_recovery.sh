#!/usr/bin/env bash
set -eo pipefail

if [ -z "$1" ]; then
    echo "Usage: ./ignite_recovery.sh <reasoning_budget>"
    echo "Example: ./ignite_recovery.sh 0     # Reasoning disabled completely"
    echo "Example: ./ignite_recovery.sh 512   # Capped 512-token reasoning budget"
    exit 1
fi

REASONING_BUDGET="$1"
WORKSPACE_DIR="/workspace/Sovereign-Epistemic-Agent"
MODEL_PATH="${WORKSPACE_DIR}/models/Qwen3.5-27B.Q8_0.gguf"
LOG_FILE="${WORKSPACE_DIR}/inference.log"

echo "=========================================================="
echo "Igniting vLLM Pod Server (Recovery Mode)"
echo "Reasoning Budget: ${REASONING_BUDGET}"
echo "Model: ${MODEL_PATH}"
echo "=========================================================="

# 1. Kill any existing instances
pkill -f llama-server || true
sleep 3

# 2. Determine reasoning flag.
# NOTE: --reasoning-budget 0 causes llama-server slot-reset bug that drops HTTP connections.
# Workaround: for budget=0, launch with NO reasoning flag at all.
# The answer-only prompt contract handles suppression at the prompt level.
# For budget>0, normal --reasoning-budget N flag is safe to use.
if [ "$REASONING_BUDGET" -eq "0" ]; then
    REASONING_FLAG=""
else
    REASONING_FLAG="--reasoning-budget ${REASONING_BUDGET}"
fi

# 3. Launch
nohup ${WORKSPACE_DIR}/bin/llama-server \
    -m "${MODEL_PATH}" \
    --flash-attn on -ngl 99 \
    --port 8000 --host 0.0.0.0 \
    --ctx-size 16384 \
    ${REASONING_FLAG} \
    > "${LOG_FILE}" 2>&1 &
SERVER_PID=$!

echo "Server ignited with PID $SERVER_PID"
echo "Tailing inference log. Wait for 'HTTP server listening'..."
sleep 2

tail -f "${LOG_FILE}" &
TAIL_PID=$!

# Wait passively for health check to pass
echo "Polling health endpoint..."
while ! curl -sf http://localhost:8000/health > /dev/null; do
    sleep 5
done

kill $TAIL_PID || true
echo ""
echo "=========================================================="
echo "Mission Ready. The H100 inference engine is alive and capped."
echo "Use configs/recovery_8192.yaml in your pipeline."
echo "=========================================================="
