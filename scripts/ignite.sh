#!/bin/bash
# =============================================================================
# ignite.sh — Unified Pod Resurrection Script for SM90a/H100
# =============================================================================
# Usage: bash /workspace/Sovereign-Epistemic-Agent/scripts/ignite.sh
#
# Expects (pre-built by restore_pod.sh):
#   $REPO_DIR/bin/llama-server
#   $REPO_DIR/models/Qwen3.5-27B.Q8_0.gguf
#   $REPO_DIR/.venv/
#
# Lessons:
#   - Flag is --flash-attn on  (NOT -fa; changed in newer llama.cpp builds)
#   - Health check needs 180s timeout: 27GB GGUF takes ~90s to load into VRAM
# =============================================================================
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_PATH="$REPO_DIR/bin/llama-server"
MODEL_PATH="$REPO_DIR/models/Qwen3.5-27B.Q8_0.gguf"
VENV_PATH="$REPO_DIR/.venv"

echo "🚀 Igniting Epistemic Agent Environment..."
cd "$REPO_DIR"

# 1. GPU Check
if ! command -v nvidia-smi &>/dev/null; then
    echo "⚠️  No NVIDIA GPU detected. Requires SM90a (H100)."
else
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
    echo "✅ Found GPU: $GPU_NAME"
fi

# 2. Venv
if [ ! -d "$VENV_PATH" ]; then
    echo "📦 Creating fresh Python venv..."
    python3 -m venv "$VENV_PATH"
    source "$VENV_PATH/bin/activate"
    pip install --upgrade pip -q
    pip install -e . -q
else
    echo "✅ venv already seated."
    source "$VENV_PATH/bin/activate"
fi

# 3. Pre-flight checks
[ -f "$BIN_PATH" ]   || { echo "❌ ERROR: llama-server not found at $BIN_PATH"; echo "   Run: bash scripts/restore_pod.sh"; exit 1; }
[ -f "$MODEL_PATH" ] || { echo "❌ ERROR: Model not found at $MODEL_PATH"; echo "   Run: HF_TOKEN=hf_xxx bash scripts/restore_pod.sh"; exit 1; }

echo "🔥 Starting Inference Server (SM90a optimised)..."
# Kill any existing server on :8000
fuser -k 8000/tcp 2>/dev/null || true
sleep 1

# Launch llama-server
#   --flash-attn on  : Flash Attention 2 (SM90a native)
#   -ngl 99          : Offload all layers to H100 VRAM
#   --port 8000      : OpenAI-compatible API
nohup "$BIN_PATH" \
    -m "$MODEL_PATH" \
    --flash-attn on \
    -ngl 99 \
    --port 8000 \
    --host 0.0.0.0 \
    --ctx-size 16384 \
    --reasoning off \
    --reasoning-budget 0 \
    > "$REPO_DIR/inference.log" 2>&1 &

SERVER_PID=$!
echo "   Server PID: $SERVER_PID"

# 4. Health Check (180s — 27GB model needs ~90s to load)
echo "🔍 Waiting for server health (up to 180s)..."
TIMEOUT=180
START=$(date +%s)
while ! curl -sf http://localhost:8000/health >/dev/null 2>&1; do
    ELAPSED=$(( $(date +%s) - START ))
    if [ "$ELAPSED" -gt "$TIMEOUT" ]; then
        echo "❌ ERROR: Server failed to start within ${TIMEOUT}s."
        echo "   Check: tail -50 $REPO_DIR/inference.log"
        exit 1
    fi
    printf "."
    sleep 2
done

echo ""
echo "✨ SUCCESS: Inference server READY at http://localhost:8000/v1"
curl -s http://localhost:8000/v1/models | python3 -c \
    "import sys,json; m=json.load(sys.stdin); print('   Model:', m['data'][0]['id'])" 2>/dev/null || true
echo "Mission Ready."
