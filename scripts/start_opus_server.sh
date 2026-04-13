#!/bin/bash
# OpenAI-Compatible Server Ignition for Opus-Distill (llama.cpp)
# Target: Sovereign-Epistemic-Agent H100 Pod

SERVER_BIN="/workspace/llama.cpp/build/bin/llama-server"
MODEL_PATH="/workspace/models/Qwen3.5-27B.Q8_0.gguf"
MMPROJ_PATH="/workspace/models/mmproj-BF16.gguf"
PORT=8000
HOST="0.0.0.0"
CONTXT_SIZE=16384 # High-density context for reasoning

echo "--- Igniting Opus-Distill API Server (H100 Optimized) ---"
# -ngl 99 ensures full offload to H100 VRAM
# --fa enables FlashAttention
# --mmproj seats the vision projector for multimodal stability
nohup "$SERVER_BIN" \
    --model "$MODEL_PATH" \
    --mmproj "$MMPROJ_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --ctx-size "$CONTXT_SIZE" \
    --n-gpu-layers 99 \
    --flash-attn on \
    --no-mmap \
    --ubatch-size 512 \
    > llama_server.log 2>&1 &

echo "--- Server Backgrounded (PID: $!) ---"
echo "Monitor logs with: tail -f llama_server.log"
sleep 5
tail -n 20 llama_server.log
