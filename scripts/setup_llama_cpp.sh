#!/bin/bash
# =============================================================================
# setup_llama_cpp.sh — Build llama-server for H100 (SM90a/CUDA arch 90a)
# =============================================================================
# Usage: bash setup_llama_cpp.sh
#
# Output: /workspace/llama.cpp/build/bin/llama-server
#
# After this, restore_pod.sh copies to $REPO_DIR/bin/llama-server
# =============================================================================
set -e

# Pin CUDA paths for H100
export PATH="/usr/local/cuda/bin:${PATH:-}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

LLAMA_DIR="/workspace/llama.cpp"
BUILD_DIR="${LLAMA_DIR}/build"

# Ensure build tools available
command -v cmake &>/dev/null || { echo "ERROR: cmake not found. Run: apt-get install -y cmake ninja-build"; exit 1; }

echo "--- Cloning llama.cpp ---"
if [ ! -d "$LLAMA_DIR" ]; then
    git clone --depth 1 https://github.com/ggerganov/llama.cpp "$LLAMA_DIR"
else
    echo "   Already cloned. Pulling latest..."
    git -C "$LLAMA_DIR" pull --ff-only || echo "   (pull failed, using cached clone)"
fi

echo "--- Configuring for H100 (SM90a, CUDA arch 90a) ---"
cmake -B "$BUILD_DIR" "$LLAMA_DIR" \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_F16=ON \
    -DCMAKE_CUDA_ARCHITECTURES=90a \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=ON \
    -DLLAMA_CURL=ON

echo "--- Building llama-server ($(nproc) cores) ---"
cmake --build "$BUILD_DIR" --config Release --target llama-server -j "$(nproc)"

echo "--- Verifying ---"
if [ -f "$BUILD_DIR/bin/llama-server" ]; then
    echo "SUCCESS: $BUILD_DIR/bin/llama-server"
    "$BUILD_DIR/bin/llama-server" --version 2>/dev/null || true
else
    echo "ERROR: llama-server binary not found after build."
    exit 1
fi
