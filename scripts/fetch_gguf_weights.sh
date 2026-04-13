#!/bin/bash
# =============================================================================
# fetch_gguf_weights.sh — Download Qwen3.5-27B GGUF weights to REPO_DIR/models/
# =============================================================================
# Usage: HF_TOKEN=hf_xxx bash fetch_gguf_weights.sh
#
# Downloads:
#   Qwen3.5-27B.Q8_0.gguf  (~27GB) — main reasoning weights
#   mmproj-BF16.gguf        (~1GB)  — multimodal projector
#
# Target: $REPO_DIR/models/ (co-located with repo, not /workspace/models/)
# =============================================================================
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$REPO_DIR/models"
REPO_ID="Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2-GGUF"

[[ -n "${HF_TOKEN:-}" ]] || { echo "ERROR: HF_TOKEN not set."; exit 1; }
export HF_TOKEN="${HF_TOKEN}"
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "$MODEL_DIR"

echo "--- Installing hf CLI if needed ---"
command -v hf &>/dev/null || pip install -q "huggingface_hub[hf_xet]"

dl_if_missing() {
    local FILE="$1"
    local DEST="$MODEL_DIR/$FILE"
    if [[ -f "$DEST" ]] && [[ $(stat -c%s "$DEST" 2>/dev/null || echo 0) -gt 1000000 ]]; then
        echo "SKIP: $FILE already present."
        return
    fi
    echo "--- Downloading $FILE ---"
    hf download "$REPO_ID" "$FILE" --local-dir "$MODEL_DIR"
    echo "OK: $FILE → $DEST"
}

dl_if_missing "Qwen3.5-27B.Q8_0.gguf"
dl_if_missing "mmproj-BF16.gguf"

echo ""
echo "--- Verification ---"
ls -lh "$MODEL_DIR/"*.gguf 2>/dev/null || echo "WARNING: No .gguf files found in $MODEL_DIR"
echo "Done."
