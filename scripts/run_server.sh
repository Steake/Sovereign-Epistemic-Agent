#!/usr/bin/env bash
# =============================================================================
# run_server.sh — Launch vLLM Server for Sovereign Epistemic Agent
# =============================================================================
set -e

# ── Load Environment ─────────────────────────────────────────────────────────
ENV_FILE="/workspace/.container_env"
VENV_PATH="/workspace/venv"

if [[ -f "${ENV_FILE}" ]]; then
    source "${ENV_FILE}"
    echo "[INFO] Loaded environment from ${ENV_FILE}"
    export VLLM_USE_V1=0
else
    echo "[ERROR] ${ENV_FILE} not found. Run setup_container.sh first."
    exit 1
fi

if [[ -d "${VENV_PATH}" ]]; then
    source "${VENV_PATH}/bin/activate"
    echo "[INFO] Activated venv at ${VENV_PATH}"
else
    echo "[ERROR] Venv not found at ${VENV_PATH}"
    exit 1
fi

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_REPO="Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2"
MODEL_DIR="${HF_HOME}/hub/models--$(echo "$MODEL_REPO" | tr '/' '--')"

# Since we used 'hf download --local-dir', the model dir is flat
REAL_MODEL_PATH="${MODEL_DIR}"

if [[ ! -f "${REAL_MODEL_PATH}/config.json" ]]; then
    echo "[ERROR] Could not find config.json in ${REAL_MODEL_PATH}"

    echo "        Ensure step 7 of setup_container.sh completed successfully."
    exit 1
fi

echo "[INFO] Starting vLLM server with model at: ${REAL_MODEL_PATH}"

# ── Launch vLLM ───────────────────────────────────────────────────────────────
# Launch vLLM
# --max-model-len: Balanced context for 27B reasoning
# --gpu-memory-utilization: High utilization for H100 (80GB)
python3 -m vllm.entrypoints.openai.api_server \
    --model "${REAL_MODEL_PATH}" \
    --served-model-name "qwen3.5-27b-reasoning" \
    --port 8000 \
    --host 0.0.0.0 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 1 \
    --trust-remote-code
