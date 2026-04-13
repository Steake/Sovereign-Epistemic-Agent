#!/usr/bin/env bash
# =============================================================================
# restore_pod.sh — Sovereign Epistemic Agent: Full Cold-Start Pod Restore
# =============================================================================
# Idempotent: safe to re-run on a pod that already has some steps done.
#
# Usage:
#   HF_TOKEN=hf_xxx bash /workspace/scripts/restore_pod.sh
#   HF_TOKEN=hf_xxx bash /workspace/scripts/restore_pod.sh --skip-download
#
# Sequence:
#   1. Install system deps    (cmake, ninja, curl, aria2, git)
#   2. Build llama-server     (llama.cpp, CUDA arch 90a for H100)
#   3. Download GGUF weights  (Qwen3.5-27B Q8_0 + mmproj)
#   4. Set up Python venv     (uv or venv + project install)
#   5. Ignite                 (llama-server on :8000, health-check waits 180s)
#
# Lessons hard-won in production (H100, SM90a):
#   - cmake must be installed before llama.cpp build
#   - flash-attn flag is --flash-attn on  (NOT -fa; syntax changed)
#   - 27GB model load takes ~90s; health check needs 180s timeout
#   - HF_TOKEN required for gated GGUF repo download
#   - REPO_DIR is /workspace/Sovereign-Epistemic-Agent
# =============================================================================
set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; RESET='\033[0m'
info()    { echo -e "${BLUE}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[ OK ]${RESET}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
die()     { echo -e "${RED}[FAIL]${RESET}  $*" >&2; exit 1; }

# ── Args ──────────────────────────────────────────────────────────────────────
SKIP_DOWNLOAD=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-download) SKIP_DOWNLOAD=true; shift ;;
    --token)         HF_TOKEN="$2"; shift 2 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

# ── Config ────────────────────────────────────────────────────────────────────
WORKSPACE=/workspace
REPO_DIR="${WORKSPACE}/Sovereign-Epistemic-Agent"
BIN_DIR="${REPO_DIR}/bin"
MODEL_DIR="${REPO_DIR}/models"
VENV_DIR="${REPO_DIR}/.venv"
LLAMA_SRC="${WORKSPACE}/llama.cpp"
LLAMA_BIN="${LLAMA_SRC}/build/bin/llama-server"
SCRIPTS_DIR="${REPO_DIR}/scripts"

# Model: Opus-Distill 27B GGUF (Q8_0)
GGUF_REPO="Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2-GGUF"
GGUF_FILE="Qwen3.5-27B.Q8_0.gguf"
MMPROJ_FILE="mmproj-BF16.gguf"

[[ -d "${WORKSPACE}" ]]       || die "/workspace is not mounted."
[[ -n "${HF_TOKEN:-}" ]]      || die "HF_TOKEN is not set. Usage: HF_TOKEN=hf_xxx bash $0"

echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}  Sovereign Epistemic Agent — Pod Restore (H100 / SM90a)${RESET}"
echo -e "${BOLD}═══════════════════════════════════════════════════════════${RESET}"
echo ""

# ── Step 1: System deps ───────────────────────────────────────────────────────
info "Step 1/5 — System dependencies..."
apt-get update -qq
apt-get install -y -qq \
    build-essential cmake ninja-build git curl wget aria2 \
    libcurl4-openssl-dev python3 python3-pip python3-venv \
    2>&1 | tail -3
success "cmake $(cmake --version | head -1 | awk '{print $3}'), ninja ready."

# ── Step 2: Build llama-server (SM90a) ────────────────────────────────────────
info "Step 2/5 — Building llama-server for H100 (CUDA arch 90a)..."

if [[ -f "${BIN_DIR}/llama-server" ]]; then
    success "llama-server already at ${BIN_DIR}/llama-server — skipping build."
else
    mkdir -p "${BIN_DIR}"

    if [[ ! -d "${LLAMA_SRC}" ]]; then
        info "  Cloning llama.cpp..."
        git clone --depth 1 https://github.com/ggerganov/llama.cpp "${LLAMA_SRC}"
    else
        info "  llama.cpp present, pulling latest..."
        git -C "${LLAMA_SRC}" pull --ff-only 2>/dev/null || warn "  git pull failed, using cached clone."
    fi

    # Pin CUDA paths for H100
    export PATH="/usr/local/cuda/bin:${PATH:-}"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

    info "  Configuring cmake (CUDA arch=90a, flash-attn, curl)..."
    cmake -B "${LLAMA_SRC}/build" "${LLAMA_SRC}" \
        -DGGML_CUDA=ON \
        -DGGML_CUDA_F16=ON \
        -DCMAKE_CUDA_ARCHITECTURES=90a \
        -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=ON \
        -DLLAMA_CURL=ON \
        2>&1 | tail -5

    info "  Building with $(nproc) cores (this takes ~5-10 min)..."
    cmake --build "${LLAMA_SRC}/build" --config Release \
          --target llama-server -j "$(nproc)" 2>&1 | tail -10

    [[ -f "${LLAMA_BIN}" ]] || die "Build failed: ${LLAMA_BIN} not found."

    cp "${LLAMA_BIN}" "${BIN_DIR}/llama-server"
    chmod +x "${BIN_DIR}/llama-server"
    success "llama-server built → ${BIN_DIR}/llama-server"
fi

# ── Step 3: Download GGUF weights ─────────────────────────────────────────────
if [[ "${SKIP_DOWNLOAD}" == true ]]; then
    warn "Step 3/5 — Skipping model download (--skip-download)."
else
    info "Step 3/5 — Downloading GGUF weights from HuggingFace..."
    mkdir -p "${MODEL_DIR}"

    export HF_TOKEN="${HF_TOKEN}"
    export PATH="${HOME}/.local/bin:${PATH}"

    # Install hf CLI if missing
    if ! command -v hf &>/dev/null; then
        pip install -q "huggingface_hub[hf_xet]" 2>&1 | tail -2
    fi

    dl_if_missing() {
        local FILE="$1"
        local DEST="${MODEL_DIR}/${FILE}"
        if [[ -f "${DEST}" ]] && [[ $(stat -c%s "${DEST}" 2>/dev/null || echo 0) -gt 1000000 ]]; then
            success "  ${FILE} already present — skipping."
            return
        fi
        info "  Downloading ${FILE}..."
        hf download "${GGUF_REPO}" "${FILE}" --local-dir "${MODEL_DIR}"
        success "  ${FILE} seated at ${DEST}"
    }

    dl_if_missing "${GGUF_FILE}"
    dl_if_missing "${MMPROJ_FILE}" || warn "mmproj download failed — text inference still works."

    success "Model weights ready."
fi

# ── Step 4: Python venv ───────────────────────────────────────────────────────
info "Step 4/5 — Python environment..."

if [[ ! -d "${VENV_DIR}" ]]; then
    python3 -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip -q

if [[ -f "${REPO_DIR}/pyproject.toml" ]]; then
    pip install -e "${REPO_DIR}" -q 2>&1 | tail -3
    success "Project installed into venv."
else
    warn "pyproject.toml not found — skipping pip install."
fi

# Persist env for future sessions
PROFILE="${WORKSPACE}/.container_env"
cat > "${PROFILE}" << EOF
export HF_TOKEN="${HF_TOKEN}"
export PYTHONPATH="${REPO_DIR}/src"
export OPENAI_BASE_URL="http://localhost:8000/v1"
export VIRTUAL_ENV="${VENV_DIR}"
export PATH="${VENV_DIR}/bin:\${PATH}"
EOF
grep -qxF "source ${PROFILE}" ~/.bashrc 2>/dev/null || echo "source ${PROFILE}" >> ~/.bashrc
success "Env persisted to ${PROFILE}"

# ── Step 5: Ignite ────────────────────────────────────────────────────────────
info "Step 5/5 — Igniting inference server..."
bash "${SCRIPTS_DIR}/ignite.sh"
