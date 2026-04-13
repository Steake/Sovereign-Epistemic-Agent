#!/usr/bin/env bash
# =============================================================================
# restore_pod.sh — Sovereign Epistemic Agent: Full Cold-Start Pod Restore
# =============================================================================
# Idempotent: safe to re-run on a pod that already has some steps done.
#
# Usage:
#   HF_TOKEN=hf_xxx bash /workspace/scripts/restore_pod.sh
#   HF_TOKEN=hf_xxx bash /workspace/scripts/restore_pod.sh --skip-download
#   HF_TOKEN=hf_xxx bash /workspace/scripts/restore_pod.sh --dataset-dir /workspace/arc_dataset
#
# Sequence:
#   1. Install system deps    (cmake, ninja, curl, aria2, git)
#   2. Build llama-server     (llama.cpp, CUDA arch 90a for H100)
#   3. Download GGUF weights  (Qwen3.5-27B Q8_0 + mmproj)
#   4. Set up Python venv     (uv or venv + project install)
#   5. Ignite                 (llama-server on :8000, health-check waits 180s)
#   6. Seat ARC dataset       (validates ARC_DATASET_PATH or prints upload cmd)
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
DATASET_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-download) SKIP_DOWNLOAD=true; shift ;;
    --token)         HF_TOKEN="$2"; shift 2 ;;
    --dataset-dir)   DATASET_DIR="$2"; shift 2 ;;
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
info "Step 4/6 — Python environment..."

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

# Persist env for future sessions (ARC_DATASET_PATH will be patched after dataset seating)
PROFILE="${WORKSPACE}/.container_env"
cat > "${PROFILE}" <<'ENVEOF'
export HF_TOKEN="__HF_TOKEN__"
export PYTHONPATH="__REPO_DIR__/src"
export OPENAI_BASE_URL="http://localhost:8000/v1"
export VIRTUAL_ENV="__VENV_DIR__"
export PATH="__VENV_DIR__/bin:${PATH}"
export ARC_DATASET_PATH="/workspace/arc_dataset"
ENVEOF
# Substitute the real paths (avoid heredoc variable expansion issues)
sed -i \
    -e "s|__HF_TOKEN__|${HF_TOKEN}|g" \
    -e "s|__REPO_DIR__|${REPO_DIR}|g" \
    -e "s|__VENV_DIR__|${VENV_DIR}|g" \
    "${PROFILE}"
grep -qxF "source ${PROFILE}" ~/.bashrc 2>/dev/null || echo "source ${PROFILE}" >> ~/.bashrc
success "Env persisted to ${PROFILE}"

# ── Step 5: Ignite ────────────────────────────────────────────────────────────
info "Step 5/6 — Igniting inference server..."
bash "${SCRIPTS_DIR}/ignite.sh"

# ── Step 6: Seat ARC benchmark dataset ───────────────────────────────────────
info "Step 6/6 — Validating ARC benchmark dataset..."

ARC_DEFAULT_DIR="/workspace/arc_dataset"
ARC_DATASET_RESOLVED=""

# Prefer explicit --dataset-dir arg
if [[ -n "${DATASET_DIR}" ]]; then
    if [[ -d "${DATASET_DIR}" ]] && ls "${DATASET_DIR}"/*.json &>/dev/null; then
        ARC_DATASET_RESOLVED="${DATASET_DIR}"
        success "ARC dataset found at ${DATASET_DIR} ($(ls "${DATASET_DIR}"/*.json | wc -l) tasks)."
    else
        die "--dataset-dir '${DATASET_DIR}' is set but contains no *.json files."
    fi
elif [[ -d "${ARC_DEFAULT_DIR}" ]] && ls "${ARC_DEFAULT_DIR}"/*.json &>/dev/null; then
    ARC_DATASET_RESOLVED="${ARC_DEFAULT_DIR}"
    success "ARC dataset found at ${ARC_DEFAULT_DIR} ($(ls "${ARC_DEFAULT_DIR}"/*.json | wc -l) tasks)."
else
    warn "ARC dataset NOT found. Upload it before running the sweep:"
    warn ""
    warn "  From your Mac:"
    warn "    scp -r '/Users/oli/Documents/Oli Works/arc_epistemic/Kaggle_Submission_Bundle/dataset/' \\"
    warn "          root@<POD_IP>:/workspace/arc_dataset/"
    warn ""
    warn "  Then set ARC_DATASET_PATH=/workspace/arc_dataset in your shell, or re-run:"
    warn "    bash ${SCRIPTS_DIR}/restore_pod.sh --dataset-dir /workspace/arc_dataset"
    ARC_DATASET_RESOLVED="${ARC_DEFAULT_DIR}"   # write placeholder so env file is still valid
fi

# Patch env file with resolved dataset path
sed -i "s|^export ARC_DATASET_PATH=.*|export ARC_DATASET_PATH=\"${ARC_DATASET_RESOLVED}\"|" "${PROFILE}" 2>/dev/null || true

echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}  🟢  Pod Restore Complete!${RESET}"
echo -e "${BOLD}═══════════════════════════════════════════════════════════${RESET}"
echo ""
echo "  Inference API : http://localhost:8000/v1"
echo "  ARC Dataset   : ${ARC_DATASET_RESOLVED}"
echo ""
echo "  Run sweep:"
echo "    source ${PROFILE}"
echo "    cd ${REPO_DIR}"
echo "    python3 scripts/full_validation.py"
echo ""
