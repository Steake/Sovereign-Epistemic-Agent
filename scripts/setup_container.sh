#!/usr/bin/env bash
# =============================================================================
# setup_container.sh — Sovereign Epistemic Agent Container Bootstrap
# =============================================================================
# Idempotent. Safe to run multiple times (skips steps already done).
# Usage:
#   HF_TOKEN=hf_xxx bash setup_container.sh
#   HF_TOKEN=hf_xxx bash setup_container.sh --skip-download
#   HF_TOKEN=hf_xxx bash setup_container.sh --download-only
#
# Requirements:
#   - /workspace mounted (RunPod network volume, ≥60GB free)
#   - HF_TOKEN env var set (or passed as --token hf_xxx)
#   - Internet access to huggingface.co
# =============================================================================
set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; RESET='\033[0m'
info()    { echo -e "${BLUE}[INFO]${RESET} $*"; }
success() { echo -e "${GREEN}[OK]${RESET}   $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET} $*"; }
die()     { echo -e "${RED}[FAIL]${RESET} $*" >&2; exit 1; }

# ── Argument Parsing ─────────────────────────────────────────────────────────
SKIP_DOWNLOAD=false
DOWNLOAD_ONLY=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-download)  SKIP_DOWNLOAD=true; shift ;;
    --download-only)  DOWNLOAD_ONLY=true; shift ;;
    --token)          HF_TOKEN="$2"; shift 2 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

# ── Config ────────────────────────────────────────────────────────────────────
WORKSPACE=/workspace
REPO_DIR="${WORKSPACE}/Sovereign-Epistemic-Agent"
VENV_DIR="${WORKSPACE}/venv"
HF_CACHE="${WORKSPACE}/.cache/huggingface"
MODEL_REPO="Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2"
MODEL_DIR="${HF_CACHE}/hub/models--$(echo "$MODEL_REPO" | tr '/' '--')"
SHARD_COUNT=11
LOG_DIR="${WORKSPACE}/logs"

# ── Validate prerequisites ────────────────────────────────────────────────────
[[ -d "${WORKSPACE}" ]] || die "/workspace is not mounted. Attach your RunPod network volume."
[[ -n "${HF_TOKEN:-}" ]]  || die "HF_TOKEN is not set. Pass it via env or --token."
mkdir -p "${LOG_DIR}"

echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}  Sovereign Epistemic Agent — Container Setup${RESET}"
echo -e "${BOLD}═══════════════════════════════════════════════════════════${RESET}"
echo ""

if [[ "$DOWNLOAD_ONLY" == false ]]; then

# ── Step 1: System dependencies ───────────────────────────────────────────────
info "Step 1/7 — Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq aria2 curl git build-essential 2>&1 | tail -n 3
success "System dependencies ready."

# ── Step 2: Node.js (NVM) — Rule 3 Compliance ────────────────────────────────
info "Step 2/7 — Installing NVM and Node.js v20..."
export NVM_DIR="/usr/local/nvm"
mkdir -p "$NVM_DIR"
if [[ ! -f "$NVM_DIR/nvm.sh" ]]; then
  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash >/dev/null
fi

# Load nvm and install node
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
nvm install 20 --silent
nvm use 20 --silent
success "Node.js $(node -v) ready."


# ── Step 3: Redirect pip/uv/HF caches to /workspace ──────────────────────────
info "Step 3/7 — Redirecting caches to /workspace..."

export HF_HOME="${HF_CACHE}"
export UV_CACHE_DIR="${WORKSPACE}/.cache/uv"
export PIP_CACHE_DIR="${WORKSPACE}/.cache/pip"
mkdir -p "${HF_CACHE}" "${UV_CACHE_DIR}" "${PIP_CACHE_DIR}"

# Persist environment across reboots
PROFILE_SNIPPET="${WORKSPACE}/.container_env"
cat > "${PROFILE_SNIPPET}" <<EOF
export HF_HOME="${HF_CACHE}"
export HF_TOKEN="${HF_TOKEN}"
export UV_CACHE_DIR="${WORKSPACE}/.cache/uv"
export PIP_CACHE_DIR="${WORKSPACE}/.cache/pip"
export PYTHONPATH="${REPO_DIR}/src"
export OPENAI_BASE_URL="http://localhost:8000/v1"
export NCCL_IGNORE_CPU_AFFINITY=1
export NVM_DIR="/usr/local/nvm"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PATH="${VENV_DIR}/bin:\$HOME/.local/bin:\$NVM_DIR/versions/node/v20.\$NODE_MINOR/bin:\$PATH"
export VIRTUAL_ENV="${VENV_DIR}"

# Load NVM if present
[ -s "\$NVM_DIR/nvm.sh" ] && \. "\$NVM_DIR/nvm.sh"
[ -s "\$NVM_DIR/bash_completion" ] && \. "\$NVM_DIR/bash_completion"
EOF


grep -qxF "source ${PROFILE_SNIPPET}" ~/.bashrc || echo "source ${PROFILE_SNIPPET}" >> ~/.bashrc
success "Cache dirs redirected. Env persisted to ${PROFILE_SNIPPET}"

# ── Step 4: Authenticate HuggingFace ─────────────────────────────────────────
info "Step 4/7 — Authenticating with HuggingFace..."

export HF_TOKEN="${HF_TOKEN}"
mkdir -p "${HF_CACHE}"
echo -n "${HF_TOKEN}" > "${HF_CACHE}/token"

# Install hf CLI if missing
if ! command -v hf &>/dev/null; then
  pip install -q "huggingface_hub[hf_xet]" 2>&1 | tail -n 2
fi
export PATH="$HOME/.local/bin:$PATH"
success "HuggingFace authenticated."

# ── Step 5: Python virtual environment ───────────────────────────────────────
info "Step 5/7 — Setting up Python environment with uv..."

if ! command -v uv &>/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi

# Force-path CUDA for H100/Hopper compilation
export PATH="/usr/local/cuda-12.4/bin:${PATH:-}"
export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_HOME="/usr/local/cuda-12.4"


# Check for Torch version mismatch before proceeding
if [[ -f "${VENV_DIR}/bin/python3" ]]; then

  CURRENT_TORCH=$("${VENV_DIR}/bin/python3" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "none")
  if [[ "$CURRENT_TORCH" != "2.5.1+cu124" ]]; then
    warn "Torch version mismatch ($CURRENT_TORCH vs 2.5.1+cu124). Nuking venv for ABI alignment..."
    rm -rf "${VENV_DIR}"
  fi
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  UV_CACHE_DIR="${UV_CACHE_DIR}" uv venv "${VENV_DIR}" --python 3.11
  success "Virtualenv created at ${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

if [[ -d "${REPO_DIR}" ]]; then
  info "Installing project dependencies from ${REPO_DIR}..."
  # Stage 1a: Clean-room Torch (forcing authoritative indexes)
  info "  Stage 1a: Isolating Torch (2.5.1+cu124) from platform mirrors..."
  UV_CACHE_DIR="${UV_CACHE_DIR}" uv pip install --force-reinstall --no-cache \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.1+cu124 \
    torchvision==0.20.1+cu124 \
    torchaudio==2.5.1+cu124 2>&1 | tail -n 3

  # Stage 1b: Build backends (authoritative index only)
  info "  Stage 1b: Installing build backends..."
  UV_CACHE_DIR="${UV_CACHE_DIR}" uv pip install --no-cache \
    --index-url https://pypi.org/simple \
    hatchling setuptools wheel cmake ninja editables packaging psutil 2>&1 | tail -n 3


  # Stage 2: Reasoning stack (authoritative isolation)
  info "  Stage 2: Installing reasoning stack (pypi.org baseline)..."
  UV_CACHE_DIR="${UV_CACHE_DIR}" uv pip install --no-cache \
    --index-url https://pypi.org/simple \
    vllm==0.6.4.post1 \
    flash-attn==2.6.3 \
    unsloth \
    einops \
    tiktoken \
    openai \
    hf_transfer \
    accelerate \
    -e "${REPO_DIR}" --no-build-isolation 2>&1 | tail -n 5

  success "Project installed. CLI: 'tribunal --help'"


else
  warn "Repo not found at ${REPO_DIR}. Skipping package install."
  info "  Clone with: git clone <your-repo-url> ${REPO_DIR}"
fi

# ── Step 6: Patch model config (architecture masquerade) ─────────────────────
info "Step 6/7 — Applying model config patches (if model dir exists)..."

python3 - <<'PYEOF'
import json, glob, os

model_dir = os.environ.get("MODEL_DIR", "")
if not os.path.isdir(model_dir):
    print(f"  Model dir not yet present ({model_dir}), skipping patches (will apply post-download).")
    exit(0)

# Patch config.json — Flatten text_config (Required for vLLM validation)
for path in glob.glob(f"{model_dir}/**/config.json", recursive=True) + [f"{model_dir}/config.json"]:
    if not os.path.isfile(path):
        continue
    with open(path) as f:
        data = json.load(f)
    if "text_config" in data:
        for k, v in data.pop("text_config").items():
            data.setdefault(k, v)
    
    # Fix vLLM Sliding Window Assertion (H100 Optimization)
    num_layers = data.get("num_hidden_layers", 64)
    data["max_window_layers"] = num_layers
    data["use_sliding_window"] = False

    if "num_attention_heads" not in data and "num_heads" in data:
        data["num_attention_heads"] = data["num_heads"]
    if "num_key_value_heads" not in data and "num_kv_heads" in data:
        data["num_key_value_heads"] = data["num_kv_heads"]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Patched config.json: {path} (Layers: {num_layers})")



# Patch tokenizer_config.json — Disable broken backend
for path in glob.glob(f"{model_dir}/**/tokenizer_config.json", recursive=True) + [f"{model_dir}/tokenizer_config.json"]:
    if not os.path.isfile(path):
        continue
    with open(path) as f:
        data = json.load(f)
    data["backend"] = None
    data["processor_class"] = None
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Patched tokenizer_config.json: {path}")
PYEOF
success "Config patches applied."

fi  # end if [[ "$DOWNLOAD_ONLY" == false ]]

# ── Step 7: Model download ────────────────────────────────────────────────────
if [[ "$SKIP_DOWNLOAD" == true ]]; then
  warn "Skipping model download (--skip-download passed)."
  exit 0
fi

info "Step 7/7 — Downloading model weights (${MODEL_REPO})..."

echo "   Target: ${MODEL_DIR}"

mkdir -p "${MODEL_DIR}"

# Build aria2 input file with all 11 shards
export HF_TOKEN="${HF_TOKEN}"
SHARD_LIST="${WORKSPACE}/shards.txt"
: > "${SHARD_LIST}"
for i in $(seq -f "%05g" 1 ${SHARD_COUNT}); do
  cat >> "${SHARD_LIST}" <<EOF
https://huggingface.co/${MODEL_REPO}/resolve/main/model.safetensors-${i}-of-$(printf "%05d" ${SHARD_COUNT}).safetensors
  header=Authorization: Bearer ${HF_TOKEN}
  out=model.safetensors-${i}-of-$(printf "%05d" ${SHARD_COUNT}).safetensors
EOF
done

# Check which shards are still missing
MISSING_SHARDS=()
for i in $(seq -f "%05g" 1 ${SHARD_COUNT}); do
  SHARD_FILE="${MODEL_DIR}/model.safetensors-${i}-of-$(printf "%05d" ${SHARD_COUNT}).safetensors"
  EXPECTED_SIZE=5368709120  # ~5GB; last shard ~2.1GB — aria2 will verify
  if [[ ! -f "${SHARD_FILE}" ]] || [[ -f "${SHARD_FILE}.aria2" ]]; then
    MISSING_SHARDS+=("$i")
  fi
done

if [[ ${#MISSING_SHARDS[@]} -eq 0 ]]; then
  success "All ${SHARD_COUNT} shards already downloaded."
else
  warn "${#MISSING_SHARDS[@]} shards missing or incomplete. Downloading..."

  # ── OOM-safe: 2 concurrent downloads, 8 connections each ──
  # This avoids the OOM killer on 2GB RAM RunPod CPU instances.
  # On GPU instances (≥80GB RAM) you can safely raise -j to 11.
  ARIA2_LOG="${LOG_DIR}/aria2_$(date +%Y%m%d_%H%M%S).log"
  aria2c \
    -i "${SHARD_LIST}" \
    -d "${MODEL_DIR}" \
    -j 2 \
    -x 8 -s 8 \
    --file-allocation=none \
    --auto-file-renaming=false \
    --allow-overwrite=false \
    --max-tries=5 \
    --retry-wait=3 \
    --log="${ARIA2_LOG}" \
    --log-level=notice \
    --summary-interval=10 \
    --console-log-level=notice

  success "All shards downloaded."
fi

# Also pull metadata/tokenizer files via hf CLI (small, fast)
info "Pulling official metadata + surrogate tokenizer files (7B Instruct)..."
# We pull the 7B-Instruct tokenizer as a surrogate because the 27B one is often broken/incomplete
hf download Qwen/Qwen2.5-7B-Instruct \
  --local-dir "${MODEL_DIR}" \
  --include "tokenizer.json" "tokenizer_config.json" "vocab.json" "merges.txt" \
  2>/dev/null || warn "Surrogate tokenizer pull had warnings."

hf download "${MODEL_REPO}" \
  --local-dir "${MODEL_DIR}" \
  --include "*.json" \
  --exclude "tokenizer*" \
  2>/dev/null || warn "hf CLI metadata pull had warnings (likely already present)."


# ── Post-download patches ──────────────────────────────────────────────────────
info "Applying post-download config patches..."
MODEL_DIR="${MODEL_DIR}" python3 - <<'PYEOF'
import json, glob, os

model_dir = os.environ["MODEL_DIR"]

for path in glob.glob(f"{model_dir}/**/config.json", recursive=True) + [f"{model_dir}/config.json"]:
    if not os.path.isfile(path):
        continue
    with open(path) as f:
        data = json.load(f)
    if "text_config" in data:
        for k, v in data.pop("text_config").items():
            data.setdefault(k, v)
    
    # Final architectural alignment for H100/vLLM
    num_layers = data.get("num_hidden_layers", 64)
    data["max_window_layers"] = num_layers
    data["use_sliding_window"] = False
    
    if "num_attention_heads" not in data and "num_heads" in data:
        data["num_attention_heads"] = data["num_heads"]
    if "num_key_value_heads" not in data and "num_kv_heads" in data:
        data["num_key_value_heads"] = data["num_kv_heads"]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Patched config: {path} (Layers: {num_layers})")


for path in glob.glob(f"{model_dir}/**/tokenizer_config.json", recursive=True) + [f"{model_dir}/tokenizer_config.json"]:
    if not os.path.isfile(path):
        continue
    with open(path) as f:
        data = json.load(f)
    data["backend"] = None
    data["processor_class"] = None
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Patched tokenizer: {path}")
PYEOF

# ── Final summary ─────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════════════${RESET}"
echo -e "${GREEN}${BOLD}  Setup Complete!${RESET}"
echo -e "${BOLD}═══════════════════════════════════════════════════════════${RESET}"
echo ""
echo -e "  Run: ${BOLD}source ${PROFILE_SNIPPET}${RESET}"
echo -e "  CLI: ${BOLD}tribunal --help${RESET}"
echo -e "  Model: ${BOLD}${MODEL_DIR}${RESET}"
du -sh "${MODEL_DIR}"
echo ""
