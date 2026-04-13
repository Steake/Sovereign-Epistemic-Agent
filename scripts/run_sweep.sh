#!/usr/bin/env bash
# =============================================================================
# run_sweep.sh — Launch the 50-Task Four-Arm Validation Sweep
# =============================================================================
# Usage: bash /workspace/Sovereign-Epistemic-Agent/scripts/run_sweep.sh
#
# Prerequisites:
#   - ignite.sh has been run and server is healthy on :8000
#   - .venv is seated
#
# What this does:
#   1. Rebuilds the manifest from actual task files (avoids ID mismatch)
#   2. Creates ledger dirs
#   3. Launches full_validation.py in background with nohup
#   4. Tails the live log
# =============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$REPO_DIR/.venv"
TASK_DIR="$REPO_DIR/data/validation_set/tasks"
MANIFEST="$REPO_DIR/data/validation_manifest_v1.txt"
LOG="$REPO_DIR/data/validation_sweep.log"

cd "$REPO_DIR"

# ── Activate env ──────────────────────────────────────────────────────────────
[ -d "$VENV_DIR" ] || { echo "❌ venv not found. Run restore_pod.sh first."; exit 1; }
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$REPO_DIR/src"
export OPENAI_BASE_URL="http://localhost:8000/v1"

# ── Server health check ───────────────────────────────────────────────────────
if ! curl -sf http://localhost:8000/health >/dev/null 2>&1; then
    echo "❌ Inference server not responding. Run ignite.sh first."
    exit 1
fi
MODEL_ID=$(curl -s http://localhost:8000/v1/models | python3 -c \
    "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "unknown")
echo "✅ Server healthy. Model: $MODEL_ID"

# ── Rebuild manifest from actual task files ────────────────────────────────────
[ -d "$TASK_DIR" ] || { echo "❌ Task dir not found: $TASK_DIR"; exit 1; }
TASK_COUNT=$(ls "$TASK_DIR"/*.json 2>/dev/null | wc -l)
[ "$TASK_COUNT" -gt 0 ] || { echo "❌ No .json tasks in $TASK_DIR"; exit 1; }

echo "📋 Rebuilding manifest from $TASK_COUNT tasks in $TASK_DIR..."
ls "$TASK_DIR"/*.json | xargs -I{} basename {} .json | head -50 > "$MANIFEST"
MANIFEST_COUNT=$(wc -l < "$MANIFEST")
echo "   Manifest: $MANIFEST_COUNT tasks"
echo "   First 3: $(head -3 "$MANIFEST" | tr '\n' ' ')"

# ── Fresh ledger dirs ─────────────────────────────────────────────────────────
mkdir -p "$REPO_DIR/configs/validation/data"
rm -f "$REPO_DIR/configs/validation/data/"*.db 2>/dev/null || true
echo "🗑️  Ledgers cleared."

# ── Kill any prior sweep ──────────────────────────────────────────────────────
pkill -f full_validation.py 2>/dev/null && echo "   Killed prior sweep." || true
sleep 1

# ── Launch sweep ──────────────────────────────────────────────────────────────
echo ""
echo "🚀 Launching 50-Task Four-Arm Validation Sweep..."
nohup python3 scripts/full_validation.py > "$LOG" 2>&1 &
SWEEP_PID=$!
echo "   Sweep PID: $SWEEP_PID"
echo "   Logging to: $LOG"
echo ""
echo "📡 Tailing live log (Ctrl+C to detach — sweep keeps running)..."
echo "────────────────────────────────────────────────────────────"
tail -f "$LOG"
