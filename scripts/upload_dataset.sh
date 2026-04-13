#!/usr/bin/env bash
# =============================================================================
# upload_dataset.sh — Upload ARC benchmark dataset from Mac to RunPod
# =============================================================================
# Usage:
#   bash scripts/upload_dataset.sh <POD_IP> [POD_PORT]
#
# Example:
#   bash scripts/upload_dataset.sh 69.87.123.45
#   bash scripts/upload_dataset.sh 69.87.123.45 22
#
# Requirements (Mac side):
#   - SSH key-based access to the pod configured (or password auth)
#   - The Kaggle bundle dataset present locally
#
# After upload, on the pod run:
#   bash /workspace/scripts/restore_pod.sh --dataset-dir /workspace/arc_dataset
# =============================================================================
set -euo pipefail

POD_IP="${1:-}"
POD_PORT="${2:-22}"
LOCAL_DATASET="/Users/oli/Documents/Oli Works/arc_epistemic/Kaggle_Submission_Bundle/dataset"
REMOTE_DIR="/workspace/arc_dataset"

if [[ -z "${POD_IP}" ]]; then
    echo "Usage: bash scripts/upload_dataset.sh <POD_IP> [POD_PORT]"
    echo ""
    echo "Get POD_IP from RunPod dashboard → Connect → SSH Command"
    exit 1
fi

if [[ ! -d "${LOCAL_DATASET}" ]]; then
    echo "ERROR: Local dataset not found at:"
    echo "  ${LOCAL_DATASET}"
    exit 1
fi

JSON_COUNT=$(ls "${LOCAL_DATASET}"/*.json 2>/dev/null | wc -l | tr -d ' ')
echo "Uploading ${JSON_COUNT} task JSON files → root@${POD_IP}:${REMOTE_DIR}"
echo ""

# Create remote dir and upload
ssh -p "${POD_PORT}" "root@${POD_IP}" "mkdir -p ${REMOTE_DIR}"
scp -P "${POD_PORT}" "${LOCAL_DATASET}"/*.json "root@${POD_IP}:${REMOTE_DIR}/"

echo ""
echo "✅ Dataset uploaded: ${JSON_COUNT} files → ${REMOTE_DIR}"
echo ""
echo "Now on the pod, run:"
echo "  bash /workspace/scripts/restore_pod.sh --dataset-dir ${REMOTE_DIR}"
echo "  # OR, if restore already ran:"
echo "  export ARC_DATASET_PATH=${REMOTE_DIR}"
echo "  cd /workspace/Sovereign-Epistemic-Agent"
echo "  python3 scripts/full_validation.py"
