#!/usr/bin/env bash
# Run 25-task ARC benchmark against DeepSeek Reasoner API on custom ARC dataset
set -euo pipefail

export OPENAI_BASE_URL="https://api.deepseek.com"
export OPENAI_API_KEY="sk-08e83c1210624090a59fee70318f2d95"
export ARC_DATASET_PATH="/Users/oli/Documents/Oli Works/arc_epistemic/Kaggle_Submission_Bundle/dataset"
export PYTHONPATH="$(pwd)/src"

echo "=== DeepSeek 25-Task Benchmark (Custom Dataset) ==="
echo "API: $OPENAI_BASE_URL"
echo "Dataset: $ARC_DATASET_PATH"
echo "Config: configs/deepseek_custom_25.yaml"
echo "Manifest: data/deepseek_custom_25_manifest.txt"
echo ""

# Run interactive live benchmark
python3 scripts/live_benchmark.py
