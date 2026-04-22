#!/usr/bin/env bash
set -e

echo "=========================================="
echo "Strange Loop Memory Efficacy Experiment v1"
echo "=========================================="
echo ""
echo "Uses the custom 25-task contested benchmark designed for tribunal testing."
echo "LLM-only generators so memory injection drives outcomes."
echo ""

# The custom ARC dataset path (where the 25 contested tasks live)
TASK_DIR="/Users/oli/Documents/Oli Works/arc_epistemic/Kaggle_Submission_Bundle/dataset"
MANIFEST="data/deepseek_custom_25_manifest.txt"
CONFIG_PATH="configs/strange_loop_experiment.yaml"

echo "Task Directory: $TASK_DIR"
echo "Manifest: $MANIFEST ($(wc -l < "$MANIFEST" | tr -d ' ') tasks)"
echo "Config: $CONFIG_PATH"
echo ""

echo "1. Cleaning up previous experiment ledgers..."
rm -f data/sl_baseline.db data/sl_control_retry.db data/sl_bad_answers_only.db data/sl_warnings_only.db data/sl_full_memory.db

echo ""
echo "2. Running Baseline Cohort..."
tribunal benchmark "$TASK_DIR" \
    --ledger data/sl_baseline.db \
    --config "$CONFIG_PATH" \
    --manifest "$MANIFEST"

echo ""
echo "3. Running Strange Loop Replays on Failed Tasks..."

echo "--> Arm 1: Control Retry (No Memory)"
tribunal replay-failed "$TASK_DIR" \
    --from-ledger data/sl_baseline.db \
    --ledger data/sl_control_retry.db \
    --mode off \
    --config "$CONFIG_PATH"

echo "--> Arm 2: Bad Answers Only"
tribunal replay-failed "$TASK_DIR" \
    --from-ledger data/sl_baseline.db \
    --ledger data/sl_bad_answers_only.db \
    --mode bad_answers_only \
    --config "$CONFIG_PATH"

echo "--> Arm 3: Warnings Only"
tribunal replay-failed "$TASK_DIR" \
    --from-ledger data/sl_baseline.db \
    --ledger data/sl_warnings_only.db \
    --mode warnings_only \
    --config "$CONFIG_PATH"

echo "--> Arm 4: Full Strange Loop Memory"
tribunal replay-failed "$TASK_DIR" \
    --from-ledger data/sl_baseline.db \
    --ledger data/sl_full_memory.db \
    --mode full_memory \
    --config "$CONFIG_PATH"

echo ""
echo "4. Generating Final Efficacy Report..."
tribunal compare-strange-loop \
    --baseline data/sl_baseline.db \
    --control data/sl_control_retry.db \
    --bad-answers data/sl_bad_answers_only.db \
    --warnings data/sl_warnings_only.db \
    --full data/sl_full_memory.db

echo ""
echo "Experiment Complete!"
