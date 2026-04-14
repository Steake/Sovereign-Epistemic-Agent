#!/bin/bash
set -a
source /workspace/.container_env
set +a
cd /workspace/Sovereign-Epistemic-Agent

for BUDGET in 0 256 512; do
    echo "========================================="
    echo "Initiating sweep for reasoning budget: $BUDGET"
    echo "========================================="
    
    bash scripts/ignite_recovery.sh $BUDGET
    sleep 5
    
    .venv/bin/python scripts/full_validation.py data/smoke_test_manifest.txt \
        > data/sweep_run_budget_${BUDGET}.log 2>&1
    
    cp data/validation_results_v1.json data/smoke_sweep_budget_${BUDGET}.json
    
    echo ""
    echo "===== Budget $BUDGET Scorecard ====="
    cat data/sweep_run_budget_${BUDGET}.log | grep -A 40 "Final Validation Scorecard" | head -45
done
echo "All budget sweeps complete."
