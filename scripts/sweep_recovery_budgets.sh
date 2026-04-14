#!/bin/bash
set -a
source /workspace/.container_env
set +a
cd /workspace/Sovereign-Epistemic-Agent

for BUDGET in 0 512 1024 2048; do
    echo "========================================="
    echo "Initiating sweep for reasoning budget: $BUDGET"
    echo "========================================="
    
    bash scripts/ignite_recovery.sh $BUDGET
    sleep 5
    
    python3 scripts/full_validation.py
    
    cp data/validation_results_v1.json data/smoke_sweep_budget_${BUDGET}.json
    
    echo "Budget $BUDGET Completed! Accuracies:"
    cat data/smoke_sweep_budget_${BUDGET}.json | grep 'accuracy' || true
done
echo "All recovery loops executed successfully."
