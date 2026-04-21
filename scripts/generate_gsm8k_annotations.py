#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate synthetic annotations for GSM8K test set to bootstrap the tribunal benchmark.
Assigns cohorts (control-trivial, contested-recoverable, contested-unrecoverable) 
deterministically so the tribunal has a mix of task difficulties to evaluate.
"""

import json
import hashlib
from pathlib import Path

def get_cohort(question: str) -> str:
    # Use deterministic hashing to assign cohorts consistently
    h = int(hashlib.md5(question.encode('utf-8')).hexdigest(), 16)
    val = h % 100
    
    # Let's say:
    # 0-39: control-trivial
    # 40-79: contested-recoverable
    # 80-99: contested-unrecoverable
    if val < 40:
        return "control-trivial"
    elif val < 80:
        return "contested-recoverable"
    else:
        return "contested-unrecoverable"

def main():
    test_path = Path("data/gsm8k/test.jsonl")
    out_path = Path("data/gsm8k/annotated.jsonl")
    
    if not test_path.exists():
        print(f"Error: {test_path} not found. Please download the GSM8K test set first.")
        return
        
    counts = {"control-trivial": 0, "contested-recoverable": 0, "contested-unrecoverable": 0}
    
    with open(test_path, "r") as f_in, open(out_path, "w") as f_out:
        for i, line in enumerate(f_in):
            line = line.strip()
            if not line:
                continue
                
            data = json.loads(line)
            question = data.get("question", "")
            
            # Keep original ID or generate one
            task_id = f"gsm8k_{i:04d}"
            data["task_id"] = task_id
            
            cohort = get_cohort(question)
            data["cohort"] = cohort
            
            # Synthetic indices for completeness
            data["contestability_index"] = 0.8 if "contested" in cohort else 0.2
            data["recoverability_index"] = 0.9 if cohort == "contested-recoverable" else 0.1
            data["structural_separability"] = 0.5
            data["plausible_hypotheses"] = 2 if "contested" in cohort else 1
            data["recoverability_status"] = "recoverable" if cohort == "contested-recoverable" else "unrecoverable"
            
            counts[cohort] += 1
            f_out.write(json.dumps(data) + "\n")
            
    print(f"Generated {sum(counts.values())} annotated tasks at {out_path}")
    for k, v in counts.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
