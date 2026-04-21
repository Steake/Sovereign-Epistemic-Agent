#!/usr/bin/env python3
import re
import csv
from pathlib import Path

def main():
    log_path = Path("data/sweep_run2.log")
    if not log_path.exists():
        print(f"Log not found at {log_path}")
        return

    # Data structure: task_id -> { "finish_reason": set(), "parse_success": bool, "shape_match": bool, ... }
    # Since the generator result is the same for the task across arms, we just track the max/latest status for the task.
    
    tasks = {}
    current_task = None
    pending_task = False
    
    with log_path.open("r") as f:
        for line in f:
            # Handle Rich wrapped logging for task IDs
            if "Running tribunal on task" in line:
                m_task = re.search(r"Running tribunal on task ([a-zA-Z0-9_]+)", line)
                if m_task:
                    candidate = m_task.group(1)
                    if candidate != "task":
                        current_task = candidate
                else:
                    # Expecting it on the next line
                    pending_task = True
                continue
            
            if pending_task:
                m_task = re.search(r"([a-zA-Z0-9_]+)", line.strip())
                if m_task:
                    current_task = m_task.group(1)
                    pending_task = False
            
            if current_task and current_task not in tasks:
                tasks[current_task] = {
                    "family": current_task.split('_')[1] if len(current_task.split('_')) > 1 else "unknown",
                    "finish_reason": "unknown",
                    "parse_failure": False,
                    "shape_mismatch": False,
                    "decisions": [],
                    "confidences": [],
                    "ground_truth_matches": 0,
                    "zero_char": False
                }
            
            if not current_task:
                continue
                
            # Finish reason
            m_finish = re.search(r"finish=(stop|length)", line)
            if m_finish:
                tasks[current_task]["finish_reason"] = m_finish.group(1)
                
            # Zero char
            if "Length: 0 chars" in line or "Length: 0 characters" in line:
                tasks[current_task]["zero_char"] = True
                
            # Parse failures
            if "response contained no JSON object" in line or "LLM response did not" in line:
                # "LLM response did not contain a valid 6x5 grid" happens during validation, but usually we see parse failures first.
                if "no JSON object" in line:
                    tasks[current_task]["parse_failure"] = True
                    
            # Shape mismatches
            if "shape mismatch" in line or "did not contain a valid" in line:
                # Sometimes shape mismatch logs as "Rejected LLM answer due to shape mismatch"
                tasks[current_task]["shape_mismatch"] = True
                
            # Decisions
            m_decision = re.search(r"Decision:\s*([a-zA-Z]+)\s*\(confidence=([0-9.]+)\)", line)
            if m_decision:
                tasks[current_task]["decisions"].append(m_decision.group(1))
                tasks[current_task]["confidences"].append(float(m_decision.group(2)))
                
            # Ground truth
            if "Ground-truth match: True" in line:
                tasks[current_task]["ground_truth_matches"] += 1

    # Output to CSV
    out_path = Path("data/forensic_summary_run2.csv")
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "task_id", "family", "finish_reason", "zero_char", 
            "parse_failure", "shape_mismatch", "primary_decision", 
            "avg_confidence", "gt_matches"
        ])
        
        for t_id, data in tasks.items():
            primary_decision = data["decisions"][0] if data["decisions"] else "none"
            avg_conf = sum(data["confidences"]) / len(data["confidences"]) if data["confidences"] else 0.0
            
            writer.writerow([
                t_id,
                data["family"],
                data["finish_reason"],
                data["zero_char"],
                data["parse_failure"],
                data["shape_mismatch"],
                primary_decision,
                f"{avg_conf:.3f}",
                data["ground_truth_matches"]
            ])
            
    print(f"Forensic summary written to {out_path} for {len(tasks)} tasks.")
    
    # Print high level stats
    truncations = sum(1 for t in tasks.values() if t["finish_reason"] == "length")
    parse_fails = sum(1 for t in tasks.values() if t["parse_failure"])
    shape_fails = sum(1 for t in tasks.values() if t["shape_mismatch"])
    zeros = sum(1 for t in tasks.values() if t["zero_char"])
    print(f"\nTotal tasks: {len(tasks)}")
    print(f"Truncations (finish=length): {truncations}")
    print(f"Zero char returns: {zeros}")
    print(f"Parse failures: {parse_fails}")
    print(f"Shape mismatches: {shape_fails}")

if __name__ == "__main__":
    main()
