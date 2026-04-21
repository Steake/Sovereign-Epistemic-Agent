import sqlite3
import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List

@dataclass
class CandidateInfo:
    trace_id: str
    overlap_pct: float
    is_tribunal_choice: bool
    answer: List[List[int]]

def calculate_overlap(grid1: List[List[int]], grid2: List[List[int]]) -> float:
    if not grid1 or not grid2: return 0.0
    h1, w1 = len(grid1), len(grid1[0])
    h2, w2 = len(grid2), len(grid2[0])
    
    if h1 != h2 or w1 != w2:
        return 0.0
        
    total_cells = h1 * w1
    matches = sum(1 for r in range(h1) for c in range(w1) if grid1[r][c] == grid2[r][c])
    return matches / total_cells

def get_ground_truth(task_id: str, dataset_dir: Path) -> List[List[int]]:
    task_file = dataset_dir / f"{task_id}.json"
    if not task_file.exists():
        # Try recursive search if not in root
        matches = list(dataset_dir.rglob(f"*{task_id}*.json"))
        if matches: task_file = matches[0]
        else: return []
        
    with open(task_file, 'r') as f:
        data = json.load(f)
        
    # Standard ARC
    if 'test' in data and len(data['test']) > 0 and 'output' in data['test'][0]:
        return data['test'][0]['output']
    # Custom format seen in some pods
    if 'ground_truth' in data:
        return data['ground_truth']
    # 'Golden' format (Scale-up pods)
    if 'golden' in data and 'expected_test_outputs' in data['golden']:
        outputs = data['golden']['expected_test_outputs']
        if outputs and len(outputs) > 0:
            return outputs[0]
            
    return []

def run_extraction():
    db_path = "/workspace/Sovereign-Epistemic-Agent/configs/validation/data/arm4_path_b.db"
    dataset_dir = Path("/workspace/arc_dataset")
    
    tasks_to_analyze = {
        "auto_control_0a7d41f9": "Identity",
        "auto_control_10a8c371": "Flip",
        "auto_control_22ae2e50": "Fill",
        "auto_control_2387060f": "Color Swap",
        "auto_selector_divergence_2ef7ff5d": "Messy"
    }
    
    if not os.path.exists(db_path):
        print(f"ERROR: DB not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    final_report = {}
    
    for task_id, task_type in tasks_to_analyze.items():
        print(f"Analyzing {task_id} ({task_type})...")
        
        ground_truth = get_ground_truth(task_id, dataset_dir)
        if not ground_truth:
            print(f"  WARNING: Ground truth not found for {task_id}")
            continue
            
        # Get tribunal choice
        cursor.execute("SELECT selected_trace_id FROM decisions WHERE task_id = ?", (task_id,))
        choice_row = cursor.fetchone()
        tribunal_choice_id = choice_row[0] if choice_row else None
        
        # Get all candidates
        cursor.execute("SELECT trace_id, answer_json FROM traces WHERE task_id = ?", (task_id,))
        rows = cursor.fetchall()
        
        candidates = []
        for tid, ans_json in rows:
            try:
                ans = json.loads(ans_json)
                # The answer field might be nested or the result itself
                if isinstance(ans, dict) and "answer" in ans:
                    ans = ans["answer"]
                
                overlap = calculate_overlap(ans, ground_truth)
                candidates.append({
                    "trace_id": tid,
                    "overlap_pct": overlap,
                    "is_tribunal_choice": (tid == tribunal_choice_id),
                    "answer_preview": ans[:2] # truncated for size
                })
            except Exception as e:
                print(f"  Error parsing trace {tid}: {e}")
        
        # Sort by overlap
        candidates.sort(key=lambda x: x["overlap_pct"], reverse=True)
        
        final_report[task_id] = {
            "type": task_type,
            "best_overlap": candidates[0]["overlap_pct"] if candidates else 0.0,
            "tribunal_overlap": next((c["overlap_pct"] for c in candidates if c["is_tribunal_choice"]), 0.0),
            "pool_size": len(candidates),
            "top_candidates": candidates[:5]
        }
    
    with open("forensic_raw_data.json", "w") as f:
        json.dump(final_report, f, indent=2)
    
    print("DONE. Report saved to forensic_raw_data.json")

if __name__ == "__main__":
    run_extraction()
