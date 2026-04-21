import json
import sqlite3
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from epistemic_tribunal.tasks.gsm8k import load_tasks_from_jsonl
from epistemic_tribunal.domains.factory import get_adapter

def audit_v4():
    db_path = "data/gsm8k/results/v4/experiment/ledger_full26.db"
    gsm8k_path = "data/gsm8k/test.jsonl"
    
    # Load tasks for ground truth
    all_tasks = {t.task_id: t for t in load_tasks_from_jsonl(gsm8k_path)}
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    # Get all decisions from Pass 2 (the last 26)
    rows = conn.execute("SELECT * FROM decisions ORDER BY created_at DESC LIMIT 26").fetchall()
    rows = rows[::-1] # back to chronological
    
    wrong_picks = []
    bad_abstentions = []
    
    for row in rows:
        task_id = row["task_id"]
        decision = row["decision"]
        
        task = all_tasks.get(task_id)
        if not task: continue
        
        gt = task.ground_truth
        adapter = get_adapter(task.domain)
        
        # Check pool for any correct
        trace_rows = conn.execute("SELECT answer_json FROM traces WHERE task_id = ?", (task_id,)).fetchall()
        any_correct = any(adapter.answers_equal(json.loads(t["answer_json"]), gt) for t in trace_rows)
        
        if decision == "select":
            json.loads(row["scores_json"]).get("selected_answer")
            # Wait, scores_json might not have it. selected_trace_id?
            sel_trace_id = row["selected_trace_id"]
            if sel_trace_id:
                sel_row = conn.execute("SELECT answer_json FROM traces WHERE trace_id = ?", (sel_trace_id,)).fetchone()
                sel_ans = json.loads(sel_row["answer_json"])
                if not adapter.answers_equal(sel_ans, gt):
                    wrong_picks.append(task_id)
            else:
                # Should not happen for select
                pass
        elif decision == "abstain":
            if any_correct:
                bad_abstentions.append(task_id)
                
    print(f"WRONG PICKS ({len(wrong_picks)}): {wrong_picks}")
    print(f"BAD ABSTENTIONS ({len(bad_abstentions)}): {bad_abstentions}")
    
    conn.close()

if __name__ == "__main__":
    audit_v4()
