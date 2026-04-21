import json
from pathlib import Path
from rich.console import Console

from epistemic_tribunal.config import TribunalSettings
from epistemic_tribunal.ledger.store import LedgerStore
from epistemic_tribunal.orchestrator import Orchestrator
from epistemic_tribunal.tasks.arc_like import load_task_from_file

def main():
    console = Console()
    repo_root = Path(__file__).parent.parent
    db_path = repo_root / "data" / "deepseek_chat_25_ledger.db"
    dataset_path = Path("/Users/oli/Documents/Oli Works/arc_epistemic/Kaggle_Submission_Bundle/dataset")
    
    annotations_file = repo_root / "data" / "deepseek_annotations.json"
    ledger_file = repo_root / "data" / "deepseek_benchmark_ledger.jsonl"
    
    # 1. Extract DeepSeek Chat runs
    task_ids = set()
    deepseek_runs = []
    
    from epistemic_tribunal.cli import _load_runs_from_ledger
    db_runs = _load_runs_from_ledger(str(db_path))
    console.print(f"Loaded {len(db_runs)} runs from {db_path}")
    
    for r in db_runs:
        task_ids.add(r.task_id)
        # Rename arm for clarity if needed
        r.metadata["arm_name"] = "deepseek_chat"
        deepseek_runs.append(r)
        
    # 2. Build annotations
    annotations = []
    task_files = []
    for tid in task_ids:
        if tid.startswith("auto_control") or tid.startswith("control"):
            cohort = "control_trivial"
            ci, ri, ss = 0, 4, 4
            status = "exact_candidate_present"
        elif "diversity_sensitive" in tid or "selector_divergence" in tid:
            cohort = "contested_recoverable"
            ci, ri, ss = 2, 4, 2
            status = "exact_candidate_present"
        elif "refinement_composition" in tid:
            cohort = "contested_unrecoverable"
            ci, ri, ss = 4, 1, 1
            status = "no_viable_candidate_present"
        else:
            cohort = "contested_recoverable" # fallback
            ci, ri, ss = 2, 4, 2
            status = "exact_candidate_present"
            
        annotations.append({
            "task_id": tid,
            "cohort": cohort,
            "contestability_index": ci,
            "recoverability_index": ri,
            "structural_separability": ss,
            "plausible_hypotheses": [],
            "recoverability_status": status,
            "annotation_notes": "auto-generated"
        })
        
        tf = dataset_path / f"{tid}.json"
        if tf.exists():
            task_files.append(tf)
        else:
            console.print(f"[red]Missing dataset file for {tid}[/red]")
            
    with open(annotations_file, "w") as f:
        json.dump(annotations, f, indent=2)
        
    # 3. Generate Greedy Baseline runs for these exact tasks
    greedy_config = TribunalSettings()
    greedy_config.generators.enabled = ["greedy"]
    greedy_config.tribunal.adjudication_strategy = "greedy"
    
    mem_store = LedgerStore(":memory:")
    greedy_orch = Orchestrator(greedy_config, mem_store)
    
    greedy_runs = []
    for tf in task_files:
        task = load_task_from_file(tf)
        g_run = greedy_orch.run(task)
        g_run.metadata["arm_name"] = "greedy"
        greedy_runs.append(g_run)
        
    # 4. Save combined runs
    all_runs = deepseek_runs + greedy_runs
    with open(ledger_file, "w") as f:
        for r in all_runs:
            f.write(r.model_dump_json() + "\n")
            
    console.print(f"[bold green]Saved {len(all_runs)} combined runs to {ledger_file}[/]")

if __name__ == "__main__":
    main()
