import json
from pathlib import Path
from rich.console import Console

from epistemic_tribunal.config import TribunalSettings
from epistemic_tribunal.ledger.store import LedgerStore
from epistemic_tribunal.orchestrator import Orchestrator
from epistemic_tribunal.tasks.arc_like import load_task_from_file

def run_scientific_experiment():
    console = Console()
    console.print("[bold cyan]Starting Scientific Experiment...[/]")
    
    repo_root = Path(__file__).parent.parent
    manifest_path = repo_root / "data" / "validation_manifest_v1.txt"
    dataset_path = Path("/Users/oli/Documents/Oli Works/arc_epistemic/Kaggle_Submission_Bundle/dataset")
    
    annotations_file = repo_root / "data" / "validation_annotations.json"
    ledger_file = repo_root / "data" / "scientific_ledger.jsonl"
    
    # 1. Parse manifest and build annotations
    with open(manifest_path) as f:
        task_ids = [line.strip() for line in f if line.strip()]
        
    annotations = []
    task_files = []
    
    for tid in task_ids:
        # Determine cohort by prefix
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
            console.print(f"[bold red]Unknown prefix for {tid}[/]")
            continue
            
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
        
        task_file = dataset_path / f"{tid}.json"
        if task_file.exists():
            task_files.append(task_file)
        else:
            console.print(f"[bold red]Missing dataset file {task_file}[/]")
            
    with open(annotations_file, "w") as f:
        json.dump(annotations, f, indent=2)
        
    console.print(f"[green]Saved {len(annotations)} annotations to {annotations_file}[/]")
    
    # 2. Run the tasks
    greedy_config = TribunalSettings()
    greedy_config.generators.enabled = ["greedy"]
    greedy_config.tribunal.adjudication_strategy = "greedy"
    
    tribunal_config = TribunalSettings()
    tribunal_config.generators.enabled = ["greedy", "diverse", "adversarial", "rule_first", "minimal_description"]
    tribunal_config.tribunal.adjudication_strategy = "standard"
    
    store = LedgerStore(":memory:")
    greedy_orch = Orchestrator(greedy_config, store)
    tribunal_orch = Orchestrator(tribunal_config, store)
    
    runs = []
    
    for i, tf in enumerate(task_files, 1):
        console.print(f"[{i}/{len(task_files)}] Processing {tf.name}...")
        task = load_task_from_file(tf)
        
        # Greedy Arm
        g_run = greedy_orch.run(task)
        g_run.metadata["arm_name"] = "greedy"
        runs.append(g_run)
        
        # Tribunal Arm
        t_run = tribunal_orch.run(task)
        t_run.metadata["arm_name"] = "tribunal"
        runs.append(t_run)
        
    # 3. Save runs
    with open(ledger_file, "w") as f:
        for r in runs:
            f.write(r.model_dump_json() + "\n")
            
    console.print(f"[bold green]Saved {len(runs)} runs to {ledger_file}[/]")

if __name__ == "__main__":
    run_scientific_experiment()
