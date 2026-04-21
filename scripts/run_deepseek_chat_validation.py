import os
import json
import time
from pathlib import Path
from rich.console import Console

from dotenv import load_dotenv
load_dotenv()

from epistemic_tribunal.config import load_config, TribunalSettings
from epistemic_tribunal.ledger.store import LedgerStore
from epistemic_tribunal.orchestrator import Orchestrator
from epistemic_tribunal.tasks.arc_like import load_task_from_file

def main():
    console = Console()
    console.print("[bold cyan]Starting DeepSeek-Chat Validation Experiment...[/]")
    
    repo_root = Path(__file__).parent.parent
    manifest_path = repo_root / "data" / "validation_manifest_v1.txt"
    dataset_path = Path("/Users/oli/Documents/Oli Works/arc_epistemic/Kaggle_Submission_Bundle/dataset")
    
    ledger_file = repo_root / "data" / "deepseek_chat_validation_ledger.jsonl"
    
    # Ensure API Key is available
    if not os.environ.get("DEEPSEEK_API_KEY"):
        console.print("[red]DEEPSEEK_API_KEY is not set in environment![/]")
        return

    # 1. Parse manifest
    with open(manifest_path) as f:
        task_ids = [line.strip() for line in f if line.strip()]
        
    task_files = []
    for tid in task_ids:
        tf = dataset_path / f"{tid}.json"
        if tf.exists():
            task_files.append(tf)
        else:
            console.print(f"[red]Missing dataset file for {tid}[/red]")
            
    # 2. Setup Configs
    # GREEDY ARM
    greedy_config = TribunalSettings()
    greedy_config.generators.enabled = ["greedy"]
    greedy_config.tribunal.adjudication_strategy = "greedy"
    
    # TRIBUNAL ARM (DeepSeek-Chat + Greedy + Diverse)
    tribunal_config_path = repo_root / "configs" / "tribunal_full_experiment.yaml"
    tribunal_config = load_config(str(tribunal_config_path))
    
    store = LedgerStore(":memory:")
    greedy_orch = Orchestrator(greedy_config, store)
    tribunal_orch = Orchestrator(tribunal_config, store)
    
    runs = []
    
    for i, tf in enumerate(task_files, 1):
        console.print(f"\n[bold yellow][{i}/{len(task_files)}] Processing {tf.name}...[/]")
        task = load_task_from_file(tf)
        
        # Run Greedy
        start_t = time.time()
        g_run = greedy_orch.run(task)
        g_run.metadata["arm_name"] = "greedy"
        runs.append(g_run)
        
        # Run Tribunal
        t_run = tribunal_orch.run(task)
        t_run.metadata["arm_name"] = "tribunal"
        runs.append(t_run)
        elapsed = time.time() - start_t
        
        # Output summary of this task
        tribunal_decision = t_run.decision.value if t_run.decision else "None"
        g_ans = g_run.ground_truth_match
        t_ans = t_run.ground_truth_match
        console.print(f"   [dim]Greedy Match: {g_ans} | Tribunal Decision: {tribunal_decision} (Match: {t_ans}) | Elapsed: {elapsed:.1f}s[/]")
        
    # 3. Save runs
    with open(ledger_file, "w") as f:
        for r in runs:
            f.write(r.model_dump_json() + "\n")
            
    console.print(f"\n[bold green]Experiment complete! Saved {len(runs)} runs to {ledger_file}[/]")
    console.print("Run: [cyan]tribunal benchmark-usefulness --runs data/deepseek_chat_validation_ledger.jsonl --annotations data/validation_annotations.json[/]")

if __name__ == "__main__":
    main()
