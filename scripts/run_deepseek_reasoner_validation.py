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
    console.print("[bold cyan]Starting DeepSeek-Reasoner (R1) Validation Experiment...[/]")
    
    repo_root = Path(__file__).parent.parent
    manifest_path = repo_root / "data" / "validation_manifest_v1.txt"
    dataset_path = Path("/Users/oli/Documents/Oli Works/arc_epistemic/Kaggle_Submission_Bundle/dataset")
    
    ledger_file = repo_root / "data" / "deepseek_r1_validation_ledger.jsonl"
    
    # Ensure API Key is available
    if not os.environ.get("DEEPSEEK_API_KEY"):
        console.print("[red]DEEPSEEK_API_KEY is not set in environment![/]")
        return

    # 1. Parse manifest
    if not manifest_path.exists():
        console.print(f"[red]Manifest not found at {manifest_path}[/red]")
        return
        
    with open(manifest_path) as f:
        task_ids = [line.strip() for line in f if line.strip()]
        
    task_files = []
    for tid in task_ids:
        tf = dataset_path / f"{tid}.json"
        if tf.exists():
            task_files.append(tf)
        else:
            console.print(f"[red]Missing dataset file for {tid}[/red]")
            
    if not task_files:
        console.print("[red]No tasks found to process![/red]")
        return

    # 2. Setup Configs
    # GREEDY ARM (Raw R1)
    greedy_config = TribunalSettings()
    greedy_config.generators.enabled = ["llm"]
    greedy_config.generators.llm.model_name = "deepseek-reasoner"
    greedy_config.generators.llm.temperature = 0.0
    greedy_config.generators.llm.max_new_tokens = 16384
    greedy_config.generators.llm.use_json_schema = False
    greedy_config.generators.llm.api_key = os.environ.get("DEEPSEEK_API_KEY")
    greedy_config.tribunal.adjudication_strategy = "standard"
    
    # TRIBUNAL ARM (DeepSeek-Reasoner + Greedy + Diverse)
    tribunal_config_path = repo_root / "configs" / "tribunal_full_experiment.yaml"
    tribunal_config = load_config(str(tribunal_config_path))
    # Override model to reasoner
    tribunal_config.generators.llm.model_name = "deepseek-reasoner"
    tribunal_config.generators.llm.max_new_tokens = 16384
    tribunal_config.generators.llm.use_json_schema = False
    tribunal_config.generators.llm.api_key = os.environ.get("DEEPSEEK_API_KEY")
    
    store = LedgerStore(":memory:")
    greedy_orch = Orchestrator(greedy_config, store)
    tribunal_orch = Orchestrator(tribunal_config, store)
    
    runs = []
    
    def progress_callback(kind, token):
        if kind == "content":
            console.print(".", end="")
        elif kind == "reasoning":
            console.print("~", end="")

    for i, tf in enumerate(task_files, 1):
        console.print(f"\n[bold yellow][{i}/{len(task_files)}] Processing {tf.name}...[/]")
        task = load_task_from_file(tf)
        
        # Run Greedy (Raw R1)
        console.print("   [dim]Running Greedy (Raw R1)...[/]")
        start_t = time.time()
        g_run = greedy_orch.run(task, on_token=progress_callback)
        g_run.metadata["arm_name"] = "greedy_r1"
        runs.append(g_run)
        
        # Run Tribunal (R1 + Pool)
        console.print("\n   [dim]Running Tribunal (R1 + Pool)...[/]")
        t_run = tribunal_orch.run(task, on_token=progress_callback)
        t_run.metadata["arm_name"] = "tribunal_r1"
        runs.append(t_run)
        elapsed = time.time() - start_t
        
        # Output summary of this task
        tribunal_decision = t_run.decision.value if t_run.decision else "None"
        g_ans = g_run.ground_truth_match
        t_ans = t_run.ground_truth_match
        console.print(f"\n   [bold blue]R1 Match: {g_ans}[/] | [bold magenta]Tribunal Decision: {tribunal_decision} (Match: {t_ans})[/] | [dim]Elapsed: {elapsed:.1f}s[/]")
        
        # Save checkpoints frequently
        with open(ledger_file, "a") as f:
            f.write(g_run.model_dump_json() + "\n")
            f.write(t_run.model_dump_json() + "\n")
            
    console.print(f"\n[bold green]Experiment complete! Saved {len(runs)} runs to {ledger_file}[/]")
    console.print(f"Run: [cyan]tribunal benchmark-usefulness --runs {ledger_file} --annotations data/validation_annotations.json[/]")

if __name__ == "__main__":
    main()
