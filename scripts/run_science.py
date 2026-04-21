"""Script to generate test runs for the Tribunal Usefulness Benchmark.
This simulates two arms (greedy and tribunal) across the example tasks,
injecting the `arm_name` into the metadata, and saving to JSONL.
"""
from pathlib import Path
from rich.console import Console

from epistemic_tribunal.config import TribunalSettings
from epistemic_tribunal.ledger.store import LedgerStore
from epistemic_tribunal.orchestrator import Orchestrator
from epistemic_tribunal.tasks.arc_like import load_task_from_file

def run_experiment():
    console = Console()
    console.print("[bold cyan]Starting Science Experiment...[/]")
    
    # Base paths
    repo_root = Path(__file__).parent.parent
    examples_dir = repo_root / "data" / "examples"
    output_file = repo_root / "data" / "science_ledger.jsonl"
    
    # Configs
    greedy_config = TribunalSettings()
    greedy_config.generators.enabled = ["greedy"]
    greedy_config.tribunal.adjudication_strategy = "greedy"
    
    tribunal_config = TribunalSettings()
    tribunal_config.generators.enabled = ["greedy", "diverse", "adversarial", "rule_first", "minimal_description"]
    tribunal_config.tribunal.adjudication_strategy = "standard"
    
    store = LedgerStore(":memory:")
    
    greedy_orch = Orchestrator(greedy_config, store)
    tribunal_orch = Orchestrator(tribunal_config, store)
    
    task_files = sorted(examples_dir.glob("*.json"))
    runs = []
    
    for tf in task_files:
        console.print(f"Processing task {tf.name}...")
        task = load_task_from_file(tf)
        
        # Greedy Arm
        g_run = greedy_orch.run(task)
        g_run.metadata["arm_name"] = "greedy"
        runs.append(g_run)
        
        # Tribunal Arm
        t_run = tribunal_orch.run(task)
        t_run.metadata["arm_name"] = "tribunal"
        runs.append(t_run)
        
    console.print(f"Writing {len(runs)} runs to {output_file}...")
    with open(output_file, "w") as f:
        for r in runs:
            # We must serialize the Pydantic model to JSON
            f.write(r.model_dump_json() + "\n")
            
    console.print("[bold green]Done![/]")

if __name__ == "__main__":
    run_experiment()
