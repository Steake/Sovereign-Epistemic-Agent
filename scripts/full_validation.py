import json
import os
import time
from pathlib import Path
from rich.console import Console
from rich.table import Table

from epistemic_tribunal.config import load_config
from epistemic_tribunal.evaluation.benchmark import BenchmarkRunner
from epistemic_tribunal.evaluation.metrics import summary_report

def run_validation():
    console = Console()
    console.print("[bold blue]Starting Full 50-Task Validation Sweep (4 Arms)...[/bold blue]")
    
    manifest_path = "data/validation_manifest_v1.txt"
    dataset_path = "/Users/oli/Documents/Oli Works/arc_epistemic/Kaggle_Submission_Bundle/dataset"
    configs_dir = Path("configs/validation")
    arms = [
        ("arm1_greedy.yaml", "Greedy"),
        ("arm2_structural.yaml", "Structural"),
        ("arm3_lockout.yaml", "Lockout"),
        ("arm4_path_b.yaml", "Path B"),
    ]
    
    final_results = {}

    for arm_cfg, label in arms:
        cfg_path = configs_dir / arm_cfg
        console.print(f"\n[bold yellow]Targeting Arm: {label} ({arm_cfg})[/bold yellow]")
        
        # Load arm-specific config
        config = load_config(str(cfg_path))
        
        # Ensure ledger is fresh if not resuming
        db_path = Path(config.ledger.path)
        if db_path.exists():
            console.print(f"Clearing existing ledger: {db_path}")
            db_path.unlink()
        
        # Initialize Runner
        runner = BenchmarkRunner(config=config)
        
        # Run 50-task sweep
        start_t = time.monotonic()
        runs = runner.run(
            dataset_path=dataset_path,
            manifest_path=manifest_path
        )
        elapsed = time.monotonic() - start_t
        
        # Report
        metrics = summary_report(runs)
        final_results[label] = metrics
        console.print(f"Arm {label} complete in {elapsed:.1f}s. Accuracy: {metrics.get('overall_accuracy')}")

    # Final Comparison Table
    table = Table(title="Final Validation Scorecard (50 Tasks)")
    table.add_column("Metric", style="cyan")
    for _, label in arms:
        table.add_column(label, justify="right")

    metric_keys = [
        "overall_accuracy",
        "resolved_accuracy",
        "coverage",
        "resample_rate",
        "wrong_pick_count",
        "override_count"
    ]

    for key in metric_keys:
        table.add_row(
            key,
            *[str(final_results[label].get(key, "-")) for _, label in arms]
        )

    console.print("\n")
    console.print(table)
    
    # Save final JSON report
    with open("data/validation_results_v1.json", "w") as f:
        json.dump(final_results, f, indent=2)
    console.print("[bold green]Final results saved to data/validation_results_v1.json[/bold green]")

if __name__ == "__main__":
    run_validation()
