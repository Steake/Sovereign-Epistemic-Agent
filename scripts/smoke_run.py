from pathlib import Path
from rich.console import Console
from rich.table import Table

from epistemic_tribunal.config import load_config
from epistemic_tribunal.evaluation.benchmark import BenchmarkRunner
from epistemic_tribunal.evaluation.metrics import summary_report

def run_smoke():
    console = Console()
    console.print("[bold blue]Starting 10-Task Multi-Arm Smoke Run...[/bold blue]")
    
    manifest_path = "data/validation_manifest_v1.txt"
    dataset_path = "/Users/oli/Documents/Oli Works/arc_epistemic/Kaggle_Submission_Bundle/dataset"
    configs_dir = Path("configs/validation")
    arms = [
        "arm1_greedy.yaml",
        "arm2_structural.yaml",
        "arm3_lockout.yaml",
        "arm4_path_b.yaml",
    ]
    
    results = {}

    for arm_cfg in arms:
        cfg_path = configs_dir / arm_cfg
        arm_name = arm_cfg.replace(".yaml", "")
        console.print(f"\n[bold yellow]Targeting Arm: {arm_name}[/bold yellow]")
        
        # Load arm-specific config
        config = load_config(str(cfg_path))
        
        # Initialize Runner
        runner = BenchmarkRunner(config=config)
        
        # Run 10-task smoke
        runs = runner.run(
            dataset_path=dataset_path,
            manifest_path=manifest_path,
            limit=10
        )
        
        # Report
        metrics = summary_report(runs)
        results[arm_name] = metrics
        console.print(f"Arm {arm_name} complete. Accuracy: {metrics.get('overall_accuracy')}")

    # Final Comparison Table
    table = Table(title="Smoke Run Metrics (10 Tasks)")
    table.add_column("Metric", style="cyan")
    table.add_column("Greedy", justify="right")
    table.add_column("Structural", justify="right")
    table.add_column("Lockout", justify="right")
    table.add_column("Path B", justify="right", style="green")

    metric_keys = [
        "overall_accuracy",
        "resolved_accuracy",
        "coverage",
        "resample_rate",
        "wrong_pick_count",
        "mean_confidence",
        "override_count"
    ]

    for key in metric_keys:
        table.add_row(
            key,
            str(results["arm1_greedy"].get(key, "-")),
            str(results["arm2_structural"].get(key, "-")),
            str(results["arm3_lockout"].get(key, "-")),
            str(results["arm4_path_b"].get(key, "-"))
        )

    console.print("\n")
    console.print(table)

if __name__ == "__main__":
    run_smoke()
