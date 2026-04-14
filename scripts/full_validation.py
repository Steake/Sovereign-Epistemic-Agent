import json
import os
import sys
import time
from pathlib import Path
from rich.console import Console
from rich.table import Table

from epistemic_tribunal.config import load_config
from epistemic_tribunal.evaluation.benchmark import BenchmarkRunner
from epistemic_tribunal.evaluation.metrics import summary_report

# ---------------------------------------------------------------------------
# Dataset path resolution
# Priority:
#   1. ARC_DATASET_PATH environment variable  (set this on the pod)
#   2. Local Mac dev path as a fallback
# ---------------------------------------------------------------------------
_LOCAL_DATASET_FALLBACK = (
    "/Users/oli/Documents/Oli Works/arc_epistemic/Kaggle_Submission_Bundle/dataset"
)
_POD_DATASET_DEFAULT = "/workspace/arc_dataset"

def _resolve_dataset_path() -> Path:
    """Return the ARC benchmark dataset directory, failing fast if not found."""
    # Explicit override always wins
    env_path = os.environ.get("ARC_DATASET_PATH", "")
    if env_path:
        p = Path(env_path)
        if p.is_dir() and list(p.glob("*.json")):
            return p
        raise SystemExit(
            f"[ERROR] ARC_DATASET_PATH='{env_path}' is set but contains no *.json files.\n"
            f"        Ensure the dataset is mounted/uploaded to that path on the pod."
        )

    # Local Mac fallback
    fallback = Path(_LOCAL_DATASET_FALLBACK)
    if fallback.is_dir() and list(fallback.glob("*.json")):
        return fallback

    # Pod default path
    pod_default = Path(_POD_DATASET_DEFAULT)
    if pod_default.is_dir() and list(pod_default.glob("*.json")):
        return pod_default

    raise SystemExit(
        "[ERROR] Could not locate the ARC benchmark dataset.\n"
        "  • Set  ARC_DATASET_PATH=/path/to/dataset  before running, OR\n"
        "  • Upload dataset JSON files to /workspace/arc_dataset on the pod, OR\n"
        f"  • Ensure local path exists: {_LOCAL_DATASET_FALLBACK}"
    )


def run_validation():
    console = Console()
    console.print("[bold blue]Starting Full 50-Task Validation Sweep (4 Arms)...[/bold blue]")

    import sys
    manifest_path = sys.argv[1] if len(sys.argv) > 1 else "data/validation_manifest_v1.txt"
    dataset_path = _resolve_dataset_path()
    console.print(f"[dim]Dataset: {dataset_path}[/dim]")
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

        # Fail loudly if no tasks ran — avoids silent zero-accuracy results
        if not runs:
            console.print(
                f"[bold red]\n[FATAL] Arm {label}: 0 tasks ran. "
                f"Check that manifest IDs match files in:\n  {dataset_path}[/bold red]"
            )
            sys.exit(1)

        # Report
        metrics = summary_report(runs)
        final_results[label] = metrics
        console.print(f"Arm {label} complete in {elapsed:.1f}s. Accuracy: {metrics.get('overall_accuracy')}")

    # Final Comparison Table
    task_count = len(runs) if runs else 0
    table = Table(title=f"Final Validation Scorecard ({task_count} Tasks)")
    table.add_column("Metric", style="cyan")
    for _, label in arms:
        table.add_column(label, justify="right")

    metric_keys = [
        "overall_accuracy",
        "resolved_accuracy",
        "coverage",
        "resample_rate",
        "wrong_pick_count",
        "override_count",
        "truncation_count",
        "json_not_found_count",
        "json_invalid_count",
        "grid_shape_invalid_count",
        "reasoning_bleed_count",
        "parse_failure_count",
        "path_b_met_gate",
        "path_b_failed_v",
        "path_b_failed_c",
        "path_b_failed_violations",
        "path_b_failed_margin"
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
