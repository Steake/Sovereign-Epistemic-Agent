"""Command-line interface for the Epistemic Tribunal.

Commands
--------
tribunal run       — evaluate a single task JSON file
tribunal benchmark — run the tribunal over a dataset directory
tribunal ledger stats   — show ledger statistics
tribunal ledger inspect — show records for a specific task ID
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from epistemic_tribunal.config import load_config
from epistemic_tribunal.evaluation import calibration
from epistemic_tribunal.evaluation.benchmark import BenchmarkRunner
from epistemic_tribunal.ledger.store import LedgerStore
from epistemic_tribunal.orchestrator import Orchestrator
from epistemic_tribunal.tasks.arc_like import load_task_from_file
from epistemic_tribunal.types import DecisionKind, ExperimentRun
from epistemic_tribunal.utils.logging import get_logger

log = get_logger(__name__)
console = Console()

app = typer.Typer(
    name="tribunal",
    help="Epistemic Tribunal — metacognitive reasoning adjudication.",
    no_args_is_help=True,
)
ledger_app = typer.Typer(help="Ledger inspection commands.")
app.add_typer(ledger_app, name="ledger")


# ---------------------------------------------------------------------------
# tribunal run
# ---------------------------------------------------------------------------


@app.command("run")
def run_task(
    task_path: Path = typer.Argument(..., help="Path to task JSON file."),
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to YAML config file."
    ),
    ledger_path: Optional[str] = typer.Option(
        None, "--ledger", "-l", help="Override ledger DB path."
    ),
    json_output: bool = typer.Option(False, "--json", help="Output result as JSON."),
) -> None:
    """Run the tribunal on a single task JSON file."""
    config = load_config(config_path)
    if ledger_path:
        config.ledger.path = ledger_path

    orchestrator = Orchestrator(config)

    try:
        task = load_task_from_file(task_path)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Error loading task:[/red] {exc}")
        raise typer.Exit(1) from exc

    result = orchestrator.run_and_format(task)

    if json_output:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_run_result(result)


def _print_run_result(result: dict) -> None:
    table = Table(title="Tribunal Result", show_header=True)
    table.add_column("Field", style="bold cyan")
    table.add_column("Value", style="white")
    for key, val in result.items():
        table.add_row(str(key), str(val))
    console.print(table)


# ---------------------------------------------------------------------------
# tribunal benchmark
# ---------------------------------------------------------------------------


@app.command("benchmark")
def run_benchmark(
    dataset_path: Path = typer.Argument(..., help="Directory of task JSON files."),
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to YAML config file."
    ),
    ledger_path: Optional[str] = typer.Option(
        None, "--ledger", "-l", help="Override ledger DB path."
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume from run_progress.json and skip task IDs that already completed.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output metrics as JSON."),
) -> None:
    """Run the tribunal over a directory of task JSON files."""
    config = load_config(config_path)
    if ledger_path:
        config.ledger.path = ledger_path

    runner = BenchmarkRunner(config=config, ledger_path=ledger_path)

    if not dataset_path.is_dir():
        console.print(f"[red]Not a directory:[/red] {dataset_path}")
        raise typer.Exit(1)

    metrics = runner.run_and_report(dataset_path, resume=resume)

    if json_output:
        print(json.dumps(metrics, indent=2, default=str))
    else:
        _print_metrics(metrics)


def _print_metrics(metrics: dict) -> None:
    table = Table(title="Benchmark Metrics", show_header=True)
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", style="white")
    for key, val in metrics.items():
        if isinstance(val, dict):
            for subkey, subval in val.items():
                table.add_row(f"  {key}.{subkey}", str(subval))
        else:
            table.add_row(str(key), str(val))
    console.print(table)


# ---------------------------------------------------------------------------
# tribunal ledger stats
# ---------------------------------------------------------------------------


@ledger_app.command("stats")
def ledger_stats(
    ledger_path: Optional[str] = typer.Option(
        None, "--ledger", "-l", help="Path to ledger DB."
    ),
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to YAML config file."
    ),
) -> None:
    """Show high-level statistics from the failure ledger."""
    config = load_config(config_path)
    db_path = ledger_path or config.ledger.path
    store = LedgerStore(db_path)
    stats = store.get_stats()
    store.close()

    table = Table(title=f"Ledger Stats — {db_path}", show_header=True)
    table.add_column("Table / Metric", style="bold cyan")
    table.add_column("Count / Value", style="white")

    for key, val in stats.items():
        if isinstance(val, dict):
            for subkey, subval in val.items():
                table.add_row(f"  decisions.{subkey}", str(subval))
        else:
            table.add_row(str(key), str(val))

    console.print(table)


# ---------------------------------------------------------------------------
# tribunal ledger inspect
# ---------------------------------------------------------------------------


@ledger_app.command("inspect")
def ledger_inspect(
    task_id: str = typer.Option(..., "--task-id", "-t", help="Task ID to inspect."),
    ledger_path: Optional[str] = typer.Option(
        None, "--ledger", "-l", help="Path to ledger DB."
    ),
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to YAML config file."
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Inspect all ledger records for a specific task ID."""
    config = load_config(config_path)
    db_path = ledger_path or config.ledger.path
    store = LedgerStore(db_path)
    summary = store.get_task_summary(task_id)
    store.close()

    if json_output:
        print(json.dumps(summary, indent=2, default=str))
        return

    if "error" in summary:
        console.print(f"[red]{summary['error']}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Task:[/bold] {task_id}")
    for section, records in summary.items():
        if section == "task":
            console.print(f"\n[bold cyan]Task Record:[/bold cyan]")
            for k, v in records.items():
                console.print(f"  {k}: {v}")
        elif isinstance(records, list):
            console.print(f"\n[bold cyan]{section.title()} ({len(records)}):[/bold cyan]")
            for rec in records:
                console.print(f"  {rec}")


# ---------------------------------------------------------------------------
# tribunal calibrate
# ---------------------------------------------------------------------------


@app.command("calibrate")
def calibrate_cmd(
    ledger_path: str = typer.Option(
        ..., "--ledger", "-l", help="Path to ledger DB."
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Compute a calibration report from historical ledger runs."""
    store = LedgerStore(ledger_path)
    rows = store.get_experiment_runs()
    store.close()

    runs: list[ExperimentRun] = []
    for row in rows:
        runs.append(
            ExperimentRun(
                run_id=row["run_id"],
                task_id=row["task_id"],
                generator_names=json.loads(row["generator_names_json"]),
                decision=DecisionKind(row["decision"]),
                selected_trace_id=row["selected_trace_id"],
                ground_truth_match=(
                    None
                    if row["ground_truth_match"] is None
                    else bool(row["ground_truth_match"])
                ),
                confidence=row.get("confidence", 0.0) or 0.0,
                duration_seconds=row["duration_seconds"],
                config_snapshot=json.loads(row["config_snapshot_json"]),
            )
        )

    if not any(r.confidence > 0 for r in runs):
        console.print(
            "[yellow]No runs with usable confidence scores found in the ledger.[/yellow]"
        )
        raise typer.Exit(0)

    ece = calibration.expected_calibration_error(runs)
    bs = calibration.brier_score(runs)
    curve = calibration.reliability_curve(runs)
    sel_acc = calibration.accuracy_at_coverage(runs, coverage_target=0.9)
    abst = calibration.abstention_quality(runs)

    if json_output:
        report = {
            "ece": round(ece, 4),
            "brier_score": round(bs, 4),
            "reliability_curve": curve,
            "selective_accuracy_90": {k: round(v, 4) for k, v in sel_acc.items()},
            "abstention_quality": {k: round(v, 4) for k, v in abst.items()},
        }
        print(json.dumps(report, indent=2))
        return

    # Scalar metrics table
    metrics_table = Table(title="Calibration Report", show_header=True)
    metrics_table.add_column("Metric", style="bold cyan")
    metrics_table.add_column("Value", style="white")
    metrics_table.add_row("ECE", f"{ece:.4f}")
    metrics_table.add_row("Brier Score", f"{bs:.4f}")
    console.print(metrics_table)

    # Reliability curve table
    if curve:
        rc_table = Table(title="Reliability Curve", show_header=True)
        rc_table.add_column("Bin Midpoint", style="bold cyan")
        rc_table.add_column("Mean Confidence", style="white")
        rc_table.add_column("Mean Accuracy", style="white")
        rc_table.add_column("Count", style="white")
        for entry in curve:
            rc_table.add_row(
                f"{entry['bin_midpoint']:.4f}",
                f"{entry['mean_confidence']:.4f}",
                f"{entry['mean_accuracy']:.4f}",
                str(entry["count"]),
            )
        console.print(rc_table)

    # Accuracy at coverage
    acc_table = Table(title="Accuracy at 90% Coverage", show_header=True)
    acc_table.add_column("Metric", style="bold cyan")
    acc_table.add_column("Value", style="white")
    acc_table.add_row("Accuracy", f"{sel_acc['accuracy']:.4f}")
    acc_table.add_row("Coverage", f"{sel_acc['coverage']:.4f}")
    acc_table.add_row("Threshold", f"{sel_acc['threshold']:.4f}")
    console.print(acc_table)

    # Abstention quality
    abst_table = Table(title="Abstention Quality", show_header=True)
    abst_table.add_column("Metric", style="bold cyan")
    abst_table.add_column("Value", style="white")
    abst_table.add_row("Abstention Rate", f"{abst['abstention_rate']:.4f}")
    abst_table.add_row("Wrong Abstention Rate", f"{abst['wrong_abstention_rate']:.4f}")
    abst_table.add_row("Correct Abstention Rate", f"{abst['correct_abstention_rate']:.4f}")
    console.print(abst_table)


if __name__ == "__main__":
    app()
