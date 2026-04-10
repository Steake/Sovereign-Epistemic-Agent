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
from epistemic_tribunal.evaluation.benchmark import BenchmarkRunner
from epistemic_tribunal.ledger.store import LedgerStore
from epistemic_tribunal.orchestrator import Orchestrator
from epistemic_tribunal.tasks.arc_like import load_task_from_file
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

    metrics = runner.run_and_report(dataset_path)

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


if __name__ == "__main__":
    app()
