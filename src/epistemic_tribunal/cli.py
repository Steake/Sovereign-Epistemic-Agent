"""Command-line interface for the Epistemic Tribunal.

Commands
--------
tribunal run                 — evaluate a single task JSON file
tribunal benchmark           — run the tribunal over a dataset directory
tribunal benchmark-usefulness — compare arms on annotated cohorts
tribunal ledger stats        — show ledger statistics
tribunal ledger inspect      — show records for a specific task ID
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from epistemic_tribunal.config import load_config
from epistemic_tribunal.evaluation import calibration
from epistemic_tribunal.evaluation.benchmark import BenchmarkRunner
from epistemic_tribunal.evaluation.benchmark import experiment_run_from_row
from epistemic_tribunal.evaluation.benchmark_annotations import (
    load_annotations,
    load_oracle_metadata,
)
from epistemic_tribunal.evaluation.benchmark_report import (
    build_report,
    records_from_runs,
)
from epistemic_tribunal.evaluation.benchmark_spec import BenchmarkCohort
from epistemic_tribunal.ledger.store import LedgerStore
from epistemic_tribunal.orchestrator import Orchestrator
from epistemic_tribunal.tasks.arc_like import load_task_from_file
from epistemic_tribunal.tribunal_types import DecisionKind, ExperimentRun
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
        if task_path.suffix == ".jsonl":
            from epistemic_tribunal.tasks.gsm8k import load_tasks_from_jsonl
            tasks = load_tasks_from_jsonl(task_path)
            if not tasks:
                raise ValueError("JSONL file is empty.")
            task = tasks[0]
            if len(tasks) > 1:
                log.warning("JSONL file has %d tasks, `run` will only process the first one. Use `benchmark` for all.", len(tasks))
        else:
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
    manifest_path: Optional[Path] = typer.Option(
        None, "--manifest", "-m", help="Text file with task IDs (one per line) to run."
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume from run_progress.json and skip task IDs that already completed.",
    ),
    forensic: bool = typer.Option(False, "--forensic", help="Display detailed candidate score breakdown."),
    json_output: bool = typer.Option(False, "--json", help="Output metrics as JSON."),
    limit: Optional[int] = typer.Option(None, "--limit", help="Cap the number of tasks to run."),
) -> None:
    """Run the tribunal over a directory of task JSON files."""
    config = load_config(config_path)
    if ledger_path:
        config.ledger.path = ledger_path
    
    # Diagnostic Block
    log.info("CONFIG BOOT: Enabled Generators: %s", config.generators.enabled)
    log.info("CONFIG BOOT: Path B Enabled: %s", config.tribunal.structural_override.enabled)

    runner = BenchmarkRunner(config=config, ledger_path=ledger_path)

    if not dataset_path.exists():
        console.print(f"[red]Path not found:[/red] {dataset_path}")
        raise typer.Exit(1)

    runs = runner.run(dataset_path, resume=resume, manifest_path=manifest_path, limit=limit)
    metrics = runner.report(runs)

    if forensic:
        _print_forensic_results(runner.last_runs, ledger_path=runner._config.ledger.path)

    if json_output:
        print(json.dumps(metrics, indent=2, default=str))
    else:
        _print_metrics(metrics)


def _print_metrics(metrics: dict) -> None:
    # 1. Main Performance Table
    summary_table = Table(title="[bold blue]Benchmark Primary Performance[/bold blue]", show_header=True, box=None)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", justify="right", style="bold white")
    
    summary_table.add_row("Total Runs", str(metrics["total_runs"]))
    summary_table.add_row("Overall Accuracy", f"{metrics['overall_accuracy']:.4f}")
    summary_table.add_row("Selective Accuracy", f"[bold green]{metrics['selective_accuracy']:.4f}[/bold green]")
    summary_table.add_row("Coverage", f"{metrics['coverage']:.4f}")
    summary_table.add_row("Wrong Picks", f"[bold red]{metrics['wrong_pick_count']}[/bold red]")
    console.print(summary_table)
    console.print()

    # 2. Cohort Stratification Table
    cohort_table = Table(title="[bold magenta]Cohort Stratification (Recoverable vs Contested)[/bold magenta]", show_header=True)
    cohort_table.add_column("Cohort", style="bold cyan")
    cohort_table.add_column("N", justify="right")
    cohort_table.add_column("Selective Acc", justify="right", style="green")
    cohort_table.add_column("Abstain Rate", justify="right", style="yellow")
    cohort_table.add_column("Wrong Picks", justify="right", style="red")

    cohort_map = {
        "control-trivial": "[dim white]control-trivial[/dim white]",
        "contested-recoverable": "[bold green]contested-recoverable[/bold green]",
        "contested-unrecoverable": "[bold yellow]contested-unrecoverable[/bold yellow]",
        "unknown": "[dim red]unknown[/dim red]"
    }

    cohort_metrics = metrics.get("cohort_metrics", {})
    for name, display in cohort_map.items():
        if name in cohort_metrics:
            m = cohort_metrics[name]
            cohort_table.add_row(
                display,
                str(m["n"]),
                f"{m['selective_acc']:.4f}",
                f"{m['abstain_rate']:.4f}",
                str(m["wrong_picks"])
            )
    console.print(cohort_table)
    console.print()

    # 3. Abstention Quality Table
    abstain_metrics = metrics.get("abstention_metrics", {})
    if abstain_metrics:
        abstain_table = Table(title="[bold yellow]Abstention Audit (Metacognitive Honesty)[/bold yellow]", show_header=True, box=None)
        abstain_table.add_column("Abstention Type", style="cyan")
        abstain_table.add_column("Count", justify="right")
        abstain_table.add_column("Description", style="dim italic")

        abstain_table.add_row(
            "Good Abstentions", 
            f"[bold green]{abstain_metrics['good_abstentions']}[/bold green]",
            "Avoided unrecoverable errors"
        )
        abstain_table.add_row(
            "Bad Abstentions", 
            f"[bold red]{abstain_metrics['bad_abstentions']}[/bold red]",
            "Missed recoverable solutions"
        )
        abstain_table.add_row(
            "Efficiency", 
            f"{abstain_metrics['abstention_efficiency']:.4f}",
            "P(No Correct Answer | Abstain)"
        )
        console.print(abstain_table)
        console.print()

    # 3.5 Tribunal Usefulness Table
    usefulness = metrics.get("tribunal_usefulness", {})
    if usefulness:
        usefulness_table = Table(title="[bold green]Tribunal Usefulness & Oracle Analysis[/bold green]", show_header=True, box=None)
        usefulness_table.add_column("Metric", style="cyan")
        usefulness_table.add_column("Value", justify="right", style="bold white")
        usefulness_table.add_column("Description", style="dim italic")

        usefulness_table.add_row(
            "Best Candidate in Pool",
            f"{usefulness['best_candidate_in_pool_accuracy']:.4f}",
            "P(Correct | Oracle Selection)"
        )
        usefulness_table.add_row(
            "Greedy Accuracy",
            f"{usefulness['greedy_accuracy']:.4f}",
            "Baseline LLM accuracy"
        )
        usefulness_table.add_row(
            "Tribunal Lift",
            f"[bold green]{usefulness['tribunal_lift_over_greedy']:+.4f}[/bold green]",
            "Selective Acc - Greedy Acc"
        )
        usefulness_table.add_row(
            "Lift on Contested-Recoverable",
            f"[bold green]{usefulness['lift_on_contested_recoverable']:+.4f}[/bold green]",
            "Tribunal value on hard-but-doable tasks"
        )
        console.print(usefulness_table)
        console.print()

    # 4. Diagnostics Table
    diag = metrics.get("diagnostics", {})
    if diag:
        diag_table = Table(title="[bold dim]System Health & Diagnostics[/bold dim]", show_header=True, box=None)
        diag_table.add_column("Metric", style="dim cyan")
        diag_table.add_column("Value", justify="right", style="dim white")
        
        diag_table.add_row("Avg Duration (s)", f"{diag['avg_duration']:.2f}")
        diag_table.add_row("Path B Overrides", str(diag['path_b_overrides']))
        diag_table.add_row("Parse Failures", str(diag['parse_failures']))
        diag_table.add_row("JSON Errors", str(diag['json_errors']))
        diag_table.add_row("Truncations", str(diag['truncations']))
        console.print(diag_table)

    eqbsl_diag = metrics.get("eqbsl_diagnostics", {})
    if eqbsl_diag:
        console.print()
        eqbsl_table = Table(title="[bold cyan]EQBSL Diagnostics[/bold cyan]", show_header=True, box=None)
        eqbsl_table.add_column("Metric", style="cyan")
        eqbsl_table.add_column("Value", justify="right", style="bold white")
        for key, value in eqbsl_diag.items():
            if isinstance(value, float):
                eqbsl_table.add_row(key, f"{value:.4f}")
            else:
                eqbsl_table.add_row(key, str(value))
        console.print(eqbsl_table)


def _print_forensic_results(runs: list[ExperimentRun], ledger_path: Optional[str] = None) -> None:
    """Print a detailed candidate score breakdown for each task."""
    coalition_rows_by_run: dict[str, list[dict]] = {}
    if ledger_path is not None:
        store = LedgerStore(ledger_path)
        try:
            rows = store.get_coalition_opinions(run_ids=[run.run_id for run in runs])
            for row in rows:
                coalition_rows_by_run.setdefault(row["run_id"], []).append(row)
        finally:
            store.close()

    for run in runs:
        forensic = run.metadata.get("forensic")
        if not forensic:
            continue
            
        table = Table(title=f"Forensic Audit — Task: {run.task_id}", show_header=True, header_style="bold magenta")
        table.add_column("Generator", style="cyan")
        table.add_column("U", justify="right")
        table.add_column("C", justify="right")
        table.add_column("M", justify="right")
        table.add_column("V", justify="right")
        table.add_column("Total", style="bold yellow", justify="right")
        table.add_column("Conf", justify="right")
        table.add_column("Rank", justify="center")

        sorted_forensic = sorted(forensic, key=lambda x: x["total"], reverse=True)
        for i, f in enumerate(sorted_forensic):
            table.add_row(
                f["generator"],
                f"{f['U']:.3f}",
                f"{f['C']:.3f}",
                f"{f['M']:.3f}",
                f"{f['V']:.3f}",
                f"{f['total']:.3f}",
                f"{f['confidence']:.3f}",
                str(i + 1)
            )
        console.print(table)
        console.print()

        eqbsl_summary = run.metadata.get("eqbsl")
        coalition_rows = coalition_rows_by_run.get(run.run_id, [])
        if eqbsl_summary and coalition_rows:
            coalition_table = Table(
                title=f"EQBSL Coalition Forensic — Task: {run.task_id}",
                show_header=True,
                header_style="bold cyan",
            )
            coalition_table.add_column("Answer Signature", style="white")
            coalition_table.add_column("Role", style="yellow")
            coalition_table.add_column("Belief", justify="right")
            coalition_table.add_column("Disbelief", justify="right")
            coalition_table.add_column("Uncertainty", justify="right")
            coalition_table.add_column("Expectation", justify="right", style="bold green")
            coalition_table.add_column("Reason", style="dim")

            for row in sorted(coalition_rows, key=lambda item: item["expectation"], reverse=True):
                coalition_table.add_row(
                    str(row["answer_signature"]),
                    str(row["decision_role"]),
                    f"{float(row['belief']):.3f}",
                    f"{float(row['disbelief']):.3f}",
                    f"{float(row['uncertainty']):.3f}",
                    f"{float(row['expectation']):.3f}",
                    str(row["decision_reason_code"]),
                )
            console.print(coalition_table)
            console.print()


# ---------------------------------------------------------------------------
# tribunal replay-failed
# ---------------------------------------------------------------------------

@app.command("replay-failed")
def replay_failed(
    dataset_path: Path = typer.Argument(..., help="Directory of task JSON files."),
    baseline_ledger: Path = typer.Option(..., "--from-ledger", help="Path to baseline ledger DB to identify failures."),
    output_ledger: Path = typer.Option(..., "--ledger", help="Path to output ledger DB for the replay."),
    mode: str = typer.Option("full_memory", "--mode", help="Strange Loop mode to inject: off, bad_answers_only, warnings_only, full_memory."),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to base YAML config file."),
    json_output: bool = typer.Option(False, "--json", help="Output metrics as JSON."),
) -> None:
    """Replay tasks that failed in the baseline under a specific Strange Loop condition."""
    import tempfile
    import os

    if not baseline_ledger.exists():
        console.print(f"[red]Baseline ledger not found:[/red] {baseline_ledger}")
        raise typer.Exit(1)

    baseline_runs = _load_runs_from_ledger(str(baseline_ledger))
    # Identify tasks that failed (wrong pick or bad abstention -> ground_truth_match is False or missing when there was a correct answer in pool)
    # The prompt explicitly mentioned: "identify failed cohort". Let's use ground_truth_match is False.
    failed_task_ids = set(
        run.task_id for run in baseline_runs
        if run.ground_truth_match is False or (run.decision == DecisionKind.ABSTAIN and run.metadata.get("any_correct") is True)
    )

    if not failed_task_ids:
        console.print("[yellow]No failed tasks found in the baseline ledger. Nothing to replay.[/yellow]")
        raise typer.Exit(0)

    console.print(f"[bold cyan]Found {len(failed_task_ids)} failed tasks in baseline. Replaying under mode '{mode}'...[/bold cyan]")

    # Create temporary manifest
    fd, tmp_manifest = tempfile.mkstemp(suffix=".txt", text=True)
    with os.fdopen(fd, "w") as f:
        for tid in sorted(failed_task_ids):
            f.write(f"{tid}\n")

    try:
        config = load_config(config_path)
        config.ledger.path = str(output_ledger)
        config.strange_loop.enabled = True
        config.failure_memory.enabled = True
        config.strange_loop.mode = mode
        config.tribunal.ledger_warmup_tasks = 0
        # Re-attach the baseline ledger path as the failure memory store path, since we want to read FROM the baseline's memory
        # Wait, the failure memory store defaults to config.ledger.path. If we change ledger.path to output_ledger,
        # it will look for failure memory in output_ledger (which is initially empty)!
        # We need to tell the FailureMemoryStore to read from the baseline ledger.
        # But wait, FailureMemoryStore only takes `path`. And Orchestrator passes `config.ledger.path` to it.
        # So we need to monkeypatch it for this run or ensure the baseline ledger is copied to the output ledger first.
        # The user says: "same ledger seed copied from baseline".
        import shutil
        if not output_ledger.exists():
            console.print(f"Copying baseline ledger {baseline_ledger} to {output_ledger} to preserve seed memory...")
            shutil.copy2(baseline_ledger, output_ledger)

        # Also, to prevent it from re-loading "completed tasks" from the copied ledger, we must NOT use --resume.
        # But wait, if we copy the ledger, the copied ledger already has `ExperimentRun` records for these task_ids.
        # If we run without `--resume`, will it overwrite or append? The ledger writer appends.
        # That means the ledger will have two runs for the same task. This is fine, but analysis scripts usually
        # take the most recent one or group by run_id.
        
        runner = BenchmarkRunner(config=config, ledger_path=str(output_ledger))
        runs = runner.run(dataset_path, manifest_path=Path(tmp_manifest), resume=False)
        metrics = runner.report(runs)

        if json_output:
            print(json.dumps(metrics, indent=2, default=str))
        else:
            _print_metrics(metrics)

    finally:
        os.remove(tmp_manifest)


# ---------------------------------------------------------------------------
# tribunal compare-strange-loop
# ---------------------------------------------------------------------------

@app.command("compare-strange-loop")
def compare_strange_loop(
    baseline_ledger: Path = typer.Option(..., "--baseline", help="Baseline ledger DB."),
    control_ledger: Path = typer.Option(..., "--control", help="Control replay ledger."),
    bad_answers_ledger: Path = typer.Option(..., "--bad-answers", help="Bad answers only replay ledger."),
    warnings_ledger: Path = typer.Option(..., "--warnings", help="Warnings only replay ledger."),
    full_ledger: Path = typer.Option(..., "--full", help="Full memory replay ledger."),
) -> None:
    """Compare the results of the 4 Strange Loop experimental replay arms."""
    if not baseline_ledger.exists():
        console.print(f"[red]Baseline ledger not found:[/red] {baseline_ledger}")
        raise typer.Exit(1)

    baseline_runs = _load_runs_from_ledger(str(baseline_ledger))
    
    # We only care about runs that failed in the baseline
    failed_baseline_runs = [
        run for run in baseline_runs 
        if run.ground_truth_match is False or (run.decision == DecisionKind.ABSTAIN and run.metadata.get("any_correct") is True)
    ]
    
    if not failed_baseline_runs:
        console.print("[yellow]No failed tasks found in the baseline ledger. Cannot compare.[/yellow]")
        raise typer.Exit(0)
        
    baseline_signatures = {}
    for run in failed_baseline_runs:
        selected_sig = None
        for outcome in run.metadata.get("generator_outcomes", []):
            if outcome["trace_id"] == run.selected_trace_id:
                selected_sig = outcome["answer_signature"]
                break
        if selected_sig:
            baseline_signatures[run.task_id] = selected_sig

    failed_task_ids = set(run.task_id for run in failed_baseline_runs)

    def _analyze_arm(ledger_path: Path, arm_name: str) -> dict:
        if not ledger_path.exists():
            return {"name": arm_name, "missing": True}
            
        runs = _load_runs_from_ledger(str(ledger_path))
        
        # Group by task_id and take the latest run for tasks that failed in baseline
        # Wait, since we copied the baseline ledger, the replay run is simply the run with the latest timestamp.
        latest_runs_by_task = {}
        for run in runs:
            if run.task_id not in failed_task_ids:
                continue
            if run.task_id not in latest_runs_by_task or run.timestamp > latest_runs_by_task[run.task_id].timestamp:
                latest_runs_by_task[run.task_id] = run
                
        replayed_runs = list(latest_runs_by_task.values())
        if not replayed_runs:
            return {"name": arm_name, "missing": True}

        total = len(replayed_runs)
        recoveries = 0
        exact_recurrence = 0
        bad_abstentions = 0
        total_duration = 0.0
        
        for run in replayed_runs:
            if run.ground_truth_match is True:
                recoveries += 1
                
            if run.decision == DecisionKind.ABSTAIN and run.metadata.get("any_correct") is True:
                bad_abstentions += 1
                
            total_duration += run.duration_seconds
            
            # Recurrence: did we pick the EXACT same wrong signature as baseline?
            selected_sig = None
            for outcome in run.metadata.get("generator_outcomes", []):
                if outcome["trace_id"] == run.selected_trace_id:
                    selected_sig = outcome["answer_signature"]
                    break
            
            baseline_sig = baseline_signatures.get(run.task_id)
            if selected_sig and baseline_sig and selected_sig == baseline_sig:
                exact_recurrence += 1

        return {
            "name": arm_name,
            "missing": False,
            "n": total,
            "recovery_rate": recoveries / total,
            "exact_recurrence_rate": exact_recurrence / total,
            "bad_abstention_rate": bad_abstentions / total,
            "mean_duration": total_duration / total,
        }

    arms = [
        _analyze_arm(control_ledger, "control_retry"),
        _analyze_arm(bad_answers_ledger, "bad_answers_only"),
        _analyze_arm(warnings_ledger, "warnings_only"),
        _analyze_arm(full_ledger, "full_memory"),
    ]
    
    control_arm = arms[0]
    
    table = Table(title="Strange Loop Efficacy (Replay of Failed Cohort)", show_header=True)
    table.add_column("Condition", style="bold cyan")
    table.add_column("N", justify="right")
    table.add_column("Recovery Rate", justify="right", style="bold green")
    table.add_column("Δ Recovery", justify="right")
    table.add_column("Exact Recurrence", justify="right", style="red")
    table.add_column("Bad Abstention", justify="right", style="yellow")
    table.add_column("Mean Duration", justify="right")
    
    for arm in arms:
        if arm.get("missing"):
            table.add_row(arm["name"], "Missing ledger", "", "", "", "", "")
            continue
            
        recovery_str = f"{arm['recovery_rate']*100:.1f}%"
        recurrence_str = f"{arm['exact_recurrence_rate']*100:.1f}%"
        abstention_str = f"{arm['bad_abstention_rate']*100:.1f}%"
        duration_str = f"{arm['mean_duration']:.1f}s"
        
        delta_str = "—"
        if arm["name"] != "control_retry" and not control_arm.get("missing"):
            delta = arm['recovery_rate'] - control_arm['recovery_rate']
            color = "green" if delta > 0 else "red" if delta < 0 else "white"
            sign = "+" if delta > 0 else ""
            delta_str = f"[{color}]{sign}{delta*100:.1f}%[/{color}]"
            
        table.add_row(
            arm["name"],
            str(arm["n"]),
            recovery_str,
            delta_str,
            recurrence_str,
            abstention_str,
            duration_str,
        )

    console.print()
    console.print(table)
    console.print()


def _load_runs_from_ledger(db_path: str) -> list[ExperimentRun]:
    store = LedgerStore(db_path)
    try:
        return [experiment_run_from_row(row) for row in store.get_experiment_runs()]
    finally:
        store.close()


def _format_float(value: float) -> str:
    return f"{value:.4f}"


def _print_reliability_curve(curve: list[dict]) -> None:
    table = Table(title="Reliability Curve", show_header=True)
    table.add_column("Bin Midpoint", style="bold cyan")
    table.add_column("Mean Confidence", style="white")
    table.add_column("Mean Accuracy", style="white")
    table.add_column("Count", style="white")
    for row in curve:
        table.add_row(
            _format_float(row["bin_midpoint"]),
            _format_float(row["mean_confidence"]),
            _format_float(row["mean_accuracy"]),
            str(row["count"]),
        )
    console.print(table)


@app.command("calibrate")
def calibrate_ledger(
    ledger_path: str = typer.Option(..., "--ledger", "-l", help="Path to ledger DB."),
) -> None:
    """Compute a calibration report from historical ledger runs."""
    runs = _load_runs_from_ledger(ledger_path)
    eligible_with_confidence = [
        run
        for run in runs
        if (
            run.decision == DecisionKind.SELECT
            and run.ground_truth_match is not None
            and run.confidence > 0.0
        )
    ]
    if not eligible_with_confidence:
        console.print(
            f"[yellow]No usable confidence-bearing runs found in ledger:[/yellow] {ledger_path}"
        )
        return

    metrics_table = Table(title=f"Calibration Report — {ledger_path}", show_header=True)
    metrics_table.add_column("Metric", style="bold cyan")
    metrics_table.add_column("Value", style="white")
    metrics_table.add_row("ECE", _format_float(calibration.expected_calibration_error(eligible_with_confidence)))
    metrics_table.add_row("Brier score", _format_float(calibration.brier_score(eligible_with_confidence)))

    acc90 = calibration.accuracy_at_coverage(eligible_with_confidence, 0.9)
    metrics_table.add_row("Accuracy@90% coverage", _format_float(acc90["accuracy"]))
    metrics_table.add_row("Coverage@90%", _format_float(acc90["coverage"]))
    metrics_table.add_row("Threshold@90%", _format_float(acc90["threshold"]))

    abstention = calibration.abstention_quality(runs)
    metrics_table.add_row("Abstention rate", _format_float(abstention["abstention_rate"]))
    metrics_table.add_row(
        "Bad abstention rate (Missed Solution)", _format_float(abstention["bad_abstention_rate"])
    )
    metrics_table.add_row(
        "Good abstention rate (Avoided Error)", _format_float(abstention["good_abstention_rate"])
    )
    console.print(metrics_table)

    _print_reliability_curve(calibration.reliability_curve(eligible_with_confidence))


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
            console.print("\n[bold cyan]Task Record:[/bold cyan]")
            for k, v in records.items():
                console.print(f"  {k}: {v}")
        elif isinstance(records, list):
            console.print(f"\n[bold cyan]{section.title()} ({len(records)}):[/bold cyan]")
            for rec in records:
                console.print(f"  {rec}")


# ---------------------------------------------------------------------------
# tribunal benchmark-usefulness
# ---------------------------------------------------------------------------


@app.command("benchmark-usefulness")
def benchmark_usefulness(
    runs_path: Optional[Path] = typer.Option(
        None, "--runs", help="Path to a JSONL file of ExperimentRun records."
    ),
    ledger_path: Optional[str] = typer.Option(
        None, "--ledger", "-l", help="Load runs from a ledger SQLite DB."
    ),
    annotations_path: Path = typer.Option(
        ..., "--annotations", "-a", help="Path to task annotation JSON/JSONL file."
    ),
    oracle_path: Optional[Path] = typer.Option(
        None, "--oracle", help="Path to oracle metadata JSON/JSONL file (optional)."
    ),
    greedy_arm: str = typer.Option(
        "greedy", "--greedy-arm", help="Arm name to use as the greedy baseline."
    ),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON report."),
) -> None:
    """Report tribunal usefulness across cohorts, comparing named arms.

    Requires a runs source (--runs or --ledger) and an annotations file.
    Each run must carry arm_name in its metadata field.
    """
    # --- Load runs ---
    if runs_path is not None:
        raw = runs_path.read_text(encoding="utf-8")
        run_dicts = []
        for line in raw.splitlines():
            line = line.strip()
            if line:
                run_dicts.append(json.loads(line))
        runs = [ExperimentRun.model_validate(d) for d in run_dicts]
    elif ledger_path is not None:
        runs = _load_runs_from_ledger(ledger_path)
    else:
        console.print("[red]Provide either --runs or --ledger.[/red]")
        raise typer.Exit(1)

    if not runs:
        console.print("[yellow]No runs found. Nothing to report.[/yellow]")
        raise typer.Exit(0)

    # --- Load annotations ---
    try:
        annotations = load_annotations(annotations_path)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]Annotation load error:[/red] {exc}")
        raise typer.Exit(1) from exc

    # --- Load optional oracle metadata ---
    oracle = None
    if oracle_path is not None:
        try:
            oracle = load_oracle_metadata(oracle_path)
        except (ValueError, FileNotFoundError) as exc:
            console.print(f"[red]Oracle load error:[/red] {exc}")
            raise typer.Exit(1) from exc

    # --- Build records and report ---
    records = records_from_runs(runs, annotations, oracle)
    if not records:
        console.print(
            "[yellow]No runs matched annotation task IDs. "
            "Check that task_id values align.[/yellow]"
        )
        raise typer.Exit(0)

    report = build_report(records, greedy_arm=greedy_arm)

    if json_output:
        print(json.dumps(report, indent=2, default=str))
        return

    _print_usefulness_report(report, greedy_arm=greedy_arm)


def _fmt_pct(v: float) -> str:
    """Format a fraction as a percentage string."""
    return f"{v * 100:.1f}%"


def _fmt_opt(v: object) -> str:
    """Format an optional numeric value for display."""
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v * 100:.1f}%"
    return str(v)


def _print_usefulness_report(report: dict, *, greedy_arm: str) -> None:
    """Render the benchmark usefulness report to the console."""
    g = report["global"]
    per_arm: dict = report["per_arm"]
    lift_map: dict = report["tribunal_lift_over_greedy"]
    interpretation: dict = report["interpretation"]

    arm_names = g["arms"]
    cohort_order = [
        BenchmarkCohort.control_trivial,
        BenchmarkCohort.contested_recoverable,
        BenchmarkCohort.contested_unrecoverable,
    ]
    cohort_labels = {
        BenchmarkCohort.control_trivial: "control-trivial",
        BenchmarkCohort.contested_recoverable: "contested-recoverable",
        BenchmarkCohort.contested_unrecoverable: "contested-unrecoverable",
    }
    cohort_colors = {
        BenchmarkCohort.control_trivial: "dim",
        BenchmarkCohort.contested_recoverable: "green",
        BenchmarkCohort.contested_unrecoverable: "yellow",
    }

    # ---- Header ----
    console.print()
    console.print(
        f"[bold blue]Tribunal Usefulness Benchmark[/bold blue]  "
        f"[dim]{g['total_records']} records · {len(arm_names)} arm(s)[/dim]"
    )
    console.print()

    # ---- Arm × Cohort selective accuracy matrix ----
    # This is the primary decision-making table.
    matrix = Table(
        title="Selective Accuracy by Arm × Cohort",
        show_header=True,
        header_style="bold",
        box=None,
        pad_edge=False,
    )
    matrix.add_column("Cohort", style="bold", min_width=28)
    for arm in arm_names:
        style = "bold cyan" if arm != greedy_arm else "dim cyan"
        matrix.add_column(arm, justify="right", style=style, min_width=12)

    for cohort in cohort_order:
        label = f"[{cohort_colors[cohort]}]{cohort_labels[cohort]}[/{cohort_colors[cohort]}]"
        row = [label]
        for arm in arm_names:
            arm_data = per_arm.get(arm, {})
            cm = arm_data.get(cohort.value, {})
            sel_acc = cm.get("selective_accuracy")
            n = cm.get("task_count", 0)
            if sel_acc is None or n == 0:
                row.append("—")
            else:
                row.append(f"{_fmt_pct(sel_acc)} ({n}n)")
        matrix.add_row(*row)

    console.print(matrix)
    console.print()

    # ---- Per-cohort detail per arm ----
    for cohort in cohort_order:
        cohort_label = cohort_labels[cohort]
        color = cohort_colors[cohort]

        detail = Table(
            title=f"[{color}]{cohort_label}[/{color}] — detail per arm",
            show_header=True,
            box=None,
            pad_edge=False,
        )
        detail.add_column("Metric", style="cyan", min_width=32)
        for arm in arm_names:
            detail.add_column(arm, justify="right", min_width=12)

        metric_rows = [
            ("Overall accuracy",    "overall_accuracy"),
            ("Selective accuracy",   "selective_accuracy"),
            ("Coverage",             "coverage"),
            ("Wrong-pick rate ↓",    "wrong_pick_rate"),
            ("Abstention rate",      "abstention_rate"),
            ("Good abstention rate ↑", "good_abstention_rate"),
            ("Bad abstention rate ↓",  "bad_abstention_rate"),
        ]

        for label, key in metric_rows:
            row = [label]
            for arm in arm_names:
                cm = per_arm.get(arm, {}).get(cohort.value, {})
                v = cm.get(key)
                row.append(_fmt_opt(v) if v is not None else "—")
            detail.add_row(*row)

        console.print(detail)
        console.print()

    # ---- Lift table (non-greedy arms vs greedy) ----
    if lift_map:
        lift_table = Table(
            title=f"Tribunal Lift over {greedy_arm!r} — contested-recoverable only",
            show_header=True,
            box=None,
            pad_edge=False,
        )
        lift_table.add_column("Arm", style="cyan", min_width=20)
        lift_table.add_column("Sel-Acc Lift", justify="right", min_width=14)
        lift_table.add_column("Verdict", justify="left", min_width=20)

        for arm, lift in lift_map.items():
            if lift is None:
                lift_str = "—"
                verdict = "[dim]no data[/dim]"
            elif lift > 0:
                lift_str = f"[green]+{lift * 100:.1f}%[/green]"
                verdict = "[green]↑ tribunal helps[/green]"
            elif lift == 0:
                lift_str = "0.0%"
                verdict = "[dim]neutral[/dim]"
            else:
                lift_str = f"[red]{lift * 100:.1f}%[/red]"
                verdict = "[red]↓ tribunal hurts[/red]"
            lift_table.add_row(arm, lift_str, verdict)

        console.print(lift_table)
        console.print()

    # ---- Interpretation flags ----
    if interpretation:
        flag_table = Table(
            title="Benchmark Thesis — Interpretation Flags",
            show_header=True,
            box=None,
            pad_edge=False,
        )
        flag_table.add_column("Arm", style="cyan", min_width=20)
        flag_table.add_column("Useful on Recoverable?", justify="center", min_width=24)
        flag_table.add_column("Honest on Unrecoverable?", justify="center", min_width=26)
        flag_table.add_column("Lift", justify="right", min_width=10)

        for arm, flags in interpretation.items():
            useful = flags["tribunal_useful_on_contested_recoverable"]
            honest = flags["tribunal_honest_on_contested_unrecoverable"]
            lift = flags["tribunal_lift_over_greedy_on_contested_recoverable"]

            useful_str = "[bold green]YES[/bold green]" if useful else "[bold red]NO[/bold red]"
            honest_str = "[bold green]YES[/bold green]" if honest else "[bold red]NO[/bold red]"
            lift_str = _fmt_opt(lift)

            flag_table.add_row(arm, useful_str, honest_str, lift_str)

        console.print(flag_table)
        console.print()


if __name__ == "__main__":
    app()
