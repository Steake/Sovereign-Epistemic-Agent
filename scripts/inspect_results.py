#!/usr/bin/env python3
"""
Epistemic ARC — Stage 2 Results Inspector
==========================================
A standalone Rich TUI for inspecting saved benchmark results JSON files.
Renders accuracy, error breakdown, calibration metrics, decision distribution,
and per-failure classification — all in a pretty terminal dashboard.

Usage:
    python3 scripts/inspect_results.py                    # auto-pick latest file
    python3 scripts/inspect_results.py data/my_results.json
    python3 scripts/inspect_results.py --compare data/a.json data/b.json
"""
import sys
import json
import argparse
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.rule import Rule
from rich.columns import Columns
from rich import box

console = Console()

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _pct_colour(value: float, thresholds=(0.4, 0.7)) -> str:
    """Red / yellow / green based on value."""
    if value >= thresholds[1]:
        return "bold green"
    if value >= thresholds[0]:
        return "bold yellow"
    return "bold red"

def _pct(value: Optional[float], colour: bool = True) -> Text:
    if value is None:
        return Text("N/A", style="dim")
    pct = value if value <= 1.0 else value / 100.0
    txt = f"{pct:.1%}"
    style = _pct_colour(pct) if colour else "white"
    return Text(txt, style=style)

def _num(value, style="white") -> Text:
    return Text(str(value), style=style)

# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _accuracy_panel(data: dict, arm: str) -> Panel:
    t = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold magenta", expand=True)
    t.add_column("Metric", style="dim", min_width=28)
    t.add_column("Value", justify="right", min_width=12)

    t.add_row("Overall Accuracy",   _pct(data.get("overall_accuracy")))
    t.add_row("Resolved Accuracy",  _pct(data.get("resolved_accuracy")))
    t.add_row("Coverage",           _pct(data.get("coverage")))
    t.add_row("Abstention Rate",    _pct(data.get("abstention_rate"), colour=False))
    t.add_row("Resample Rate",      _pct(data.get("resample_rate"),   colour=False))
    t.add_row("Mean Confidence",    _pct(data.get("mean_confidence"), colour=False))
    t.add_row("Avg Duration (s)",   _num(f"{data.get('avg_duration_seconds', 0):.2f}"))
    t.add_row("Total Tasks",        _num(data.get("total_runs", 0), "bold white"))

    if "ece" in data:
        t.add_section()
        t.add_row("ECE ↓",          _num(f"{data['ece']:.4f}",       "cyan"))
        t.add_row("Brier Score ↓",  _num(f"{data['brier_score']:.4f}", "cyan"))

    return Panel(t, title=f"[bold green]Accuracy & Coverage — [white]{arm}[/]", border_style="green", padding=(0, 1))


def _error_panel(data: dict) -> Panel:
    t = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold red", expand=True)
    t.add_column("Error Category", style="dim", min_width=26)
    t.add_column("Count", justify="right", min_width=8)

    rows = [
        ("Wrong Picks",             data.get("wrong_pick_count", 0),         "red"),
        ("Grid Shape Invalid",      data.get("grid_shape_invalid_count", 0), "red"),
        ("JSON Not Found",          data.get("json_not_found_count", 0),     "orange1"),
        ("JSON Invalid",            data.get("json_invalid_count", 0),       "orange1"),
        ("Parse Failures",          data.get("parse_failure_count", 0),      "orange1"),
        ("Reasoning Bleed",         data.get("reasoning_bleed_count", 0),    "yellow"),
        ("Token Truncation",        data.get("truncation_count", 0),         "yellow"),
    ]
    for label, count, colour in rows:
        style = colour if count > 0 else "dim"
        t.add_row(label, Text(str(count), style=style))

    return Panel(t, title="[bold red]Error Breakdown", border_style="red", padding=(0, 1))


def _decision_distribution_panel(data: dict, total: int) -> Panel:
    dist: dict = data.get("decision_distribution", {})
    t = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold blue", expand=True)
    t.add_column("Decision", style="dim", min_width=12)
    t.add_column("Count", justify="right", min_width=6)
    t.add_column("Share", justify="right", min_width=8)
    t.add_column("Bar", min_width=20)

    colour_map = {"select": "green", "abstain": "red", "resample": "yellow"}
    for kind, count in sorted(dist.items(), key=lambda x: -x[1]):
        share = count / total if total else 0.0
        colour = colour_map.get(kind, "white")
        bar_width = max(1, int(share * 20))
        bar_str = "█" * bar_width
        t.add_row(
            Text(kind.upper(), style=f"bold {colour}"),
            Text(str(count)),
            Text(f"{share:.1%}"),
            Text(bar_str, style=colour),
        )

    return Panel(t, title="[bold blue]Decision Distribution", border_style="blue", padding=(0, 1))


def _path_b_panel(data: dict) -> Panel:
    t = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold magenta", expand=True)
    t.add_column("Path-B Gate", style="dim", min_width=22)
    t.add_column("Count", justify="right")

    rows = [
        ("Met Gate Potential",  data.get("path_b_met_gate", 0),      "cyan"),
        ("Failed V threshold",  data.get("path_b_failed_v", 0),       "red"),
        ("Failed C threshold",  data.get("path_b_failed_c", 0),       "red"),
        ("Failed Margin",       data.get("path_b_failed_margin", 0),  "orange1"),
        ("Failed Violations",   data.get("path_b_failed_violations", 0), "orange1"),
        ("Override Triggered",  data.get("override_count", 0),        "green"),
    ]
    for label, count, colour in rows:
        style = colour if count > 0 else "dim"
        t.add_row(label, Text(str(count), style=style))

    return Panel(t, title="[bold magenta]Path-B Structural Override", border_style="magenta", padding=(0, 1))


def _calibration_panel(data: dict) -> Optional[Panel]:
    if "selective_accuracy_90" not in data:
        return None
    
    sa90 = data["selective_accuracy_90"]
    aq = data.get("abstention_quality", {})
    
    t = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold cyan", expand=True)
    t.add_column("Metric", style="dim", min_width=28)
    t.add_column("Value", justify="right")

    t.add_row("Sel. Accuracy @ 90% cov.",  _pct(sa90.get("accuracy")))
    t.add_row("Tasks at 90% cov.",         _num(sa90.get("tasks_included", "N/A")))
    if aq:
        t.add_section()
        t.add_row("Abstention Precision",  _pct(aq.get("abstention_precision")))
        t.add_row("Abstention Recall",     _pct(aq.get("abstention_recall")))

    return Panel(t, title="[bold cyan]Calibration", border_style="cyan", padding=(0, 1))


def _diagnosis_panel(data: dict) -> Panel:
    """Heuristic text summary of the most likely problem."""
    total = data.get("total_runs", 0)
    accuracy = data.get("overall_accuracy", 0.0)
    coverage = data.get("coverage", 0.0)
    grid_errs = data.get("grid_shape_invalid_count", 0)
    resample_rate = data.get("resample_rate", 0.0)
    abstention_rate = data.get("abstention_rate", 0.0)

    lines = []

    if coverage == 0.0:
        lines.append("[bold red]⚠ ZERO coverage: the tribunal abstained or resampled on every task.[/]")
        lines.append("  → The model never produced a valid answer that passed tribunal gates.")

    if grid_errs > total * 0.3:
        lines.append(f"[yellow]▸ Grid shape failures dominate ({grid_errs}/{total} tasks).[/]")
        lines.append("  → Consider relaxing shape validation or increasing max_new_tokens.")

    if resample_rate > 0.5:
        lines.append(f"[yellow]▸ High resample rate ({resample_rate:.0%}): tribunal is requesting more diversity.[/]")
        lines.append("  → With only 1 generator, resample gate can never be met. Try adjudication_strategy: greedy.")

    if abstention_rate > 0.3:
        lines.append(f"[orange1]▸ High abstention ({abstention_rate:.0%}): tribunal is refusing uncertain answers.[/]")
        lines.append("  → Confidence thresholds may be too strict for single-generator mode.")

    if accuracy > 0.5:
        lines.append("[bold green]✓ Accuracy looks promising — good signal generation![/]")
    elif accuracy == 0.0 and coverage > 0.0:
        lines.append("[bold red]⚠ Zero accuracy but selections were made — systematic prediction error.[/]")

    if not lines:
        lines.append("[dim]No specific anomalies detected.[/]")

    text = "\n".join(lines)
    return Panel(
        Text.from_markup(text),
        title="[bold white]Diagnostic Summary",
        border_style="white",
        padding=(1, 2),
    )


# ---------------------------------------------------------------------------
# Comparison mode
# ---------------------------------------------------------------------------

def _compare_table(results) -> Table:
    """Side-by-side comparison of two or more results files."""
    keys = [
        ("overall_accuracy",       "Overall Accuracy",  True),
        ("resolved_accuracy",      "Resolved Accuracy", True),
        ("coverage",               "Coverage",          True),
        ("abstention_rate",        "Abstention Rate",   False),
        ("resample_rate",          "Resample Rate",     False),
        ("mean_confidence",        "Mean Confidence",   False),
        ("grid_shape_invalid_count","Grid Errs",        False),
        ("wrong_pick_count",       "Wrong Picks",       False),
        ("avg_duration_seconds",   "Avg Dur (s)",       False),
        ("total_runs",             "Total Tasks",       False),
    ]
    
    t = Table(box=box.DOUBLE_EDGE, show_header=True, header_style="bold white", expand=True)
    t.add_column("Metric", style="dim", min_width=22)
    for _, arm, _ in results:
        t.add_column(arm, justify="right", min_width=14)

    for key, label, is_acc in keys:
        row_cells = [Text(label)]
        for _, _, data in results:
            val = data.get(key)
            if isinstance(val, float):
                row_cells.append(_pct(val, colour=is_acc) if key not in ("avg_duration_seconds",) else _num(f"{val:.2f}"))
            else:
                row_cells.append(_num(val or 0))
        t.add_row(*row_cells)

    return t


# ---------------------------------------------------------------------------
# Main display logic
# ---------------------------------------------------------------------------

def _display_single(path: Path, data: dict):
    arm, metrics = next(iter(data.items()))
    total = metrics.get("total_runs", 0)

    console.print()
    console.print(Rule(f"[bold green] Epistemic ARC — Results Inspector [/] · [dim]{path.name}[/]"))
    console.print()

    # Row 1: accuracy + error + decisions
    row1 = Columns([
        _accuracy_panel(metrics, arm),
        _error_panel(metrics),
        _decision_distribution_panel(metrics, total),
    ], equal=True, expand=True)
    console.print(row1)

    # Row 2: Path-B + calibration (if available)
    panels_r2 = [_path_b_panel(metrics)]
    cal = _calibration_panel(metrics)
    if cal:
        panels_r2.append(cal)
    console.print(Columns(panels_r2, equal=True, expand=True))

    # Row 3: diagnosis
    console.print(_diagnosis_panel(metrics))
    console.print()


def _display_compare(paths_and_data: list[tuple[Path, dict]]):
    console.print()
    console.print(Rule("[bold white] Epistemic ARC — Multi-Arm Comparison [/]"))
    console.print()

    results = []
    for path, data in paths_and_data:
        arm, metrics = next(iter(data.items()))
        results.append((path.name, arm, metrics))

    t = _compare_table(results)
    console.print(Panel(t, title="[bold white]Cross-Arm Comparison", border_style="white", padding=(0, 1)))

    # Per-arm diagnosis
    for path, data in paths_and_data:
        arm, metrics = next(iter(data.items()))
        console.print(_diagnosis_panel(metrics))

    console.print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Epistemic ARC Stage 2 Results Inspector")
    parser.add_argument("files", nargs="*", help="Path(s) to results JSON file(s). Auto-detects latest if omitted.")
    parser.add_argument("--compare", action="store_true", help="Force comparison mode across all supplied files.")
    args = parser.parse_args()

    data_dir = Path("data")

    # Auto-discover latest results file if nothing supplied
    if not args.files:
        candidates = sorted(data_dir.glob("*_results.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            console.print("[red]No *_results.json files found in data/. Run the benchmark first.[/]")
            sys.exit(1)
        selected = candidates[0]
        console.print(f"[dim]Auto-selected latest results file: [bold]{selected}[/][/]")
        args.files = [str(selected)]

    loaded: list[tuple[Path, dict]] = []
    for fp in args.files:
        path = Path(fp)
        if not path.exists():
            console.print(f"[red]File not found: {path}[/]")
            sys.exit(1)
        with path.open() as f:
            loaded.append((path, json.load(f)))

    if len(loaded) == 1 and not args.compare:
        _display_single(loaded[0][0], loaded[0][1])
    else:
        _display_compare(loaded)


if __name__ == "__main__":
    main()
