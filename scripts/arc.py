#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║          Epistemic ARC — Unified Benchmark CLI                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  arc.py run      — Execute a live benchmark with streaming UI       ║
║  arc.py inspect  — Inspect & analyse a saved results file           ║
║  arc.py compare  — Compare two or more result files side-by-side    ║
║  arc.py list     — List all saved result files with quick summary   ║
╚══════════════════════════════════════════════════════════════════════╝

Examples
--------
  # Run the non-reasoning benchmark, auto-open inspector when done
  python3 scripts/arc.py run \\
      --config configs/deepseek_chat_25.yaml \\
      --manifest data/deepseek_custom_25_manifest.txt

  # Run the reasoning benchmark, skip inspector
  python3 scripts/arc.py run \\
      --config configs/deepseek_synthesis_chat.yaml \\
      --no-inspect

  # Inspect the latest results file (auto-detected)
  python3 scripts/arc.py inspect

  # Inspect a specific file
  python3 scripts/arc.py inspect data/deepseek_chat_25_results.json

  # Compare two arms
  python3 scripts/arc.py compare data/deepseek_25_results.json \\
                                  data/deepseek_chat_25_results.json

  # List all saved results
  python3 scripts/arc.py list
"""
from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Bootstrap: ensure src/ is always on sys.path regardless of cwd
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# Rich imports (used by the list subcommand — others delegate to modules)
# ---------------------------------------------------------------------------
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

_console = Console()

# ---------------------------------------------------------------------------
# Subcommand: RUN
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> None:
    """Launch the live benchmark TUI, then optionally open the inspector."""
    from live_benchmark import LiveBenchmarkUI

    dataset = args.dataset or os.environ.get("ARC_DATASET_PATH", "")
    if not dataset:
        _console.print("[red]Error:[/] --dataset not supplied and ARC_DATASET_PATH env var is not set.")
        _console.print("  Set it with: [bold]export ARC_DATASET_PATH=/path/to/dataset[/]")
        sys.exit(1)

    ui = LiveBenchmarkUI(
        config_path=args.config,
        dataset_path=dataset,
        manifest_path=args.manifest,
    )
    results_path = ui.run()

    if not args.no_inspect and results_path:
        _console.print()
        _console.rule("[bold cyan]⟶  Stage 2: Results Inspector[/]")
        _console.print()
        # Re-use this process — just call inspect directly
        from inspect_results import _display_single
        path = Path(results_path)
        with path.open() as f:
            data = json.load(f)
        _display_single(path, data)


# ---------------------------------------------------------------------------
# Subcommand: INSPECT
# ---------------------------------------------------------------------------

def cmd_inspect(args: argparse.Namespace) -> None:
    """Inspect a single results JSON with full Rich panels."""
    from inspect_results import _display_single

    path = _resolve_results_path(args.file)
    with path.open() as f:
        data = json.load(f)
    _display_single(path, data)


# ---------------------------------------------------------------------------
# Subcommand: COMPARE
# ---------------------------------------------------------------------------

def cmd_compare(args: argparse.Namespace) -> None:
    """Side-by-side comparison of two or more results files."""
    from inspect_results import _display_compare

    if len(args.files) < 2:
        _console.print("[red]compare requires at least 2 result files.[/]")
        _console.print("  Example: [bold]arc.py compare data/a.json data/b.json[/]")
        sys.exit(1)

    loaded = []
    for fp in args.files:
        path = Path(fp)
        if not path.exists():
            _console.print(f"[red]File not found:[/] {path}")
            sys.exit(1)
        with path.open() as f:
            loaded.append((path, json.load(f)))

    _display_compare(loaded)


# ---------------------------------------------------------------------------
# Subcommand: LIST
# ---------------------------------------------------------------------------

def cmd_list(args: argparse.Namespace) -> None:
    """List all saved *_results.json files with a quick summary table."""
    data_dir = _REPO_ROOT / "data"
    files = sorted(data_dir.glob("*_results.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not files:
        _console.print("[yellow]No results files found in data/[/]")
        _console.print("  Run a benchmark first: [bold]arc.py run --config ...[/]")
        return

    t = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold white",
        expand=True,
        title="[bold cyan]Saved Benchmark Results[/]",
    )
    t.add_column("#",          justify="right",   style="dim",        min_width=3)
    t.add_column("File",       justify="left",                        min_width=32)
    t.add_column("Arm",        justify="left",    style="dim",        min_width=24)
    t.add_column("Tasks",      justify="right",                       min_width=7)
    t.add_column("Accuracy",   justify="right",                       min_width=10)
    t.add_column("Coverage",   justify="right",                       min_width=10)
    t.add_column("Dur/task",   justify="right",                       min_width=10)
    t.add_column("Modified",   justify="right",   style="dim",        min_width=20)

    from datetime import datetime

    def _pct_text(v: float, good: bool = True) -> Text:
        if v >= 0.7 and good:
            return Text(f"{v:.1%}", style="bold green")
        if v >= 0.4:
            return Text(f"{v:.1%}", style="bold yellow")
        return Text(f"{v:.1%}", style="bold red" if good else "white")

    for idx, path in enumerate(files, 1):
        mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        try:
            with path.open() as f:
                data = json.load(f)
            arm, metrics = next(iter(data.items()))
            tasks    = metrics.get("total_runs", "?")
            acc      = metrics.get("overall_accuracy", 0.0)
            cov      = metrics.get("coverage", 0.0)
            dur      = metrics.get("avg_duration_seconds", 0.0)
        except Exception:
            arm, tasks, acc, cov, dur = "parse error", "?", 0.0, 0.0, 0.0

        t.add_row(
            str(idx),
            path.name,
            arm,
            str(tasks),
            _pct_text(acc),
            _pct_text(cov),
            Text(f"{dur:.1f}s"),
            mtime,
        )

    _console.print()
    _console.print(t)
    _console.print()
    _console.print("[dim]Inspect with:[/]  [bold]python3 scripts/arc.py inspect data/<file>[/]")
    _console.print("[dim]Compare with:[/]  [bold]python3 scripts/arc.py compare data/a.json data/b.json[/]")
    _console.print()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_results_path(file_arg: Optional[str]) -> Path:
    """Return a Path to the results file, auto-detecting the latest if not given."""
    if file_arg:
        path = Path(file_arg)
        if not path.exists():
            _console.print(f"[red]File not found:[/] {path}")
            sys.exit(1)
        return path

    data_dir = _REPO_ROOT / "data"
    candidates = sorted(data_dir.glob("*_results.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        _console.print("[red]No *_results.json files found in data/. Run the benchmark first.[/]")
        sys.exit(1)
    latest = candidates[0]
    _console.print(f"[dim]Auto-selected: [bold]{latest.name}[/][/]")
    return latest


# ---------------------------------------------------------------------------
# Splash banner
# ---------------------------------------------------------------------------

def _print_banner() -> None:
    _console.print(Panel(
        "[bold cyan]Epistemic ARC[/] — Unified Benchmark CLI\n"
        "[dim]DeepSeek · Tribunal · Calibration · Comparison[/]",
        border_style="cyan",
        padding=(0, 4),
    ))


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arc.py",
        description="Epistemic ARC — Unified Benchmark CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ── run ──────────────────────────────────────────────────────────────
    p_run = sub.add_parser(
        "run",
        help="Execute a live benchmark with streaming UI",
        description=(
            "Run the Epistemic Tribunal benchmark against a task manifest.\n"
            "Streams token output and tribunal telemetry in real-time.\n"
            "Automatically opens the inspector when complete."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_run.add_argument(
        "--config", "-c",
        default="configs/deepseek_custom_25.yaml",
        metavar="PATH",
        help="Path to the benchmark config YAML  (default: configs/deepseek_custom_25.yaml)",
    )
    p_run.add_argument(
        "--manifest", "-m",
        default="data/phase3_test_manifest.txt",
        metavar="PATH",
        help="Path to the task manifest .txt file  (default: data/phase3_test_manifest.txt)",
    )
    p_run.add_argument(
        "--dataset", "-d",
        default=None,
        metavar="PATH",
        help="Path to the ARC dataset root  (default: $ARC_DATASET_PATH env var)",
    )
    p_run.add_argument(
        "--no-inspect",
        action="store_true",
        help="Skip the Stage 2 results inspector after the benchmark completes",
    )
    p_run.set_defaults(func=cmd_run)

    # ── inspect ──────────────────────────────────────────────────────────
    p_inspect = sub.add_parser(
        "inspect",
        help="Inspect & analyse a saved results file",
        description=(
            "Render a full Rich TUI dashboard from a saved *_results.json file.\n"
            "Shows accuracy, error breakdown, decision distribution, calibration,\n"
            "Path-B gate stats, and a heuristic diagnostic summary.\n\n"
            "If no file is given, the most recently modified *_results.json is used."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_inspect.add_argument(
        "file",
        nargs="?",
        default=None,
        metavar="FILE",
        help="Path to results JSON  (optional — auto-detects latest if omitted)",
    )
    p_inspect.set_defaults(func=cmd_inspect)

    # ── compare ──────────────────────────────────────────────────────────
    p_compare = sub.add_parser(
        "compare",
        help="Compare two or more result files side-by-side",
        description=(
            "Render a cross-arm comparison table across two or more results files.\n"
            "Each arm's accuracy, coverage, error counts, and duration are displayed\n"
            "in a single aligned table with per-arm diagnostic summaries below."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_compare.add_argument(
        "files",
        nargs="+",
        metavar="FILE",
        help="Two or more results JSON file paths to compare",
    )
    p_compare.set_defaults(func=cmd_compare)

    # ── list ─────────────────────────────────────────────────────────────
    p_list = sub.add_parser(
        "list",
        help="List all saved result files with a quick summary",
        description=(
            "Scan data/ for all *_results.json files and display them in a\n"
            "summary table sorted by modification time (newest first).\n"
            "Shows accuracy, coverage, task count, and avg duration per file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_list.set_defaults(func=cmd_list)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Ensure scripts/ is on the path so local imports (live_benchmark, inspect_results) work
    _SCRIPTS = Path(__file__).parent
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))

    parser = build_parser()

    # Print banner for all commands except when piped
    import sys as _sys
    if _sys.stdout.isatty():
        _print_banner()

    args = parser.parse_args()
    args.func(args)
