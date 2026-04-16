#!/usr/bin/env python3
"""
Interactive ARC Benchmark Runner with Streaming Reasoning Traces.
Uses `rich.live` to provide a real-time dashboard of the tribunal execution.
"""
import os
import sys
import json
import time
import argparse
import logging
from collections import deque
from pathlib import Path
from typing import Optional, List

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.text import Text
from rich.markup import escape
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.markdown import Markdown

# Ensure src/ is in PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent / "src"))

from epistemic_tribunal.config import load_config
from epistemic_tribunal.evaluation.benchmark import BenchmarkRunner
from epistemic_tribunal.orchestrator import Orchestrator
from epistemic_tribunal.tasks.arc_like import load_task_from_file
from epistemic_tribunal.evaluation.metrics import summary_report
from epistemic_tribunal.tribunal_types import ExperimentRun, DecisionKind

class TribunalLogHandler(logging.Handler):
    """Custom logging handler to hook into Live UI."""
    def __init__(self, ui):
        super().__init__()
        self.ui = ui
        
    def emit(self, record):
        msg = self.format(record)
        self.ui.add_tribunal_log(msg)

class LiveBenchmarkUI:
    def __init__(self, config_path: str, dataset_path: str, manifest_path: str):
        self.console = Console()
        self.config_path = config_path  # Added for save generation logic
        self.config = load_config(config_path)
        self.dataset_path = Path(dataset_path)
        self.manifest_path = Path(manifest_path)
        
        # Clear ledger for clean run
        db = Path(self.config.ledger.path)
        if db.exists():
            db.unlink()
            
        self.orchestrator = Orchestrator(self.config)
        self.runs: List[ExperimentRun] = []
        self.current_task_id = "Initialising..."
        self.current_reasoning = ""
        self.current_content = ""
        
        # Timing
        self.start_time = time.monotonic()
        
        # Telemetry
        self.tribunal_logs = deque(maxlen=20)
        self.log_handler = TribunalLogHandler(self)
        self.log_handler.setFormatter(logging.Formatter('[%(name)s] %(message)s'))
        self.log_handler.setLevel(logging.INFO)
        
        # All back-end modules use get_logger(__name__), creating per-module child
        # loggers with propagate=False (see utils/logging.py). We must attach
        # directly to every existing child AND monkey-patch get_logger so future
        # child loggers also receive our handler automatically.
        self._attach_handler_to_all_epistemic_loggers()

    def _attach_handler_to_all_epistemic_loggers(self):
        """Attach the TribunalLogHandler to every current and future child logger."""
        # 1. Attach to all already-initialised child loggers
        for name, logger in logging.Logger.manager.loggerDict.items():
            if name.startswith("epistemic_tribunal") and isinstance(logger, logging.Logger):
                # Avoid duplicates
                if self.log_handler not in logger.handlers:
                    logger.addHandler(self.log_handler)
        
        # 2. Monkey-patch get_logger so future loggers also get the handler
        import epistemic_tribunal.utils.logging as _et_logging
        _original_get_logger = _et_logging.get_logger
        _handler_ref = self.log_handler
        
        def _patched_get_logger(name="epistemic_tribunal", level=None):
            logger = _original_get_logger(name, level)
            if _handler_ref not in logger.handlers:
                logger.addHandler(_handler_ref)
            return logger
        
        _et_logging.get_logger = _patched_get_logger
        
    def add_tribunal_log(self, text: str):
        self.tribunal_logs.append(text)
        if hasattr(self, 'layout'):
            self.layout["tribunal"].update(self.make_tribunal_panel())

    def on_token(self, token_type: str, text: str):
        """Callback from Orchestrator/Generator."""
        if token_type == "reasoning":
            self.current_reasoning += text
        else:
            self.current_content += text
            
        if hasattr(self, 'layout'):
            self.layout["streaming"].update(self.make_reasoning_panel())

    def make_header(self) -> Panel:
        grid = Table.grid(expand=True)
        grid.add_column(justify="left", ratio=1)
        grid.add_column(justify="right")
        
        elapsed = time.monotonic() - self.start_time
        grid.add_row(
            Text.from_markup(f"[bold blue]ARC Epistemic Benchmark — [white]{self.config.generators.llm.model_name}[/]"),
            Text.from_markup(f"[cyan]Elapsed: [white]{elapsed:.1f}s[/]")
        )
        return Panel(grid, style="blue")

    def make_metrics_table(self) -> Table:
        metrics = summary_report(self.runs) if self.runs else {}
        table = Table(title="Overall Metrics")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        
        table.add_row("Tasks Run", str(len(self.runs)))
        table.add_row("Accuracy", f"{metrics.get('overall_accuracy', 0.0):.1%}")
        table.add_row("Coverage", f"{metrics.get('coverage', 0.0):.1%}")
        table.add_row("Resample Rate", f"{metrics.get('resample_rate', 0.0):.1%}")
        
        # Failure breakdown
        table.add_row("[red]Wrong Picks", str(metrics.get("wrong_pick_count", 0)), style="dim")
        table.add_row("[red]Parse Fails", str(metrics.get("parse_failure_count", 0)), style="dim")
        
        return table

    def make_reasoning_panel(self) -> Panel:
        # Support display logic for both reasoning and non-reasoning endpoints
        sections = []
        if self.current_reasoning:
            sections.append(f"[bold cyan]Reasoning Trace:[/]\n[dim]{escape(self.current_reasoning)}[/]\n")
        if self.current_content:
            sections.append(f"[bold green]LLM Content Stream:[/]\n{escape(self.current_content)}")
            
        display_text = "\n".join(sections)
        if not display_text:
            display_text = "[italic dim]Awaiting LLM generation tokens...[/italic dim]"
            
        # Limit display size gracefully avoiding massive terminal hang breaks
        lines = display_text.splitlines()
        if len(lines) > 30:
            display_text = "...\n" + "\n".join(lines[-30:])
            
        return Panel(
            Text.from_markup(display_text),
            title=f"Terminal Output: [yellow]{self.current_task_id}[/]",
            border_style="yellow",
            padding=(1, 1),
            expand=True
        )

    def make_tribunal_panel(self) -> Panel:
        escaped_logs = [escape(str(log)) for log in self.tribunal_logs]
        display_text = "\n".join(escaped_logs) if escaped_logs else "[dim]Awaiting tribunal decisions...[/]"
        return Panel(
            Text.from_markup(display_text, emoji=False),
            title="[blue]Tribunal Telemetry[/]",
            border_style="blue",
            padding=(1, 1),
            expand=True
        )

    def make_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        layout["main"].split_row(
            Layout(name="left_column", ratio=1),
            Layout(name="streaming", ratio=2)
        )
        layout["left_column"].split_column(
            Layout(name="metrics", ratio=1),
            Layout(name="tribunal", ratio=2)
        )
        return layout

    def run(self):
        # Load tasks
        with self.manifest_path.open("r") as f:
            task_ids = [line.strip() for line in f if line.strip()]
            
        self.layout = self.make_layout()
        
        with Live(self.layout, refresh_per_second=10, screen=True) as live:
            try:
                for idx, tid in enumerate(task_ids):
                    self.current_task_id = tid
                    self.current_reasoning = ""
                    self.current_content = ""
                    
                    # Update UI
                    self.layout["header"].update(self.make_header())
                    self.layout["footer"].update(Panel(Text(f"Task {idx+1}/{len(task_ids)} — {tid}"), style="dim"))
                    self.layout["metrics"].update(self.make_metrics_table())
                    self.layout["tribunal"].update(self.make_tribunal_panel())
                    self.layout["streaming"].update(self.make_reasoning_panel())
                    
                    # Find task file
                    matches = list(self.dataset_path.glob(f"**/{tid}.json"))
                    if not matches:
                        continue
                    
                    task = load_task_from_file(matches[0])
                    
                    # Execute orchestrated run with streaming callback
                    try:
                        run = self.orchestrator.run(task, on_token=self.on_token)
                        self.runs.append(run)
                    except Exception as e:
                        # Log failure to reasoning pane for visibility
                        self.current_reasoning += f"\n[bold red]FATAL CRASH on task {tid}: {escape(str(e))}[/]"
                        self.layout["streaming"].update(self.make_reasoning_panel())
                        time.sleep(2) # Let user see it
                        
                    # Update metrics
                    self.layout["metrics"].update(self.make_metrics_table())
                    self.layout["tribunal"].update(self.make_tribunal_panel())
                    self.layout["streaming"].update(self.make_reasoning_panel())
            except KeyboardInterrupt:
                self.current_reasoning += "\n\n[bold yellow blink]KeyboardInterrupt detected! Safely exiting and saving partial benchmark results...[/]"
                self.layout["streaming"].update(self.make_reasoning_panel())
                try:
                    time.sleep(1.5)
                except KeyboardInterrupt:
                    pass

            # Final Summary
            live.stop()
            self.console.print("\n[bold green]Benchmark Complete![/bold green]")
            self.console.print(self.make_metrics_table())
            
            # Save final results
            results_path = None
            if self.runs:
                metrics = summary_report(self.runs)
                
                # Derive output path from config base name so we don't overwrite blindly
                config_name = str(self.config_path).split('/')[-1].replace('.yaml', '')
                results_path = f"data/{config_name}_results.json"
                
                with open(results_path, "w") as f:
                    json.dump({f"DeepSeek ({config_name})": metrics}, f, indent=2)
                self.console.print(f"[green]Results saved → {results_path}[/]")
            else:
                self.console.print("[dim]No tasks completed. Metrics were aborted.[/]")
            
            return results_path

