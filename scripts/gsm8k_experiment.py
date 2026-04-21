#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSM8K Tribunal Experiment - Live TUI
=====================================
Runs the GSM8K benchmark with the LLM-as-a-judge critic enabled,
showing real-time resampling loops, judge rubric scores, and
a live decision dashboard.

Usage:
    python scripts/gsm8k_experiment.py [--questions N] [--judge/--no-judge]

Options:
    --questions N     Number of GSM8K questions to run (default: 5)
    --judge           Enable LLM-as-a-judge (default: True)
    --no-judge        Disable LLM-as-a-judge (A/B comparison mode)
    --config PATH     Config file path (default: configs/gsm8k_sample.yaml)
    --out PATH        Results output path (default: data/gsm8k/results/)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

from rich.markup import escape
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    RichLog,
    Static,
)

# ---------------------------------------------------------------------------
# Logging bridge — pipes tribunal logs into the TUI
# ---------------------------------------------------------------------------

class TribunalLogHandler(logging.Handler):
    """Captures tribunal log records and routes them into the TUI telemetry panel."""

    _SUBSYSTEM_STYLES: dict[str, tuple[str, str]] = {
        "generator":    ("bright_green",  "⚡"),
        "orchestrator": ("bright_cyan",   "🧠"),
        "aggregator":   ("bright_magenta","⚖️"),
        "trace_critic": ("bright_yellow", "🔍"),
        "uncertainty":  ("bright_blue",   "📊"),
        "ledger":       ("dim white",     "💾"),
    }

    def __init__(self, app: "GSM8KExperimentApp") -> None:
        super().__init__()
        self.app = app

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if "Initialized remote API client" in msg:
            return  # suppress noisy init line

        subsystem = record.name.split(".")[-1]
        color, icon = self._SUBSYSTEM_STYLES.get(subsystem, ("white", "ℹ"))
        level_tag = "[bold red]ERR [/]" if record.levelno >= logging.ERROR else ""
        formatted = f"[{color}]{icon} {subsystem:<14}[/] {level_tag}{escape(msg)}"
        self.app.call_from_thread(self.app.push_telemetry, formatted)


# ---------------------------------------------------------------------------
# TUI App
# ---------------------------------------------------------------------------

class GSM8KExperimentApp(App):
    """Live experiment dashboard for the GSM8K Tribunal benchmark."""

    TITLE = "GSM8K Tribunal Experiment"

    BINDINGS = [
        ("[", "shrink_bottom", "Shrink Bottom Panel"),
        ("]", "grow_bottom", "Grow Bottom Panel"),
    ]

    CSS = """
    /* ── Layout ─────────────────────────────────────────────── */
    Screen { layout: vertical; }

    #body { height: 1fr; layout: vertical; }

    #top_row { 
        height: 1fr; 
        border-bottom: solid $primary-darken-2;
    }
    
    #bottom_row { 
        height: 18; 
        min-height: 5;
    }

    #metrics_container {
        width: 44;
        border-right: solid $primary-darken-2;
        background: $surface;
    }

    /* ── Metric table ────────────────────────────────────────── */
    #metrics_table {
        height: 1fr;
    }

    /* ── Telemetry feed ──────────────────────────────────────── */
    #telemetry_log {
        width: 1fr;
        height: 1fr;
        padding: 0 1;
        background: $surface-darken-1;
    }

    /* ── Current question panel ──────────────────────────────── */
    #question_panel {
        height: auto;
        padding: 1 2;
        background: $boost;
        border-bottom: solid $primary-darken-2;
    }

    #question_text {
        color: $text;
        text-style: italic;
    }

    /* ── Attempt stream ──────────────────────────────────────── */
    #attempt_header {
        height: 3;
        align: center middle;
        background: $surface;
        border-bottom: solid $accent;
        padding: 0 2;
    }

    #attempt_badge {
        color: $accent;
        text-style: bold;
        margin-right: 2;
    }

    #phase_badge { color: $warning; }

    #stream_scroll { height: 1fr; padding: 1 2; }
    #stream_content { }

    /* ── Judge score bar ─────────────────────────────────────── */
    #judge_panel {
        height: auto;
        border-top: solid $accent-darken-2;
        padding: 1 2;
        background: $surface-darken-2;
    }

    /* ── Decision banner ─────────────────────────────────────── */
    #decision_banner {
        height: 3;
        align: center middle;
        border-top: solid $primary;
    }
    """

    def __init__(self, config_path: str, data_path: str, n_questions: int, use_judge: bool, out_dir: str, ledger_override: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config_path = config_path
        self.data_path = Path(data_path)
        self.n_questions = n_questions
        self.use_judge = use_judge
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_override = ledger_override
        self.active_ledger_path: str | None = None

        # Runtime state
        self.runs: list = []
        self.current_q_text: str = ""
        self.current_attempt: int = 0
        self.current_phase: str = "Initialising"
        self.current_trace_text: str = ""
        self.judge_scores: dict[str, float] = {}
        self.last_decision: str = "—"
        self.last_gt_match: Optional[bool] = None

        # Log bridge
        self._log_handler = TribunalLogHandler(self)
        self._log_handler.setLevel(logging.DEBUG)
        self._attach_log_handler()

    def _attach_log_handler(self) -> None:
        """Wire the TUI handler into all tribunal loggers."""
        for name, logger in logging.Logger.manager.loggerDict.items():
            if name.startswith("epistemic_tribunal") and isinstance(logger, logging.Logger):
                logger.handlers = [self._log_handler]
        import epistemic_tribunal.utils.logging as _et_logging
        _orig = _et_logging.get_logger
        def _patched(name="epistemic_tribunal", level=None):
            lg = _orig(name, level)
            if self._log_handler not in lg.handlers:
                lg.handlers = [self._log_handler]
            return lg
        _et_logging.get_logger = _patched

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="body"):
            # Top: current task view
            with Vertical(id="top_row"):
                yield Static("", id="question_panel")
                with Horizontal(id="attempt_header"):
                    yield Static("", id="attempt_badge")
                    yield Static("", id="phase_badge")
                with VerticalScroll(id="stream_scroll"):
                    yield Static("", id="stream_content")
                yield Static("", id="judge_panel")
                yield Static("", id="decision_banner")
            
            # Bottom: metrics + log (full width)
            with Horizontal(id="bottom_row"):
                with Vertical(id="metrics_container"):
                    yield DataTable(id="metrics_table", show_cursor=False)
                yield RichLog(id="telemetry_log", markup=True, highlight=True, wrap=True, max_lines=400)
        yield Footer()

    def on_mount(self) -> None:
        tbl = self.query_one("#metrics_table", DataTable)
        tbl.add_column("Metric", width=22)
        tbl.add_column("Value", width=12)
        self._refresh_metrics()
        self.sub_title = f"{self.n_questions} questions · judge={'ON' if self.use_judge else 'OFF'}"
        self.run_experiment_thread()

    def action_shrink_bottom(self) -> None:
        bottom = self.query_one("#bottom_row")
        h = bottom.styles.height.value if bottom.styles.height else 18
        if h > 5:
            bottom.styles.height = h - 2

    def action_grow_bottom(self) -> None:
        bottom = self.query_one("#bottom_row")
        h = bottom.styles.height.value if bottom.styles.height else 18
        if h < 40:
            bottom.styles.height = h + 2

    # ------------------------------------------------------------------
    # Metric table
    # ------------------------------------------------------------------

    def _refresh_metrics(self) -> None:
        from epistemic_tribunal.evaluation.metrics import summary_report
        from epistemic_tribunal.ledger.store import LedgerStore
        tbl = self.query_one("#metrics_table", DataTable)
        tbl.clear(columns=False)
        runs = self.runs

        if not runs:
            tbl.add_row("Questions Run", "0")
            tbl.add_row("Accuracy (sel.)", "—")
            tbl.add_row("Coverage", "—")
            tbl.add_row("Abstentions", "—")
            tbl.add_row("Avg Duration", "—")
            return

        coalition_rows = None
        if self.active_ledger_path:
            store = LedgerStore(self.active_ledger_path)
            try:
                coalition_rows = store.get_coalition_opinions(run_ids=[r.run_id for r in runs])
            finally:
                store.close()
        m = summary_report(runs, coalition_rows=coalition_rows)
        n_sel = sum(1 for r in runs if r.decision.value == "select")
        n_abs = sum(1 for r in runs if r.decision.value == "abstain")
        n_ok  = sum(1 for r in runs if r.ground_truth_match is True)
        avg_dur = m.get("diagnostics", {}).get("avg_duration", 0.0)

        sel_acc = m.get("selective_accuracy", 0.0)
        cov     = m.get("coverage", 0.0)
        abst_eff = m.get("abstention_metrics", {}).get("abstention_efficiency", 1.0)

        tbl.add_row("Questions Run",    str(len(runs)))
        tbl.add_row("✓ Correct",        f"{n_ok}/{len(runs)}")
        tbl.add_row("Selective Acc",    f"{sel_acc*100:.1f}%")
        tbl.add_row("Coverage (SEL)",   f"{cov*100:.1f}%")
        tbl.add_row("Selected",         str(n_sel))
        tbl.add_row("Abstained",        str(n_abs))
        tbl.add_row("Abst. Efficiency", f"{abst_eff*100:.1f}%")
        tbl.add_row("Avg Duration",     f"{avg_dur:.1f}s")
        if self.last_decision:
            color = {
                "select":  "bright_green",
                "abstain": "bright_red",
                "resample":"bright_yellow",
            }.get(self.last_decision.lower(), "white")
            tbl.add_row(
                "Last Decision",
                Text(self.last_decision.upper(), style=color),
            )
        gt_str = "✓" if self.last_gt_match is True else ("✗" if self.last_gt_match is False else "—")
        tbl.add_row("Last GT Match", gt_str)

    # ------------------------------------------------------------------
    # Telemetry helpers
    # ------------------------------------------------------------------

    def push_telemetry(self, markup: str) -> None:
        self.query_one("#telemetry_log", RichLog).write(Text.from_markup(markup))

    # ------------------------------------------------------------------
    # Live view helpers
    # ------------------------------------------------------------------

    def _update_question(self, text: str) -> None:
        self.query_one("#question_panel", Static).update(
            Text.assemble(
                Text("Question  ", style="bold cyan"),
                Text(text[:220] + ("…" if len(text) > 220 else ""), style="italic"),
            )
        )

    def _update_attempt_header(self) -> None:
        self.query_one("#attempt_badge", Static).update(
            Text(f"Attempt {self.current_attempt}", style="bold bright_cyan")
        )
        phase_colors = {
            "generating":  "bright_green",
            "judging":     "bright_yellow",
            "adjudicating":"bright_magenta",
            "done":        "bright_white",
        }
        color = phase_colors.get(self.current_phase.lower(), "white")
        self.query_one("#phase_badge", Static).update(
            Text(f"● {self.current_phase.upper()}", style=f"bold {color}")
        )

    def _update_stream(self, text: str) -> None:
        # Show last 1800 chars to keep the view clean
        display = text[-1800:] if len(text) > 1800 else text
        self.query_one("#stream_content", Static).update(
            Text.assemble(
                Text("Reasoning Trace\n", style="bold dim"),
                Text(display, style="dim"),
            )
        )

    def _update_judge_panel(self) -> None:
        if not self.judge_scores:
            self.query_one("#judge_panel", Static).update(
                Text("LLM Judge  —  not yet evaluated", style="dim")
            )
            return

        def _bar(v: float, width: int = 20) -> str:
            filled = int(v * width)
            empty  = width - filled
            color  = "bright_green" if v >= 0.75 else ("bright_yellow" if v >= 0.50 else "bright_red")
            return f"[{color}]{'█' * filled}[/][dim]{'░' * empty}[/]"

        lines: list[str] = ["[bold]LLM Judge Rubric[/]\n"]
        labels = {
            "arithmetic_consistency":  "Arithmetic   ",
            "logical_consistency":     "Logic        ",
            "answer_trace_alignment":  "Trace Align  ",
            "final_rule_coherence":    "Coherence ★  ",
        }
        for key, label in labels.items():
            v = self.judge_scores.get(key, 0.0)
            lines.append(f"[dim]{label}[/] {_bar(v)} [bold]{v:.2f}[/]")
        if "rationale" in self.judge_scores:
            rat = str(self.judge_scores["rationale"])[:120]
            lines.append(f"\n[dim italic]Judge: {escape(rat)}[/]")
        self.query_one("#judge_panel", Static).update(
            Text.from_markup("\n".join(lines))
        )

    def _update_decision_banner(self, decision: str, gt_match: Optional[bool]) -> None:
        colors = {
            "select":  ("bright_green",  "✓  SELECT"),
            "abstain": ("bright_red",    "⊘  ABSTAIN"),
            "resample":("bright_yellow", "↻  RESAMPLE"),
        }
        color, label = colors.get(decision.lower(), ("white", decision.upper()))
        gt_text = ""
        if gt_match is True:
            gt_text = "  [bold bright_green]Ground Truth ✓[/]"
        elif gt_match is False:
            gt_text = "  [bold bright_red]Ground Truth ✗[/]"
        self.query_one("#decision_banner", Static).update(
            Text.from_markup(f"[bold {color}]{label}[/]{gt_text}")
        )

    # ------------------------------------------------------------------
    # Benchmark worker (background thread)
    # ------------------------------------------------------------------

    @work(thread=True)
    def run_experiment_thread(self) -> None:
        from epistemic_tribunal.config import load_config
        from epistemic_tribunal.tasks.gsm8k import load_tasks_from_jsonl
        from epistemic_tribunal.orchestrator import Orchestrator

        config = load_config(self.config_path)

        # Patch judge flag from CLI arg
        config.critic.use_llm_judge_for_math = self.use_judge

        ts = time.strftime("%Y%m%d_%H%M%S")
        if self.ledger_override:
            config.ledger.path = self.ledger_override
        else:
            # Fresh ledger per experiment run
            ledger_path = str(self.out_dir / f"gsm8k_judge{'_ON' if self.use_judge else '_OFF'}_{ts}.db")
            config.ledger.path = ledger_path
        self.active_ledger_path = config.ledger.path

        orch = Orchestrator(config)
        tasks = load_tasks_from_jsonl(self.data_path)[: self.n_questions]

        self.call_from_thread(
            self.push_telemetry,
            f"[bold cyan]Starting GSM8K experiment[/] — {len(tasks)} questions, "
            f"judge={'[bright_green]ON[/]' if self.use_judge else '[bright_red]OFF[/]'}"
        )

        for i, task in enumerate(tasks):
            self.call_from_thread(self._start_question, i, len(tasks), task.test_input)

            def on_token(token_type: str, text: str, _task=task) -> None:
                self.call_from_thread(self._handle_token, token_type, text)

            try:
                run = orch.run(task, on_token=on_token)
                self.runs.append(run)
                self.call_from_thread(
                    self._finish_question, run, task
                )
            except Exception as exc:
                self.call_from_thread(
                    self.push_telemetry,
                    f"[bold red]CRASH on q{i+1}: {escape(str(exc))}[/]"
                )
                time.sleep(1)
                self.call_from_thread(self._refresh_metrics)

        self.call_from_thread(self._finalise, ts)

    def _start_question(self, idx: int, total: int, question: str) -> None:
        self.current_q_text = question
        self.current_attempt = 0
        self.current_phase = "Generating"
        self.current_trace_text = ""
        self.judge_scores = {}
        self.sub_title = f"Q{idx+1}/{total} · judge={'ON' if self.use_judge else 'OFF'}"
        self._update_question(question)
        self._update_attempt_header()
        self._update_stream("")
        self._update_judge_panel()
        self.query_one("#decision_banner", Static).update(
            Text("Awaiting decision…", style="dim italic")
        )

    def _handle_token(self, token_type: str, text: str) -> None:
        if token_type == "generator_start":
            self.current_attempt += 1
            self.current_phase = "Generating"
            self.current_trace_text = ""
            self._update_attempt_header()
            return
        if token_type == "reasoning":
            self.current_trace_text += text
            self._update_stream(self.current_trace_text)
        elif token_type == "content":
            self.current_trace_text += text
            self._update_stream(self.current_trace_text)
        # Detect when the judge fires (phase change via log or metadata)
        if "llm_judge" in token_type:
            self.current_phase = "Judging"
            self._update_attempt_header()

    def _finish_question(self, run, task) -> None:
        self.last_decision = run.decision.value
        self.last_gt_match = run.ground_truth_match

        # Pull judge rubric from decision metadata (bubbled up by orchestrator)
        judge_rubric = run.metadata.get("judge_rubric", {})
        if judge_rubric:
            self.judge_scores = {
                "arithmetic_consistency": judge_rubric.get("arithmetic_consistency", 0.0),
                "logical_consistency":    judge_rubric.get("logical_consistency", 0.0),
                "answer_trace_alignment": judge_rubric.get("answer_trace_alignment", 0.0),
                "final_rule_coherence":   judge_rubric.get("final_rule_coherence", 0.0),
                "rationale":              judge_rubric.get("rationale", ""),
            }

        self.current_phase = "Done"
        self._update_attempt_header()
        self._update_judge_panel()
        self._update_decision_banner(run.decision.value, run.ground_truth_match)
        self._refresh_metrics()

        gt = "✓" if run.ground_truth_match is True else ("✗" if run.ground_truth_match is False else "—")
        d_color = {"select": "bright_green", "abstain": "bright_red"}.get(run.decision.value, "yellow")
        self.push_telemetry(
            f"[{d_color}]{'✓' if run.ground_truth_match else '✗'} "
            f"{run.decision.value.upper():<8}[/] "
            f"[dim]{task.task_id}[/] conf={run.confidence:.2f} gt={gt}"
        )

    def _finalise(self, ts: str) -> None:
        from epistemic_tribunal.evaluation.metrics import summary_report
        from epistemic_tribunal.ledger.store import LedgerStore
        self.current_phase = "Done"
        self._update_attempt_header()

        out_file = self.out_dir / f"gsm8k_results_judge{'ON' if self.use_judge else 'OFF'}_{ts}.json"
        coalition_rows = None
        if self.runs and self.active_ledger_path:
            store = LedgerStore(self.active_ledger_path)
            try:
                coalition_rows = store.get_coalition_opinions(run_ids=[r.run_id for r in self.runs])
            finally:
                store.close()
        report = summary_report(self.runs, coalition_rows=coalition_rows) if self.runs else {}
        try:
            with out_file.open("w") as f:
                json.dump({
                    "config": self.config_path,
                    "judge_enabled": self.use_judge,
                    "n_questions": self.n_questions,
                    "timestamp": ts,
                    "summary": report,
                    "runs": [r.model_dump(mode="json") for r in self.runs],
                }, f, indent=2)
        except OSError:
            pass

        self.push_telemetry(f"[bold bright_green]Experiment complete![/] Results → {out_file}")
        self._refresh_metrics()

        # Auto-exit after 5 s so user can read final state
        time.sleep(5)
        self.exit(return_code=0, result=str(out_file))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="GSM8K Tribunal Experiment TUI")
    parser.add_argument("--questions", type=int, default=20, help="Number of questions (default: 20)")
    parser.add_argument("--judge", dest="judge", action="store_true", default=True)
    parser.add_argument("--no-judge", dest="judge", action="store_false")
    parser.add_argument("--config", default="configs/gsm8k_sample.yaml")
    parser.add_argument("--data",   default="data/gsm8k/test.jsonl")
    parser.add_argument("--out",    default="data/gsm8k/results")
    parser.add_argument("--ledger", default=None, help="Explicit ledger DB to use (for multi-pass experiments)")
    args = parser.parse_args()

    app = GSM8KExperimentApp(
        config_path=args.config,
        data_path=args.data,
        n_questions=args.questions,
        use_judge=args.judge,
        out_dir=args.out,
        ledger_override=args.ledger,
    )
    result_path = app.run()

    if result_path:
        from rich.console import Console
        from rich.table import Table
        c = Console()
        try:
            with open(result_path) as f:
                data = json.load(f)
            t = Table(title="Final Output Statistics")
            t.add_column("Metric")
            t.add_column("Value", justify="right")
            for k, v in data.get("summary", {}).items():
                if isinstance(v, float):
                    t.add_row(k, f"{v:.4f}")
                elif isinstance(v, dict):
                    t.add_row(k, str(v))
                else:
                    t.add_row(k, str(v))
            c.print("\n[bold green]Experiment Complete![/]")
            c.print(t)
            c.print(f"\n[dim]Full results saved to: {result_path}[/]")
        except Exception as e:
            c.print(f"[red]Failed to print results: {e}[/]")

if __name__ == "__main__":
    main()
