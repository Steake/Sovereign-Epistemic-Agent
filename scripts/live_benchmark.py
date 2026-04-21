import json
import time
import logging
from pathlib import Path
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Header, Footer, Static, RichLog, Button, DataTable, Label
from textual import work
from rich.text import Text
from rich.markup import escape
from rich.markdown import Markdown
from rich.syntax import Syntax

from epistemic_tribunal.config import load_config
from epistemic_tribunal.tasks.arc_like import load_task_from_file
from epistemic_tribunal.orchestrator import Orchestrator
from epistemic_tribunal.evaluation.metrics import summary_report

class TribunalLogHandler(logging.Handler):
    def __init__(self, ui_app):
        super().__init__()
        self.ui_app = ui_app
        
    def emit(self, record):
        msg = record.getMessage()
        if "Initialized remote API client" in msg: return
            
        module_name = record.name.split('.')[-1]
        
        if "generator" in record.name:
            color = "green"
            icon = "⚡"
        elif "orchestrator" in record.name:
            color = "cyan"
            icon = "🧠"
        elif "aggregator" in record.name:
            color = "magenta"
            icon = "⚖️"
        else:
            color = "white"
            icon = "ℹ"
            module_name = "system"

        formatted_msg = f"[{color}]{icon} {module_name:<12}[/] {escape(msg)}"
        self.ui_app.call_from_thread(self.ui_app.write_telemetry, formatted_msg)

class TextualLiveBenchmarkUI(App):
    CSS = """
    #left_pane { width: 1fr; border-right: solid ansi_blue; }
    #streaming_view { width: 2fr; }
    #metrics { height: auto; max-height: 12; border-bottom: solid ansi_blue; }
    #telemetry { height: 1fr; padding: 0 1; }
    #switcher { height: 3; align: center middle; background: $boost; border-bottom: solid ansi_yellow; }
    #prev_gen, #next_gen { min-width: 16; margin: 0 2; }
    #gen_label { text-align: center; content-align: center middle; min-width: 30; }
    #content_area { height: 1fr; padding: 0 1; overflow-y: scroll; }
    """
    
    def __init__(self, config_path, dataset_path, manifest_path):
        super().__init__()
        self.config_path = config_path
        self.dataset_path = Path(dataset_path)
        self.manifest_path = Path(manifest_path)
        self.config = load_config(config_path)
        self.orchestrator = Orchestrator(self.config)
        self.runs = []
        self.results_path = None
        self.generations = []
        self.view_idx = -1
        self.current_task_id = ""
        self.log_handler = TribunalLogHandler(self)
        self.log_handler.setLevel(logging.INFO)
        self._attach_handler()
        
    def _attach_handler(self):
        for name, logger in logging.Logger.manager.loggerDict.items():
            if name.startswith("epistemic_tribunal") and isinstance(logger, logging.Logger):
                logger.handlers.clear()
                logger.addHandler(self.log_handler)
        import epistemic_tribunal.utils.logging as _et_logging
        _orig_get_logger = _et_logging.get_logger
        def _patched_get_logger(name="epistemic_tribunal", level=None):
            logger = _orig_get_logger(name, level)
            logger.handlers.clear()
            logger.addHandler(self.log_handler)
            return logger
        _et_logging.get_logger = _patched_get_logger

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            with Vertical(id="left_pane"):
                yield DataTable(id="metrics")
                yield RichLog(id="telemetry", markup=True, highlight=True, wrap=True)
            with Vertical(id="streaming_view"):
                with Horizontal(id="switcher"):
                    yield Button("← Previous", id="prev_gen", variant="primary")
                    yield Label("Awaiting generation...", id="gen_label")
                    yield Button("Next →", id="next_gen", variant="primary")
                with VerticalScroll(id="content_area"):
                    yield Static("", id="streaming_static")
        yield Footer()

    def on_mount(self):
        table = self.query_one("#metrics", DataTable)
        table.add_column("Metric"); table.add_column("Value")
        self._update_metrics()
        with self.manifest_path.open("r") as f:
            self.task_ids = [line.strip() for line in f if line.strip()]
        self.title = "Epistemic Tribunal"
        self.run_benchmark_thread()

    def write_telemetry(self, markup_text: str):
        self.query_one("#telemetry", RichLog).write(Text.from_markup(markup_text))
        
    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "prev_gen" and self.view_idx > 0:
            self.view_idx -= 1
            self._render_view()
        elif event.button.id == "next_gen" and self.view_idx < len(self.generations) - 1:
            self.view_idx += 1
            self._render_view()
                
    def _update_metrics(self):
        m = summary_report(self.runs) if self.runs else {}
        table = self.query_one("#metrics", DataTable)
        table.clear(columns=False)
        table.add_row("Tasks Run", str(len(self.runs)))
        table.add_row("Accuracy", f"{m.get('Accuracy', 0.0):.1f}%")
        table.add_row("Coverage", f"{m.get('Coverage', 0.0):.1f}%")
        table.add_row("Resample Rate", f"{m.get('Resample Rate', 0.0):.1f}%")
        table.add_row("Wrong Picks", str(m.get("Wrong Picks", 0)))
        table.add_row("Parse Fails", str(m.get("Parse Fails", 0)))

    @work(thread=True)
    def run_benchmark_thread(self):
        for idx, tid in enumerate(self.task_ids):
            self.call_from_thread(self._start_task, tid, idx)
            matches = list(self.dataset_path.glob(f"**/{tid}.json"))
            if not matches:
                self.call_from_thread(self.write_telemetry, f"[red]Task not found: {tid}[/red]")
                continue
            task = load_task_from_file(matches[0])
            try:
                run = self.orchestrator.run(task, on_token=self._token_router)
                self.runs.append(run)
            except Exception as e:
                self.call_from_thread(self.write_telemetry, f"[red bold]FATAL CRASH on {tid}: {e}[/red bold]")
                time.sleep(2)
            self.call_from_thread(self._update_metrics)
        self.call_from_thread(self._save_and_quit)

    def _start_task(self, tid, idx):
        self.current_task_id = tid
        self.sub_title = f"Task {idx+1}/{len(self.task_ids)} - {tid}"
        self.generations = []
        self.view_idx = -1
        self._render_view()
        
    def _token_router(self, token_type: str, text: str):
        self.call_from_thread(self._handle_token, token_type, text)
        
    def _handle_token(self, token_type: str, text: str):
        if token_type == "generator_start":
            self.generations.append({"name": text, "reasoning": "", "sandbox_code": "", "sandbox_state": "", "sandbox_result": "", "content": ""})
            self.view_idx = len(self.generations) - 1
            self._render_view()
            return
        if not self.generations: return
        active_gen = self.generations[-1]
        
        if token_type == "reasoning": active_gen["reasoning"] += text
        elif token_type == "sandbox_code": active_gen["sandbox_code"] = text
        elif token_type == "sandbox_state": active_gen["sandbox_state"] = text
        elif token_type == "sandbox_result": active_gen["sandbox_result"] = text
        else: active_gen["content"] += text
            
        if self.view_idx == len(self.generations) - 1:
            self._render_view()
            
    def _render_view(self):
        lbl = self.query_one("#gen_label", Label)
        stat = self.query_one("#streaming_static", Static)
        if not self.generations or self.view_idx < 0:
            lbl.update(f"Awaiting Task {self.current_task_id}...")
            stat.update(Markdown("[dim italic]Awaiting LLM generation tokens...[/]"))
            return
            
        current = self.generations[self.view_idx]
        is_live = (self.view_idx == len(self.generations) - 1)
        status_flag = "[blink green]● Live[/]" if is_live else "[dim]⚪ History[/]"
        lbl.update(f"Gen {self.view_idx+1}/{len(self.generations)} ({current['name']}) {status_flag}")
        
        from rich.console import Group
        components = []
        if current["sandbox_code"]:
            components.append(Text.from_markup("[bold magenta]Python Synthesis Pipeline[/]"))
            components.append(Syntax(current["sandbox_code"], "python", theme="monokai", line_numbers=True, word_wrap=True))
            style = "bold green" if "Success" in current["sandbox_state"] else "bold red" if "Error" in current["sandbox_state"] else "yellow"
            st_text = f"[{style}]Sandbox Status: {escape(current['sandbox_state'])}[/{style}]"
            if current["sandbox_result"]:
                res = escape(current["sandbox_result"])
                st_text += f"\n[dim]{res[:500] + ('...' if len(res) > 500 else '')}[/]"
            components.append(Text.from_markup(st_text))
        else:
            r, c = current["reasoning"], current["content"]
            if len(r) > 1500: r = "...\n" + r[-1500:]
            if len(c) > 1500: c = "...\n" + c[-1500:]
            if r: components.append(Text.from_markup("[bold cyan]Reasoning Trace:[/]\n[dim]" + escape(r) + "[/]"))
            if c: components.append(Text.from_markup("[bold green]Content Stream:[/]\n" + escape(c)))
                
        stat.update(Group(*components))
        
    def _save_and_quit(self):
        if self.runs:
            c_name = str(self.config_path).split('/')[-1].replace('.yaml', '')
            ts = time.strftime("%Y%m%d_%H%M%S")
            p = Path("results"); p.mkdir(exist_ok=True)
            self.results_path = str(p / f"benchmark_{c_name}_{ts}.json")
            try:
                with open(self.results_path, "w") as f:
                    json.dump({"metrics": summary_report(self.runs), "config": str(self.config_path), "timestamp": ts, "runs": [r.model_dump(mode='json') for r in self.runs]}, f, indent=2)
            except OSError:
                pass # Prevent total crash if disk full
        self.exit(return_code=0)

class LiveBenchmarkUI:
    def __init__(self, config_path, manifest_path, dataset_path):
        self.app = TextualLiveBenchmarkUI(config_path, dataset_path, manifest_path)
    def run(self):
        self.app.run()
        if hasattr(self.app, 'results_path') and self.app.results_path:
            import os
            os.system("clear")
            from rich.console import Console
            from rich.table import Table
            c = Console()
            try:
                with open(self.app.results_path) as f:
                    data = json.load(f)
            except OSError:
                data = {"metrics": summary_report(self.app.runs) if self.app.runs else {}}
            t = Table(title="Overall Metrics")
            t.add_column("Metric"); t.add_column("Value", justify="right")
            for k, v in data["metrics"].items():
                if isinstance(v, float): t.add_row(k, f"{v:.1f}%")
                else: t.add_row(k, str(v))
            c.print("\n[bold green]Benchmark Complete![/]")
            c.print(t)
            return self.app.results_path
        return None
