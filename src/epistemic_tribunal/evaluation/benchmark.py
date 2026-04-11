"""Benchmark runner — evaluates the tribunal over a directory of task files.

Each JSON file in the dataset directory is treated as one task.
Results are collected into :class:`ExperimentRun` objects and metrics are
computed via :mod:`epistemic_tribunal.evaluation.metrics`.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

from epistemic_tribunal.config import TribunalSettings, load_config
from epistemic_tribunal.evaluation.metrics import summary_report
from epistemic_tribunal.orchestrator import Orchestrator
from epistemic_tribunal.tasks.arc_like import load_task_from_file
from epistemic_tribunal.types import DecisionKind, ExperimentRun
from epistemic_tribunal.utils.logging import get_logger

log = get_logger(__name__)


class BenchmarkRunner:
    """Runs the tribunal over an entire dataset directory.

    Parameters
    ----------
    config:
        Application settings.  Loaded from ``configs/default.yaml`` if omitted.
    ledger_path:
        Override the ledger database path.
    """

    def __init__(
        self,
        config: Optional[TribunalSettings] = None,
        ledger_path: Optional[str] = None,
    ) -> None:
        self._config = config or load_config()
        if ledger_path:
            self._config.ledger.path = ledger_path
        self._orchestrator = Orchestrator(self._config)
        self._store = self._orchestrator.store
        self._progress_path = Path(self._config.ledger.path).with_name("run_progress.json")
        self._checkpoint_path = Path(self._config.ledger.path).with_suffix(".checkpoint.sqlite3")

    def run(self, dataset_path: Path | str, *, resume: bool = False) -> list[ExperimentRun]:
        """Run the tribunal over every ``*.json`` file in *dataset_path*.

        Parameters
        ----------
        dataset_path:
            Directory containing task JSON files.

        Returns
        -------
        list[ExperimentRun]
            One run record per task file.
        """
        dataset_path = Path(dataset_path)
        task_files = sorted(dataset_path.glob("*.json"))
        start_time = time.monotonic()
        prior_progress = self._load_progress() if resume else {}
        completed_task_ids = set(prior_progress.get("completed_task_ids", []))
        prior_runs = self._load_prior_runs(completed_task_ids) if (resume and completed_task_ids) else []
        elapsed_before_resume = float(prior_progress.get("elapsed_time_seconds", 0.0))

        if not task_files:
            log.warning("No *.json task files found in %s", dataset_path)
            return []

        runs: list[ExperimentRun] = list(prior_runs)
        processed_since_checkpoint = 0
        for task_file in task_files:
            try:
                task = load_task_from_file(task_file)
                if task.task_id in completed_task_ids:
                    log.info("Skipping completed task %s due to resume state.", task.task_id)
                    continue
                log.info("Running tribunal on task %s", task.task_id)
                run = self._orchestrator.run(task)
                runs.append(run)
                completed_task_ids.add(task.task_id)
                processed_since_checkpoint += 1
                if self._should_checkpoint(processed_since_checkpoint):
                    self._checkpoint_progress(
                        completed_task_ids=completed_task_ids,
                        runs=runs,
                        elapsed_time_seconds=elapsed_before_resume + (time.monotonic() - start_time),
                    )
                    processed_since_checkpoint = 0
            except Exception as exc:
                log.error("Failed to process %s: %s", task_file.name, exc)

        if self._config.benchmark.checkpoint_every_n_tasks > 0:
            self._checkpoint_progress(
                completed_task_ids=completed_task_ids,
                runs=runs,
                elapsed_time_seconds=elapsed_before_resume + (time.monotonic() - start_time),
            )
        return runs

    def report(self, runs: list[ExperimentRun]) -> dict:
        """Compute and return summary metrics for a list of runs."""
        return summary_report(runs)

    def run_and_report(self, dataset_path: Path | str, *, resume: bool = False) -> dict:
        """Convenience: run the full benchmark and return the metrics dict."""
        runs = self.run(dataset_path, resume=resume)
        metrics = self.report(runs)
        log.info("Benchmark complete: %s", metrics)
        return metrics

    def _should_checkpoint(self, processed_since_checkpoint: int) -> bool:
        checkpoint_every = self._config.benchmark.checkpoint_every_n_tasks
        return checkpoint_every > 0 and processed_since_checkpoint >= checkpoint_every

    def _checkpoint_progress(
        self,
        *,
        completed_task_ids: set[str],
        runs: list[ExperimentRun],
        elapsed_time_seconds: float,
    ) -> None:
        self._store.checkpoint(self._checkpoint_path)
        payload = {
            "completed_task_ids": sorted(completed_task_ids),
            "solve_count": sum(1 for run in runs if run.ground_truth_match is True),
            "elapsed_time_seconds": round(elapsed_time_seconds, 4),
        }
        self._progress_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_name = tempfile.mkstemp(
            dir=self._progress_path.parent, suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w") as fh:
                json.dump(payload, fh, indent=2)
            os.replace(tmp_name, self._progress_path)
        except (OSError, ValueError):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        log.info(
            "Checkpointed benchmark progress to %s after %d completed task(s).",
            self._progress_path,
            len(completed_task_ids),
        )

    def _load_progress(self) -> dict:
        if not self._progress_path.exists():
            log.warning(
                "Resume requested but progress file %s was not found; starting fresh.",
                self._progress_path,
            )
            return {}
        try:
            return json.loads(self._progress_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            log.warning(
                "Failed to read progress file %s (%s); starting fresh.",
                self._progress_path,
                exc,
            )
            return {}

    def _load_prior_runs(self, completed_task_ids: set[str]) -> list[ExperimentRun]:
        rows = self._store.get_experiment_runs(completed_task_ids)
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
        return runs
