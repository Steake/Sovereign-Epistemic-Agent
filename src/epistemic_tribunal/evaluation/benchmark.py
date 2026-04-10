"""Benchmark runner — evaluates the tribunal over a directory of task files.

Each JSON file in the dataset directory is treated as one task.
Results are collected into :class:`ExperimentRun` objects and metrics are
computed via :mod:`epistemic_tribunal.evaluation.metrics`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from epistemic_tribunal.config import TribunalSettings, load_config
from epistemic_tribunal.evaluation.metrics import summary_report
from epistemic_tribunal.orchestrator import Orchestrator
from epistemic_tribunal.tasks.arc_like import load_task_from_file
from epistemic_tribunal.types import ExperimentRun
from epistemic_tribunal.utils.logging import get_logger

log = get_logger(__name__)

_PROGRESS_FILE = "run_progress.json"


class BenchmarkRunner:
    """Runs the tribunal over an entire dataset directory.

    Parameters
    ----------
    config:
        Application settings.  Loaded from ``configs/default.yaml`` if omitted.
    ledger_path:
        Override the ledger database path.
    resume:
        If ``True``, read ``run_progress.json`` from the dataset directory
        and skip already-completed task IDs.
    """

    def __init__(
        self,
        config: Optional[TribunalSettings] = None,
        ledger_path: Optional[str] = None,
        resume: bool = False,
    ) -> None:
        self._config = config or load_config()
        if ledger_path:
            self._config.ledger.path = ledger_path
        self._orchestrator = Orchestrator(self._config)
        self._resume = resume

    def run(self, dataset_path: Path | str) -> list[ExperimentRun]:
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

        # Exclude the progress file itself
        task_files = [f for f in task_files if f.name != _PROGRESS_FILE]

        if not task_files:
            log.warning("No *.json task files found in %s", dataset_path)
            return []

        # Resume support: load already-completed task IDs
        completed_ids: set[str] = set()
        progress_path = dataset_path / _PROGRESS_FILE
        if self._resume and progress_path.exists():
            try:
                progress = json.loads(progress_path.read_text())
                completed_ids = set(progress.get("completed_task_ids", []))
                log.info(
                    "Resuming benchmark: %d tasks already completed.",
                    len(completed_ids),
                )
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("Failed to read progress file: %s", exc)

        checkpoint_n = self._config.ledger.checkpoint_every_n_tasks
        start_time = time.monotonic()
        runs: list[ExperimentRun] = []
        solve_count = 0

        for task_file in task_files:
            try:
                task = load_task_from_file(task_file)

                # Skip already-completed tasks on resume
                if task.task_id in completed_ids:
                    log.debug("Skipping already-completed task %s", task.task_id)
                    continue

                log.info("Running tribunal on task %s", task.task_id)
                run = self._orchestrator.run(task)
                runs.append(run)
                completed_ids.add(task.task_id)

                if run.ground_truth_match is True:
                    solve_count += 1

                # Periodic checkpoint
                if checkpoint_n > 0 and len(runs) % checkpoint_n == 0:
                    self._write_checkpoint(
                        progress_path,
                        completed_ids,
                        solve_count,
                        time.monotonic() - start_time,
                    )

            except Exception as exc:
                log.error("Failed to process %s: %s", task_file.name, exc)

        # Final checkpoint
        if checkpoint_n > 0 and runs:
            self._write_checkpoint(
                progress_path,
                completed_ids,
                solve_count,
                time.monotonic() - start_time,
            )

        return runs

    def report(self, runs: list[ExperimentRun]) -> dict:
        """Compute and return summary metrics for a list of runs."""
        return summary_report(runs)

    def run_and_report(self, dataset_path: Path | str) -> dict:
        """Convenience: run the full benchmark and return the metrics dict."""
        runs = self.run(dataset_path)
        metrics = self.report(runs)
        log.info("Benchmark complete: %s", metrics)
        return metrics

    @staticmethod
    def _write_checkpoint(
        path: Path,
        completed_ids: set[str],
        solve_count: int,
        elapsed_seconds: float,
    ) -> None:
        """Write current progress to a JSON checkpoint file."""
        payload = {
            "completed_task_ids": sorted(completed_ids),
            "completed_count": len(completed_ids),
            "solve_count": solve_count,
            "elapsed_seconds": round(elapsed_seconds, 2),
        }
        try:
            path.write_text(json.dumps(payload, indent=2))
            log.info(
                "Checkpoint written: %d tasks completed, %d solved (%.1fs elapsed).",
                len(completed_ids),
                solve_count,
                elapsed_seconds,
            )
        except OSError as exc:
            log.error("Failed to write checkpoint: %s", exc)
