"""Benchmark runner — evaluates the tribunal over a directory of task files.

Each JSON file in the dataset directory is treated as one task.
Results are collected into :class:`ExperimentRun` objects and metrics are
computed via :mod:`epistemic_tribunal.evaluation.metrics`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from epistemic_tribunal.config import TribunalSettings, load_config
from epistemic_tribunal.evaluation.metrics import summary_report
from epistemic_tribunal.orchestrator import Orchestrator
from epistemic_tribunal.tasks.arc_like import load_task_from_file
from epistemic_tribunal.types import ExperimentRun
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

        if not task_files:
            log.warning("No *.json task files found in %s", dataset_path)
            return []

        runs: list[ExperimentRun] = []
        for task_file in task_files:
            try:
                task = load_task_from_file(task_file)
                log.info("Running tribunal on task %s", task.task_id)
                run = self._orchestrator.run(task)
                runs.append(run)
            except Exception as exc:
                log.error("Failed to process %s: %s", task_file.name, exc)

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
