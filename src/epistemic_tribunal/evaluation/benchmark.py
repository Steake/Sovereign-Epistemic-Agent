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
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from epistemic_tribunal.config import TribunalSettings, load_config
from epistemic_tribunal.evaluation.metrics import summary_report
from epistemic_tribunal.orchestrator import Orchestrator
from epistemic_tribunal.tasks.arc_like import load_task_from_file
from epistemic_tribunal.tribunal_types import DecisionKind, ExperimentRun
from epistemic_tribunal.utils.logging import get_logger

log = get_logger(__name__)


def experiment_run_from_row(row: dict) -> ExperimentRun:
    """Reconstruct an ExperimentRun from a persisted ledger row."""
    created_at_str = row.get("created_at")
    if created_at_str:
        try:
            ts = datetime.fromisoformat(created_at_str)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            ts = datetime.now(timezone.utc)
    else:
        ts = datetime.now(timezone.utc)
    return ExperimentRun(
        run_id=row["run_id"],
        task_id=row["task_id"],
        generator_names=json.loads(row["generator_names_json"]),
        decision=DecisionKind(row["decision"]),
        confidence=float(row.get("confidence", 0.0) or 0.0),
        selected_trace_id=row["selected_trace_id"],
        ground_truth_match=(
            None
            if row["ground_truth_match"] is None
            else bool(row["ground_truth_match"])
        ),
        duration_seconds=row["duration_seconds"],
        config_snapshot=json.loads(row["config_snapshot_json"]),
        metadata=json.loads(row.get("metadata_json", "{}")),
        timestamp=ts,
    )


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
        self.last_runs: list[ExperimentRun] = []

    def run(
        self, 
        dataset_path: Path | str, 
        *, 
        resume: bool = False,
        manifest_path: Optional[str | Path] = None,
        limit: Optional[int] = None,
    ) -> list[ExperimentRun]:
        """Run the tribunal over tasks in *dataset_path*.

        Parameters
        ----------
        dataset_path:
            Directory containing task JSON files.
        resume:
            Whether to attempt resuming from a prior checkpoint.
        manifest_path:
            Optional path to a text file containing task IDs (one per line).
            If provided, this manifest determines the authoritative ordering.
        limit:
            Optional cap on the number of tasks to run.

        Returns
        -------
        list[ExperimentRun]
            One run record per task file.
        """
        dataset_path = Path(dataset_path)
        
        # Determine task files and ordering
        task_metadata_map = {}
        if manifest_path:
            manifest_path = Path(manifest_path)
            log.info("Loading tasks from authoritative manifest: %s", manifest_path)
            
            task_ids = []
            if manifest_path.suffix == ".json":
                with manifest_path.open("r") as f:
                    manifest_data = json.load(f)
                for entry in manifest_data:
                    tid = entry["task_id"]
                    task_ids.append(tid)
                    task_metadata_map[tid] = entry
            else:
                with manifest_path.open("r") as f:
                    task_ids = [line.strip() for line in f if line.strip()]
            
            # Map IDs to local files (preserving order)
            task_files = []
            for tid in task_ids:
                matches = list(dataset_path.glob(f"**/{tid}.json")) + list(dataset_path.glob(f"**/{tid}.jsonl"))
                if matches:
                    task_files.append(matches[0])
                else:
                    log.warning("Task %s from manifest not found in %s.", tid, dataset_path)
        else:
            if dataset_path.is_file():
                task_files = [dataset_path]
            else:
                task_files = sorted(dataset_path.glob("*.json")) + sorted(dataset_path.glob("*.jsonl"))

        # Apply limit if specified
        if limit:
            task_files = task_files[:limit]

        start_time = time.monotonic()
        prior_progress = self._load_progress() if resume else {}
        completed_task_ids = set(prior_progress.get("completed_task_ids", []))
        prior_runs = self._load_prior_runs(completed_task_ids) if (resume and completed_task_ids) else []
        elapsed_before_resume = float(prior_progress.get("elapsed_time_seconds", 0.0))

        if not task_files:
            log.warning("No *.json or *.jsonl task files found in %s", dataset_path)
            return []

        runs: list[ExperimentRun] = list(prior_runs)
        processed_since_checkpoint = 0
        # Count tasks already loaded from resume so limit is relative to fresh work
        fresh_task_count = 0
        for task_file in task_files:
            if limit is not None and fresh_task_count >= limit:
                break
            try:
                if task_file.suffix == ".jsonl":
                    from epistemic_tribunal.tasks.gsm8k import load_tasks_from_jsonl
                    tasks_to_run = load_tasks_from_jsonl(task_file)
                else:
                    tasks_to_run = [load_task_from_file(task_file)]

                for task in tasks_to_run:
                    if limit is not None and fresh_task_count >= limit:
                        break
                    if task.task_id in completed_task_ids:
                        log.info("Skipping completed task %s due to resume state.", task.task_id)
                        continue
                    
                    if task.task_id in task_metadata_map:
                        task.metadata.update(task_metadata_map[task.task_id])
                        
                    log.info("Running tribunal on task %s", task.task_id)
                    run = self._orchestrator.run(task)
                    runs.append(run)
                    completed_task_ids.add(task.task_id)
                    fresh_task_count += 1
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
        self.last_runs = runs
        return runs

    def report(self, runs: list[ExperimentRun]) -> dict:
        """Compute and return summary metrics for a list of runs."""
        coalition_rows = self._store.get_coalition_opinions(
            run_ids=[run.run_id for run in runs]
        )
        return summary_report(runs, coalition_rows=coalition_rows)

    def run_and_report(
        self,
        dataset_path: Path | str,
        *,
        resume: bool = False,
        manifest_path: Optional[str | Path] = None,
    ) -> dict:
        """Convenience: run the full benchmark and return the metrics dict."""
        runs = self.run(dataset_path, resume=resume, manifest_path=manifest_path)
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
        return [experiment_run_from_row(row) for row in rows]
