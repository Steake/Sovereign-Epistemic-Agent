"""Tests for benchmark checkpointing and resume support."""

from __future__ import annotations

import json
from pathlib import Path

from epistemic_tribunal.config import TribunalSettings
from epistemic_tribunal.evaluation.benchmark import BenchmarkRunner
from epistemic_tribunal.ledger.store import LedgerStore


def _write_task(path: Path, task_id: str, colour: int) -> None:
    path.write_text(
        json.dumps(
            {
                "task_id": task_id,
                "description": "identity",
                "train": [
                    {
                        "input": [[colour, 0], [0, colour]],
                        "output": [[colour, 0], [0, colour]],
                    }
                ],
                "test": [{"input": [[colour, 0], [0, colour]]}],
                "ground_truth": [[colour, 0], [0, colour]],
            }
        )
    )


def test_benchmark_writes_progress_and_checkpoint(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    _write_task(dataset_path / "task_1.json", "task_1", 1)
    _write_task(dataset_path / "task_2.json", "task_2", 2)

    ledger_path = tmp_path / "ledger.db"
    config = TribunalSettings()
    config.ledger.path = str(ledger_path)
    config.benchmark.checkpoint_every_n_tasks = 1
    config.tribunal.selection_threshold = 0.0
    config.tribunal.resample_threshold = 0.0
    config.tribunal.diversity_floor = 1.0

    runner = BenchmarkRunner(config=config)
    runs = runner.run(dataset_path)

    assert len(runs) == 2
    progress_path = ledger_path.with_name("run_progress.json")
    checkpoint_path = ledger_path.with_suffix(".checkpoint.sqlite3")
    assert progress_path.exists()
    assert checkpoint_path.exists()

    progress = json.loads(progress_path.read_text())
    assert sorted(progress["completed_task_ids"]) == ["task_1", "task_2"]
    assert "elapsed_time_seconds" in progress


def test_benchmark_resume_skips_completed_tasks(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    _write_task(dataset_path / "task_1.json", "task_1", 1)
    _write_task(dataset_path / "task_2.json", "task_2", 2)

    ledger_path = tmp_path / "ledger.db"
    config = TribunalSettings()
    config.ledger.path = str(ledger_path)
    config.benchmark.checkpoint_every_n_tasks = 1
    config.tribunal.selection_threshold = 0.0
    config.tribunal.resample_threshold = 0.0
    config.tribunal.diversity_floor = 1.0

    runner = BenchmarkRunner(config=config)
    first_runs = runner.run(dataset_path)
    store = LedgerStore(ledger_path)
    first_stats = store.get_stats()
    resumed_runs = runner.run(dataset_path, resume=True)
    second_stats = store.get_stats()
    store.close()

    assert len(first_runs) == 2
    assert len(resumed_runs) == 2
    assert first_stats["experiment_runs"] == 2
    assert second_stats["experiment_runs"] == 2
