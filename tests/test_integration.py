"""Integration test — full tribunal pipeline from task file to ledger."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from epistemic_tribunal.config import TribunalSettings
from epistemic_tribunal.evaluation.benchmark import BenchmarkRunner
from epistemic_tribunal.ledger.store import LedgerStore
from epistemic_tribunal.orchestrator import Orchestrator
from epistemic_tribunal.tasks.arc_like import load_task_from_file
from epistemic_tribunal.tribunal_types import DecisionKind, ExperimentRun


# ---------------------------------------------------------------------------
# Integration: single task end-to-end
# ---------------------------------------------------------------------------


EXAMPLES_DIR = Path(__file__).parent.parent / "data" / "examples"


@pytest.mark.parametrize(
    "task_file",
    [
        "colour_swap_001.json",
        "fill_background_002.json",
        "copy_identity_003.json",
        "horizontal_flip_004.json",
        "vertical_flip_005.json",
    ],
)
def test_single_task_pipeline(task_file: str) -> None:
    """Each example task should run through the full pipeline without errors."""
    task_path = EXAMPLES_DIR / task_file
    assert task_path.exists(), f"Example task not found: {task_path}"

    task = load_task_from_file(task_path)

    config = TribunalSettings()
    config.tribunal.selection_threshold = 0.0  # ensure we always get a decision
    config.tribunal.resample_threshold = 0.0

    store = LedgerStore(":memory:")
    orch = Orchestrator(config=config, ledger_store=store)
    run = orch.run(task)

    # Basic assertions
    assert isinstance(run, ExperimentRun)
    assert run.task_id == task.task_id
    assert run.decision in DecisionKind
    assert run.duration_seconds >= 0.0

    # Ledger should have been populated
    stats = store.get_stats()
    assert stats["tasks"] >= 1
    assert stats["traces"] >= 5
    assert stats["decisions"] >= 1
    assert stats["experiment_runs"] >= 1


# ---------------------------------------------------------------------------
# Integration: benchmark runner over example dataset
# ---------------------------------------------------------------------------


def test_benchmark_over_examples_dir(tmp_path: Path) -> None:
    """BenchmarkRunner should process all example tasks and return metrics."""
    config = TribunalSettings()
    config.tribunal.selection_threshold = 0.0
    config.tribunal.resample_threshold = 0.0
    config.ledger.path = str(tmp_path / "test_ledger.db")

    runner = BenchmarkRunner(config=config)
    runs = runner.run(EXAMPLES_DIR)

    assert len(runs) == 5
    metrics = runner.report(runs)

    assert metrics["total_runs"] == 5
    assert 0.0 <= metrics["coverage"] <= 1.0
    assert 0.0 <= metrics["overall_accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# Integration: ledger persistence across two orchestrator calls
# ---------------------------------------------------------------------------


def test_ledger_accumulates_across_runs() -> None:
    """Multiple runs on different tasks should accumulate in the same store."""
    store = LedgerStore(":memory:")
    config = TribunalSettings()
    config.tribunal.selection_threshold = 0.0
    config.tribunal.resample_threshold = 0.0

    orch = Orchestrator(config=config, ledger_store=store)

    for task_file in ("colour_swap_001.json", "copy_identity_003.json"):
        task = load_task_from_file(EXAMPLES_DIR / task_file)
        orch.run(task)

    stats = store.get_stats()
    assert stats["tasks"] == 2
    assert stats["experiment_runs"] == 2
    assert stats["traces"] == 10  # 5 generators × 2 tasks


# ---------------------------------------------------------------------------
# Integration: JSON output round-trip
# ---------------------------------------------------------------------------


def test_run_and_format_json_serialisable() -> None:
    task = load_task_from_file(EXAMPLES_DIR / "colour_swap_001.json")
    config = TribunalSettings()
    config.tribunal.selection_threshold = 0.0
    config.tribunal.resample_threshold = 0.0

    store = LedgerStore(":memory:")
    orch = Orchestrator(config=config, ledger_store=store)
    result = orch.run_and_format(task)

    # Should be JSON-serialisable
    serialised = json.dumps(result, default=str)
    parsed = json.loads(serialised)
    assert parsed["task_id"] == task.task_id
    assert "decision" in parsed
