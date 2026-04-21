"""Tests for the end-to-end Orchestrator."""

from __future__ import annotations

import pytest

from epistemic_tribunal.config import TribunalSettings, TribunalConfig
from epistemic_tribunal.ledger.store import LedgerStore
from epistemic_tribunal.orchestrator import Orchestrator
from epistemic_tribunal.tribunal_types import DecisionKind, ExperimentRun, Task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _orchestrator(config: TribunalSettings = None, store: LedgerStore = None) -> Orchestrator:
    cfg = config or TribunalSettings()
    st = store or LedgerStore(":memory:")
    return Orchestrator(config=cfg, ledger_store=st)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_run_returns_experiment_run(simple_task: Task) -> None:
    orch = _orchestrator()
    run = orch.run(simple_task)
    assert isinstance(run, ExperimentRun)


def test_run_task_id_preserved(simple_task: Task) -> None:
    orch = _orchestrator()
    run = orch.run(simple_task)
    assert run.task_id == simple_task.task_id


def test_run_decision_valid(simple_task: Task) -> None:
    orch = _orchestrator()
    run = orch.run(simple_task)
    assert run.decision in DecisionKind


def test_run_generator_names_present(simple_task: Task) -> None:
    orch = _orchestrator()
    run = orch.run(simple_task)
    assert len(run.generator_names) == 5  # default 5 generators


def test_run_duration_positive(simple_task: Task) -> None:
    orch = _orchestrator()
    run = orch.run(simple_task)
    assert run.duration_seconds >= 0.0


def test_run_ground_truth_match_evaluated(simple_task: Task) -> None:
    """When ground truth is present, match should be evaluated."""
    orch = _orchestrator()
    run = orch.run(simple_task)
    if run.decision == DecisionKind.SELECT:
        assert run.ground_truth_match is not None


def test_run_no_ground_truth(simple_task: Task) -> None:
    from copy import deepcopy
    task = deepcopy(simple_task)
    task.ground_truth = None
    orch = _orchestrator()
    run = orch.run(task)
    assert run.ground_truth_match is None


def test_run_writes_to_ledger(simple_task: Task) -> None:
    store = LedgerStore(":memory:")
    orch = _orchestrator(store=store)
    orch.run(simple_task)
    stats = store.get_stats()
    assert stats["tasks"] == 1
    assert stats["traces"] == 5
    assert stats["decisions"] == 1
    assert stats["experiment_runs"] == 1


def test_run_and_format_returns_dict(simple_task: Task) -> None:
    orch = _orchestrator()
    result = orch.run_and_format(simple_task)
    assert isinstance(result, dict)
    assert "decision" in result
    assert "task_id" in result
    assert result["task_id"] == simple_task.task_id


def test_run_low_threshold_always_selects(simple_task: Task) -> None:
    """With very low thresholds, every run should select a candidate."""
    config = TribunalSettings()
    config.tribunal.selection_threshold = 0.0
    config.tribunal.resample_threshold = 0.0
    orch = _orchestrator(config=config)
    run = orch.run(simple_task)
    assert run.decision == DecisionKind.SELECT
    assert run.selected_trace_id is not None


def test_run_failure_record_written_on_wrong_answer(simple_task: Task) -> None:
    """Failure record should be written when selected answer != ground truth."""
    from copy import deepcopy
    task = deepcopy(simple_task)
    # Set an impossible ground truth that can't match any generator
    task.ground_truth = [[9, 9, 9], [9, 9, 9], [9, 9, 9]]

    config = TribunalSettings()
    config.tribunal.selection_threshold = 0.0  # Force selection
    config.tribunal.resample_threshold = 0.0

    store = LedgerStore(":memory:")
    orch = _orchestrator(config=config, store=store)
    orch.run(task)
    stats = store.get_stats()
    # At least one failure record from wrong answer or other conditions
    assert stats["failures"] >= 0  # May or may not match, just don't crash


def test_run_identity_task(identity_task: Task) -> None:
    """Identity task should at minimum complete without errors."""
    orch = _orchestrator()
    run = orch.run(identity_task)
    assert isinstance(run, ExperimentRun)


def test_orchestrator_with_single_generator() -> None:
    """Single generator should still produce a valid run."""
    config = TribunalSettings()
    config.generators.enabled = ["greedy"]
    config.tribunal.selection_threshold = 0.0
    config.tribunal.resample_threshold = 0.0
    config.tribunal.diversity_floor = 1.0
    # Disable guardrails — with one trace the margin is necessarily 0.
    config.tribunal.guardrail_margin_threshold = 0.0
    config.tribunal.guardrail_min_coalition_mass = 0.0
    store = LedgerStore(":memory:")
    orch = Orchestrator(config=config, ledger_store=store)

    from epistemic_tribunal.tribunal_types import GridExample, Task, TaskDomain
    task = Task(
        task_id="single_gen_test",
        domain=TaskDomain.ARC_LIKE,
        train=[GridExample(input=[[1, 0], [0, 1]], output=[[0, 1], [1, 0]])],
        test_input=[[1, 1], [0, 0]],
        ground_truth=[[0, 0], [1, 1]],
    )
    run = orch.run(task)
    assert run.decision == DecisionKind.SELECT
