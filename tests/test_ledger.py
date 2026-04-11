"""Tests for the failure ledger (store + writer)."""

from __future__ import annotations

import json
import uuid
from datetime import datetime

import pytest

from epistemic_tribunal.ledger.models import (
    DecisionRecord,
    ExperimentRunRecord,
    FailureRecordRow,
    InvariantViolationRecord,
    TaskRecord,
    TraceRecord,
)
from epistemic_tribunal.ledger.store import LedgerStore
from epistemic_tribunal.ledger.writer import LedgerWriter
from epistemic_tribunal.types import (
    CandidateTrace,
    CritiqueResult,
    DecisionKind,
    ExperimentRun,
    FailureRecord,
    Task,
    TribunalDecision,
)


# ---------------------------------------------------------------------------
# Store-level tests
# ---------------------------------------------------------------------------


def test_store_initialises(in_memory_store: LedgerStore) -> None:
    stats = in_memory_store.get_stats()
    assert stats["tasks"] == 0
    assert stats["failures"] == 0


def test_upsert_task(in_memory_store: LedgerStore) -> None:
    rec = TaskRecord(
        task_id="t001",
        domain="arc_like",
        description="Test task",
        train_examples_count=2,
    )
    in_memory_store.upsert_task(rec)
    stats = in_memory_store.get_stats()
    assert stats["tasks"] == 1


def test_upsert_task_idempotent(in_memory_store: LedgerStore) -> None:
    rec = TaskRecord(task_id="t001", domain="arc_like", description="", train_examples_count=1)
    in_memory_store.upsert_task(rec)
    in_memory_store.upsert_task(rec)
    assert in_memory_store.get_stats()["tasks"] == 1


def test_insert_trace(in_memory_store: LedgerStore) -> None:
    in_memory_store.upsert_task(TaskRecord(task_id="t1", domain="arc_like", description="", train_examples_count=0))
    rec = TraceRecord(
        trace_id=str(uuid.uuid4()),
        task_id="t1",
        generator_name="greedy",
        confidence_score=0.75,
        answer_json=json.dumps([[1, 2], [3, 4]]),
        reasoning_steps_json=json.dumps(["step 1"]),
    )
    in_memory_store.insert_trace(rec)
    assert in_memory_store.get_stats()["traces"] == 1


def test_insert_failure(in_memory_store: LedgerStore) -> None:
    in_memory_store.upsert_task(TaskRecord(task_id="t1", domain="arc_like", description="", train_examples_count=0))
    rec = FailureRecordRow(
        failure_id=str(uuid.uuid4()),
        task_id="t1",
        selected_trace_id=None,
        all_candidate_trace_ids_json=json.dumps([]),
        violated_invariants_json=json.dumps(["object_count_preserved"]),
        disagreement_pattern="3/5 unique",
        diagnosis="Test failure",
        notes="",
        ground_truth_match=0,
    )
    in_memory_store.insert_failure(rec)
    assert in_memory_store.get_stats()["failures"] == 1


def test_get_failure_patterns_filtered(in_memory_store: LedgerStore) -> None:
    in_memory_store.upsert_task(TaskRecord(task_id="t1", domain="arc_like", description="", train_examples_count=0))
    in_memory_store.upsert_task(TaskRecord(task_id="t2", domain="arc_like", description="", train_examples_count=0))
    for tid in ("t1", "t1", "t2"):
        in_memory_store.insert_failure(FailureRecordRow(
            failure_id=str(uuid.uuid4()),
            task_id=tid,
            selected_trace_id=None,
            all_candidate_trace_ids_json="[]",
            violated_invariants_json="[]",
            disagreement_pattern="",
            diagnosis="",
            notes="",
            ground_truth_match=None,
        ))
    patterns = in_memory_store.get_failure_patterns("t1")
    assert len(patterns) == 2
    assert all(p["task_id"] == "t1" for p in patterns)


def test_get_task_summary_not_found(in_memory_store: LedgerStore) -> None:
    summary = in_memory_store.get_task_summary("nonexistent")
    assert "error" in summary


def test_get_task_summary_found(in_memory_store: LedgerStore) -> None:
    in_memory_store.upsert_task(TaskRecord(task_id="t1", domain="arc_like", description="", train_examples_count=1))
    summary = in_memory_store.get_task_summary("t1")
    assert "task" in summary
    assert summary["task"]["task_id"] == "t1"


def test_insert_experiment_run(in_memory_store: LedgerStore) -> None:
    in_memory_store.upsert_task(TaskRecord(task_id="t1", domain="arc_like", description="", train_examples_count=0))
    rec = ExperimentRunRecord(
        run_id=str(uuid.uuid4()),
        task_id="t1",
        decision="select",
        confidence=0.75,
        selected_trace_id=str(uuid.uuid4()),
        ground_truth_match=1,
        duration_seconds=0.5,
        generator_names_json=json.dumps(["greedy"]),
        config_snapshot_json="{}",
    )
    in_memory_store.insert_run(rec)
    assert in_memory_store.get_stats()["experiment_runs"] == 1


# ---------------------------------------------------------------------------
# Writer-level tests
# ---------------------------------------------------------------------------


def test_writer_write_task(in_memory_store: LedgerStore, simple_task: Task) -> None:
    writer = LedgerWriter(in_memory_store)
    writer.write_task(simple_task)
    assert in_memory_store.get_stats()["tasks"] == 1


def test_writer_write_traces(in_memory_store: LedgerStore, simple_task: Task) -> None:
    from epistemic_tribunal.generators.greedy import GreedyGenerator
    writer = LedgerWriter(in_memory_store)
    writer.write_task(simple_task)
    traces = [GreedyGenerator(seed=42).generate(simple_task)]
    writer.write_traces(simple_task, traces)
    assert in_memory_store.get_stats()["traces"] == 1


def test_writer_write_failure(in_memory_store: LedgerStore, simple_task: Task) -> None:
    writer = LedgerWriter(in_memory_store)
    writer.write_task(simple_task)
    failure = FailureRecord(
        task_id=simple_task.task_id,
        diagnosis="test failure",
        notes="written by test",
    )
    writer.write_failure(failure)
    assert in_memory_store.get_stats()["failures"] == 1


def test_writer_write_run(in_memory_store: LedgerStore, simple_task: Task) -> None:
    writer = LedgerWriter(in_memory_store)
    writer.write_task(simple_task)
    run = ExperimentRun(
        task_id=simple_task.task_id,
        generator_names=["greedy"],
        decision=DecisionKind.SELECT,
        duration_seconds=0.1,
    )
    writer.write_run(run)
    assert in_memory_store.get_stats()["experiment_runs"] == 1
