"""Tests for calibration metrics, persistence, and CLI reporting."""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path

import pytest
from typer.testing import CliRunner

from epistemic_tribunal.cli import app
from epistemic_tribunal.config import TribunalSettings
from epistemic_tribunal.evaluation.benchmark import BenchmarkRunner, experiment_run_from_row
from epistemic_tribunal.evaluation.calibration import (
    abstention_quality,
    accuracy_at_coverage,
    brier_score,
    expected_calibration_error,
    reliability_curve,
)
from epistemic_tribunal.evaluation.metrics import summary_report
from epistemic_tribunal.ledger.models import TaskRecord
from epistemic_tribunal.ledger.store import LedgerStore
from epistemic_tribunal.ledger.writer import LedgerWriter
from epistemic_tribunal.types import DecisionKind, ExperimentRun, Task

runner = CliRunner()


def _run(
    *,
    decision: DecisionKind = DecisionKind.SELECT,
    confidence: float = 0.0,
    match: bool | None = True,
    task_id: str | None = None,
) -> ExperimentRun:
    return ExperimentRun(
        task_id=task_id or str(uuid.uuid4()),
        generator_names=["heuristic"],
        decision=decision,
        confidence=confidence,
        ground_truth_match=match,
        duration_seconds=0.1,
    )


def test_perfect_calibration_metrics() -> None:
    runs = [
        _run(confidence=1.0, match=True),
        _run(confidence=1.0, match=True),
        _run(confidence=0.0, match=False),
        _run(confidence=0.0, match=False),
    ]

    assert expected_calibration_error(runs, n_bins=2) == 0.0
    assert brier_score(runs) == 0.0
    assert reliability_curve(runs, n_bins=2) == [
        {
            "bin_midpoint": 0.25,
            "mean_confidence": 0.0,
            "mean_accuracy": 0.0,
            "count": 2,
        },
        {
            "bin_midpoint": 0.75,
            "mean_confidence": 1.0,
            "mean_accuracy": 1.0,
            "count": 2,
        },
    ]


def test_overconfident_model_metrics() -> None:
    runs = [
        _run(confidence=0.9, match=True),
        _run(confidence=0.9, match=False),
    ]

    assert expected_calibration_error(runs, n_bins=10) == pytest.approx(0.4)
    assert brier_score(runs) == pytest.approx(0.41)


def test_underconfident_model_metrics() -> None:
    runs = [
        _run(confidence=0.2, match=True),
        _run(confidence=0.2, match=True),
    ]

    assert expected_calibration_error(runs, n_bins=5) == pytest.approx(0.8)
    assert brier_score(runs) == pytest.approx(0.64)


def test_all_abstentions_and_empty_inputs() -> None:
    abstain_runs = [
        _run(decision=DecisionKind.ABSTAIN, confidence=0.8, match=False),
        _run(decision=DecisionKind.ABSTAIN, confidence=0.4, match=True),
    ]

    assert expected_calibration_error(abstain_runs) == 0.0
    assert brier_score(abstain_runs) == 0.0
    assert reliability_curve(abstain_runs) == []
    assert accuracy_at_coverage(abstain_runs, 0.9) == {
        "accuracy": 0.0,
        "coverage": 0.0,
        "threshold": 0.0,
    }
    assert abstention_quality(abstain_runs) == {
        "abstention_rate": 1.0,
        "wrong_abstention_rate": 0.5,
        "correct_abstention_rate": 0.5,
    }
    assert abstention_quality([]) == {
        "abstention_rate": 0.0,
        "wrong_abstention_rate": 0.0,
        "correct_abstention_rate": 0.0,
    }


def test_accuracy_at_coverage_sorts_by_confidence() -> None:
    runs = [
        _run(confidence=0.9, match=True),
        _run(confidence=0.7, match=False),
        _run(confidence=0.2, match=True),
    ]

    report = accuracy_at_coverage(runs, 0.9)
    assert report == {
        "accuracy": pytest.approx(2 / 3),
        "coverage": 1.0,
        "threshold": 0.2,
    }


def test_summary_report_gates_on_real_confidence_values() -> None:
    old_runs = [
        _run(decision=DecisionKind.ABSTAIN, confidence=0.0, match=False),
        _run(decision=DecisionKind.RESAMPLE, confidence=0.0, match=True),
    ]
    mixed_runs = old_runs + [_run(confidence=0.8, match=True)]

    old_report = summary_report(old_runs)
    mixed_report = summary_report(mixed_runs)

    assert "ece" not in old_report
    assert "brier_score" not in old_report
    assert mixed_report["ece"] == pytest.approx(0.2)
    assert mixed_report["brier_score"] == pytest.approx(0.04)
    assert mixed_report["selective_accuracy_90"]["threshold"] == pytest.approx(0.8)


def test_confidence_persists_and_reads_back(in_memory_store: LedgerStore, simple_task: Task) -> None:
    in_memory_store.upsert_task(
        TaskRecord(
            task_id=simple_task.task_id,
            domain=simple_task.domain.value,
            description=simple_task.description,
            train_examples_count=len(simple_task.train),
        )
    )
    run = ExperimentRun(
        task_id=simple_task.task_id,
        generator_names=["greedy"],
        decision=DecisionKind.SELECT,
        confidence=0.73,
        ground_truth_match=True,
        duration_seconds=0.1,
    )

    LedgerWriter(in_memory_store).write_run(run)
    row = in_memory_store.get_experiment_runs()[0]
    restored = experiment_run_from_row(row)

    assert row["confidence"] == pytest.approx(0.73)
    assert restored.confidence == pytest.approx(0.73)


def test_legacy_ledger_schema_is_migrated_for_confidence(tmp_path: Path) -> None:
    ledger_path = tmp_path / "legacy.sqlite3"
    conn = sqlite3.connect(ledger_path)
    conn.executescript(
        """
        CREATE TABLE experiment_runs (
            run_id TEXT PRIMARY KEY,
            task_id TEXT,
            decision TEXT,
            selected_trace_id TEXT,
            ground_truth_match INTEGER,
            duration_seconds REAL,
            generator_names_json TEXT,
            config_snapshot_json TEXT,
            created_at TEXT
        );
        """
    )
    conn.execute(
        """
        INSERT INTO experiment_runs
        (run_id, task_id, decision, selected_trace_id, ground_truth_match,
         duration_seconds, generator_names_json, config_snapshot_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            "legacy-task",
            "select",
            None,
            1,
            0.1,
            json.dumps(["greedy"]),
            "{}",
            "2026-01-01T00:00:00+00:00",
        ),
    )
    conn.commit()
    conn.close()

    store = LedgerStore(ledger_path)
    try:
        rows = store.get_experiment_runs()
    finally:
        store.close()

    assert rows[0]["confidence"] == 0.0
    restored = experiment_run_from_row(rows[0])
    assert restored.confidence == 0.0


def test_cli_calibrate_prints_report_for_confident_runs(tmp_path: Path) -> None:
    ledger_path = tmp_path / "calibration.sqlite3"
    store = LedgerStore(ledger_path)
    store.upsert_task(TaskRecord(task_id="t1", domain="arc_like", description="", train_examples_count=0))
    writer = LedgerWriter(store)
    writer.write_run(_run(task_id="t1", confidence=0.8, match=True))
    writer.write_run(_run(task_id="t1", confidence=0.6, match=False))
    store.close()

    result = runner.invoke(app, ["calibrate", "--ledger", str(ledger_path)])

    assert result.exit_code == 0
    assert "Calibration Report" in result.stdout
    assert "Reliability Curve" in result.stdout
    assert "ECE" in result.stdout
    assert "Brier score" in result.stdout


def test_cli_calibrate_handles_missing_confidence(tmp_path: Path) -> None:
    ledger_path = tmp_path / "no_confidence.sqlite3"
    store = LedgerStore(ledger_path)
    store.upsert_task(TaskRecord(task_id="t1", domain="arc_like", description="", train_examples_count=0))
    LedgerWriter(store).write_run(_run(task_id="t1", confidence=0.0, match=True))
    store.close()

    result = runner.invoke(app, ["calibrate", "--ledger", str(ledger_path)])

    assert result.exit_code == 0
    assert "No usable confidence-bearing runs found" in result.stdout


def test_benchmark_resume_restores_confidence(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    task_path = dataset_path / "task.json"
    task_path.write_text(
        json.dumps(
            {
                "task_id": "task_resume",
                "description": "identity",
                "train": [{"input": [[1]], "output": [[1]]}],
                "test": [{"input": [[1]]}],
                "ground_truth": [[1]],
            }
        )
    )

    ledger_path = tmp_path / "benchmark.sqlite3"
    config = TribunalSettings()
    config.ledger.path = str(ledger_path)
    config.benchmark.checkpoint_every_n_tasks = 1
    config.tribunal.selection_threshold = 0.0
    config.tribunal.resample_threshold = 0.0
    config.tribunal.diversity_floor = 1.0

    benchmark = BenchmarkRunner(config=config)
    benchmark.run(dataset_path)
    resumed = benchmark.run(dataset_path, resume=True)

    assert len(resumed) == 1
    assert resumed[0].confidence > 0.0
