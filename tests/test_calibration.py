"""Tests for confidence calibration metrics and integration.

Covers:
- perfect calibration
- overconfident model
- underconfident model
- all abstentions
- empty input
- mixed old (confidence=0.0) and new (real confidence) records
- persistence/readback of confidence
- tribunal calibrate CLI behaviour
- summary report gating behaviour
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pytest
from typer.testing import CliRunner

from epistemic_tribunal.cli import app
from epistemic_tribunal.evaluation import calibration
from epistemic_tribunal.evaluation.metrics import summary_report
from epistemic_tribunal.ledger.models import ExperimentRunRecord
from epistemic_tribunal.ledger.store import LedgerStore
from epistemic_tribunal.ledger.writer import LedgerWriter
from epistemic_tribunal.types import DecisionKind, ExperimentRun


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(
    *,
    decision: DecisionKind = DecisionKind.SELECT,
    confidence: float = 0.5,
    ground_truth_match: Optional[bool] = True,
    task_id: str = "t",
) -> ExperimentRun:
    return ExperimentRun(
        task_id=task_id,
        decision=decision,
        confidence=confidence,
        ground_truth_match=ground_truth_match,
        generator_names=["greedy"],
    )


# ---------------------------------------------------------------------------
# expected_calibration_error
# ---------------------------------------------------------------------------

class TestECE:
    def test_empty(self) -> None:
        assert calibration.expected_calibration_error([]) == 0.0

    def test_no_eligible(self) -> None:
        runs = [_run(decision=DecisionKind.ABSTAIN)]
        assert calibration.expected_calibration_error(runs) == 0.0

    def test_perfect_calibration(self) -> None:
        """When confidence == accuracy in every bin, ECE should be 0."""
        runs = (
            [_run(confidence=0.25, ground_truth_match=True)] * 1
            + [_run(confidence=0.25, ground_truth_match=False)] * 3
            + [_run(confidence=0.75, ground_truth_match=True)] * 3
            + [_run(confidence=0.75, ground_truth_match=False)] * 1
        )
        ece = calibration.expected_calibration_error(runs)
        assert ece == pytest.approx(0.0, abs=1e-6)

    def test_overconfident(self) -> None:
        """High confidence but all wrong -> ECE close to confidence."""
        runs = [_run(confidence=0.95, ground_truth_match=False) for _ in range(20)]
        ece = calibration.expected_calibration_error(runs)
        assert ece > 0.8

    def test_underconfident(self) -> None:
        """Low confidence but all correct -> ECE close to 1 - confidence."""
        runs = [_run(confidence=0.05, ground_truth_match=True) for _ in range(20)]
        ece = calibration.expected_calibration_error(runs)
        assert ece > 0.8


# ---------------------------------------------------------------------------
# brier_score
# ---------------------------------------------------------------------------

class TestBrierScore:
    def test_empty(self) -> None:
        assert calibration.brier_score([]) == 0.0

    def test_perfect(self) -> None:
        runs = [_run(confidence=1.0, ground_truth_match=True) for _ in range(10)]
        assert calibration.brier_score(runs) == pytest.approx(0.0)

    def test_worst(self) -> None:
        runs = [_run(confidence=1.0, ground_truth_match=False) for _ in range(10)]
        assert calibration.brier_score(runs) == pytest.approx(1.0)

    def test_middle(self) -> None:
        runs = [_run(confidence=0.5, ground_truth_match=True) for _ in range(10)]
        assert calibration.brier_score(runs) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# reliability_curve
# ---------------------------------------------------------------------------

class TestReliabilityCurve:
    def test_empty(self) -> None:
        assert calibration.reliability_curve([]) == []

    def test_single_bin(self) -> None:
        runs = [_run(confidence=0.55, ground_truth_match=True) for _ in range(5)]
        curve = calibration.reliability_curve(runs)
        assert len(curve) == 1
        assert curve[0]["count"] == 5
        assert curve[0]["mean_accuracy"] == pytest.approx(1.0)

    def test_structure(self) -> None:
        runs = [
            _run(confidence=0.15, ground_truth_match=True),
            _run(confidence=0.85, ground_truth_match=False),
        ]
        curve = calibration.reliability_curve(runs)
        assert len(curve) == 2
        for entry in curve:
            assert "bin_midpoint" in entry
            assert "mean_confidence" in entry
            assert "mean_accuracy" in entry
            assert "count" in entry


# ---------------------------------------------------------------------------
# accuracy_at_coverage
# ---------------------------------------------------------------------------

class TestAccuracyAtCoverage:
    def test_empty(self) -> None:
        result = calibration.accuracy_at_coverage([], coverage_target=0.9)
        assert result == {"accuracy": 0.0, "coverage": 0.0, "threshold": 0.0}

    def test_full_coverage(self) -> None:
        runs = [_run(confidence=0.8, ground_truth_match=True) for _ in range(10)]
        result = calibration.accuracy_at_coverage(runs, coverage_target=1.0)
        assert result["accuracy"] == pytest.approx(1.0)
        assert result["coverage"] == pytest.approx(1.0)

    def test_selective(self) -> None:
        runs = (
            [_run(confidence=0.9, ground_truth_match=True)] * 5
            + [_run(confidence=0.1, ground_truth_match=False)] * 5
        )
        result = calibration.accuracy_at_coverage(runs, coverage_target=0.5)
        assert result["accuracy"] == pytest.approx(1.0)
        assert result["threshold"] >= 0.9


# ---------------------------------------------------------------------------
# abstention_quality
# ---------------------------------------------------------------------------

class TestAbstentionQuality:
    def test_empty(self) -> None:
        result = calibration.abstention_quality([])
        assert result["abstention_rate"] == 0.0

    def test_no_abstentions(self) -> None:
        runs = [_run(decision=DecisionKind.SELECT, ground_truth_match=True)]
        result = calibration.abstention_quality(runs)
        assert result["abstention_rate"] == 0.0
        assert result["wrong_abstention_rate"] == 0.0
        assert result["correct_abstention_rate"] == 0.0

    def test_all_abstentions(self) -> None:
        """All runs are ABSTAIN with ground_truth_match set."""
        runs = [
            _run(decision=DecisionKind.ABSTAIN, ground_truth_match=False),
            _run(decision=DecisionKind.ABSTAIN, ground_truth_match=False),
        ]
        result = calibration.abstention_quality(runs)
        assert result["abstention_rate"] == pytest.approx(1.0)
        assert result["correct_abstention_rate"] == pytest.approx(1.0)
        assert result["wrong_abstention_rate"] == pytest.approx(0.0)

    def test_mixed(self) -> None:
        runs = [
            _run(decision=DecisionKind.SELECT, ground_truth_match=True),
            _run(decision=DecisionKind.ABSTAIN, ground_truth_match=True),
            _run(decision=DecisionKind.ABSTAIN, ground_truth_match=False),
        ]
        result = calibration.abstention_quality(runs)
        assert result["abstention_rate"] == pytest.approx(2 / 3)
        assert result["wrong_abstention_rate"] == pytest.approx(0.5)
        assert result["correct_abstention_rate"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Mixed old and new records (confidence=0.0 vs real confidence)
# ---------------------------------------------------------------------------

class TestMixedRecords:
    def test_calibration_with_old_records(self) -> None:
        """Old records (confidence=0.0) mixed with new records should not crash."""
        old_runs = [_run(confidence=0.0, ground_truth_match=True) for _ in range(5)]
        new_runs = [_run(confidence=0.7, ground_truth_match=True) for _ in range(5)]
        all_runs = old_runs + new_runs

        # Should not crash
        ece = calibration.expected_calibration_error(all_runs)
        bs = calibration.brier_score(all_runs)
        curve = calibration.reliability_curve(all_runs)
        sel = calibration.accuracy_at_coverage(all_runs, coverage_target=0.9)
        abst = calibration.abstention_quality(all_runs)

        assert isinstance(ece, float)
        assert isinstance(bs, float)
        assert isinstance(curve, list)
        assert isinstance(sel, dict)
        assert isinstance(abst, dict)

    def test_summary_report_gates_on_confidence(self) -> None:
        """summary_report should include calibration keys only when confidence > 0."""
        old_runs = [_run(confidence=0.0, ground_truth_match=True) for _ in range(5)]
        report = summary_report(old_runs)
        assert "ece" not in report
        assert "brier_score" not in report

    def test_summary_report_includes_calibration(self) -> None:
        """summary_report should include calibration keys when confidence > 0."""
        new_runs = [_run(confidence=0.8, ground_truth_match=True) for _ in range(5)]
        report = summary_report(new_runs)
        assert "ece" in report
        assert "brier_score" in report
        assert "selective_accuracy_90" in report
        assert "abstention_quality" in report


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_confidence_persisted_and_read_back(self, tmp_path: Path) -> None:
        """Confidence written to ledger should be readable."""
        db_path = tmp_path / "test.db"
        store = LedgerStore(str(db_path))
        writer = LedgerWriter(store)

        run = ExperimentRun(
            task_id="persist_test",
            decision=DecisionKind.SELECT,
            confidence=0.87,
            ground_truth_match=True,
            generator_names=["greedy"],
        )
        writer.write_run(run)

        rows = store.get_experiment_runs()
        store.close()

        assert len(rows) == 1
        assert rows[0]["confidence"] == pytest.approx(0.87)

    def test_old_ledger_without_confidence_column(self, tmp_path: Path) -> None:
        """Opening a ledger created before the confidence column should work."""
        import sqlite3
        db_path = tmp_path / "old.db"
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY, domain TEXT, description TEXT,
                train_examples_count INTEGER, created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS traces (
                trace_id TEXT PRIMARY KEY, task_id TEXT, generator_name TEXT,
                confidence_score REAL, answer_json TEXT, reasoning_steps_json TEXT,
                created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS decisions (
                decision_id TEXT PRIMARY KEY, task_id TEXT, decision TEXT,
                selected_trace_id TEXT, confidence REAL, reasoning TEXT,
                scores_json TEXT, created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS failures (
                failure_id TEXT PRIMARY KEY, task_id TEXT, selected_trace_id TEXT,
                all_candidate_trace_ids_json TEXT, violated_invariants_json TEXT,
                disagreement_pattern TEXT, diagnosis TEXT, notes TEXT,
                ground_truth_match INTEGER, created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS invariant_violations (
                violation_id TEXT PRIMARY KEY, task_id TEXT, trace_id TEXT,
                invariant_name TEXT, note TEXT, created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS experiment_runs (
                run_id TEXT PRIMARY KEY, task_id TEXT, decision TEXT,
                selected_trace_id TEXT, ground_truth_match INTEGER,
                duration_seconds REAL, generator_names_json TEXT,
                config_snapshot_json TEXT, created_at TEXT
            );
        """)
        # Insert a row without the confidence column
        conn.execute(
            """INSERT INTO experiment_runs
               (run_id, task_id, decision, selected_trace_id,
                ground_truth_match, duration_seconds,
                generator_names_json, config_snapshot_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("r1", "t1", "select", None, 1, 0.5, '["greedy"]', '{}', "2024-01-01"),
        )
        conn.commit()
        conn.close()

        # LedgerStore should migrate schema and read old rows
        store = LedgerStore(str(db_path))
        rows = store.get_experiment_runs()
        store.close()

        assert len(rows) == 1
        # confidence should default to 0.0 or None for old rows
        assert rows[0].get("confidence", 0.0) in (0.0, None)


# ---------------------------------------------------------------------------
# CLI calibrate command
# ---------------------------------------------------------------------------

runner = CliRunner()


class TestCalibrateCLI:
    def _setup_ledger(self, tmp_path: Path, *, with_confidence: bool = True) -> Path:
        db_path = tmp_path / "cal_ledger.db"
        store = LedgerStore(str(db_path))
        writer = LedgerWriter(store)
        for i in range(10):
            run = ExperimentRun(
                task_id=f"task_{i}",
                decision=DecisionKind.SELECT,
                confidence=0.7 if with_confidence else 0.0,
                ground_truth_match=(i % 2 == 0),
                generator_names=["greedy"],
            )
            writer.write_run(run)
        store.close()
        return db_path

    def test_calibrate_json(self, tmp_path: Path) -> None:
        db_path = self._setup_ledger(tmp_path)
        result = runner.invoke(app, ["calibrate", "--ledger", str(db_path), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "ece" in data
        assert "brier_score" in data
        assert "reliability_curve" in data
        assert "selective_accuracy_90" in data
        assert "abstention_quality" in data

    def test_calibrate_table(self, tmp_path: Path) -> None:
        db_path = self._setup_ledger(tmp_path)
        result = runner.invoke(app, ["calibrate", "--ledger", str(db_path)])
        assert result.exit_code == 0
        assert "Calibration Report" in result.output

    def test_calibrate_no_confidence(self, tmp_path: Path) -> None:
        db_path = self._setup_ledger(tmp_path, with_confidence=False)
        result = runner.invoke(app, ["calibrate", "--ledger", str(db_path)])
        assert result.exit_code == 0
        assert "No runs with usable confidence" in result.output
