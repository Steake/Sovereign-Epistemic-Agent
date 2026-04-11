import uuid
from typer.testing import CliRunner

from epistemic_tribunal.cli import app
from epistemic_tribunal.evaluation import calibration
from epistemic_tribunal.ledger.models import ExperimentRunRecord, TaskRecord
from epistemic_tribunal.ledger.store import LedgerStore
from epistemic_tribunal.types import DecisionKind, ExperimentRun

def _run(decision, match, conf):
    return ExperimentRun(
        run_id=str(uuid.uuid4()),
        task_id="t1",
        decision=decision,
        selected_trace_id="tr1" if decision == DecisionKind.SELECT else None,
        ground_truth_match=match,
        confidence=conf,
    )

def test_empty_input():
    runs = []
    assert calibration.expected_calibration_error(runs) == 0.0
    assert calibration.brier_score(runs) == 0.0
    assert calibration.reliability_curve(runs) == []
    acc_cov = calibration.accuracy_at_coverage(runs, 0.9)
    assert acc_cov["accuracy"] == 0.0
    assert acc_cov["coverage"] == 0.0
    
    abst_qual = calibration.abstention_quality(runs)
    assert abst_qual["abstention_rate"] == 0.0
    assert abst_qual["wrong_abstention_rate"] == 0.0

def test_perfect_calibration():
    runs = [
        _run(DecisionKind.SELECT, True, 1.0),
        _run(DecisionKind.SELECT, True, 1.0),
        _run(DecisionKind.SELECT, False, 0.0),
        _run(DecisionKind.SELECT, False, 0.0),
    ]
    assert calibration.expected_calibration_error(runs) == 0.0
    assert calibration.brier_score(runs) == 0.0

def test_overconfident():
    runs = [
        _run(DecisionKind.SELECT, False, 1.0),
        _run(DecisionKind.SELECT, False, 0.9),
    ]
    # conf=1.0, acc=0 -> ECE = 1.0
    # conf=0.9, acc=0 -> ECE = 0.9
    ece = calibration.expected_calibration_error(runs)
    assert ece > 0.9

def test_underconfident():
    runs = [
        _run(DecisionKind.SELECT, True, 0.1),
        _run(DecisionKind.SELECT, True, 0.2),
    ]
    ece = calibration.expected_calibration_error(runs)
    assert ece > 0.7

def test_all_abstentions():
    runs = [
        _run(DecisionKind.ABSTAIN, True, 0.0),
        _run(DecisionKind.ABSTAIN, False, 0.0),
    ]
    assert calibration.expected_calibration_error(runs) == 0.0
    
    abst_qual = calibration.abstention_quality(runs)
    assert abst_qual["abstention_rate"] == 1.0
    assert abst_qual["wrong_abstention_rate"] == 0.5
    assert abst_qual["correct_abstention_rate"] == 0.5

def test_mixed_old_and_new_records():
    runs = [
        _run(DecisionKind.SELECT, True, 0.0), # old record
        _run(DecisionKind.SELECT, False, 0.0), # old record
        _run(DecisionKind.SELECT, True, 0.9),
        _run(DecisionKind.SELECT, False, 0.1),
    ]
    ece = calibration.expected_calibration_error(runs)
    assert ece >= 0.0
    # ensure it doesn't crash
    curve = calibration.reliability_curve(runs)
    assert len(curve) > 0

def test_summary_report_gating():
    from epistemic_tribunal.evaluation import metrics
    runs_old = [
        _run(DecisionKind.SELECT, True, 0.0),
        _run(DecisionKind.SELECT, False, 0.0),
    ]
    report_old = metrics.summary_report(runs_old)
    assert "ece" not in report_old
    assert "brier_score" not in report_old

    runs_new = [
        _run(DecisionKind.SELECT, True, 0.9),
        _run(DecisionKind.SELECT, False, 0.1),
    ]
    report_new = metrics.summary_report(runs_new)
    assert "ece" in report_new
    assert "brier_score" in report_new
    assert "selective_accuracy_90" in report_new
    assert "abstention_quality" in report_new


def test_cli_calibrate_empty_ledger(tmp_path):
    ledger_file = tmp_path / "ledger.db"
    store = LedgerStore(ledger_file)
    store.close()
    
    runner = CliRunner()
    result = runner.invoke(app, ["calibrate", "--ledger", str(ledger_file)])
    assert result.exit_code == 0
    assert "no runs with usable confidence scores" in result.stdout

def test_cli_calibrate_with_records(tmp_path):
    ledger_file = tmp_path / "ledger.db"
    store = LedgerStore(ledger_file)
    
    store.upsert_task(TaskRecord(task_id="t1", domain="arc_like", description="", train_examples_count=0))
    store.insert_run(ExperimentRunRecord(
        run_id=str(uuid.uuid4()),
        task_id="t1",
        decision="select",
        selected_trace_id="tr1",
        ground_truth_match=1,
        confidence=0.95,
        duration_seconds=1.0,
        generator_names_json="[]",
        config_snapshot_json="{}"
    ))
    store.close()
    
    runner = CliRunner()
    result = runner.invoke(app, ["calibrate", "--ledger", str(ledger_file)])
    assert result.exit_code == 0
    assert "Calibration Report" in result.stdout
    assert "ECE:" in result.stdout
    assert "Brier Score:" in result.stdout
