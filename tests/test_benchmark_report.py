"""Tests for benchmark_report metric computation and CLI smoke tests.

Covers:
- cohort_metrics computes correctly for each cohort
- good/bad abstention rates computed correctly
- tribunal_lift_over_greedy computed correctly
- interpretation flags (useful, honest) set correctly
- missing oracle metadata: report still works, oracle metrics are null
- CLI: human-readable output works
- CLI: --json output works
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Optional

import pytest
from typer.testing import CliRunner

from epistemic_tribunal.cli import app
from epistemic_tribunal.evaluation.benchmark_report import (
    _bad_abstention_rate,
    _good_abstention_rate,
    _overall_accuracy,
    _selective_accuracy,
    _wrong_pick_rate,
    build_report,
    cohort_metrics,
    tribunal_honest_on_contested_unrecoverable,
    tribunal_lift_over_greedy,
    tribunal_useful_on_contested_recoverable,
)
from epistemic_tribunal.evaluation.benchmark_spec import (
    BenchmarkCohort,
    RecoverabilityStatus,
    TaskBenchmarkAnnotation,
    TaskOracleMetadata,
    TribunalBenchmarkRecord,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ann(
    task_id: str,
    cohort: BenchmarkCohort = BenchmarkCohort.contested_recoverable,
    ci: int = 2,
) -> TaskBenchmarkAnnotation:
    return TaskBenchmarkAnnotation(
        task_id=task_id,
        cohort=cohort,
        contestability_index=ci,
        recoverability_index=3,
        structural_separability=1,
        plausible_hypotheses=[],
        recoverability_status=RecoverabilityStatus.exact_candidate_present,
    )


def _rec(
    task_id: str,
    *,
    arm: str = "greedy",
    decision: str = "select",
    correct: Optional[bool] = True,
    any_correct: Optional[bool] = True,
    cohort: BenchmarkCohort = BenchmarkCohort.contested_recoverable,
    oracle: bool = False,
) -> TribunalBenchmarkRecord:
    ann = _ann(task_id, cohort=cohort)
    oracle_meta: Optional[TaskOracleMetadata] = None
    if oracle:
        oracle_meta = TaskOracleMetadata(
            task_id=task_id,
            oracle_exact_candidate_present=(correct is True),
            oracle_structurally_defensible_candidate_present=True,
        )
    return TribunalBenchmarkRecord(
        task_id=task_id,
        arm_name=arm,
        decision=decision,
        ground_truth_match=correct,
        any_correct_in_pool=any_correct,
        annotation=ann,
        oracle=oracle_meta,
    )


# ---------------------------------------------------------------------------
# cohort_metrics — control_trivial
# ---------------------------------------------------------------------------


def test_cohort_metrics_control_trivial_all_correct() -> None:
    records = [
        _rec(f"t{i}", cohort=BenchmarkCohort.control_trivial, arm="greedy")
        for i in range(5)
    ]
    m = cohort_metrics(records)
    assert m["task_count"] == 5
    assert m["overall_accuracy"] == pytest.approx(1.0)
    assert m["selective_accuracy"] == pytest.approx(1.0)
    assert m["coverage"] == pytest.approx(1.0)
    assert m["wrong_pick_rate"] == pytest.approx(0.0)
    assert m["abstention_rate"] == pytest.approx(0.0)


def test_cohort_metrics_empty() -> None:
    m = cohort_metrics([])
    assert m == {"task_count": 0}


# ---------------------------------------------------------------------------
# cohort_metrics — contested_recoverable
# ---------------------------------------------------------------------------


def test_cohort_metrics_contested_recoverable_mixed() -> None:
    records = [
        _rec("t1", correct=True),     # SELECT correct
        _rec("t2", correct=False),    # SELECT wrong
        _rec("t3", decision="abstain", correct=None, any_correct=True),   # bad abstention
    ]
    m = cohort_metrics(records)
    assert m["task_count"] == 3
    # 1 correct SELECT / 2 SELECT = 0.5
    assert m["selective_accuracy"] == pytest.approx(0.5)
    # 1 correct / 3 total = 0.333...
    assert m["overall_accuracy"] == pytest.approx(1 / 3, rel=1e-3)
    # 2 SELECT / 3 = 0.666...
    assert m["coverage"] == pytest.approx(2 / 3, rel=1e-3)
    # 1 wrong SELECT / 3 = 0.333...
    assert m["wrong_pick_rate"] == pytest.approx(1 / 3, rel=1e-3)
    # 1 non-select / 3 = 0.333...
    assert m["abstention_rate"] == pytest.approx(1 / 3, rel=1e-3)
    # bad abstention: 1 non-select where any_correct=True → 1/1 = 1.0
    assert m["bad_abstention_rate"] == pytest.approx(1.0)
    # good abstention: 0 non-selects where any_correct=False → 0
    assert m["good_abstention_rate"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# cohort_metrics — contested_unrecoverable
# ---------------------------------------------------------------------------


def test_cohort_metrics_contested_unrecoverable_good_abstentions() -> None:
    records = [
        _rec("u1", cohort=BenchmarkCohort.contested_unrecoverable,
             decision="abstain", correct=None, any_correct=False),
        _rec("u2", cohort=BenchmarkCohort.contested_unrecoverable,
             decision="abstain", correct=None, any_correct=False),
        _rec("u3", cohort=BenchmarkCohort.contested_unrecoverable,
             decision="select", correct=False, any_correct=False),
    ]
    m = cohort_metrics(records)
    assert m["task_count"] == 3
    assert m["abstention_rate"] == pytest.approx(2 / 3, rel=1e-3)
    # 2 abstentions where any_correct=False → good_abstention_rate = 2/2 = 1.0
    assert m["good_abstention_rate"] == pytest.approx(1.0)
    assert m["bad_abstention_rate"] == pytest.approx(0.0)
    # 1 wrong select / 3 tasks
    assert m["wrong_pick_rate"] == pytest.approx(1 / 3, rel=1e-3)


# ---------------------------------------------------------------------------
# Abstention quality classification
# ---------------------------------------------------------------------------


def test_good_abstention_rate_only_good() -> None:
    records = [
        _rec("u1", decision="abstain", correct=None, any_correct=False),
        _rec("u2", decision="abstain", correct=None, any_correct=False),
    ]
    assert _good_abstention_rate(records) == pytest.approx(1.0)
    assert _bad_abstention_rate(records) == pytest.approx(0.0)


def test_bad_abstention_rate_only_bad() -> None:
    records = [
        _rec("r1", decision="abstain", correct=None, any_correct=True),
        _rec("r2", decision="abstain", correct=None, any_correct=True),
    ]
    assert _good_abstention_rate(records) == pytest.approx(0.0)
    assert _bad_abstention_rate(records) == pytest.approx(1.0)


def test_no_abstentions_rates_are_zero() -> None:
    records = [_rec("t1", decision="select", correct=True)]
    assert _good_abstention_rate(records) == pytest.approx(0.0)
    assert _bad_abstention_rate(records) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# tribunal_lift_over_greedy
# ---------------------------------------------------------------------------


def test_lift_positive_when_tribunal_better() -> None:
    tribunal = [_rec("t1", correct=True), _rec("t2", correct=True)]    # 100%
    greedy = [_rec("t1", correct=True), _rec("t2", correct=False)]     # 50%
    lift = tribunal_lift_over_greedy(tribunal, greedy)
    assert lift == pytest.approx(0.5)


def test_lift_negative_when_tribunal_worse() -> None:
    tribunal = [_rec("t1", correct=False), _rec("t2", correct=False)]  # 0%
    greedy = [_rec("t1", correct=True), _rec("t2", correct=True)]      # 100%
    lift = tribunal_lift_over_greedy(tribunal, greedy)
    assert lift == pytest.approx(-1.0)


def test_lift_zero_when_equal() -> None:
    records = [_rec("t1", correct=True), _rec("t2", correct=False)]
    lift = tribunal_lift_over_greedy(records, records)
    assert lift == pytest.approx(0.0)


def test_lift_none_when_empty() -> None:
    assert tribunal_lift_over_greedy([], [_rec("t1")]) is None
    assert tribunal_lift_over_greedy([_rec("t1")], []) is None


# ---------------------------------------------------------------------------
# Interpretation flags
# ---------------------------------------------------------------------------


def test_useful_flag_true_when_lift_positive_and_wpr_not_worse() -> None:
    tribunal_cr = [_rec("t1", correct=True), _rec("t2", correct=True)]
    greedy_cr = [_rec("t1", correct=True), _rec("t2", correct=False)]
    lift = tribunal_lift_over_greedy(tribunal_cr, greedy_cr)
    assert tribunal_useful_on_contested_recoverable(lift, tribunal_cr, greedy_cr) is True


def test_useful_flag_false_when_lift_zero() -> None:
    records = [_rec("t1", correct=True)]
    lift = tribunal_lift_over_greedy(records, records)
    assert tribunal_useful_on_contested_recoverable(lift, records, records) is False


def test_honest_flag_true_when_high_good_abstention_and_low_wpr() -> None:
    cu = [
        _rec("u1", cohort=BenchmarkCohort.contested_unrecoverable,
             decision="abstain", correct=None, any_correct=False),
        _rec("u2", cohort=BenchmarkCohort.contested_unrecoverable,
             decision="abstain", correct=None, any_correct=False),
        _rec("u3", cohort=BenchmarkCohort.contested_unrecoverable,
             decision="select", correct=True),
    ]
    assert tribunal_honest_on_contested_unrecoverable(cu) is True


def test_honest_flag_false_when_wpr_too_high() -> None:
    cu = [
        _rec(f"u{i}", cohort=BenchmarkCohort.contested_unrecoverable,
             decision="select", correct=False)
        for i in range(5)
    ]
    assert tribunal_honest_on_contested_unrecoverable(cu) is False


def test_honest_flag_false_on_empty() -> None:
    assert tribunal_honest_on_contested_unrecoverable([]) is False


# ---------------------------------------------------------------------------
# build_report — integration
# ---------------------------------------------------------------------------


def _build_multi_arm_records() -> list[TribunalBenchmarkRecord]:
    records = []
    # greedy arm: 50% sel acc on recoverable
    records += [
        _rec("t1", arm="greedy", correct=True),
        _rec("t2", arm="greedy", correct=False),
        _rec("u1", arm="greedy", cohort=BenchmarkCohort.contested_unrecoverable,
             decision="select", correct=False, any_correct=False),
    ]
    # structural arm: 100% sel acc on recoverable + good abstentions on unrecoverable
    records += [
        _rec("t1", arm="structural", correct=True),
        _rec("t2", arm="structural", correct=True),
        _rec("u1", arm="structural", cohort=BenchmarkCohort.contested_unrecoverable,
             decision="abstain", correct=None, any_correct=False),
    ]
    return records


def test_build_report_global_shape() -> None:
    records = _build_multi_arm_records()
    report = build_report(records, greedy_arm="greedy")
    assert "global" in report
    assert "per_arm" in report
    assert "tribunal_lift_over_greedy" in report
    assert "interpretation" in report


def test_build_report_per_arm_keys() -> None:
    records = _build_multi_arm_records()
    report = build_report(records, greedy_arm="greedy")
    assert "greedy" in report["per_arm"]
    assert "structural" in report["per_arm"]


def test_build_report_lift_computed() -> None:
    records = _build_multi_arm_records()
    report = build_report(records, greedy_arm="greedy")
    lift = report["tribunal_lift_over_greedy"].get("structural")
    assert lift is not None
    assert lift > 0  # structural beats greedy on recoverable


def test_build_report_interpretation_flags() -> None:
    records = _build_multi_arm_records()
    report = build_report(records, greedy_arm="greedy")
    flags = report["interpretation"]["structural"]
    assert flags["tribunal_useful_on_contested_recoverable"] is True
    assert flags["tribunal_honest_on_contested_unrecoverable"] is True


# ---------------------------------------------------------------------------
# Missing oracle metadata
# ---------------------------------------------------------------------------


def test_report_works_without_oracle() -> None:
    records = [
        _rec("t1", correct=True, oracle=False),
        _rec("t2", correct=False, oracle=False),
    ]
    report = build_report(records)
    # Should complete without error; no oracle key in per-arm
    assert "per_arm" in report
    arm_data = report["per_arm"].get("greedy", {})
    # oracle_best_candidate_in_pool_accuracy should NOT be present
    assert "oracle_best_candidate_in_pool_accuracy" not in arm_data


def test_report_with_oracle_includes_metric() -> None:
    records = [
        _rec("t1", correct=True, oracle=True),
        _rec("t2", correct=False, oracle=True),
    ]
    report = build_report(records)
    arm_data = report["per_arm"].get("greedy", {})
    assert "oracle_best_candidate_in_pool_accuracy" in arm_data


# ---------------------------------------------------------------------------
# CLI smoke tests
# ---------------------------------------------------------------------------


runner = CliRunner()


def _write_runs_jsonl(path: Path, records: list[dict]) -> None:
    lines = "\n".join(json.dumps(r) for r in records)
    path.write_text(lines, encoding="utf-8")


def _write_annotations_json(path: Path, records: list[dict]) -> None:
    path.write_text(json.dumps(records), encoding="utf-8")


def _minimal_run(
    task_id: str,
    arm: str = "greedy",
    decision: str = "select",
    correct: Optional[bool] = True,
) -> dict:
    from datetime import datetime, timezone
    return {
        "run_id": f"run-{task_id}-{arm}",
        "task_id": task_id,
        "generator_names": ["mock"],
        "decision": decision,
        "confidence": 0.8,
        "selected_trace_id": None,
        "ground_truth_match": correct,
        "duration_seconds": 0.1,
        "config_snapshot": {},
        "metadata": {"arm_name": arm},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _minimal_ann(task_id: str, cohort: str = "contested_recoverable") -> dict:
    return {
        "task_id": task_id,
        "cohort": cohort,
        "contestability_index": 2,
        "recoverability_index": 3,
        "structural_separability": 1,
        "plausible_hypotheses": [],
        "recoverability_status": "exact_candidate_present",
    }


def test_cli_human_output(tmp_path: Path) -> None:
    runs_path = tmp_path / "runs.jsonl"
    ann_path = tmp_path / "anns.json"

    runs = [
        _minimal_run("t1", arm="greedy", correct=True),
        _minimal_run("t2", arm="greedy", correct=False),
        _minimal_run("t1", arm="structural", correct=True),
        _minimal_run("t2", arm="structural", correct=True),
    ]
    anns = [_minimal_ann("t1"), _minimal_ann("t2")]

    _write_runs_jsonl(runs_path, runs)
    _write_annotations_json(ann_path, anns)

    result = runner.invoke(
        app,
        ["benchmark-usefulness", "--runs", str(runs_path), "--annotations", str(ann_path)],
    )
    assert result.exit_code == 0, result.output
    assert "Selective Accuracy" in result.output
    assert "contested-recoverable" in result.output


def test_cli_json_output(tmp_path: Path) -> None:
    runs_path = tmp_path / "runs.jsonl"
    ann_path = tmp_path / "anns.json"

    runs = [_minimal_run("t1", correct=True)]
    anns = [_minimal_ann("t1")]

    _write_runs_jsonl(runs_path, runs)
    _write_annotations_json(ann_path, anns)

    result = runner.invoke(
        app,
        ["benchmark-usefulness", "--runs", str(runs_path), "--annotations", str(ann_path), "--json"],
    )
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.output)
    assert "global" in parsed
    assert "per_arm" in parsed
    assert "interpretation" in parsed


def test_cli_no_matching_annotations(tmp_path: Path) -> None:
    runs_path = tmp_path / "runs.jsonl"
    ann_path = tmp_path / "anns.json"

    runs = [_minimal_run("t1")]
    anns = [_minimal_ann("t99")]  # different task_id, no overlap

    _write_runs_jsonl(runs_path, runs)
    _write_annotations_json(ann_path, anns)

    result = runner.invoke(
        app,
        ["benchmark-usefulness", "--runs", str(runs_path), "--annotations", str(ann_path)],
    )
    assert result.exit_code == 0
    assert "No runs matched" in result.output


def test_cli_requires_runs_or_ledger(tmp_path: Path) -> None:
    ann_path = tmp_path / "anns.json"
    ann_path.write_text(json.dumps([_minimal_ann("t1")]))

    result = runner.invoke(
        app,
        ["benchmark-usefulness", "--annotations", str(ann_path)],
    )
    assert result.exit_code == 1
    assert "Provide either" in result.output
