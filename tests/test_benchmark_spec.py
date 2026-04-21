"""Tests for benchmark_spec domain models.

Covers:
- Valid annotation construction
- Invalid cohort string raises ValidationError
- CI/RI/SS out-of-range raises ValidationError
- Cohort consistency guard (contested + CI=0)
- RecoverabilityStatus values
- TaskOracleMetadata construction
- TribunalBenchmarkRecord construction
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from epistemic_tribunal.evaluation.benchmark_spec import (
    BenchmarkCohort,
    RecoverabilityStatus,
    TaskBenchmarkAnnotation,
    TaskOracleMetadata,
    TribunalBenchmarkRecord,
)


# ---------------------------------------------------------------------------
# BenchmarkCohort
# ---------------------------------------------------------------------------


def test_cohort_enum_values() -> None:
    assert BenchmarkCohort.control_trivial.value == "control_trivial"
    assert BenchmarkCohort.contested_recoverable.value == "contested_recoverable"
    assert BenchmarkCohort.contested_unrecoverable.value == "contested_unrecoverable"


def test_cohort_from_string() -> None:
    assert BenchmarkCohort("control_trivial") == BenchmarkCohort.control_trivial


# ---------------------------------------------------------------------------
# RecoverabilityStatus
# ---------------------------------------------------------------------------


def test_recoverability_status_values() -> None:
    assert RecoverabilityStatus("exact_candidate_present") == RecoverabilityStatus.exact_candidate_present
    assert RecoverabilityStatus("no_viable_candidate_present") == RecoverabilityStatus.no_viable_candidate_present


# ---------------------------------------------------------------------------
# TaskBenchmarkAnnotation — valid
# ---------------------------------------------------------------------------


def _valid_annotation(**overrides) -> TaskBenchmarkAnnotation:
    base = dict(
        task_id="task_001",
        cohort=BenchmarkCohort.contested_recoverable,
        contestability_index=2,
        recoverability_index=3,
        structural_separability=1,
        plausible_hypotheses=["swap colours", "transpose"],
        recoverability_status=RecoverabilityStatus.exact_candidate_present,
    )
    base.update(overrides)
    return TaskBenchmarkAnnotation.model_validate(base)


def test_valid_annotation_constructs() -> None:
    ann = _valid_annotation()
    assert ann.task_id == "task_001"
    assert ann.cohort == BenchmarkCohort.contested_recoverable
    assert ann.contestability_index == 2
    assert ann.annotation_notes is None


def test_annotation_notes_optional() -> None:
    ann = _valid_annotation(annotation_notes="needs review")
    assert ann.annotation_notes == "needs review"


def test_control_trivial_ci_zero_allowed() -> None:
    # control_trivial cohort is allowed to have CI=0
    ann = _valid_annotation(
        cohort=BenchmarkCohort.control_trivial,
        contestability_index=0,
    )
    assert ann.contestability_index == 0


# ---------------------------------------------------------------------------
# TaskBenchmarkAnnotation — invalid
# ---------------------------------------------------------------------------


def test_invalid_cohort_raises() -> None:
    with pytest.raises(ValidationError):
        TaskBenchmarkAnnotation.model_validate(
            dict(
                task_id="t",
                cohort="unknown_cohort",
                contestability_index=1,
                recoverability_index=1,
                structural_separability=1,
                plausible_hypotheses=[],
                recoverability_status="exact_candidate_present",
            )
        )


@pytest.mark.parametrize("field", ["contestability_index", "recoverability_index", "structural_separability"])
def test_ordinal_too_high_raises(field: str) -> None:
    data = dict(
        task_id="t",
        cohort="control_trivial",
        contestability_index=0,
        recoverability_index=0,
        structural_separability=0,
        plausible_hypotheses=[],
        recoverability_status="exact_candidate_present",
    )
    data[field] = 5  # max is 4
    with pytest.raises(ValidationError):
        TaskBenchmarkAnnotation.model_validate(data)


@pytest.mark.parametrize("field", ["contestability_index", "recoverability_index", "structural_separability"])
def test_ordinal_negative_raises(field: str) -> None:
    data = dict(
        task_id="t",
        cohort="control_trivial",
        contestability_index=0,
        recoverability_index=0,
        structural_separability=0,
        plausible_hypotheses=[],
        recoverability_status="exact_candidate_present",
    )
    data[field] = -1
    with pytest.raises(ValidationError):
        TaskBenchmarkAnnotation.model_validate(data)


def test_contested_with_ci_zero_raises() -> None:
    with pytest.raises(ValidationError, match="contestability_index=0"):
        TaskBenchmarkAnnotation.model_validate(
            dict(
                task_id="t",
                cohort="contested_recoverable",
                contestability_index=0,  # invalid for contested
                recoverability_index=2,
                structural_separability=1,
                plausible_hypotheses=[],
                recoverability_status="exact_candidate_present",
            )
        )


# ---------------------------------------------------------------------------
# TaskOracleMetadata
# ---------------------------------------------------------------------------


def test_oracle_metadata_constructs() -> None:
    meta = TaskOracleMetadata.model_validate(
        dict(
            task_id="task_001",
            oracle_exact_candidate_present=True,
            oracle_best_candidate_overlap=0.95,
            oracle_structurally_defensible_candidate_present=True,
        )
    )
    assert meta.oracle_exact_candidate_present is True
    assert meta.oracle_best_candidate_overlap == pytest.approx(0.95)


def test_oracle_overlap_out_of_range_raises() -> None:
    with pytest.raises(ValidationError):
        TaskOracleMetadata.model_validate(
            dict(
                task_id="t",
                oracle_exact_candidate_present=False,
                oracle_best_candidate_overlap=1.5,  # > 1.0
                oracle_structurally_defensible_candidate_present=False,
            )
        )


# ---------------------------------------------------------------------------
# TribunalBenchmarkRecord
# ---------------------------------------------------------------------------


def test_record_constructs() -> None:
    ann = _valid_annotation()
    rec = TribunalBenchmarkRecord(
        task_id="task_001",
        arm_name="greedy",
        decision="select",
        ground_truth_match=True,
        any_correct_in_pool=True,
        annotation=ann,
    )
    assert rec.oracle is None
    assert rec.arm_name == "greedy"


def test_record_with_oracle() -> None:
    ann = _valid_annotation()
    oracle = TaskOracleMetadata.model_validate(
        dict(
            task_id="task_001",
            oracle_exact_candidate_present=True,
            oracle_structurally_defensible_candidate_present=True,
        )
    )
    rec = TribunalBenchmarkRecord(
        task_id="task_001",
        arm_name="structural",
        decision="select",
        ground_truth_match=False,
        annotation=ann,
        oracle=oracle,
    )
    assert rec.oracle is not None
    assert rec.oracle.oracle_exact_candidate_present is True
