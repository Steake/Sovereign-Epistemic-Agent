"""Domain models for the Tribunal Usefulness Benchmark.

These models are the authoritative schema for benchmark annotations,
oracle metadata, and per-run benchmark records.  They are intentionally
separate from the core pipeline types so that the benchmark layer can
evolve independently without touching the main data flow.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class BenchmarkCohort(str, Enum):
    """The three benchmark cohorts that stratify task difficulty and recoverability."""

    control_trivial = "control_trivial"
    contested_recoverable = "contested_recoverable"
    contested_unrecoverable = "contested_unrecoverable"


class RecoverabilityStatus(str, Enum):
    """How close the candidate pool came to a correct answer on this task."""

    exact_candidate_present = "exact_candidate_present"
    near_miss_candidate_present = "near_miss_candidate_present"
    structurally_defensible_candidate_present = "structurally_defensible_candidate_present"
    no_viable_candidate_present = "no_viable_candidate_present"


# ---------------------------------------------------------------------------
# Ordinal score bounds
# ---------------------------------------------------------------------------

_ORDINAL_MIN = 0
_ORDINAL_MAX = 4  # CI / RI / SS are integers in [0, 4]


def _validate_ordinal(value: int, name: str) -> int:
    if not (_ORDINAL_MIN <= value <= _ORDINAL_MAX):
        raise ValueError(
            f"{name} must be an integer in [{_ORDINAL_MIN}, {_ORDINAL_MAX}], got {value}"
        )
    return value


# ---------------------------------------------------------------------------
# Core annotation model
# ---------------------------------------------------------------------------


class TaskBenchmarkAnnotation(BaseModel):
    """Per-task annotation that assigns cohort membership and difficulty signals.

    Parameters
    ----------
    task_id:
        Must match the ``task_id`` used in the tribunal run ledger.
    cohort:
        Which of the three benchmark cohorts this task belongs to.
    contestability_index:
        Ordinal 0–4.  How much the generator pool disagrees on this task.
        0 = unanimous, 4 = maximally contested.
    recoverability_index:
        Ordinal 0–4.  How close the best candidate is to the correct answer.
        0 = no candidate even close, 4 = exact match present.
    structural_separability:
        Ordinal 0–4.  How well structural signals distinguish the best candidate.
        0 = signals are useless, 4 = unambiguous structural winner.
    plausible_hypotheses:
        Informal list of competing interpretations a human annotator considered.
    recoverability_status:
        Categorical label for the quality of the best candidate in the pool.
    annotation_notes:
        Optional free-text comment from the annotator.
    """

    task_id: str
    cohort: BenchmarkCohort
    contestability_index: int = Field(..., ge=_ORDINAL_MIN, le=_ORDINAL_MAX)
    recoverability_index: int = Field(..., ge=_ORDINAL_MIN, le=_ORDINAL_MAX)
    structural_separability: int = Field(..., ge=_ORDINAL_MIN, le=_ORDINAL_MAX)
    plausible_hypotheses: list[str] = Field(default_factory=list)
    recoverability_status: RecoverabilityStatus
    annotation_notes: Optional[str] = None

    @model_validator(mode="after")
    def _check_cohort_consistency(self) -> "TaskBenchmarkAnnotation":
        """Warn-level guard: contested cohorts should have CI >= 1."""
        contested = {
            BenchmarkCohort.contested_recoverable,
            BenchmarkCohort.contested_unrecoverable,
        }
        if self.cohort in contested and self.contestability_index == 0:
            raise ValueError(
                f"task_id={self.task_id!r}: cohort={self.cohort.value!r} "
                "but contestability_index=0 (must be >= 1 for contested cohorts)."
            )
        return self


# ---------------------------------------------------------------------------
# Oracle metadata model
# ---------------------------------------------------------------------------


class TaskOracleMetadata(BaseModel):
    """Ground-truth oracle signals for one task.

    These are derived from the gold answer and provide an upper-bound view
    of what an ideal adjudicator could have achieved.
    """

    task_id: str
    oracle_exact_candidate_present: bool
    oracle_best_candidate_overlap: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Cell-level overlap fraction between best candidate and gold answer.",
    )
    oracle_structurally_defensible_candidate_present: bool
    oracle_notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Per-run joined record
# ---------------------------------------------------------------------------


class TribunalBenchmarkRecord(BaseModel):
    """One evaluated run joined with its benchmark annotation and optional oracle data.

    Consumers should treat ``oracle`` as nullable; report logic must degrade
    gracefully when oracle data is absent.
    """

    # Slim copy of run fields we need for metric computation
    task_id: str
    arm_name: str = Field(
        default="default",
        description="Which benchmark arm produced this run (e.g. 'greedy', 'structural').",
    )
    decision: str = Field(description="DecisionKind value: 'select', 'resample', 'abstain'.")
    ground_truth_match: Optional[bool] = None
    any_correct_in_pool: Optional[bool] = Field(
        default=None,
        description=(
            "True if any candidate in the pool matched ground truth. "
            "Populated from ExperimentRun.metadata['any_correct']."
        ),
    )

    annotation: TaskBenchmarkAnnotation
    oracle: Optional[TaskOracleMetadata] = None
