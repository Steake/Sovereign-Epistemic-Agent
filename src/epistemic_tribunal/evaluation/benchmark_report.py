"""Cohort-aware benchmark reporting for the Tribunal Usefulness Benchmark.

All metric functions are pure functions operating on lists of
:class:`TribunalBenchmarkRecord`.  The report is structured so that:

- trivial tasks (control_trivial) expose baseline accuracy
- recoverable contested tasks expose whether adjudication helps
- unrecoverable contested tasks expose whether abstention is honest

The benchmark thesis is only satisfied when:
  tribunal_useful_on_contested_recoverable = True
  tribunal_honest_on_contested_unrecoverable = True

These booleans are computed transparently from the underlying metrics.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

from epistemic_tribunal.evaluation.benchmark_spec import (
    BenchmarkCohort,
    TribunalBenchmarkRecord,
)

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

_SELECT = "select"
_NON_SELECT = {"resample", "abstain"}


def _is_select(r: TribunalBenchmarkRecord) -> bool:
    return r.decision == _SELECT


def _is_non_select(r: TribunalBenchmarkRecord) -> bool:
    return r.decision in _NON_SELECT


def _overall_accuracy(records: list[TribunalBenchmarkRecord]) -> float:
    """Fraction of ALL records that produced a correct selection."""
    if not records:
        return 0.0
    return sum(1 for r in records if r.ground_truth_match is True) / len(records)


def _selective_accuracy(records: list[TribunalBenchmarkRecord]) -> float:
    """Accuracy on SELECT-only decisions.

    Reads as: "of the tasks we committed to, what fraction were correct?"
    Returns 0.0 when there are no SELECT decisions (not 1.0, to avoid
    misreading silence as success).
    """
    selected = [r for r in records if _is_select(r) and r.ground_truth_match is not None]
    if not selected:
        return 0.0
    return sum(1 for r in selected if r.ground_truth_match is True) / len(selected)


def _coverage(records: list[TribunalBenchmarkRecord]) -> float:
    """Fraction of tasks where the tribunal committed to a selection."""
    if not records:
        return 0.0
    return sum(1 for r in records if _is_select(r)) / len(records)


def _wrong_pick_rate(records: list[TribunalBenchmarkRecord]) -> float:
    """Fraction of all tasks where we made an incorrect selection.

    Reads as: "of every task in scope, what fraction did we get wrong by
    picking something?"  Different from 1 - selective_accuracy because it
    counts abstentions as neither wrong nor right.
    """
    if not records:
        return 0.0
    wrong = sum(
        1
        for r in records
        if _is_select(r) and r.ground_truth_match is False
    )
    return wrong / len(records)


def _abstention_rate(records: list[TribunalBenchmarkRecord]) -> float:
    """Fraction of tasks that ended in abstain or resample."""
    if not records:
        return 0.0
    return sum(1 for r in records if _is_non_select(r)) / len(records)


def _good_abstention_rate(records: list[TribunalBenchmarkRecord]) -> float:
    """Fraction of non-select decisions that were correct to make.

    "Correct" = the pool had no viable candidate (any_correct_in_pool is False).
    High value = the system abstains when it should.
    """
    non_select = [r for r in records if _is_non_select(r)]
    if not non_select:
        return 0.0
    good = sum(1 for r in non_select if r.any_correct_in_pool is False)
    return good / len(non_select)


def _bad_abstention_rate(records: list[TribunalBenchmarkRecord]) -> float:
    """Fraction of non-select decisions that missed a recoverable solution.

    "Missed" = the pool had a correct candidate but we abstained anyway.
    High value = the system is overly cautious and leaves wins on the table.
    """
    non_select = [r for r in records if _is_non_select(r)]
    if not non_select:
        return 0.0
    bad = sum(1 for r in non_select if r.any_correct_in_pool is True)
    return bad / len(non_select)


# ---------------------------------------------------------------------------
# Per-cohort metrics block
# ---------------------------------------------------------------------------


def cohort_metrics(records: list[TribunalBenchmarkRecord]) -> dict:
    """Compute the full metric block for a single cohort slice.

    Returns a flat dict with unambiguous key names.  All rates are in [0, 1].
    """
    n = len(records)
    if n == 0:
        return {"task_count": 0}

    return {
        "task_count": n,
        "overall_accuracy": round(_overall_accuracy(records), 4),
        "selective_accuracy": round(_selective_accuracy(records), 4),
        "coverage": round(_coverage(records), 4),
        "wrong_pick_rate": round(_wrong_pick_rate(records), 4),
        "abstention_rate": round(_abstention_rate(records), 4),
        "good_abstention_rate": round(_good_abstention_rate(records), 4),
        "bad_abstention_rate": round(_bad_abstention_rate(records), 4),
    }


# ---------------------------------------------------------------------------
# Oracle-dependent metrics
# ---------------------------------------------------------------------------


def best_candidate_in_pool_accuracy(
    records: list[TribunalBenchmarkRecord],
) -> Optional[float]:
    """Upper-bound accuracy if the system always picked the best candidate.

    Returns None when no oracle data is available in the record set.
    """
    oracle_records = [r for r in records if r.oracle is not None]
    if not oracle_records:
        return None
    hits = sum(1 for r in oracle_records if r.oracle.oracle_exact_candidate_present)  # type: ignore[union-attr]
    return round(hits / len(oracle_records), 4)


# ---------------------------------------------------------------------------
# Tribunal-lift metric
# ---------------------------------------------------------------------------


def tribunal_lift_over_greedy(
    tribunal_records: list[TribunalBenchmarkRecord],
    greedy_records: list[TribunalBenchmarkRecord],
) -> Optional[float]:
    """Selective accuracy difference: tribunal minus greedy.

    A positive value means the tribunal picked more correct answers than
    greedy selection on the same set of tasks.  None if either pool is empty.
    """
    if not tribunal_records or not greedy_records:
        return None
    return round(
        _selective_accuracy(tribunal_records) - _selective_accuracy(greedy_records), 4
    )


# ---------------------------------------------------------------------------
# Utility booleans
# ---------------------------------------------------------------------------


_USEFUL_LIFT_THRESHOLD = 0.0   # strict: any positive lift counts
_HONEST_GOOD_ABSTENTION_MIN = 0.5  # at least half of abstentions should be on unrecoverable tasks
_HONEST_WRONG_PICK_MAX = 0.15      # less than 15% wrong picks on unrecoverable cohort


def tribunal_useful_on_contested_recoverable(
    lift: Optional[float],
    tribunal_cr_records: list[TribunalBenchmarkRecord],
    greedy_cr_records: list[TribunalBenchmarkRecord],
) -> bool:
    """True when tribunal beats greedy on the contested_recoverable cohort.

    Checks both selective accuracy lift AND wrong-pick reduction so that
    a system cannot game this by refusing to pick anything.
    """
    if lift is None:
        return False
    if lift <= _USEFUL_LIFT_THRESHOLD:
        return False
    # Also require that wrong-pick rate is not worse than greedy
    tribunal_wpr = _wrong_pick_rate(tribunal_cr_records)
    greedy_wpr = _wrong_pick_rate(greedy_cr_records)
    return tribunal_wpr <= greedy_wpr


def tribunal_honest_on_contested_unrecoverable(
    cu_records: list[TribunalBenchmarkRecord],
) -> bool:
    """True when the tribunal shows honest abstention behaviour on unrecoverable tasks.

    Conditions (both must hold):
    1. Good abstention rate >= 0.5  (abstains mostly on tasks with no viable candidate)
    2. Wrong pick rate < 0.15       (does not commit to wrong answers)
    """
    if not cu_records:
        return False
    gar = _good_abstention_rate(cu_records)
    wpr = _wrong_pick_rate(cu_records)
    return gar >= _HONEST_GOOD_ABSTENTION_MIN and wpr < _HONEST_WRONG_PICK_MAX


# ---------------------------------------------------------------------------
# Full benchmark report
# ---------------------------------------------------------------------------


def build_report(
    records: list[TribunalBenchmarkRecord],
    *,
    greedy_arm: str = "greedy",
) -> dict:
    """Build the complete benchmark usefulness report.

    Parameters
    ----------
    records:
        All ``TribunalBenchmarkRecord`` objects across all arms.
    greedy_arm:
        Name of the arm to treat as the greedy baseline for lift computation.

    Returns
    -------
    dict
        Structured report with global metrics, per-arm metrics, per-cohort
        metrics, tribunal-specific utility metrics, and interpretation flags.
    """
    # Group by arm
    by_arm: dict[str, list[TribunalBenchmarkRecord]] = defaultdict(list)
    for r in records:
        by_arm[r.arm_name].append(r)

    arm_names = sorted(by_arm)

    # --- Global metrics (across all arms, all cohorts) ---
    global_metrics: dict = {
        "total_records": len(records),
        "arms": arm_names,
        "overall_accuracy": round(_overall_accuracy(records), 4),
        "selective_accuracy": round(_selective_accuracy(records), 4),
        "coverage": round(_coverage(records), 4),
        "wrong_pick_rate": round(_wrong_pick_rate(records), 4),
        "abstention_rate": round(_abstention_rate(records), 4),
    }

    # --- Per-arm metrics (each arm independently) ---
    per_arm: dict[str, dict] = {}
    for arm, arm_records in by_arm.items():
        # Split by cohort within this arm
        by_cohort: dict[str, list[TribunalBenchmarkRecord]] = defaultdict(list)
        for r in arm_records:
            by_cohort[r.annotation.cohort.value].append(r)

        arm_entry: dict = {
            "overall": cohort_metrics(arm_records),
        }
        for cohort in BenchmarkCohort:
            slice_records = by_cohort.get(cohort.value, [])
            arm_entry[cohort.value] = cohort_metrics(slice_records)

        # Oracle-based metric (best possible accuracy given the pool)
        bcp = best_candidate_in_pool_accuracy(arm_records)
        if bcp is not None:
            arm_entry["oracle_best_candidate_in_pool_accuracy"] = bcp

        per_arm[arm] = arm_entry

    # --- Tribunal-lift over greedy (contested_recoverable only) ---
    lift_results: dict = {}
    if greedy_arm in by_arm:
        greedy_records = by_arm[greedy_arm]
        greedy_cr = [
            r for r in greedy_records
            if r.annotation.cohort == BenchmarkCohort.contested_recoverable
        ]
        for arm, arm_records in by_arm.items():
            if arm == greedy_arm:
                continue
            arm_cr = [
                r for r in arm_records
                if r.annotation.cohort == BenchmarkCohort.contested_recoverable
            ]
            lift = tribunal_lift_over_greedy(arm_cr, greedy_cr)
            lift_results[arm] = lift

    # --- Interpretation flags ---
    # Computed for each non-greedy arm against greedy
    interpretation: dict[str, dict] = {}
    if greedy_arm in by_arm:
        greedy_records = by_arm[greedy_arm]
        greedy_cr = [
            r for r in greedy_records
            if r.annotation.cohort == BenchmarkCohort.contested_recoverable
        ]
        for arm in arm_names:
            if arm == greedy_arm:
                continue
            arm_records = by_arm[arm]
            arm_cr = [
                r for r in arm_records
                if r.annotation.cohort == BenchmarkCohort.contested_recoverable
            ]
            arm_cu = [
                r for r in arm_records
                if r.annotation.cohort == BenchmarkCohort.contested_unrecoverable
            ]
            lift = lift_results.get(arm)
            interpretation[arm] = {
                "tribunal_useful_on_contested_recoverable": tribunal_useful_on_contested_recoverable(
                    lift, arm_cr, greedy_cr
                ),
                "tribunal_honest_on_contested_unrecoverable": tribunal_honest_on_contested_unrecoverable(
                    arm_cu
                ),
                "tribunal_lift_over_greedy_on_contested_recoverable": lift,
            }

    return {
        "global": global_metrics,
        "per_arm": per_arm,
        "tribunal_lift_over_greedy": lift_results,
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# Records construction helper
# ---------------------------------------------------------------------------


def records_from_runs(
    runs: list,  # list[ExperimentRun] — avoid circular import
    annotations: "dict[str, object]",
    oracle: "Optional[dict[str, object]]" = None,
) -> list[TribunalBenchmarkRecord]:
    """Convert ExperimentRun objects into TribunalBenchmarkRecord objects.

    Runs without a matching annotation are silently skipped (they cannot
    be placed in a cohort).

    Parameters
    ----------
    runs:
        ExperimentRun objects from the tribunal pipeline.
    annotations:
        Mapping of task_id → TaskBenchmarkAnnotation (from load_annotations).
    oracle:
        Optional mapping of task_id → TaskOracleMetadata.

    Returns
    -------
    list[TribunalBenchmarkRecord]
    """
    from epistemic_tribunal.evaluation.benchmark_spec import TaskBenchmarkAnnotation, TaskOracleMetadata

    result: list[TribunalBenchmarkRecord] = []
    for run in runs:
        ann = annotations.get(run.task_id)
        if ann is None:
            continue
        assert isinstance(ann, TaskBenchmarkAnnotation)

        oracle_meta: Optional[TaskOracleMetadata] = None
        if oracle:
            o = oracle.get(run.task_id)
            if o is not None:
                assert isinstance(o, TaskOracleMetadata)
                oracle_meta = o

        arm = run.metadata.get("arm_name", "default")

        result.append(
            TribunalBenchmarkRecord(
                task_id=run.task_id,
                arm_name=arm,
                decision=run.decision.value,
                ground_truth_match=run.ground_truth_match,
                any_correct_in_pool=run.metadata.get("any_correct"),
                annotation=ann,
                oracle=oracle_meta,
            )
        )
    return result
