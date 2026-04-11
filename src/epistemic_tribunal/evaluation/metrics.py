"""Evaluation metrics for Epistemic Tribunal benchmark runs.

All metrics operate on lists of :class:`ExperimentRun` objects collected
after running the tribunal over a dataset.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

from epistemic_tribunal.evaluation import calibration
from epistemic_tribunal.types import DecisionKind, ExperimentRun


def accuracy(runs: list[ExperimentRun]) -> float:
    """Fraction of runs where the selected answer matched the ground truth.

    Only counts runs where both a selection was made *and* a ground-truth
    label is available.
    """
    evaluated = [
        r for r in runs
        if r.decision == DecisionKind.SELECT and r.ground_truth_match is not None
    ]
    if not evaluated:
        return 0.0
    return sum(1 for r in evaluated if r.ground_truth_match) / len(evaluated)


def coverage(runs: list[ExperimentRun]) -> float:
    """Fraction of runs where the tribunal made a selection (not abstained/resampled)."""
    if not runs:
        return 0.0
    selected = sum(1 for r in runs if r.decision == DecisionKind.SELECT)
    return selected / len(runs)


def abstention_rate(runs: list[ExperimentRun]) -> float:
    """Fraction of runs that ended in ABSTAIN."""
    if not runs:
        return 0.0
    return sum(1 for r in runs if r.decision == DecisionKind.ABSTAIN) / len(runs)


def resample_rate(runs: list[ExperimentRun]) -> float:
    """Fraction of runs that requested a resample."""
    if not runs:
        return 0.0
    return sum(1 for r in runs if r.decision == DecisionKind.RESAMPLE) / len(runs)


def decision_distribution(runs: list[ExperimentRun]) -> dict[str, int]:
    """Return a count of each decision kind."""
    return dict(Counter(r.decision.value for r in runs))


def average_duration(runs: list[ExperimentRun]) -> float:
    """Mean run duration in seconds."""
    if not runs:
        return 0.0
    return sum(r.duration_seconds for r in runs) / len(runs)


def summary_report(runs: list[ExperimentRun]) -> dict[str, float | int | dict]:
    """Produce a full summary metrics dictionary."""
    report: dict[str, float | int | dict] = {
        "total_runs": len(runs),
        "accuracy": round(accuracy(runs), 4),
        "coverage": round(coverage(runs), 4),
        "abstention_rate": round(abstention_rate(runs), 4),
        "resample_rate": round(resample_rate(runs), 4),
        "decision_distribution": decision_distribution(runs),
        "avg_duration_seconds": round(average_duration(runs), 4),
    }
    has_confidence = any(
        run.confidence > 0.0
        for run in runs
        if run.decision == DecisionKind.SELECT and run.ground_truth_match is not None
    )
    if has_confidence:
        report["ece"] = round(calibration.expected_calibration_error(runs), 4)
        report["brier_score"] = round(calibration.brier_score(runs), 4)
        report["selective_accuracy_90"] = {
            key: round(value, 4)
            for key, value in calibration.accuracy_at_coverage(runs, 0.9).items()
        }
        report["abstention_quality"] = {
            key: round(value, 4)
            for key, value in calibration.abstention_quality(runs).items()
        }
    return report
