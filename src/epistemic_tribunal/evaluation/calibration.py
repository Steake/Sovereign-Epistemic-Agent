"""Confidence calibration metrics for Epistemic Tribunal benchmark runs.

All metrics operate on lists of :class:`ExperimentRun` objects.
Eligibility rules are applied per-function as documented in each docstring.
"""

from __future__ import annotations

from math import ceil
from typing import Any

from epistemic_tribunal.types import DecisionKind, ExperimentRun


def _eligible_selected(runs: list[ExperimentRun]) -> list[ExperimentRun]:
    """Return runs where decision == SELECT and ground_truth_match is not None."""
    return [
        r for r in runs
        if r.decision == DecisionKind.SELECT and r.ground_truth_match is not None
    ]


def _evaluable(runs: list[ExperimentRun]) -> list[ExperimentRun]:
    """Return all runs where ground_truth_match is not None."""
    return [r for r in runs if r.ground_truth_match is not None]


def expected_calibration_error(
    runs: list[ExperimentRun], n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error (ECE) using equal-width bins.

    Only includes runs with decision == SELECT and ground_truth_match is not None.
    Returns 0.0 if no eligible runs.
    """
    eligible = _eligible_selected(runs)
    if not eligible:
        return 0.0

    total = len(eligible)
    ece = 0.0
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        bin_runs = [
            r for r in eligible
            if lo <= r.confidence < hi or (i == n_bins - 1 and r.confidence == 1.0 and hi == 1.0)
        ]
        if not bin_runs:
            continue
        mean_conf = sum(r.confidence for r in bin_runs) / len(bin_runs)
        mean_acc = sum(1.0 for r in bin_runs if r.ground_truth_match) / len(bin_runs)
        ece += (len(bin_runs) / total) * abs(mean_conf - mean_acc)

    return ece


def brier_score(runs: list[ExperimentRun]) -> float:
    """Compute the Brier score (mean squared error between confidence and outcome).

    Only includes runs with decision == SELECT and ground_truth_match is not None.
    Returns 0.0 if no eligible runs.
    """
    eligible = _eligible_selected(runs)
    if not eligible:
        return 0.0

    total_sq = sum(
        (r.confidence - (1.0 if r.ground_truth_match else 0.0)) ** 2
        for r in eligible
    )
    return total_sq / len(eligible)


def reliability_curve(
    runs: list[ExperimentRun], n_bins: int = 10
) -> list[dict[str, Any]]:
    """Compute the reliability curve as a list of bin dicts.

    Only includes runs with decision == SELECT and ground_truth_match is not None.
    Returns [] if no eligible runs.
    """
    eligible = _eligible_selected(runs)
    if not eligible:
        return []

    curve: list[dict[str, Any]] = []
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        bin_runs = [
            r for r in eligible
            if lo <= r.confidence < hi or (i == n_bins - 1 and r.confidence == 1.0 and hi == 1.0)
        ]
        if not bin_runs:
            continue
        mean_conf = sum(r.confidence for r in bin_runs) / len(bin_runs)
        mean_acc = sum(1.0 for r in bin_runs if r.ground_truth_match) / len(bin_runs)
        curve.append({
            "bin_midpoint": round((lo + hi) / 2, 4),
            "mean_confidence": round(mean_conf, 4),
            "mean_accuracy": round(mean_acc, 4),
            "count": len(bin_runs),
        })

    return curve


def accuracy_at_coverage(
    runs: list[ExperimentRun], coverage_target: float
) -> dict[str, float]:
    """Compute accuracy at a given coverage target.

    Sort eligible selected runs by confidence descending, retain top coverage_target
    fraction, and report accuracy/coverage/threshold of that subset.

    coverage_target must be in (0, 1].
    Returns zeros if no eligible runs.
    """
    eligible = _eligible_selected(runs)
    if not eligible:
        return {"accuracy": 0.0, "coverage": 0.0, "threshold": 0.0}

    sorted_runs = sorted(eligible, key=lambda r: r.confidence, reverse=True)
    retained_count = max(1, ceil(len(eligible) * coverage_target))
    retained = sorted_runs[:retained_count]

    acc = sum(1.0 for r in retained if r.ground_truth_match) / len(retained)
    cov = retained_count / len(eligible)
    threshold = retained[-1].confidence

    return {
        "accuracy": acc,
        "coverage": cov,
        "threshold": threshold,
    }


def abstention_quality(runs: list[ExperimentRun]) -> dict[str, float]:
    """Compute abstention quality metrics.

    Uses all runs where ground_truth_match is not None (does NOT exclude abstentions).

    Returns abstention_rate, wrong_abstention_rate, correct_abstention_rate.
    """
    evaluable = _evaluable(runs)
    if not evaluable:
        return {
            "abstention_rate": 0.0,
            "wrong_abstention_rate": 0.0,
            "correct_abstention_rate": 0.0,
        }

    abstentions = [r for r in evaluable if r.decision == DecisionKind.ABSTAIN]
    abstention_rate = len(abstentions) / len(evaluable)

    if not abstentions:
        return {
            "abstention_rate": abstention_rate,
            "wrong_abstention_rate": 0.0,
            "correct_abstention_rate": 0.0,
        }

    # An abstention on a wrong answer (ground_truth_match == False) is a "correct" abstention
    # An abstention on a correct answer (ground_truth_match == True) is a "wrong" abstention
    wrong_abstentions = sum(1 for r in abstentions if r.ground_truth_match is True)
    correct_abstentions = sum(1 for r in abstentions if r.ground_truth_match is False)

    return {
        "abstention_rate": abstention_rate,
        "wrong_abstention_rate": wrong_abstentions / len(abstentions),
        "correct_abstention_rate": correct_abstentions / len(abstentions),
    }
