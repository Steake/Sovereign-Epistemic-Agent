"""Confidence calibration metrics for benchmark experiment runs."""

from __future__ import annotations

import math

from epistemic_tribunal.types import DecisionKind, ExperimentRun


def _clamp_confidence(value: float) -> float:
    return min(1.0, max(0.0, value))


def _eligible_selected_runs(runs: list[ExperimentRun]) -> list[ExperimentRun]:
    return [
        run
        for run in runs
        if run.decision == DecisionKind.SELECT and run.ground_truth_match is not None
    ]


def _evaluable_runs(runs: list[ExperimentRun]) -> list[ExperimentRun]:
    return [run for run in runs if run.ground_truth_match is not None]


def _bin_index(confidence: float, n_bins: int) -> int:
    return min(int(_clamp_confidence(confidence) * n_bins), n_bins - 1)


def expected_calibration_error(runs: list[ExperimentRun], n_bins: int = 10) -> float:
    if n_bins <= 0:
        raise ValueError(f"n_bins must be a positive integer, got {n_bins}.")
    eligible_runs = _eligible_selected_runs(runs)
    if not eligible_runs:
        return 0.0

    bins: list[list[ExperimentRun]] = [[] for _ in range(n_bins)]
    for run in eligible_runs:
        bins[_bin_index(run.confidence, n_bins)].append(run)

    total = len(eligible_runs)
    ece = 0.0
    for bucket in bins:
        if not bucket:
            continue
        mean_confidence = sum(_clamp_confidence(run.confidence) for run in bucket) / len(bucket)
        mean_accuracy = sum(1.0 if run.ground_truth_match else 0.0 for run in bucket) / len(bucket)
        ece += abs(mean_confidence - mean_accuracy) * (len(bucket) / total)
    return ece


def brier_score(runs: list[ExperimentRun]) -> float:
    eligible_runs = _eligible_selected_runs(runs)
    if not eligible_runs:
        return 0.0

    return sum(
        (_clamp_confidence(run.confidence) - (1.0 if run.ground_truth_match else 0.0)) ** 2
        for run in eligible_runs
    ) / len(eligible_runs)


def reliability_curve(runs: list[ExperimentRun], n_bins: int = 10) -> list[dict]:
    if n_bins <= 0:
        raise ValueError(f"n_bins must be a positive integer, got {n_bins}.")
    eligible_runs = _eligible_selected_runs(runs)
    if not eligible_runs:
        return []

    bins: list[list[ExperimentRun]] = [[] for _ in range(n_bins)]
    for run in eligible_runs:
        bins[_bin_index(run.confidence, n_bins)].append(run)

    curve: list[dict] = []
    for index, bucket in enumerate(bins):
        if not bucket:
            continue
        mean_confidence = sum(_clamp_confidence(run.confidence) for run in bucket) / len(bucket)
        mean_accuracy = sum(1.0 if run.ground_truth_match else 0.0 for run in bucket) / len(bucket)
        curve.append(
            {
                "bin_midpoint": (index + 0.5) / n_bins,
                "mean_confidence": mean_confidence,
                "mean_accuracy": mean_accuracy,
                "count": len(bucket),
            }
        )
    return curve


def accuracy_at_coverage(runs: list[ExperimentRun], coverage_target: float) -> dict:
    if not 0.0 < coverage_target <= 1.0:
        raise ValueError("coverage_target must be in (0, 1].")

    eligible_runs = _eligible_selected_runs(runs)
    if not eligible_runs:
        return {"accuracy": 0.0, "coverage": 0.0, "threshold": 0.0}

    ranked = sorted(
        eligible_runs,
        key=lambda run: (_clamp_confidence(run.confidence), run.timestamp),
        reverse=True,
    )
    retained_count = max(1, math.ceil(len(ranked) * coverage_target))
    retained = ranked[:retained_count]
    accuracy = sum(1.0 if run.ground_truth_match else 0.0 for run in retained) / len(retained)
    threshold = min(_clamp_confidence(run.confidence) for run in retained)
    return {
        "accuracy": accuracy,
        "coverage": retained_count / len(ranked),
        "threshold": threshold,
    }


def abstention_quality(runs: list[ExperimentRun]) -> dict:
    evaluable_runs = _evaluable_runs(runs)
    if not evaluable_runs:
        return {
            "abstention_rate": 0.0,
            "wrong_abstention_rate": 0.0,
            "correct_abstention_rate": 0.0,
        }

    abstentions = [
        run for run in evaluable_runs if run.decision == DecisionKind.ABSTAIN
    ]
    abstention_rate = len(abstentions) / len(evaluable_runs)
    if not abstentions:
        return {
            "abstention_rate": abstention_rate,
            "wrong_abstention_rate": 0.0,
            "correct_abstention_rate": 0.0,
        }

    wrong_abstentions = sum(1 for run in abstentions if run.ground_truth_match is False)
    correct_abstentions = sum(1 for run in abstentions if run.ground_truth_match is True)
    return {
        "abstention_rate": abstention_rate,
        "wrong_abstention_rate": wrong_abstentions / len(abstentions),
        "correct_abstention_rate": correct_abstentions / len(abstentions),
    }
