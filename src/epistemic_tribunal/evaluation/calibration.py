from typing import Sequence
import math

from epistemic_tribunal.types import DecisionKind, ExperimentRun


def expected_calibration_error(runs: Sequence[ExperimentRun], n_bins: int = 10) -> float:
    eligible_runs = [r for r in runs if r.decision == DecisionKind.SELECT and r.ground_truth_match is not None]
    if not eligible_runs:
        return 0.0

    bins: list[list[ExperimentRun]] = [[] for _ in range(n_bins)]
    for r in eligible_runs:
        # Bin 1.0 goes to the last bin
        bin_idx = min(int(r.confidence * n_bins), n_bins - 1)
        bins[bin_idx].append(r)

    total_runs = len(eligible_runs)
    ece = 0.0

    for bin_runs in bins:
        if not bin_runs:
            continue
        mean_conf = sum(r.confidence for r in bin_runs) / len(bin_runs)
        mean_acc = sum(1.0 if r.ground_truth_match else 0.0 for r in bin_runs) / len(bin_runs)
        weight = len(bin_runs) / total_runs
        ece += weight * abs(mean_conf - mean_acc)

    return ece


def brier_score(runs: Sequence[ExperimentRun]) -> float:
    eligible_runs = [r for r in runs if r.decision == DecisionKind.SELECT and r.ground_truth_match is not None]
    if not eligible_runs:
        return 0.0

    mse = sum((r.confidence - (1.0 if r.ground_truth_match else 0.0)) ** 2 for r in eligible_runs) / len(eligible_runs)
    return mse


def reliability_curve(runs: Sequence[ExperimentRun], n_bins: int = 10) -> list[dict]:
    eligible_runs = [r for r in runs if r.decision == DecisionKind.SELECT and r.ground_truth_match is not None]
    if not eligible_runs:
        return []

    bins: list[list[ExperimentRun]] = [[] for _ in range(n_bins)]
    for r in eligible_runs:
        bin_idx = min(int(r.confidence * n_bins), n_bins - 1)
        bins[bin_idx].append(r)

    curve = []
    for i, bin_runs in enumerate(bins):
        if not bin_runs:
            continue
        
        # equal-width bins on confidence over [0, 1]
        bin_start = i / n_bins
        bin_end = (i + 1) / n_bins
        bin_midpoint = (bin_start + bin_end) / 2.0

        mean_conf = sum(r.confidence for r in bin_runs) / len(bin_runs)
        mean_acc = sum(1.0 if r.ground_truth_match else 0.0 for r in bin_runs) / len(bin_runs)
        
        curve.append({
            "bin_midpoint": bin_midpoint,
            "mean_confidence": mean_conf,
            "mean_accuracy": mean_acc,
            "count": len(bin_runs),
        })

    return curve


def accuracy_at_coverage(runs: Sequence[ExperimentRun], coverage_target: float) -> dict:
    eligible_runs = [r for r in runs if r.decision == DecisionKind.SELECT and r.ground_truth_match is not None]
    if not eligible_runs:
        return {
            "accuracy": 0.0,
            "coverage": 0.0,
            "threshold": 0.0,
        }

    # sort descending
    sorted_runs = sorted(eligible_runs, key=lambda r: r.confidence, reverse=True)
    retained_count = max(1, math.ceil(len(eligible_runs) * coverage_target))
    
    retained_subset = sorted_runs[:retained_count]
    accuracy = sum(1.0 if r.ground_truth_match else 0.0 for r in retained_subset) / len(retained_subset)
    coverage = retained_count / len(eligible_runs)
    threshold = min(r.confidence for r in retained_subset)

    return {
        "accuracy": accuracy,
        "coverage": coverage,
        "threshold": threshold,
    }


def abstention_quality(runs: Sequence[ExperimentRun]) -> dict:
    eligible_runs = [r for r in runs if r.ground_truth_match is not None]
    if not eligible_runs:
        return {
            "abstention_rate": 0.0,
            "wrong_abstention_rate": 0.0,
            "correct_abstention_rate": 0.0,
        }

    abstentions = [r for r in eligible_runs if r.decision == DecisionKind.ABSTAIN]
    abstention_rate = len(abstentions) / len(eligible_runs)

    if not abstentions:
        return {
            "abstention_rate": abstention_rate,
            "wrong_abstention_rate": 0.0,
            "correct_abstention_rate": 0.0,
        }

    wrong_abstentions = sum(1 for r in abstentions if r.ground_truth_match is False)
    correct_abstentions = sum(1 for r in abstentions if r.ground_truth_match is True)
    
    return {
        "abstention_rate": abstention_rate,
        "wrong_abstention_rate": wrong_abstentions / len(abstentions),
        "correct_abstention_rate": correct_abstentions / len(abstentions),
    }