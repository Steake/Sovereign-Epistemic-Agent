"""Evaluation metrics for Epistemic Tribunal benchmark runs.

All metrics operate on lists of :class:`ExperimentRun` objects collected
after running the tribunal over a dataset.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Optional

from epistemic_tribunal.evaluation import calibration
from epistemic_tribunal.tribunal_types import DecisionKind, ExperimentRun

def best_in_pool_accuracy(runs: list[ExperimentRun]) -> float:
    """Fraction of runs where at least one candidate trace matched the ground truth."""
    if not runs:
        return 0.0
    correct = sum(1 for r in runs if r.metadata.get("any_correct") is True)
    return correct / len(runs)

def greedy_accuracy(runs: list[ExperimentRun]) -> float:
    """Fraction of runs where the 'llm' (greedy) baseline generator matched the ground truth."""
    if not runs:
        return 0.0
    correct = sum(1 for r in runs if r.metadata.get("greedy_correct") is True)
    return correct / len(runs)


def resolved_accuracy(runs: list[ExperimentRun]) -> float:
    """Fraction of runs where the SELECTED answer matched the ground truth.
    Only counts runs where we made a selection.
    """
    evaluated = [
        r for r in runs
        if r.decision == DecisionKind.SELECT and r.ground_truth_match is not None
    ]
    if not evaluated:
        return 0.0
    return sum(1 for r in evaluated if r.ground_truth_match) / len(evaluated)


def overall_accuracy(runs: list[ExperimentRun]) -> float:
    """Fraction of ALL runs that resulted in a correct selection."""
    if not runs:
        return 0.0
    return sum(1 for r in runs if r.ground_truth_match is True) / len(runs)


def wrong_pick_count(runs: list[ExperimentRun]) -> int:
    """Total number of tasks where the tribunal selected an incorrect answer."""
    return sum(1 for r in runs if r.ground_truth_match is False and r.decision == DecisionKind.SELECT)


def override_count(runs: list[ExperimentRun]) -> int:
    """Total number of tasks where the Path B structural override was triggered."""
    return sum(1 for r in runs if r.metadata.get("path_b_override") is True)


def mean_confidence(runs: list[ExperimentRun]) -> float:
    """Mean confidence score for selected tasks."""
    selected = [r for r in runs if r.decision == DecisionKind.SELECT]
    if not selected:
        return 0.0
    return sum(r.confidence for r in selected) / len(selected)


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


def truncation_count(runs: list[ExperimentRun]) -> int:
    return sum(r.metadata.get("generation_stats", {}).get("truncation_count", 0) for r in runs)

def json_not_found_count(runs: list[ExperimentRun]) -> int:
    return sum(r.metadata.get("generation_stats", {}).get("json_not_found_count", 0) for r in runs)

def json_invalid_count(runs: list[ExperimentRun]) -> int:
    return sum(r.metadata.get("generation_stats", {}).get("json_invalid_count", 0) for r in runs)

def grid_shape_invalid_count(runs: list[ExperimentRun]) -> int:
    return sum(r.metadata.get("generation_stats", {}).get("grid_shape_invalid_count", 0) for r in runs)

def reasoning_bleed_count(runs: list[ExperimentRun]) -> int:
    return sum(r.metadata.get("generation_stats", {}).get("reasoning_bleed_count", 0) for r in runs)

def parse_failure_count(runs: list[ExperimentRun]) -> int:
    return sum(r.metadata.get("generation_stats", {}).get("parse_failure_count", 0) for r in runs)

def path_b_met_gate(runs: list[ExperimentRun]) -> int:
    return sum(1 for r in runs if r.metadata.get("path_b_stats", {}).get("met_gate_potential", False))

def path_b_failed_v(runs: list[ExperimentRun]) -> int:
    return sum(1 for r in runs if r.metadata.get("path_b_stats", {}).get("failed_V", False))

def path_b_failed_c(runs: list[ExperimentRun]) -> int:
    return sum(1 for r in runs if r.metadata.get("path_b_stats", {}).get("failed_C", False))

def path_b_failed_margin(runs: list[ExperimentRun]) -> int:
    return sum(1 for r in runs if r.metadata.get("path_b_stats", {}).get("failed_margin", False))

def path_b_failed_violations(runs: list[ExperimentRun]) -> int:
    return sum(1 for r in runs if r.metadata.get("path_b_stats", {}).get("failed_violations", False))

def _parse_json_field(raw: Any, default: Any) -> Any:
    if raw is None:
        return default
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return default


def summary_report(
    runs: list[ExperimentRun],
    coalition_rows: Optional[list[dict[str, Any]]] = None,
) -> dict[str, float | int | dict]:
    """Produce a full summary metrics dictionary with cohort stratification."""
    
    # 1. Base Accuracy & Coverage
    total = len(runs)
    selected = [r for r in runs if r.decision == DecisionKind.SELECT]
    not_selected = [r for r in runs if r.decision != DecisionKind.SELECT]
    
    report: dict[str, float | int | dict] = {
        "total_runs": total,
        "overall_accuracy": round(overall_accuracy(runs), 4),
        "selective_accuracy": round(resolved_accuracy(runs), 4),
        "coverage": round(coverage(runs), 4),
        "wrong_pick_count": wrong_pick_count(runs),
    }

    # 2. Cohort Stratification
    cohort_stats = {}
    for c_name in ["same_task_recoverable", "same_task_unrecoverable", "control_trivial", "processing_confounded", "unknown"]:
        c_runs = [r for r in runs if r.metadata.get("cohort") == c_name]
        if not c_runs:
            continue
            
        c_sel = [r for r in c_runs if r.decision == DecisionKind.SELECT]
        c_acc = sum(1 for r in c_sel if r.ground_truth_match) / len(c_sel) if c_sel else 0.0
        
        cohort_stats[c_name] = {
            "n": len(c_runs),
            "selective_acc": round(c_acc, 4),
            "abstain_rate": round(sum(1 for r in c_runs if r.decision != DecisionKind.SELECT) / len(c_runs), 4),
            "wrong_picks": sum(1 for r in c_sel if r.ground_truth_match is False),
        }
    report["cohort_metrics"] = cohort_stats

    # 3. Abstention Quality (Impossible to misread)
    # Good Abstention: System avoided a task where NO candidate was correct.
    # Bad Abstention: System avoided a task where AT LEAST ONE candidate was correct.
    good_abstains = sum(1 for r in not_selected if r.metadata.get("any_correct") is False)
    bad_abstains = sum(1 for r in not_selected if r.metadata.get("any_correct") is True)
    
    report["abstention_metrics"] = {
        "total_abstentions": len(not_selected),
        "good_abstentions": good_abstains,  # Avoided unrecoverable errors
        "bad_abstentions": bad_abstains,    # Missed recoverable solutions
        "abstention_efficiency": round(good_abstains / len(not_selected), 4) if not_selected else 0.0,
    }

    # 4. Calibration & Quality
    eligible_with_confidence = [
        run for run in selected
        if run.ground_truth_match is not None
        and run.confidence > 0.0
    ]
    if eligible_with_confidence:
        report["calibration"] = {
            "ece": round(calibration.expected_calibration_error(eligible_with_confidence), 4),
            "brier": round(calibration.brier_score(eligible_with_confidence), 4),
            "acc_at_90_cov": round(calibration.accuracy_at_coverage(eligible_with_confidence, 0.9)["accuracy"], 4),
        }

    # 5. Diagnostics & Infrastructure
    report["diagnostics"] = {
        "avg_duration": round(average_duration(runs), 4),
        "parse_failures": parse_failure_count(runs),
        "path_b_overrides": override_count(runs),
        "truncations": truncation_count(runs),
        "json_errors": json_invalid_count(runs) + json_not_found_count(runs),
    }

    # 6. Tribunal Usefulness & Oracle Analysis
    bip_acc = best_in_pool_accuracy(runs)
    gre_acc = greedy_accuracy(runs)
    cr_runs = [r for r in runs if r.metadata.get("cohort") == "same_task_recoverable"]
    
    if cr_runs:
        cr_sel = [r for r in cr_runs if r.decision == DecisionKind.SELECT]
        cr_tribunal_acc = sum(1 for r in cr_sel if r.ground_truth_match) / len(cr_sel) if cr_sel else 0.0
        cr_greedy_acc = greedy_accuracy(cr_runs)
        lift_cr = cr_tribunal_acc - cr_greedy_acc
        cr_bip_acc = best_in_pool_accuracy(cr_runs)
    else:
        cr_tribunal_acc = 0.0
        lift_cr = 0.0
        cr_bip_acc = 0.0

    report["tribunal_usefulness"] = {
        "best_candidate_in_pool_accuracy": round(bip_acc, 4),
        "greedy_accuracy": round(gre_acc, 4),
        "good_abstention_rate": round(good_abstains / len(not_selected), 4) if not_selected else 0.0,
        "bad_abstention_rate": round(bad_abstains / len(not_selected), 4) if not_selected else 0.0,
        "tribunal_lift_over_greedy": round((resolved_accuracy(runs) - gre_acc), 4),
        "best_candidate_in_pool_accuracy_on_recoverable": round(cr_bip_acc, 4),
        "tribunal_selected_accuracy_on_recoverable": round(cr_tribunal_acc, 4),
        "lift_on_recoverable": round(lift_cr, 4),
    }

    if coalition_rows:
        mean_belief = sum(float(row.get("belief", 0.0)) for row in coalition_rows) / len(coalition_rows)
        mean_disbelief = sum(float(row.get("disbelief", 0.0)) for row in coalition_rows) / len(coalition_rows)
        mean_uncertainty = sum(float(row.get("uncertainty", 0.0)) for row in coalition_rows) / len(coalition_rows)

        high_uncertainty_abstains = 0
        same_answer_tie_selected = 0
        structural_uncertainty_cases = 0
        exact_memory_changed_winner = 0

        run_by_id = {run.run_id: run for run in runs}
        winner_rows = [row for row in coalition_rows if row.get("decision_role") == "winner"]
        for row in winner_rows:
            run = run_by_id.get(row.get("run_id"))
            if run is not None:
                eqbsl_meta = run.metadata.get("eqbsl", {})
                reason = eqbsl_meta.get("task_reason", {})
                if reason.get("reason_code") == "abstain_high_uncertainty":
                    high_uncertainty_abstains += 1
                if eqbsl_meta.get("exact_memory_changed_winner") is True:
                    exact_memory_changed_winner += 1
                if eqbsl_meta.get("same_answer_tie_case") and run.decision == DecisionKind.SELECT:
                    same_answer_tie_selected += 1

            source_opinions = _parse_json_field(row.get("source_opinions_json"), {})
            memory_source = source_opinions.get("M", {})
            memory_opinion = memory_source.get("opinion", {})
            memory_meta = memory_opinion.get("metadata", {})
            if (
                memory_meta.get("structural_hits", 0) > 0
                and memory_meta.get("exact_hits", 0) == 0
                and float(memory_opinion.get("uncertainty", 0.0)) > float(memory_opinion.get("disbelief", 0.0))
            ):
                structural_uncertainty_cases += 1

        report["eqbsl_diagnostics"] = {
            "mean_coalition_belief": round(mean_belief, 4),
            "mean_coalition_disbelief": round(mean_disbelief, 4),
            "mean_coalition_uncertainty": round(mean_uncertainty, 4),
            "abstentions_caused_by_high_uncertainty": high_uncertainty_abstains,
            "same_answer_tie_cases_correctly_selected": same_answer_tie_selected,
            "structural_evidence_uncertainty_without_disbelief": structural_uncertainty_cases,
            "exact_answer_failure_changed_winner": exact_memory_changed_winner,
        }

    return report
