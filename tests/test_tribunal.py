"""Tests for the Tribunal aggregator."""

from __future__ import annotations

import pytest

from epistemic_tribunal.config import TribunalConfig, TribunalWeights
from epistemic_tribunal.critics.trace_critic import TraceCritic
from epistemic_tribunal.generators.base import build_generators
from epistemic_tribunal.invariants.extractor import InvariantExtractor
from epistemic_tribunal.tribunal.aggregator import TribunalAggregator
from epistemic_tribunal.tribunal.scoring import compute_trace_score, normalise_weights
from epistemic_tribunal.types import (
    CandidateTrace,
    CritiqueResult,
    DecisionKind,
    TribunalDecision,
    UncertaintyReport,
)
from epistemic_tribunal.uncertainty.analyzer import UncertaintyAnalyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_tribunal(simple_task, config=None):
    gens = build_generators(
        ["greedy", "diverse", "adversarial", "rule_first", "minimal_description"]
    )
    traces = [g.generate(simple_task) for g in gens]
    inv_set = InvariantExtractor(confidence_threshold=0.4).extract(simple_task)
    critiques = [TraceCritic().critique(simple_task, t, inv_set) for t in traces]
    uncertainty = UncertaintyAnalyzer().analyze(simple_task, traces)
    tribunal = TribunalAggregator(config=config)
    return tribunal.adjudicate(simple_task, traces, critiques, uncertainty, inv_set)


# ---------------------------------------------------------------------------
# Scoring utility tests
# ---------------------------------------------------------------------------


def test_normalise_weights_sums_to_one() -> None:
    a, b, c, d = normalise_weights(1, 2, 3, 4)
    assert a + b + c + d == pytest.approx(1.0)


def test_normalise_weights_all_zero() -> None:
    a, b, c, d = normalise_weights(0, 0, 0, 0)
    assert a + b + c + d == pytest.approx(1.0)


def test_compute_trace_score_bounded(simple_task) -> None:
    trace = build_generators(["greedy"])[0].generate(simple_task)
    critique = TraceCritic().critique(simple_task, trace)
    uncertainty = UncertaintyAnalyzer().analyze(simple_task, [trace])
    score = compute_trace_score(trace, critique, uncertainty, alpha=0.25, beta=0.35, gamma=0.15, delta=0.25)
    assert 0.0 <= score.total <= 1.0
    assert 0.0 <= score.U <= 1.0
    assert 0.0 <= score.C <= 1.0
    assert 0.0 <= score.M <= 1.0
    assert 0.0 <= score.V <= 1.0


# ---------------------------------------------------------------------------
# Aggregator tests
# ---------------------------------------------------------------------------


def test_adjudicate_returns_decision(simple_task) -> None:
    decision = _run_tribunal(simple_task)
    assert isinstance(decision, TribunalDecision)


def test_adjudicate_decision_is_valid(simple_task) -> None:
    decision = _run_tribunal(simple_task)
    assert decision.decision in DecisionKind


def test_adjudicate_confidence_bounded(simple_task) -> None:
    decision = _run_tribunal(simple_task)
    assert 0.0 <= decision.confidence <= 1.0


def test_adjudicate_scores_dict_nonempty(simple_task) -> None:
    decision = _run_tribunal(simple_task)
    assert len(decision.scores) > 0


def test_adjudicate_no_traces(simple_task) -> None:
    tribunal = TribunalAggregator()
    decision = tribunal.adjudicate(simple_task, [], [], UncertaintyReport(
        entropy=0, margin=0, coalition_mass=0, disagreement_rate=1
    ))
    assert decision.decision == DecisionKind.ABSTAIN


def test_adjudicate_high_threshold_abstains(simple_task) -> None:
    """Setting selection_threshold=1.0 should force abstain or resample."""
    config = TribunalConfig(
        selection_threshold=1.0,
        resample_threshold=0.99,
    )
    decision = _run_tribunal(simple_task, config=config)
    assert decision.decision in (DecisionKind.ABSTAIN, DecisionKind.RESAMPLE)


def test_adjudicate_low_threshold_selects(simple_task) -> None:
    """Setting selection_threshold=0.0 should guarantee a selection."""
    config = TribunalConfig(
        selection_threshold=0.0,
        resample_threshold=0.0,
    )
    decision = _run_tribunal(simple_task, config=config)
    assert decision.decision == DecisionKind.SELECT
    assert decision.selected_trace_id is not None
    assert decision.selected_answer is not None


def test_adjudicate_selected_answer_is_grid(simple_task) -> None:
    config = TribunalConfig(selection_threshold=0.0, resample_threshold=0.0)
    decision = _run_tribunal(simple_task, config=config)
    if decision.decision == DecisionKind.SELECT:
        assert isinstance(decision.selected_answer, list)
        assert all(isinstance(row, list) for row in decision.selected_answer)


def test_adjudicate_reasoning_nonempty(simple_task) -> None:
    decision = _run_tribunal(simple_task)
    assert decision.reasoning != ""


def test_adjudicate_task_id_preserved(simple_task) -> None:
    decision = _run_tribunal(simple_task)
    assert decision.task_id == simple_task.task_id


def test_adjudicate_resamples_when_single_generator_type_dominates(simple_task) -> None:
    tribunal = TribunalAggregator(
        config=TribunalConfig(
            selection_threshold=0.4,
            resample_threshold=0.2,
            diversity_floor=0.9,
        )
    )
    shared_answer = [[1, 1], [1, 1]]
    traces = [
        CandidateTrace(
            trace_id="t1",
            generator_name="llm",
            answer=shared_answer,
            reasoning_steps=["a"],
            confidence_score=0.9,
        ),
        CandidateTrace(
            trace_id="t2",
            generator_name="greedy",
            answer=[[0, 0], [0, 0]],
            reasoning_steps=["b"],
            confidence_score=0.2,
        ),
    ]
    critiques = [
        CritiqueResult(
            trace_id="t1",
            consistency_score=0.9,
            rule_coherence_score=0.9,
            morphology_score=0.9,
            failure_similarity_penalty=0.0,
            invariant_compliance_score=1.0,
            aggregate_score=0.95,
        ),
        CritiqueResult(
            trace_id="t2",
            consistency_score=0.2,
            rule_coherence_score=0.2,
            morphology_score=0.2,
            failure_similarity_penalty=0.0,
            invariant_compliance_score=1.0,
            aggregate_score=0.2,
        ),
    ]
    uncertainty = UncertaintyReport(
        entropy=0.0,
        margin=1.0,
        coalition_mass=0.95,
        disagreement_rate=1.0,
        per_trace_quality={"t1": 0.9, "t2": 0.1},
    )
    decision = tribunal.adjudicate(simple_task, traces, critiques, uncertainty)
    assert decision.decision == DecisionKind.RESAMPLE
