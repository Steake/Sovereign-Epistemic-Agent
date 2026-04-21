"""Tests for the TraceCritic."""

from __future__ import annotations

import pytest

from epistemic_tribunal.critics.trace_critic import TraceCritic
from epistemic_tribunal.generators.greedy import GreedyGenerator
from epistemic_tribunal.generators.adversarial import AdversarialGenerator
from epistemic_tribunal.invariants.extractor import InvariantExtractor
from epistemic_tribunal.tribunal_types import CandidateTrace, CritiqueResult, Task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _critic() -> TraceCritic:
    return TraceCritic()


def _extract(task: Task):
    return InvariantExtractor(confidence_threshold=0.4).extract(task)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_critique_returns_result(simple_task: Task) -> None:
    critic = _critic()
    trace = GreedyGenerator(seed=42).generate(simple_task)
    inv_set = _extract(simple_task)
    result = critic.critique(simple_task, trace, inv_set)
    assert isinstance(result, CritiqueResult)


def test_critique_scores_bounded(simple_task: Task) -> None:
    critic = _critic()
    trace = GreedyGenerator(seed=42).generate(simple_task)
    result = critic.critique(simple_task, trace)
    assert 0.0 <= result.consistency_score <= 1.0
    assert 0.0 <= result.rule_coherence_score <= 1.0
    assert 0.0 <= result.morphology_score <= 1.0
    assert 0.0 <= result.failure_similarity_penalty <= 1.0
    assert 0.0 <= result.invariant_compliance_score <= 1.0
    assert 0.0 <= result.aggregate_score <= 1.0


def test_critique_trace_id_preserved(simple_task: Task) -> None:
    critic = _critic()
    trace = GreedyGenerator(seed=42).generate(simple_task)
    result = critic.critique(simple_task, trace)
    assert result.trace_id == trace.trace_id


def test_critique_adversarial_lower_confidence(simple_task: Task) -> None:
    critic = _critic()
    greedy_trace = GreedyGenerator(seed=42).generate(simple_task)
    adv_trace = AdversarialGenerator(seed=42).generate(simple_task)
    inv_set = _extract(simple_task)
    greedy_result = critic.critique(simple_task, greedy_trace, inv_set)
    adv_result = critic.critique(simple_task, adv_trace, inv_set)
    # Adversarial has lower self-reported confidence → lower consistency
    assert adv_result.consistency_score <= greedy_result.consistency_score + 0.3


def test_critique_with_failure_patterns(simple_task: Task) -> None:
    """Failure patterns for a generator should penalise similar traces."""
    critic = _critic()
    adv_trace = AdversarialGenerator(seed=42).generate(simple_task)
    patterns = [
        {"generator_name": "adversarial", "diagnosis": "adversarial failure detected"}
    ]
    result = critic.critique(simple_task, adv_trace, ledger_failure_patterns=patterns)
    assert result.failure_similarity_penalty > 0.0


def test_critique_empty_failure_patterns(simple_task: Task) -> None:
    critic = _critic()
    trace = GreedyGenerator(seed=42).generate(simple_task)
    result = critic.critique(simple_task, trace, ledger_failure_patterns=[])
    assert result.failure_similarity_penalty == 0.0


def test_critique_notes_nonempty(simple_task: Task) -> None:
    critic = _critic()
    trace = GreedyGenerator(seed=42).generate(simple_task)
    result = critic.critique(simple_task, trace)
    assert result.notes != ""


def test_critique_violated_invariants_list(simple_task: Task) -> None:
    critic = _critic()
    inv_set = _extract(simple_task)
    # An all-zero answer should violate some invariants
    trace = CandidateTrace(
        generator_name="test",
        answer=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        reasoning_steps=["Fill with zeros."],
        confidence_score=0.1,
    )
    result = critic.critique(simple_task, trace, inv_set)
    assert isinstance(result.violated_invariants, list)


def test_critique_all_generators(simple_task: Task) -> None:
    """All five generators should produce critiqueable traces."""
    from epistemic_tribunal.generators.base import build_generators
    critic = _critic()
    inv_set = _extract(simple_task)
    gens = build_generators(
        ["greedy", "diverse", "adversarial", "rule_first", "minimal_description"]
    )
    for gen in gens:
        trace = gen.generate(simple_task)
        result = critic.critique(simple_task, trace, inv_set)
        assert 0.0 <= result.aggregate_score <= 1.0, f"Score out of range for {gen.name}"
