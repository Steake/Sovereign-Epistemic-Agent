"""Tests for the UncertaintyAnalyzer."""

from __future__ import annotations

import pytest

from epistemic_tribunal.generators.base import build_generators
from epistemic_tribunal.types import CandidateTrace, UncertaintyReport
from epistemic_tribunal.uncertainty.analyzer import UncertaintyAnalyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _analyzer() -> UncertaintyAnalyzer:
    return UncertaintyAnalyzer()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_analyze_returns_report(simple_task) -> None:
    gens = build_generators(["greedy", "diverse", "adversarial", "rule_first", "minimal_description"])
    traces = [g.generate(simple_task) for g in gens]
    analyzer = _analyzer()
    report = analyzer.analyze(simple_task, traces)
    assert isinstance(report, UncertaintyReport)


def test_report_fields_bounded(simple_task) -> None:
    gens = build_generators(["greedy", "diverse"])
    traces = [g.generate(simple_task) for g in gens]
    analyzer = _analyzer()
    report = analyzer.analyze(simple_task, traces)
    assert 0.0 <= report.entropy <= 1.0
    assert 0.0 <= report.margin <= 1.0
    assert 0.0 <= report.coalition_mass <= 1.0
    assert 0.0 <= report.disagreement_rate <= 1.0


def test_empty_traces(simple_task) -> None:
    analyzer = _analyzer()
    report = analyzer.analyze(simple_task, [])
    assert report.entropy == 0.0
    assert report.disagreement_rate == 1.0


def test_single_trace(simple_task) -> None:
    from epistemic_tribunal.generators.greedy import GreedyGenerator
    trace = GreedyGenerator(seed=42).generate(simple_task)
    analyzer = _analyzer()
    report = analyzer.analyze(simple_task, [trace])
    # Single trace → zero disagreement
    assert report.disagreement_rate == 0.0
    # Coalition mass of 1 → all agree
    assert report.coalition_mass == pytest.approx(1.0)


def test_all_identical_traces(simple_task) -> None:
    """When all generators produce the same answer, disagreement_rate should be 0."""
    from epistemic_tribunal.generators.greedy import GreedyGenerator
    trace = GreedyGenerator(seed=42).generate(simple_task)
    # Create copies with different IDs but same answer
    traces = []
    for i in range(3):
        t = CandidateTrace(
            generator_name=f"gen_{i}",
            answer=[row[:] for row in trace.answer],
            reasoning_steps=["same"],
            confidence_score=0.8,
        )
        traces.append(t)
    analyzer = _analyzer()
    report = analyzer.analyze(simple_task, traces)
    assert report.disagreement_rate == pytest.approx(0.0)
    assert report.coalition_mass == pytest.approx(1.0)


def test_per_trace_quality_all_present(simple_task) -> None:
    gens = build_generators(["greedy", "diverse", "adversarial"])
    traces = [g.generate(simple_task) for g in gens]
    analyzer = _analyzer()
    report = analyzer.analyze(simple_task, traces)
    for trace in traces:
        assert trace.trace_id in report.per_trace_quality


def test_per_trace_quality_bounded(simple_task) -> None:
    gens = build_generators(["greedy", "diverse", "adversarial", "rule_first", "minimal_description"])
    traces = [g.generate(simple_task) for g in gens]
    analyzer = _analyzer()
    report = analyzer.analyze(simple_task, traces)
    for trace_id, quality in report.per_trace_quality.items():
        assert 0.0 <= quality <= 1.0, f"Quality out of range for {trace_id}"


def test_high_disagreement_scenario(simple_task) -> None:
    """All distinct answers → high disagreement rate."""
    # Build 3 traces with maximally different answers
    traces = [
        CandidateTrace(
            generator_name=f"gen_{i}",
            answer=[[i, i + 1, i], [i, i, i], [i + 1, i, i]],
            reasoning_steps=["step"],
            confidence_score=0.5,
        )
        for i in range(3)
    ]
    analyzer = _analyzer()
    report = analyzer.analyze(simple_task, traces)
    assert report.disagreement_rate == pytest.approx(1.0)
