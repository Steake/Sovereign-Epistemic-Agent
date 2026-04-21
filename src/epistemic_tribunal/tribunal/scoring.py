"""Scoring utilities for the tribunal aggregator.

The tribunal scoring function is:

    S(T_i) = alpha * U_i + beta * C_i + gamma * M_i + delta * V_i

Where:
    U_i = uncertainty-derived quality (1 - entropy proxy + margin bonus)
    C_i = critic aggregate score
    M_i = memory / failure-similarity bonus (1 - failure_penalty)
    V_i = invariant compliance score

Weights are normalised to sum to 1.0 internally.
"""

from __future__ import annotations

from dataclasses import dataclass

from epistemic_tribunal.tribunal_types import (
    CandidateTrace,
    CritiqueResult,
    InvariantSet,
    UncertaintyReport,
)


@dataclass
class TraceScore:
    """Scored candidate ready for tribunal selection."""

    trace_id: str
    generator_name: str
    U: float  # Uncertainty quality
    C: float  # Critic score
    M: float  # Memory / failure-similarity
    V: float  # Invariant compliance
    total: float


def compute_trace_score(
    trace: CandidateTrace,
    critique: CritiqueResult,
    uncertainty: UncertaintyReport,
    *,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
) -> TraceScore:
    """Compute the weighted tribunal score for a single trace.

    Parameters
    ----------
    trace, critique, uncertainty:
        The relevant data objects for this trace.
    alpha, beta, gamma, delta:
        Pre-normalised weights for U, C, M, V respectively.
    """
    # U: uncertainty quality — fraction of pool that agrees with this candidate's answer.
    # per_trace_quality is now candidate-specific (set in UncertaintyAnalyzer).
    # Majority-coalition trace: u_quality ≈ 0.667 (2/3 agree)
    # Minority trace:           u_quality ≈ 0.333 (1/3 agree)
    # We preserve the pool-level margin as a secondary boost but do not multiply
    # it in a way that collapses all candidates to the same value.
    u_quality = uncertainty.per_trace_quality.get(trace.trace_id, 0.5)
    U = u_quality

    # C: critic aggregate
    C = critique.aggregate_score

    # M: memory/failure-similarity bonus (inverted penalty)
    M = 1.0 - critique.failure_similarity_penalty

    # V: invariant compliance
    V = critique.invariant_compliance_score

    total = alpha * U + beta * C + gamma * M + delta * V
    total = max(0.0, min(1.0, total))

    return TraceScore(
        trace_id=trace.trace_id,
        generator_name=trace.generator_name,
        U=round(U, 4),
        C=round(C, 4),
        M=round(M, 4),
        V=round(V, 4),
        total=round(total, 4),
    )


def normalise_weights(
    alpha: float, beta: float, gamma: float, delta: float
) -> tuple[float, float, float, float]:
    """Normalise weights to sum to 1.0."""
    total = alpha + beta + gamma + delta
    if total <= 0.0:
        return 0.25, 0.25, 0.25, 0.25
    return alpha / total, beta / total, gamma / total, delta / total
