from __future__ import annotations

from typing import Protocol

from epistemic_tribunal.tribunal_types import AnswerType, CandidateTrace


class DomainAdapter(Protocol):
    """Protocol defining domain-specific operations for the epistemic tribunal."""

    def normalize_answer(self, answer: AnswerType) -> AnswerType:
        """Normalize an answer for canonical comparison."""
        ...

    def answers_equal(self, a: AnswerType, b: AnswerType) -> bool:
        """Return True if two answers are semantically equal."""
        ...

    def compute_disagreement(self, traces: list[CandidateTrace]) -> float:
        """Compute the disagreement rate among a pool of candidate traces.
        
        Returns a float between 0.0 (unanimous agreement) and 1.0 (total disagreement).
        """
        ...

    def cluster_answers(self, traces: list[CandidateTrace]) -> list[list[CandidateTrace]]:
        """Group traces into clusters that represent the same answer."""
        ...

    def get_cluster_key(self, answer: AnswerType) -> tuple:
        """Return a hashable key for an answer, suitable for clustering."""
        ...
