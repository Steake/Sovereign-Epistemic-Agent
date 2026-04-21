from __future__ import annotations

from typing import cast

from epistemic_tribunal.tasks.base import grids_equal
from epistemic_tribunal.tribunal_types import AnswerType, CandidateTrace
from epistemic_tribunal.domains.base import DomainAdapter


class ArcGridAdapter(DomainAdapter):
    """Domain adapter for ARC-like grid tasks."""

    def normalize_answer(self, answer: AnswerType) -> AnswerType:
        return answer

    def answers_equal(self, a: AnswerType, b: AnswerType) -> bool:
        if not isinstance(a, list) or not isinstance(b, list):
            return False
        return grids_equal(a, b)

    def compute_disagreement(self, traces: list[CandidateTrace]) -> float:
        """Compute structural disagreement for ARC grids."""
        n = len(traces)
        if n < 2:
            return 0.0

        disagreements = 0
        pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                pairs += 1
                if not self.answers_equal(traces[i].answer, traces[j].answer):
                    disagreements += 1
        return disagreements / pairs if pairs > 0 else 0.0

    def cluster_answers(self, traces: list[CandidateTrace]) -> list[list[CandidateTrace]]:
        """Group traces into clusters based on grid equality."""
        clusters: dict[tuple, list[CandidateTrace]] = {}
        for trace in traces:
            key = self.get_cluster_key(trace.answer)
            if key not in clusters:
                clusters[key] = []
            clusters[key].append(trace)
        return list(clusters.values())

    def get_cluster_key(self, answer: AnswerType) -> tuple:
        """Map a grid to a hashable tuple."""
        if not isinstance(answer, list):
            return tuple()
        return tuple(tuple(row) for row in answer)
