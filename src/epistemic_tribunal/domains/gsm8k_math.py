from __future__ import annotations

from typing import cast

from epistemic_tribunal.tribunal_types import AnswerType, CandidateTrace
from epistemic_tribunal.domains.base import DomainAdapter


class Gsm8kMathAdapter(DomainAdapter):
    """Domain adapter for GSM8K-style mathematical tasks."""

    def normalize_answer(self, answer: AnswerType) -> AnswerType:
        """Attempt to convert string answers to a float or int, stripping commas."""
        if isinstance(answer, (int, float)):
            return answer
        if isinstance(answer, str):
            clean_str = answer.replace(",", "").strip()
            try:
                if "." in clean_str:
                    return float(clean_str)
                return int(clean_str)
            except ValueError:
                return answer # Keep as string if parsing fails
        return answer

    def answers_equal(self, a: AnswerType, b: AnswerType) -> bool:
        norm_a = self.normalize_answer(a)
        norm_b = self.normalize_answer(b)
        
        if type(norm_a) in (int, float) and type(norm_b) in (int, float):
            return abs(norm_a - norm_b) < 1e-9 # Float tolerance
        return norm_a == norm_b

    def compute_disagreement(self, traces: list[CandidateTrace]) -> float:
        """Compute disagreement for scalar math answers."""
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
        """Group traces into clusters based on scalar equivalence."""
        clusters: list[list[CandidateTrace]] = []
        for trace in traces:
            placed = False
            for cluster in clusters:
                if self.answers_equal(cluster[0].answer, trace.answer):
                    cluster.append(trace)
                    placed = True
                    break
            if not placed:
                clusters.append([trace])
        return clusters

    def get_cluster_key(self, answer: AnswerType) -> tuple:
        """Map a scalar answer to a hashable tuple."""
        norm_ans = self.normalize_answer(answer)
        if isinstance(norm_ans, float):
            # Quantize slightly to avoid float hashing issues, though equality should handle this in cluster_answers
            return (round(norm_ans, 8),)
        if isinstance(norm_ans, list):
            return tuple(tuple(row) for row in norm_ans)
        return (norm_ans,)
