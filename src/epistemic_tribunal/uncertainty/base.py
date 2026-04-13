"""Abstract base for uncertainty analyzers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from epistemic_tribunal.tribunal_types import CandidateTrace, Task, UncertaintyReport


class BaseUncertaintyAnalyzer(ABC):
    """Computes uncertainty signals across a pool of candidate traces."""

    @abstractmethod
    def analyze(
        self,
        task: Task,
        traces: list[CandidateTrace],
    ) -> UncertaintyReport:
        """Return an :class:`UncertaintyReport` for the given trace pool."""
