"""Abstract base class for trace critics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from epistemic_tribunal.types import (
    CandidateTrace,
    CritiqueResult,
    InvariantSet,
    Task,
)


class BaseCritic(ABC):
    """Scores a single candidate trace on multiple quality dimensions."""

    name: str = "base"

    @abstractmethod
    def critique(
        self,
        task: Task,
        trace: CandidateTrace,
        invariant_set: Optional[InvariantSet] = None,
        ledger_failure_patterns: Optional[list[dict]] = None,
    ) -> CritiqueResult:
        """Produce a :class:`CritiqueResult` for *trace*.

        Parameters
        ----------
        task:
            The original task.
        trace:
            The candidate trace to score.
        invariant_set:
            Pre-extracted invariants for the task, if available.
        ledger_failure_patterns:
            Past failure records that can penalise similar traces.
        """
