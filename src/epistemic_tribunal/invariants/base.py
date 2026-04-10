"""Abstract base for invariant checkers.

Each checker takes a task and a candidate answer and returns whether the
invariant holds (and a confidence score for the assertion).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from epistemic_tribunal.types import Task


class BaseInvariantChecker(ABC):
    """One structural constraint that a valid answer should satisfy."""

    #: Stable identifier for this checker.
    name: str = "base"

    @abstractmethod
    def check(
        self,
        task: Task,
        candidate_answer: Optional[list[list[int]]] = None,
    ) -> tuple[bool, float, str]:
        """Evaluate whether the invariant holds.

        Parameters
        ----------
        task:
            The task being evaluated (provides training pairs for inference).
        candidate_answer:
            Optional predicted output grid.  When *None*, the checker should
            infer whether the invariant is expected to hold at all.

        Returns
        -------
        (holds, confidence, note)
            *holds* — True if the invariant is satisfied (or not applicable).
            *confidence* — How confident we are that the invariant should hold
                (0.0 = no signal, 1.0 = certain).
            *note* — Human-readable explanation.
        """
