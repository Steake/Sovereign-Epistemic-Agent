"""Diverse generator — introduces controlled variation to explore the output space.

Mock logic:
- Start from the greedy mapping.
- Randomly permute a small fraction of colour assignments to diversify.
- Useful for breaking ties and generating alternative hypotheses.
"""

from __future__ import annotations
from typing import Callable, Optional

import random

from epistemic_tribunal.failure_memory.models import FailureConstraints
from epistemic_tribunal.generators.base import BaseGenerator
from epistemic_tribunal.generators.greedy import GreedyGenerator
from epistemic_tribunal.tasks.base import colour_counts, grid_shape, object_count
from epistemic_tribunal.tribunal_types import CandidateTrace, Task


class DiverseGenerator(BaseGenerator):
    """Builds on the greedy answer with stochastic colour perturbations."""

    name = "diverse"

    #: Fraction of cells to perturb.
    perturbation_rate: float = 0.10

    def generate(
        self,
        task: Task,
        on_token: Optional[Callable[[str, str], None]] = None,
        failure_constraints: Optional[FailureConstraints] = None,
    ) -> CandidateTrace:
        rng = random.Random(self.seed + 1)  # offset seed for diversity

        # Start from greedy answer
        base_trace = GreedyGenerator(seed=self.seed).generate(task)
        base_answer = [row[:] for row in base_trace.answer]

        rows, cols = grid_shape(base_answer)
        all_colours = list(
            {cell for row in base_answer for cell in row}
            | {cell for row in task.test_input for cell in row}
        )

        # Perturb a fraction of cells
        perturbations = 0
        for r in range(rows):
            for c in range(cols):
                if rng.random() < self.perturbation_rate:
                    original = base_answer[r][c]
                    replacement = rng.choice(all_colours)
                    base_answer[r][c] = replacement
                    if replacement != original:
                        perturbations += 1

        confidence = max(0.1, base_trace.confidence_score or 0.5 - 0.05 * perturbations)

        steps = [
            "Initialised from greedy answer.",
            f"Perturbed {perturbations} cell(s) at rate {self.perturbation_rate}.",
            f"Final answer differs from greedy in {perturbations} position(s).",
        ]

        return CandidateTrace(
            generator_name=self.name,
            answer=base_answer,
            reasoning_steps=steps,
            raw_trace="\n".join(steps),
            token_count=len(" ".join(steps).split()),
            confidence_score=round(min(1.0, confidence), 4),
            derived_features={
                "perturbations": perturbations,
                "perturbation_rate": self.perturbation_rate,
                "object_count_in": object_count(task.test_input),
                "object_count_out": object_count(base_answer),
                "colour_counts_out": colour_counts(base_answer),
            },
        )
