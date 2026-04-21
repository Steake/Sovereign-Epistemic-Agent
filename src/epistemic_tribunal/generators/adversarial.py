"""Adversarial generator — deliberately proposes counter-intuitive outputs.

Mock logic:
- Invert the greedy colour mapping (swap src↔dst).
- Useful for stress-testing the tribunal: can it correctly reject a plausible
  but wrong hypothesis?
"""

from __future__ import annotations
from typing import Callable, Optional

import random

from epistemic_tribunal.generators.base import BaseGenerator
from epistemic_tribunal.generators.greedy import GreedyGenerator
from epistemic_tribunal.tasks.base import colour_counts, grid_shape, object_count
from epistemic_tribunal.tribunal_types import CandidateTrace, Task


class AdversarialGenerator(BaseGenerator):
    """Proposes an adversarial answer by inverting the greedy colour map."""

    name = "adversarial"

    def generate(
        self, 
        task: Task, 
        on_token: Optional[Callable[[str, str], None]] = None
    ) -> CandidateTrace:
        rng = random.Random(self.seed + 2)

        # Get greedy mapping features
        base_trace = GreedyGenerator(seed=self.seed).generate(task)
        greedy_map: dict[int, int] = base_trace.derived_features.get("colour_mapping", {})

        # Invert the mapping: dst → src
        inverted: dict[int, int] = {v: k for k, v in greedy_map.items()}

        rows, cols = grid_shape(task.test_input)
        answer: list[list[int]] = []
        for r in range(rows):
            row_out = []
            for c in range(cols):
                src = task.test_input[r][c]
                dst = inverted.get(src, src)
                row_out.append(dst)
            answer.append(row_out)

        # Deliberately low confidence — adversarial trace is meant to lose
        confidence = round(rng.uniform(0.10, 0.30), 4)

        steps = [
            f"Computed inverted colour mapping: {inverted}.",
            "Applied inverse mapping to introduce adversarial hypothesis.",
            "Low confidence assigned; designed to test tribunal rejection.",
        ]

        return CandidateTrace(
            generator_name=self.name,
            answer=answer,
            reasoning_steps=steps,
            raw_trace="\n".join(steps),
            token_count=len(" ".join(steps).split()),
            confidence_score=confidence,
            derived_features={
                "inverted_mapping": inverted,
                "original_mapping": greedy_map,
                "object_count_in": object_count(task.test_input),
                "object_count_out": object_count(answer),
                "colour_counts_out": colour_counts(answer),
            },
        )
