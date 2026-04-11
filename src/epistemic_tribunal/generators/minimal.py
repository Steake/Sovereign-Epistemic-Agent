"""Minimal-description generator — applies Occam's-razor compression heuristic.

Mock logic:
- Finds the output grid with the smallest number of distinct colours that is
  also consistent with the training transformation pattern.
- Prefers simpler outputs (fewer distinct colours, smaller bounding box).
"""

from __future__ import annotations

import random

from epistemic_tribunal.generators.base import BaseGenerator
from epistemic_tribunal.generators.greedy import GreedyGenerator
from epistemic_tribunal.tasks.base import (
    colour_counts,
    grid_shape,
    object_count,
    unique_colours,
)
from epistemic_tribunal.types import CandidateTrace, Task


class MinimalDescriptionGenerator(BaseGenerator):
    """Applies a minimal-description-length heuristic to simplify the output."""

    name = "minimal_description"

    def generate(self, task: Task) -> CandidateTrace:
        rng = random.Random(self.seed + 4)

        # Start from greedy
        base_trace = GreedyGenerator(seed=self.seed).generate(task)
        candidate = [row[:] for row in base_trace.answer]

        # Compute MDL proxy: prefer fewer distinct colours
        in_colours = unique_colours(task.test_input)
        out_colours = unique_colours(candidate)

        # If output has strictly more colours than input, attempt to simplify
        simplification_applied = False
        if len(out_colours) > len(in_colours) and in_colours:
            # Map all extra colours to the most common input colour
            most_common_in = max(
                colour_counts(task.test_input),
                key=lambda k: colour_counts(task.test_input)[k],
            )
            extra = out_colours - in_colours
            rows, cols = grid_shape(candidate)
            for r in range(rows):
                for c in range(cols):
                    if candidate[r][c] in extra:
                        candidate[r][c] = most_common_in
            simplification_applied = True

        # Compute MDL score as inverse of distinct colours
        final_colours = unique_colours(candidate)
        mdl_score = 1.0 / (1.0 + len(final_colours))

        confidence = round(0.35 + 0.4 * mdl_score + rng.uniform(-0.02, 0.02), 4)

        steps = [
            f"Input has {len(in_colours)} distinct colour(s): {sorted(in_colours)}.",
            f"Greedy answer has {len(out_colours)} colour(s): {sorted(out_colours)}.",
            f"Simplification applied: {simplification_applied}.",
            f"Final answer uses {len(final_colours)} colour(s). MDL score={mdl_score:.3f}.",
        ]

        return CandidateTrace(
            generator_name=self.name,
            answer=candidate,
            reasoning_steps=steps,
            raw_trace="\n".join(steps),
            token_count=len(" ".join(steps).split()),
            confidence_score=min(1.0, confidence),
            derived_features={
                "mdl_score": mdl_score,
                "simplification_applied": simplification_applied,
                "distinct_colours_out": len(final_colours),
                "object_count_in": object_count(task.test_input),
                "object_count_out": object_count(candidate),
                "colour_counts_out": colour_counts(candidate),
            },
        )
