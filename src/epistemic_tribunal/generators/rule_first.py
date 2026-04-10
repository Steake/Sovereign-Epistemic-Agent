"""Rule-first generator — attempts to extract explicit transformation rules.

Mock logic:
- Looks for row/column patterns such as:
  * Fill entire output with the most common colour.
  * Copy input unchanged.
  * Rotate or transpose the grid if shapes permit.
- Selects the rule that best fits training examples.
"""

from __future__ import annotations

import random
from typing import Callable

from epistemic_tribunal.generators.base import BaseGenerator
from epistemic_tribunal.tasks.base import (
    colour_counts,
    grid_shape,
    grids_equal,
    object_count,
)
from epistemic_tribunal.types import CandidateTrace, Task

Grid = list[list[int]]


# ---------------------------------------------------------------------------
# Simple rule implementations
# ---------------------------------------------------------------------------


def _rule_copy(test_input: Grid) -> Grid:
    return [row[:] for row in test_input]


def _rule_fill_most_common(test_input: Grid) -> Grid:
    counts = colour_counts(test_input)
    dominant = max(counts, key=lambda k: counts[k])
    rows, cols = grid_shape(test_input)
    return [[dominant] * cols for _ in range(rows)]


def _rule_transpose(test_input: Grid) -> Grid:
    rows, cols = grid_shape(test_input)
    return [[test_input[r][c] for r in range(rows)] for c in range(cols)]


def _rule_flip_horizontal(test_input: Grid) -> Grid:
    return [row[::-1] for row in test_input]


def _rule_flip_vertical(test_input: Grid) -> Grid:
    return test_input[::-1]


_RULES: list[tuple[str, Callable[[Grid], Grid]]] = [
    ("copy", _rule_copy),
    ("fill_most_common", _rule_fill_most_common),
    ("transpose", _rule_transpose),
    ("flip_horizontal", _rule_flip_horizontal),
    ("flip_vertical", _rule_flip_vertical),
]


class RuleFirstGenerator(BaseGenerator):
    """Selects the simplest rule that best fits the training examples."""

    name = "rule_first"

    def generate(self, task: Task) -> CandidateTrace:
        rng = random.Random(self.seed + 3)

        best_rule_name = "copy"
        best_score = -1.0
        best_answer: Grid = _rule_copy(task.test_input)

        for rule_name, rule_fn in _RULES:
            score = self._evaluate_rule(rule_fn, task)
            if score > best_score:
                best_score = score
                best_rule_name = rule_name
                best_answer = rule_fn(task.test_input)

        confidence = round(0.3 + 0.5 * best_score + rng.uniform(-0.02, 0.02), 4)

        steps = [
            f"Evaluated {len(_RULES)} candidate rules against {len(task.train)} training pairs.",
            f"Best rule: {best_rule_name!r} with training fit={best_score:.2f}.",
            "Applied selected rule to test input.",
        ]

        return CandidateTrace(
            generator_name=self.name,
            answer=best_answer,
            reasoning_steps=steps,
            raw_trace="\n".join(steps),
            token_count=len(" ".join(steps).split()),
            confidence_score=min(1.0, confidence),
            derived_features={
                "selected_rule": best_rule_name,
                "rule_fit": best_score,
                "object_count_in": object_count(task.test_input),
                "object_count_out": object_count(best_answer),
                "colour_counts_out": colour_counts(best_answer),
            },
        )

    @staticmethod
    def _evaluate_rule(rule_fn: Callable[[Grid], Grid], task: Task) -> float:
        """Score a rule by how many training outputs it exactly matches."""
        if not task.train:
            return 0.0
        matches = 0
        for ex in task.train:
            try:
                predicted = rule_fn(ex.input)
                if grids_equal(predicted, ex.output):
                    matches += 1
            except Exception:
                pass
        return matches / len(task.train)
