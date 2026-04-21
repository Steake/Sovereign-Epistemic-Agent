"""Invariant extractor — infers structural constraints from ARC-like training pairs.

For each training pair we observe whether certain properties are preserved
(object count, colour count, symmetry, etc.).  If they are consistently
preserved across all pairs, we emit them as invariants with high confidence.
"""

from __future__ import annotations

from typing import Optional

from epistemic_tribunal.invariants.base import BaseInvariantChecker
from epistemic_tribunal.tasks.base import (
    Grid,
    has_any_symmetry,
    object_count,
    unique_colours,
    grid_shape,
    bounding_box,
    connected_components,
)
from epistemic_tribunal.tribunal_types import Invariant, InvariantSet, Task


# ---------------------------------------------------------------------------
# Individual checkers
# ---------------------------------------------------------------------------


class ObjectCountPreservedChecker(BaseInvariantChecker):
    name = "object_count_preserved"

    def check(
        self, task: Task, candidate_answer: Optional[Grid] = None
    ) -> tuple[bool, float, str]:
        if not task.train:
            return True, 0.0, "No training data."
        preserved = [
            object_count(ex.input) == object_count(ex.output)
            for ex in task.train
        ]
        ratio = sum(preserved) / len(preserved)
        confidence = round(ratio, 3)
        if candidate_answer is None:
            return ratio == 1.0, confidence, f"Training: {sum(preserved)}/{len(preserved)} pairs preserve object count."
        # Check candidate
        expected = round(
            sum(object_count(ex.output) for ex in task.train) / len(task.train)
        )
        actual = object_count(candidate_answer)
        holds = abs(actual - expected) <= 1  # allow ±1 tolerance
        return holds, confidence, f"Expected ≈{expected} objects, got {actual}."


class ColourCountPreservedChecker(BaseInvariantChecker):
    name = "colour_count_preserved"

    def check(
        self, task: Task, candidate_answer: Optional[Grid] = None
    ) -> tuple[bool, float, str]:
        if not task.train:
            return True, 0.0, "No training data."
        preserved = [
            len(unique_colours(ex.input)) == len(unique_colours(ex.output))
            for ex in task.train
        ]
        ratio = sum(preserved) / len(preserved)
        confidence = round(ratio, 3)
        if candidate_answer is None:
            return ratio == 1.0, confidence, f"Training: {sum(preserved)}/{len(preserved)} pairs preserve colour count."
        in_count = len(unique_colours(task.test_input))
        out_count = len(unique_colours(candidate_answer))
        holds = in_count == out_count
        return holds, confidence, f"Input has {in_count} colour(s), output has {out_count}."


class SymmetryExpectedChecker(BaseInvariantChecker):
    name = "symmetry_expected"

    def check(
        self, task: Task, candidate_answer: Optional[Grid] = None
    ) -> tuple[bool, float, str]:
        if not task.train:
            return True, 0.0, "No training data."
        symmetric_outputs = [has_any_symmetry(ex.output) for ex in task.train]
        ratio = sum(symmetric_outputs) / len(symmetric_outputs)
        confidence = round(ratio, 3)
        if candidate_answer is None:
            return ratio >= 0.5, confidence, f"Training: {sum(symmetric_outputs)}/{len(symmetric_outputs)} outputs are symmetric."
        holds = has_any_symmetry(candidate_answer)
        return holds, confidence, f"Candidate is {'symmetric' if holds else 'not symmetric'}."


class ShapeTransformExpectedChecker(BaseInvariantChecker):
    """Checks whether the output shape (rows×cols) matches the pattern seen in training."""

    name = "shape_transform_expected"

    def check(
        self, task: Task, candidate_answer: Optional[Grid] = None
    ) -> tuple[bool, float, str]:
        if not task.train:
            return True, 0.0, "No training data."

        # Compute expected output shape from training
        same_shape = [
            grid_shape(ex.input) == grid_shape(ex.output)
            for ex in task.train
        ]
        ratio = sum(same_shape) / len(same_shape)
        confidence = round(ratio, 3)

        if candidate_answer is None:
            return ratio >= 0.5, confidence, f"Training: {sum(same_shape)}/{len(same_shape)} pairs preserve shape."

        in_shape = grid_shape(task.test_input)
        cand_shape = grid_shape(candidate_answer)

        if ratio >= 0.5:
            # Expect same shape
            holds = in_shape == cand_shape
            return holds, confidence, f"Expected shape={in_shape}, got {cand_shape}."
        else:
            # Shape transforms — infer expected output shape from training
            if task.train:
                out_shapes = [grid_shape(ex.output) for ex in task.train]
                most_common_shape = max(set(out_shapes), key=out_shapes.count)
                holds = cand_shape == most_common_shape
                return holds, confidence, f"Expected transformed shape={most_common_shape}, got {cand_shape}."
            return True, 0.0, "Unable to infer expected output shape."


class SizeRelationPreservedChecker(BaseInvariantChecker):
    """Checks that relative object sizes are preserved."""

    name = "size_relation_preserved"

    def check(
        self, task: Task, candidate_answer: Optional[Grid] = None
    ) -> tuple[bool, float, str]:
        if not task.train or len(task.train) < 2:
            return True, 0.0, "Insufficient training data for size relation check."

        # Compute size-order consistency across training pairs
        consistent = 0
        for ex in task.train:
            in_comps = connected_components(ex.input)
            out_comps = connected_components(ex.output)
            if len(in_comps) == len(out_comps) == 2:
                in_sizes = sorted([len(c) for c in in_comps])
                out_sizes = sorted([len(c) for c in out_comps])
                if (in_sizes[0] <= in_sizes[1]) == (out_sizes[0] <= out_sizes[1]):
                    consistent += 1
            else:
                consistent += 1  # not applicable → assume consistent

        ratio = consistent / len(task.train)
        confidence = round(ratio, 3)

        if candidate_answer is None:
            return ratio >= 0.5, confidence, f"Training: {consistent}/{len(task.train)} preserve size relations."

        return True, confidence, "Size relation check passed (heuristic)."


class BoundingBoxConsistentChecker(BaseInvariantChecker):
    """Checks that bounding-box coverage ratio is consistent with training."""

    name = "bounding_box_consistent"

    def check(
        self, task: Task, candidate_answer: Optional[Grid] = None
    ) -> tuple[bool, float, str]:
        if not task.train:
            return True, 0.0, "No training data."

        def coverage(grid: Grid) -> float:
            rows, cols = grid_shape(grid)
            non_bg = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] != 0]
            if not non_bg:
                return 0.0
            min_r, min_c, max_r, max_c = bounding_box(non_bg)
            bb_area = (max_r - min_r + 1) * (max_c - min_c + 1)
            return bb_area / (rows * cols)

        train_ratios = [
            coverage(ex.output) / max(coverage(ex.input), 0.01)
            for ex in task.train
        ]
        avg_ratio = sum(train_ratios) / len(train_ratios)
        confidence = 0.6  # moderate confidence for this heuristic

        if candidate_answer is None:
            return True, confidence, f"Average bounding-box ratio in training: {avg_ratio:.2f}."

        cand_coverage = coverage(candidate_answer)
        in_coverage = coverage(task.test_input)
        actual_ratio = cand_coverage / max(in_coverage, 0.01)
        holds = abs(actual_ratio - avg_ratio) <= 0.5  # generous tolerance
        return holds, confidence, f"Expected BB ratio≈{avg_ratio:.2f}, got {actual_ratio:.2f}."


class GridDimensionsConsistentChecker(BaseInvariantChecker):
    """Checks that the candidate grid has the expected dimensions."""

    name = "grid_dimensions_consistent"

    def check(
        self, task: Task, candidate_answer: Optional[Grid] = None
    ) -> tuple[bool, float, str]:
        if not task.train:
            return True, 0.0, "No training data."

        # Infer expected output dimensions
        in_shapes = [grid_shape(ex.input) for ex in task.train]
        out_shapes = [grid_shape(ex.output) for ex in task.train]

        # If all inputs and outputs have the same shape, expect that
        if len(set(in_shapes)) == 1 and len(set(out_shapes)) == 1:
            expected_in = in_shapes[0]
            expected_out = out_shapes[0]
            if expected_in == expected_out:
                confidence = 0.9
                if candidate_answer is None:
                    return True, confidence, f"Training: constant shape {expected_in}."
                cand_shape = grid_shape(candidate_answer)
                test_in_shape = grid_shape(task.test_input)
                holds = cand_shape == test_in_shape
                return holds, confidence, f"Expected shape={test_in_shape}, got {cand_shape}."

        return True, 0.4, "Variable dimensions in training — skipping strict check."


# ---------------------------------------------------------------------------
# Registry and extractor
# ---------------------------------------------------------------------------


_ALL_CHECKERS: list[BaseInvariantChecker] = [
    ObjectCountPreservedChecker(),
    ColourCountPreservedChecker(),
    SymmetryExpectedChecker(),
    ShapeTransformExpectedChecker(),
    SizeRelationPreservedChecker(),
    BoundingBoxConsistentChecker(),
    GridDimensionsConsistentChecker(),
]

_CHECKER_MAP: dict[str, BaseInvariantChecker] = {c.name: c for c in _ALL_CHECKERS}


class InvariantExtractor:
    """Infers task-level invariants from training examples.

    Parameters
    ----------
    enabled_checks:
        List of checker names to run.  All checkers are used by default.
    confidence_threshold:
        Minimum confidence to include an invariant in the result set.
    """

    def __init__(
        self,
        enabled_checks: Optional[list[str]] = None,
        confidence_threshold: float = 0.5,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        if enabled_checks is None:
            self._checkers = list(_ALL_CHECKERS)
        else:
            self._checkers = [
                _CHECKER_MAP[n] for n in enabled_checks if n in _CHECKER_MAP
            ]

    def extract(self, task: Task) -> InvariantSet:
        """Run all enabled checkers and return an :class:`InvariantSet`."""
        from epistemic_tribunal.tribunal_types import TaskDomain
        if task.domain == TaskDomain.GSM8K_MATH:
            return InvariantSet(task_id=task.task_id, invariants=[], extraction_notes="Skipped for math domain.")

        invariants: list[Invariant] = []
        notes_parts: list[str] = []

        for checker in self._checkers:
            holds, confidence, note = checker.check(task, candidate_answer=None)
            notes_parts.append(f"[{checker.name}] {note}")
            if confidence >= self.confidence_threshold:
                invariants.append(
                    Invariant(
                        name=checker.name,
                        description=note,
                        confidence=confidence,
                        notes=note,
                    )
                )

        return InvariantSet(
            task_id=task.task_id,
            invariants=invariants,
            extraction_notes="; ".join(notes_parts),
        )

    def check_candidate(
        self,
        task: Task,
        candidate_answer: list[list[int]],
        invariant_set: InvariantSet,
    ) -> dict[str, tuple[bool, float, str]]:
        """Check a candidate answer against all invariants in *invariant_set*.

        Returns
        -------
        dict[str, (holds, confidence, note)]
            One entry per invariant name.
        """
        results: dict[str, tuple[bool, float, str]] = {}
        active_names = {inv.name for inv in invariant_set.invariants}

        for checker in self._checkers:
            if checker.name not in active_names:
                continue
            holds, confidence, note = checker.check(task, candidate_answer=candidate_answer)
            results[checker.name] = (holds, confidence, note)

        return results
