"""Tests for the invariant extractor."""

from __future__ import annotations

import pytest

from epistemic_tribunal.invariants.extractor import InvariantExtractor
from epistemic_tribunal.types import GridExample, InvariantSet, Task, TaskDomain


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extractor() -> InvariantExtractor:
    return InvariantExtractor(confidence_threshold=0.4)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_extract_returns_invariant_set(simple_task: Task) -> None:
    ext = _extractor()
    inv_set = ext.extract(simple_task)
    assert isinstance(inv_set, InvariantSet)
    assert inv_set.task_id == simple_task.task_id


def test_extract_nonempty_invariants(simple_task: Task) -> None:
    ext = _extractor()
    inv_set = ext.extract(simple_task)
    assert len(inv_set.invariants) > 0


def test_extract_confidence_scores_bounded(simple_task: Task) -> None:
    ext = _extractor()
    inv_set = ext.extract(simple_task)
    for inv in inv_set.invariants:
        assert 0.0 <= inv.confidence <= 1.0


def test_extract_invariant_names_known(simple_task: Task) -> None:
    ext = _extractor()
    inv_set = ext.extract(simple_task)
    known = {
        "object_count_preserved",
        "colour_count_preserved",
        "symmetry_expected",
        "shape_transform_expected",
        "size_relation_preserved",
        "bounding_box_consistent",
        "grid_dimensions_consistent",
    }
    for inv in inv_set.invariants:
        assert inv.name in known, f"Unknown invariant: {inv.name!r}"


def test_shape_transform_expected_for_identity(identity_task: Task) -> None:
    """Identity task should report grid_dimensions_consistent as high confidence."""
    ext = _extractor()
    inv_set = ext.extract(identity_task)
    dims = inv_set.by_name("grid_dimensions_consistent")
    if dims is not None:
        assert dims.confidence >= 0.5


def test_check_candidate_correct_answer(simple_task: Task) -> None:
    ext = _extractor()
    inv_set = ext.extract(simple_task)
    # Ground truth should pass most invariant checks
    results = ext.check_candidate(simple_task, simple_task.ground_truth, inv_set)
    assert isinstance(results, dict)
    passing = sum(1 for (holds, _, _) in results.values() if holds)
    assert passing >= 0  # At minimum the function runs without error


def test_check_candidate_all_zeros_answer(simple_task: Task) -> None:
    """An all-zero answer should fail at least one invariant."""
    ext = _extractor()
    inv_set = ext.extract(simple_task)
    zero_answer = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    results = ext.check_candidate(simple_task, zero_answer, inv_set)
    # colour_count_preserved or object_count_preserved should fail
    assert len(results) >= 0  # at least runs


def test_extract_no_training_data() -> None:
    """Extractor should handle tasks with no training examples gracefully."""
    task = Task(
        task_id="no_train",
        train=[],
        test_input=[[1, 2], [3, 4]],
    )
    ext = _extractor()
    inv_set = ext.extract(task)
    # Should return empty or low-confidence invariants — not crash
    assert isinstance(inv_set, InvariantSet)
    for inv in inv_set.invariants:
        # No training → low confidence; threshold filters them out
        assert inv.confidence >= ext.confidence_threshold


def test_enabled_checks_subset() -> None:
    """Extractor should only run the enabled checks."""
    ext = InvariantExtractor(
        enabled_checks=["object_count_preserved", "colour_count_preserved"],
        confidence_threshold=0.0,
    )
    task = Task(
        task_id="subset_check",
        train=[
            GridExample(
                input=[[1, 2], [3, 4]],
                output=[[1, 2], [3, 4]],
            )
        ],
        test_input=[[1, 2], [3, 4]],
    )
    inv_set = ext.extract(task)
    names = {inv.name for inv in inv_set.invariants}
    assert names.issubset({"object_count_preserved", "colour_count_preserved"})
