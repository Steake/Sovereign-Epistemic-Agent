"""Tests for Strange Loop Memory v1 — pre-generation failure constraint injection.

Covers:
- FailureConstraints data model
- FailureConstraintBuilder query and extraction logic
- LLM prompt injection formatting
- Orchestrator wiring (constraint building and pass-through)
"""

from __future__ import annotations

import pytest

from epistemic_tribunal.failure_memory.constraint_builder import FailureConstraintBuilder
from epistemic_tribunal.failure_memory.models import (
    FailureConstraints,
    FailureSignature,
    FailureType,
)
from epistemic_tribunal.failure_memory.store import FailureMemoryStore
from epistemic_tribunal.generators.llm import LLMGenerator
from epistemic_tribunal.tribunal_types import Task, TaskDomain, GridExample


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_task(task_id: str = "test_task_001", domain: TaskDomain = TaskDomain.ARC_LIKE) -> Task:
    """Create a minimal test task."""
    return Task(
        task_id=task_id,
        domain=domain,
        train=[
            GridExample(
                input=[[0, 1], [1, 0]],
                output=[[1, 0], [0, 1]],
            )
        ],
        test_input=[[0, 1], [1, 0]],
        ground_truth=[[1, 0], [0, 1]],
    )


def _make_failure_signature(
    task_id: str = "test_task_001",
    domain: str = "arc_like",
    failure_type: FailureType = FailureType.WRONG_PICK,
    answer_signature: str = "((0, 0), (0, 0))",
    false_majority: bool = False,
    minority_correct: bool = False,
    rationale_present: bool = True,
    parse_issue: bool = False,
    disagreement_rate: float = 0.5,
) -> FailureSignature:
    """Create a test failure signature."""
    return FailureSignature(
        domain=domain,
        task_id=task_id,
        failure_type=failure_type,
        answer_signature=answer_signature,
        coalition_context={
            "majority_size": 3,
            "n_clusters": 2,
            "coalition_mass": 0.6,
            "false_majority": false_majority,
            "minority_correct": minority_correct,
            "total_candidates": 5,
            "parse_issue_present": parse_issue,
        },
        trace_quality_features={
            "rationale_present": rationale_present,
            "reasoning_step_count": 3 if rationale_present else 0,
            "trace_length": 100,
            "finish_reason": "stop",
            "generator_name": "llm",
        },
        critic_context={
            "aggregate_score": 0.7,
            "all_flat": False,
        },
        disagreement_rate=disagreement_rate,
        structural_margin=0.05,
        outcome_label="wrong_pick",
    )


@pytest.fixture
def memory_store():
    """In-memory failure store."""
    return FailureMemoryStore(path=":memory:")


@pytest.fixture
def builder(memory_store):
    """Constraint builder with default settings."""
    return FailureConstraintBuilder(
        store=memory_store,
        max_bad_answers=5,
        max_warnings=3,
        min_similarity=0.3,
        same_task_boost=1.5,
    )


# ---------------------------------------------------------------------------
# FailureConstraints model tests
# ---------------------------------------------------------------------------


class TestFailureConstraintsModel:
    def test_empty_constraints_have_no_guidance(self):
        c = FailureConstraints()
        assert not c.has_constraints

    def test_constraints_with_bad_answers(self):
        c = FailureConstraints(bad_answers=["((0, 0), (0, 0))"])
        assert c.has_constraints

    def test_constraints_with_warnings(self):
        c = FailureConstraints(structural_warnings=["some warning"])
        assert c.has_constraints

    def test_constraint_strength_bounds(self):
        c = FailureConstraints(constraint_strength=0.5)
        assert 0.0 <= c.constraint_strength <= 1.0

    def test_metadata_is_empty_by_default(self):
        c = FailureConstraints()
        assert c.metadata == {}


# ---------------------------------------------------------------------------
# ConstraintBuilder tests
# ---------------------------------------------------------------------------


class TestConstraintBuilder:
    def test_empty_store_produces_no_constraints(self, builder, memory_store):
        """Cold start — no failure signatures in memory."""
        task = _make_task()
        constraints = builder.build(task)
        assert not constraints.has_constraints
        assert constraints.bad_answers == []
        assert constraints.structural_warnings == []

    def test_same_task_wrong_pick_produces_bad_answer(self, builder, memory_store):
        """Exact task_id match with a wrong_pick should produce bad-answer avoidance."""
        sig = _make_failure_signature(
            task_id="test_task_001",
            answer_signature="((0, 0), (0, 0))",
        )
        memory_store.store(sig)

        task = _make_task(task_id="test_task_001")
        constraints = builder.build(task)

        assert constraints.has_constraints
        assert "((0, 0), (0, 0))" in constraints.bad_answers
        assert "test_task_001" in constraints.source_task_ids

    def test_different_domain_not_matched(self, builder, memory_store):
        """Failures from a different domain should not be returned."""
        sig = _make_failure_signature(domain="gsm8k_math")
        memory_store.store(sig)

        task = _make_task(domain=TaskDomain.ARC_LIKE)
        constraints = builder.build(task)
        assert not constraints.has_constraints

    def test_correct_selections_ignored(self, builder, memory_store):
        """Only wrong_pick and bad_abstention should produce constraints."""
        sig = _make_failure_signature(failure_type=FailureType.CORRECT_SELECT)
        memory_store.store(sig)

        task = _make_task()
        constraints = builder.build(task)
        assert not constraints.has_constraints

    def test_false_majority_produces_warning(self, builder, memory_store):
        """A prior false-majority failure should generate a structural warning."""
        sig = _make_failure_signature(false_majority=True)
        memory_store.store(sig)

        task = _make_task()
        constraints = builder.build(task)

        assert constraints.has_constraints
        assert any("majority" in w.lower() for w in constraints.structural_warnings)

    def test_minority_correct_produces_warning(self, builder, memory_store):
        """A prior minority-correct failure should generate a structural warning."""
        sig = _make_failure_signature(minority_correct=True)
        memory_store.store(sig)

        task = _make_task()
        constraints = builder.build(task)

        assert constraints.has_constraints
        assert any("minority" in w.lower() for w in constraints.structural_warnings)

    def test_no_rationale_produces_warning(self, builder, memory_store):
        """A prior wrong_pick without rationale should produce a reasoning warning."""
        sig = _make_failure_signature(rationale_present=False)
        memory_store.store(sig)

        task = _make_task()
        constraints = builder.build(task)

        assert constraints.has_constraints
        assert any("reasoning" in w.lower() for w in constraints.structural_warnings)

    def test_constraint_strength_scales_with_similarity(self, builder, memory_store):
        """constraint_strength should be > 0 when there are matches."""
        sig = _make_failure_signature(task_id="test_task_001")
        memory_store.store(sig)

        task = _make_task(task_id="test_task_001")
        constraints = builder.build(task)

        assert constraints.constraint_strength > 0.0
        assert constraints.constraint_strength <= 1.0

    def test_max_bad_answers_respected(self, memory_store):
        """Builder should cap bad answers at max_bad_answers."""
        builder = FailureConstraintBuilder(
            store=memory_store,
            max_bad_answers=2,
            max_warnings=3,
            min_similarity=0.0,
        )
        for i in range(10):
            sig = _make_failure_signature(
                task_id="test_task_001",
                answer_signature=f"answer_{i}",
            )
            memory_store.store(sig)

        task = _make_task()
        constraints = builder.build(task)
        assert len(constraints.bad_answers) <= 2

    def test_max_warnings_respected(self, memory_store):
        """Builder should cap structural warnings at max_warnings."""
        builder = FailureConstraintBuilder(
            store=memory_store,
            max_bad_answers=5,
            max_warnings=1,
            min_similarity=0.0,
        )
        # Store signatures with multiple warning patterns
        sig = _make_failure_signature(
            false_majority=True,
            minority_correct=True,
            rationale_present=False,
            parse_issue=True,
        )
        memory_store.store(sig)

        task = _make_task()
        constraints = builder.build(task)
        assert len(constraints.structural_warnings) <= 1

    def test_deduplicate_bad_answers(self, builder, memory_store):
        """Duplicate answer signatures should be deduplicated."""
        for _ in range(5):
            sig = _make_failure_signature(answer_signature="((1, 1), (1, 1))")
            memory_store.store(sig)

        task = _make_task()
        constraints = builder.build(task)

        dupes = [a for a in constraints.bad_answers if a == "((1, 1), (1, 1))"]
        assert len(dupes) <= 1


# ---------------------------------------------------------------------------
# LLM prompt injection tests
# ---------------------------------------------------------------------------


class TestLLMConstraintBlock:
    def test_no_constraints_produces_empty_string(self):
        result = LLMGenerator._build_constraint_block(None)
        assert result == ""

    def test_empty_constraints_produces_empty_string(self):
        c = FailureConstraints()
        result = LLMGenerator._build_constraint_block(c)
        assert result == ""

    def test_bad_answers_injected(self):
        c = FailureConstraints(bad_answers=["((0, 0), (0, 0))"])
        result = LLMGenerator._build_constraint_block(c)
        assert "FAILURE MEMORY" in result
        assert "((0, 0), (0, 0))" in result
        assert "WRONG" in result

    def test_warnings_injected(self):
        c = FailureConstraints(
            structural_warnings=["the majority was a trap"]
        )
        result = LLMGenerator._build_constraint_block(c)
        assert "FAILURE MEMORY" in result
        assert "the majority was a trap" in result
        assert "⚠" in result

    def test_both_injected(self):
        c = FailureConstraints(
            bad_answers=["answer_1"],
            structural_warnings=["warning_1"],
        )
        result = LLMGenerator._build_constraint_block(c)
        assert "answer_1" in result
        assert "warning_1" in result
        assert "END FAILURE MEMORY" in result

    def test_block_ends_with_delimiter(self):
        c = FailureConstraints(bad_answers=["x"])
        result = LLMGenerator._build_constraint_block(c)
        assert "=== END FAILURE MEMORY ===" in result


# ---------------------------------------------------------------------------
# Orchestrator integration tests
# ---------------------------------------------------------------------------


class TestOrchestratorStrangeLoop:
    def test_strange_loop_disabled_by_default(self):
        """Default config has strange_loop disabled."""
        from epistemic_tribunal.config import TribunalSettings

        settings = TribunalSettings()
        assert settings.strange_loop.enabled is False

    def test_strange_loop_config_fields(self):
        """Verify config model fields and defaults."""
        from epistemic_tribunal.config import StrangeLoopConfig

        cfg = StrangeLoopConfig()
        assert cfg.enabled is False
        assert cfg.max_bad_answers == 5
        assert cfg.max_warnings == 3
        assert cfg.min_similarity == 0.3
        assert cfg.same_task_boost == 1.5

    def test_orchestrator_initialises_with_strange_loop(self):
        """Orchestrator should initialise the constraint builder when enabled."""
        from epistemic_tribunal.config import TribunalSettings
        from epistemic_tribunal.orchestrator import Orchestrator

        settings = TribunalSettings(
            failure_memory={"enabled": True},
            strange_loop={"enabled": True},
            ledger={"path": ":memory:"},
            generators={"enabled": ["greedy"]},
        )
        orch = Orchestrator(config=settings)
        assert orch._strange_loop_enabled is True
        assert orch._constraint_builder is not None

    def test_orchestrator_skips_when_failure_memory_disabled(self):
        """Strange loop requires failure_memory to also be enabled."""
        from epistemic_tribunal.config import TribunalSettings
        from epistemic_tribunal.orchestrator import Orchestrator

        settings = TribunalSettings(
            failure_memory={"enabled": False},
            strange_loop={"enabled": True},
            ledger={"path": ":memory:"},
            generators={"enabled": ["greedy"]},
        )
        orch = Orchestrator(config=settings)
        assert orch._strange_loop_enabled is False
        assert orch._constraint_builder is None

    def test_orchestrator_run_with_strange_loop_cold(self):
        """Run with strange_loop enabled but empty memory — no crash."""
        from epistemic_tribunal.config import TribunalSettings
        from epistemic_tribunal.orchestrator import Orchestrator

        settings = TribunalSettings(
            failure_memory={"enabled": True},
            strange_loop={"enabled": True},
            ledger={"path": ":memory:"},
            generators={"enabled": ["greedy"]},
            tribunal={"ledger_warmup_tasks": 0},
        )
        orch = Orchestrator(config=settings)

        task = _make_task()
        run = orch.run(task)

        # Should complete without error
        assert run.task_id == "test_task_001"
        # No strange_loop metadata expected (empty store, no constraints)
        assert "strange_loop" not in run.metadata
