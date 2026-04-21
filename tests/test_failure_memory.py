"""Tests for the failure-memory metacognitive layer.

Covers:
1. FailureSignature extraction from different outcome types
2. FailureMemoryStore persistence round-trip
3. FailureProbe building (observable-only)
4. Similarity query matching
5. Per-trace penalty computation
6. Integration: penalties flow through M channel into scoring
7. ARC regression: failure memory is additive
"""

from __future__ import annotations

import pytest


from epistemic_tribunal.failure_memory.extractor import FailureSignatureExtractor
from epistemic_tribunal.failure_memory.models import (
    FailureProbe,
    FailureSignature,
    FailureType,
)
from epistemic_tribunal.failure_memory.query import FailureMemoryQuery
from epistemic_tribunal.failure_memory.store import FailureMemoryStore
from epistemic_tribunal.tribunal.scoring import compute_trace_score, normalise_weights
from epistemic_tribunal.tribunal_types import (
    CandidateTrace,
    CritiqueResult,
    DecisionKind,

    Task,
    TaskDomain,
    TribunalDecision,
    UncertaintyReport,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def gsm8k_task() -> Task:
    return Task(
        task_id="test_fm_001",
        domain=TaskDomain.GSM8K_MATH,
        description="A math problem",
        train=[],
        test_input="If a baker has 12 loaves and sells 5, how many remain?",
        ground_truth="7",
    )


@pytest.fixture()
def wrong_pick_pool(gsm8k_task: Task):
    """Pool where majority (llm + llm_warm) agrees on wrong answer,
    minority (llm_cot) has correct answer."""
    traces = [
        CandidateTrace(
            trace_id="t1", generator_name="llm", answer="10",
            reasoning_steps=[], raw_trace="10",
        ),
        CandidateTrace(
            trace_id="t2", generator_name="llm_warm", answer="10",
            reasoning_steps=[], raw_trace="10",
        ),
        CandidateTrace(
            trace_id="t3", generator_name="llm_cot", answer="7",
            reasoning_steps=["12 - 5 = 7", "Answer: 7"],
            raw_trace="Step 1: 12 - 5 = 7. Answer: 7",
        ),
    ]
    critiques = [
        CritiqueResult(
            trace_id="t1", consistency_score=0.5, rule_coherence_score=1.0,
            morphology_score=1.0, failure_similarity_penalty=0.0,
            invariant_compliance_score=1.0, aggregate_score=0.444,
        ),
        CritiqueResult(
            trace_id="t2", consistency_score=0.5, rule_coherence_score=1.0,
            morphology_score=1.0, failure_similarity_penalty=0.0,
            invariant_compliance_score=1.0, aggregate_score=0.444,
        ),
        CritiqueResult(
            trace_id="t3", consistency_score=0.7, rule_coherence_score=1.0,
            morphology_score=1.0, failure_similarity_penalty=0.0,
            invariant_compliance_score=1.0, aggregate_score=0.444,
        ),
    ]
    uncertainty = UncertaintyReport(
        entropy=0.6,
        margin=0.333,
        coalition_mass=0.667,
        disagreement_rate=0.667,
        per_trace_quality={"t1": 0.667, "t2": 0.667, "t3": 0.333},
    )
    decision = TribunalDecision(
        task_id=gsm8k_task.task_id,
        decision=DecisionKind.SELECT,
        selected_trace_id="t1",
        selected_answer="10",
        scores={"t1": 0.6, "t2": 0.6, "t3": 0.36},
        confidence=0.42,
        metadata={"structural_margin": 0.0},
    )
    return traces, critiques, uncertainty, decision


@pytest.fixture()
def bad_abstention_pool(gsm8k_task: Task):
    """Pool where all 3 generators disagree, one has correct answer."""
    traces = [
        CandidateTrace(
            trace_id="t4", generator_name="llm", answer="10",
            reasoning_steps=[], raw_trace="10",
        ),
        CandidateTrace(
            trace_id="t5", generator_name="llm_warm", answer="8",
            reasoning_steps=[], raw_trace="8",
        ),
        CandidateTrace(
            trace_id="t6", generator_name="llm_cot", answer="7",
            reasoning_steps=["12 - 5 = 7"],
            raw_trace="Step 1: 12 - 5 = 7",
        ),
    ]
    critiques = [
        CritiqueResult(
            trace_id="t4", consistency_score=0.5, rule_coherence_score=1.0,
            morphology_score=1.0, failure_similarity_penalty=0.0,
            invariant_compliance_score=1.0, aggregate_score=0.444,
        ),
        CritiqueResult(
            trace_id="t5", consistency_score=0.5, rule_coherence_score=1.0,
            morphology_score=1.0, failure_similarity_penalty=0.0,
            invariant_compliance_score=1.0, aggregate_score=0.444,
        ),
        CritiqueResult(
            trace_id="t6", consistency_score=0.7, rule_coherence_score=1.0,
            morphology_score=1.0, failure_similarity_penalty=0.0,
            invariant_compliance_score=1.0, aggregate_score=0.444,
        ),
    ]
    uncertainty = UncertaintyReport(
        entropy=1.0,
        margin=0.0,
        coalition_mass=0.333,
        disagreement_rate=1.0,
        per_trace_quality={"t4": 0.333, "t5": 0.333, "t6": 0.333},
    )
    decision = TribunalDecision(
        task_id=gsm8k_task.task_id,
        decision=DecisionKind.ABSTAIN,
        confidence=0.33,
        metadata={"structural_margin": 0.0},
    )
    return traces, critiques, uncertainty, decision


@pytest.fixture()
def memory_store():
    return FailureMemoryStore(":memory:")


# ---------------------------------------------------------------------------
# 1. Signature extraction
# ---------------------------------------------------------------------------


class TestSignatureExtraction:

    def test_extract_wrong_pick(self, gsm8k_task, wrong_pick_pool):
        traces, critiques, uncertainty, decision = wrong_pick_pool
        extractor = FailureSignatureExtractor()
        sig = extractor.extract(
            gsm8k_task, traces, critiques, decision, uncertainty,
            ground_truth_match=False, any_correct=True,
        )
        assert sig is not None
        assert sig.failure_type == FailureType.WRONG_PICK
        assert sig.coalition_context["false_majority"] is True
        assert sig.coalition_context["minority_correct"] is True
        assert sig.domain == "gsm8k_math"

    def test_extract_bad_abstention(self, gsm8k_task, bad_abstention_pool):
        traces, critiques, uncertainty, decision = bad_abstention_pool
        extractor = FailureSignatureExtractor()
        sig = extractor.extract(
            gsm8k_task, traces, critiques, decision, uncertainty,
            ground_truth_match=None, any_correct=True,
        )
        assert sig is not None
        assert sig.failure_type == FailureType.BAD_ABSTENTION
        assert sig.disagreement_rate == 1.0

    def test_extract_correct_select(self, gsm8k_task, wrong_pick_pool):
        traces, critiques, uncertainty, decision = wrong_pick_pool
        # Override decision to be correct
        decision.selected_trace_id = "t3"
        decision.selected_answer = "7"
        extractor = FailureSignatureExtractor()
        sig = extractor.extract(
            gsm8k_task, traces, critiques, decision, uncertainty,
            ground_truth_match=True, any_correct=True,
        )
        assert sig is not None
        assert sig.failure_type == FailureType.CORRECT_SELECT

    def test_extract_returns_none_without_ground_truth(self, gsm8k_task, wrong_pick_pool):
        traces, critiques, uncertainty, decision = wrong_pick_pool
        extractor = FailureSignatureExtractor()
        sig = extractor.extract(
            gsm8k_task, traces, critiques, decision, uncertainty,
            ground_truth_match=None, any_correct=None,
        )
        assert sig is None


# ---------------------------------------------------------------------------
# 2. Store persistence
# ---------------------------------------------------------------------------


class TestFailureMemoryStore:

    def test_store_and_retrieve(self, memory_store):
        sig = FailureSignature(
            domain="gsm8k_math",
            task_id="t1",
            failure_type=FailureType.WRONG_PICK,
            coalition_context={"false_majority": True, "minority_correct": True},
        )
        memory_store.store(sig)
        all_sigs = memory_store.get_all()
        assert len(all_sigs) == 1
        assert all_sigs[0].signature_id == sig.signature_id
        assert all_sigs[0].failure_type == FailureType.WRONG_PICK

    def test_store_multiple_and_stats(self, memory_store):
        for i, ft in enumerate([FailureType.WRONG_PICK, FailureType.BAD_ABSTENTION, FailureType.CORRECT_SELECT]):
            sig = FailureSignature(
                domain="gsm8k_math", task_id=f"t{i}", failure_type=ft,
            )
            memory_store.store(sig)
        stats = memory_store.get_stats()
        assert stats["total_signatures"] == 3
        assert stats["by_type"]["wrong_pick"] == 1
        assert stats["by_type"]["bad_abstention"] == 1

    def test_duplicate_handling(self, memory_store):
        sig = FailureSignature(
            domain="gsm8k_math", task_id="t1", failure_type=FailureType.WRONG_PICK,
        )
        memory_store.store(sig)
        memory_store.store(sig)  # Same ID = replace
        assert len(memory_store.get_all()) == 1

    def test_domain_filtering(self, memory_store):
        memory_store.store(FailureSignature(
            domain="gsm8k_math", task_id="t1", failure_type=FailureType.WRONG_PICK,
        ))
        memory_store.store(FailureSignature(
            domain="arc_like", task_id="t2", failure_type=FailureType.WRONG_PICK,
        ))
        gsm = memory_store.get_all("gsm8k_math")
        arc = memory_store.get_all("arc_like")
        assert len(gsm) == 1
        assert len(arc) == 1


# ---------------------------------------------------------------------------
# 3. Similarity query
# ---------------------------------------------------------------------------


class TestSimilarityQuery:

    def test_query_returns_similar(self, memory_store):
        """A past wrong_pick with false_majority should match a probe with
        similar coalition shape."""
        sig = FailureSignature(
            domain="gsm8k_math",
            task_id="past_1",
            failure_type=FailureType.WRONG_PICK,
            coalition_context={
                "false_majority": True,
                "minority_correct": True,
                "coalition_mass": 0.667,
                "n_clusters": 2,
            },
            trace_quality_features={"rationale_present": False},
            critic_context={"all_flat": True},
            disagreement_rate=0.667,
            structural_margin=0.0,
        )
        memory_store.store(sig)

        probe = FailureProbe(
            domain="gsm8k_math",
            n_candidates=3,
            n_clusters=2,
            coalition_mass=0.667,
            disagreement_rate=0.667,
            majority_has_rationale=False,
            minority_has_rationale=True,
            all_critics_flat=True,
            structural_margin=0.0,
        )
        matches = memory_store.query_similar(probe)
        assert len(matches) >= 1
        assert matches[0].similarity > 0.0

    def test_query_does_not_match_correct_selects(self, memory_store):
        """Correct selections should NOT be returned as failure matches."""
        sig = FailureSignature(
            domain="gsm8k_math",
            task_id="past_ok",
            failure_type=FailureType.CORRECT_SELECT,
            disagreement_rate=0.667,
        )
        memory_store.store(sig)

        probe = FailureProbe(
            domain="gsm8k_math",
            disagreement_rate=0.667,
        )
        matches = memory_store.query_similar(probe)
        assert len(matches) == 0

    def test_query_does_not_cross_domains(self, memory_store):
        """ARC failures should not match a GSM8K probe."""
        sig = FailureSignature(
            domain="arc_like",
            task_id="arc_1",
            failure_type=FailureType.WRONG_PICK,
            disagreement_rate=0.667,
        )
        memory_store.store(sig)

        probe = FailureProbe(domain="gsm8k_math", disagreement_rate=0.667)
        matches = memory_store.query_similar(probe)
        assert len(matches) == 0


# ---------------------------------------------------------------------------
# 4. Per-trace penalty computation
# ---------------------------------------------------------------------------


class TestPenaltyComputation:

    def test_no_penalties_from_empty_memory(self, gsm8k_task, wrong_pick_pool):
        traces, critiques, uncertainty, decision = wrong_pick_pool
        store = FailureMemoryStore(":memory:")
        query = FailureMemoryQuery(store, penalty_scale=0.3)
        penalties = query.query_penalties(gsm8k_task, traces, critiques, uncertainty)
        assert all(p == 0.0 for p in penalties.values())

    def test_majority_penalised_after_false_majority_stored(self, gsm8k_task, wrong_pick_pool):
        traces, critiques, uncertainty, _ = wrong_pick_pool
        store = FailureMemoryStore(":memory:")

        # Store a past false-majority failure
        sig = FailureSignature(
            domain="gsm8k_math",
            task_id="past_fm",
            failure_type=FailureType.WRONG_PICK,
            coalition_context={
                "false_majority": True,
                "minority_correct": True,
                "coalition_mass": 0.667,
                "n_clusters": 2,
            },
            trace_quality_features={"rationale_present": False},
            critic_context={"all_flat": True},
            disagreement_rate=0.667,
            structural_margin=0.0,
        )
        store.store(sig)

        query = FailureMemoryQuery(store, penalty_scale=0.3)
        penalties = query.query_penalties(gsm8k_task, traces, critiques, uncertainty)

        # Majority candidates (t1, t2) should get penalised more than minority (t3)
        assert penalties["t1"] > 0.0 or penalties["t2"] > 0.0
        # Minority candidate with rationale should not be penalised
        assert penalties["t3"] == 0.0

    def test_penalties_capped_at_08(self, gsm8k_task, wrong_pick_pool):
        traces, critiques, uncertainty, _ = wrong_pick_pool
        store = FailureMemoryStore(":memory:")

        # Store many similar failures to try to exceed cap
        for i in range(20):
            sig = FailureSignature(
                domain="gsm8k_math",
                task_id=f"past_{i}",
                failure_type=FailureType.WRONG_PICK,
                coalition_context={
                    "false_majority": True, "minority_correct": True,
                    "coalition_mass": 0.667, "n_clusters": 2,
                },
                trace_quality_features={"rationale_present": False},
                critic_context={"all_flat": True},
                disagreement_rate=0.667,
                structural_margin=0.0,
            )
            store.store(sig)

        query = FailureMemoryQuery(store, penalty_scale=0.3)
        penalties = query.query_penalties(gsm8k_task, traces, critiques, uncertainty)
        assert all(p <= 0.8 for p in penalties.values())


# ---------------------------------------------------------------------------
# 5. Integration: M channel
# ---------------------------------------------------------------------------


class TestMChannelIntegration:

    def test_penalty_flows_into_scoring(self):
        """Verify that a non-zero failure_similarity_penalty reduces the M score."""
        trace = CandidateTrace(
            trace_id="t1", generator_name="llm", answer="10",
        )
        # Critique with penalty
        critique_penalised = CritiqueResult(
            trace_id="t1", consistency_score=0.5, rule_coherence_score=1.0,
            morphology_score=1.0, failure_similarity_penalty=0.3,
            invariant_compliance_score=1.0, aggregate_score=0.444,
        )
        # Critique without penalty
        critique_clean = CritiqueResult(
            trace_id="t1", consistency_score=0.5, rule_coherence_score=1.0,
            morphology_score=1.0, failure_similarity_penalty=0.0,
            invariant_compliance_score=1.0, aggregate_score=0.444,
        )
        uncertainty = UncertaintyReport(
            entropy=0.5, margin=0.333, coalition_mass=0.667,
            disagreement_rate=0.667,
            per_trace_quality={"t1": 0.667},
        )
        alpha, beta, gamma, delta = normalise_weights(0.65, 0.25, 0.10, 0.0)

        score_penalised = compute_trace_score(
            trace, critique_penalised, uncertainty,
            alpha=alpha, beta=beta, gamma=gamma, delta=delta,
        )
        score_clean = compute_trace_score(
            trace, critique_clean, uncertainty,
            alpha=alpha, beta=beta, gamma=gamma, delta=delta,
        )

        # Penalised should have lower total
        assert score_penalised.total < score_clean.total
        # M component should be lower
        assert score_penalised.M < score_clean.M


# ---------------------------------------------------------------------------
# 6. ARC regression
# ---------------------------------------------------------------------------


class TestArcRegression:

    def test_arc_task_unaffected_by_failure_memory(self):
        """An ARC task with failure memory disabled should behave identically
        to the existing path."""
        task = Task(
            task_id="arc_test",
            domain=TaskDomain.ARC_LIKE,
            description="Identity transform",
            train=[],
            test_input=[[1, 2], [3, 4]],
            ground_truth=[[1, 2], [3, 4]],
        )
        # Just verify extraction works without error for ARC domain
        traces = [
            CandidateTrace(
                trace_id="arc_t1", generator_name="greedy",
                answer=[[1, 2], [3, 4]], reasoning_steps=[],
            ),
        ]
        critiques = [
            CritiqueResult(
                trace_id="arc_t1", consistency_score=0.5,
                rule_coherence_score=1.0, morphology_score=1.0,
                failure_similarity_penalty=0.0,
                invariant_compliance_score=1.0, aggregate_score=0.8,
            ),
        ]
        uncertainty = UncertaintyReport(
            entropy=0.0, margin=1.0, coalition_mass=1.0,
            disagreement_rate=0.0,
            per_trace_quality={"arc_t1": 1.0},
        )
        decision = TribunalDecision(
            task_id=task.task_id,
            decision=DecisionKind.SELECT,
            selected_trace_id="arc_t1",
            selected_answer=[[1, 2], [3, 4]],
            confidence=0.9,
            metadata={"structural_margin": 1.0},
        )

        extractor = FailureSignatureExtractor()
        sig = extractor.extract(
            task, traces, critiques, decision, uncertainty,
            ground_truth_match=True, any_correct=True,
        )
        assert sig is not None
        assert sig.failure_type == FailureType.CORRECT_SELECT
        assert sig.domain == "arc_like"
        assert sig.domain_features.get("grid_domain") is True

    def test_arc_probe_builds_without_error(self):
        """Verify FailureMemoryQuery can build a probe for ARC tasks."""
        task = Task(
            task_id="arc_test_probe",
            domain=TaskDomain.ARC_LIKE,
            train=[], test_input=[[0]], ground_truth=[[1]],
        )
        traces = [
            CandidateTrace(
                trace_id="arc_p1", generator_name="greedy",
                answer=[[1]], reasoning_steps=[],
            ),
        ]
        critiques = [
            CritiqueResult(
                trace_id="arc_p1", consistency_score=0.5,
                rule_coherence_score=1.0, morphology_score=1.0,
                failure_similarity_penalty=0.0,
                invariant_compliance_score=1.0, aggregate_score=0.8,
            ),
        ]
        uncertainty = UncertaintyReport(
            entropy=0.0, margin=1.0, coalition_mass=1.0,
            disagreement_rate=0.0,
            per_trace_quality={"arc_p1": 1.0},
        )
        store = FailureMemoryStore(":memory:")
        query = FailureMemoryQuery(store)
        probe = query.build_probe(task, traces, critiques, uncertainty)
        assert probe.domain == "arc_like"
        assert probe.n_candidates == 1


# ---------------------------------------------------------------------------
# 7. FailureProbe does not contain ground truth
# ---------------------------------------------------------------------------


class TestProbeGroundTruthGuardrail:

    def test_probe_has_no_ground_truth_fields(self, gsm8k_task, wrong_pick_pool):
        """Verify the FailureProbe model does not contain any ground-truth fields."""
        traces, critiques, uncertainty, _ = wrong_pick_pool
        store = FailureMemoryStore(":memory:")
        query = FailureMemoryQuery(store)
        query.build_probe(gsm8k_task, traces, critiques, uncertainty)

        # The probe should NOT have these retrospective fields
        probe_fields = set(FailureProbe.model_fields.keys())
        assert "failure_type" not in probe_fields
        assert "false_majority" not in probe_fields
        assert "minority_correct" not in probe_fields
        assert "ground_truth_match" not in probe_fields
        assert "any_correct" not in probe_fields
