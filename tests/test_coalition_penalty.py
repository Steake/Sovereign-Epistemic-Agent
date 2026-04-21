"""Tests for coalition-level failure-memory penalty propagation.

Covers:
1. Coalition propagation: sibling traces with same answer all get penalised
2. Trace vs coalition distinction: coalition penalty changes answer ranking
3. Metadata: coalition-level fields present and correct
4. ARC regression: no destructive changes
"""

from __future__ import annotations

import pytest

from epistemic_tribunal.failure_memory.models import FailureSignature, FailureType
from epistemic_tribunal.failure_memory.query import FailureMemoryQuery
from epistemic_tribunal.failure_memory.store import FailureMemoryStore
from epistemic_tribunal.tribunal_types import (
    CandidateTrace,
    CritiqueResult,
    Task,
    TaskDomain,
    UncertaintyReport,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def gsm8k_task() -> Task:
    return Task(
        task_id="coal_test_001",
        domain=TaskDomain.GSM8K_MATH,
        description="Math problem",
        train=[],
        test_input="5 + 3 = ?",
        ground_truth="8",
    )


@pytest.fixture()
def coalition_pool():
    """Pool where two generators (llm, llm_warm) propose the SAME wrong answer
    '42', and one minority generator (llm_cot) proposes the correct answer '8'.

    This is the canonical false-majority shape that coalition-level memory
    must learn to suppress: penalising only one of the '42' traces must NOT
    cause the other '42' trace to be selected instead.
    """
    traces = [
        CandidateTrace(
            trace_id="t_llm", generator_name="llm", answer="42",
            reasoning_steps=[], raw_trace="42",
        ),
        CandidateTrace(
            trace_id="t_llm_warm", generator_name="llm_warm", answer="42",
            reasoning_steps=[], raw_trace="42",
        ),
        CandidateTrace(
            trace_id="t_llm_cot", generator_name="llm_cot", answer="8",
            reasoning_steps=["5 + 3 = 8"],
            raw_trace="Step 1: 5 + 3 = 8. Answer: 8",
        ),
    ]
    critiques = [
        CritiqueResult(
            trace_id="t_llm", consistency_score=0.5, rule_coherence_score=1.0,
            morphology_score=1.0, failure_similarity_penalty=0.0,
            invariant_compliance_score=1.0, aggregate_score=0.5,
        ),
        CritiqueResult(
            trace_id="t_llm_warm", consistency_score=0.5, rule_coherence_score=1.0,
            morphology_score=1.0, failure_similarity_penalty=0.0,
            invariant_compliance_score=1.0, aggregate_score=0.5,
        ),
        CritiqueResult(
            trace_id="t_llm_cot", consistency_score=0.7, rule_coherence_score=1.0,
            morphology_score=1.0, failure_similarity_penalty=0.0,
            invariant_compliance_score=1.0, aggregate_score=0.7,
        ),
    ]
    uncertainty = UncertaintyReport(
        entropy=0.6,
        margin=0.333,
        coalition_mass=0.667,
        disagreement_rate=0.667,
        per_trace_quality={"t_llm": 0.667, "t_llm_warm": 0.667, "t_llm_cot": 0.333},
    )
    return traces, critiques, uncertainty


def _store_with_wrong_pick_42(answer_sig: str = "(42,)") -> FailureMemoryStore:
    """Return a store containing a wrong_pick signature for the given answer_sig.

    The answer_sig must already be in the normalized cluster-key format:
    str(adapter.get_cluster_key(answer)).  For GSM8K integer answers that is
    '(42,)', not '42'.
    """
    store = FailureMemoryStore(":memory:")
    sig = FailureSignature(
        domain="gsm8k_math",
        task_id="past_task_001",
        failure_type=FailureType.WRONG_PICK,
        answer_signature=answer_sig,
        coalition_context={"false_majority": True, "minority_correct": True,
                           "coalition_mass": 0.667, "n_clusters": 2},
        trace_quality_features={"rationale_present": False},
        critic_context={"all_flat": False},
        disagreement_rate=0.667,
        structural_margin=0.0,
    )
    store.store(sig)
    return store


# ---------------------------------------------------------------------------
# 1. Coalition propagation
# ---------------------------------------------------------------------------


class TestCoalitionPropagation:

    def test_all_same_answer_traces_penalised(self, gsm8k_task, coalition_pool):
        """If memory flags answer '42' as a past wrong_pick, BOTH t_llm and
        t_llm_warm must receive a non-zero penalty — not just one of them."""
        traces, critiques, uncertainty = coalition_pool
        store = _store_with_wrong_pick_42("(42,)")
        query = FailureMemoryQuery(store, penalty_scale=0.3)

        penalties = query.query_penalties(gsm8k_task, traces, critiques, uncertainty)

        assert penalties["t_llm"] > 0.0, "t_llm (answer='42') must be penalised"
        assert penalties["t_llm_warm"] > 0.0, "t_llm_warm (answer='42') must be penalised"

    def test_correct_answer_trace_not_penalised(self, gsm8k_task, coalition_pool):
        """t_llm_cot proposes a different answer ('8'), so it must NOT be penalised
        by the coalition sweep for answer '42'."""
        traces, critiques, uncertainty = coalition_pool
        store = _store_with_wrong_pick_42("(42,)")
        query = FailureMemoryQuery(store, penalty_scale=0.3)

        penalties = query.query_penalties(gsm8k_task, traces, critiques, uncertainty)

        assert penalties["t_llm_cot"] == 0.0, "Minority correct trace must NOT receive coalition penalty"

    def test_sibling_penalty_equal(self, gsm8k_task, coalition_pool):
        """Both siblings proposing the same wrong answer must receive IDENTICAL
        coalition penalties (the sweep applies uniformly)."""
        traces, critiques, uncertainty = coalition_pool
        store = _store_with_wrong_pick_42("(42,)")
        query = FailureMemoryQuery(store, penalty_scale=0.3)

        penalties = query.query_penalties(gsm8k_task, traces, critiques, uncertainty)

        assert penalties["t_llm"] == penalties["t_llm_warm"], (
            "All traces in the bad coalition must receive the same base penalty"
        )

    def test_no_match_means_no_coalition_penalty_for_unrelated_answer(self, gsm8k_task, coalition_pool):
        """Store a wrong_pick signature with answer_signature='99'.  No pool
        trace proposes '99', so the *coalition exact-answer sweep* must NOT fire.
        The metadata field coalitions_penalised must be 0."""
        traces, critiques, uncertainty = coalition_pool
        # Store wrong pick for answer "(99,)" — not present in the pool
        store = _store_with_wrong_pick_42("(99,)")
        query = FailureMemoryQuery(store, penalty_scale=0.3)

        _, metadata = query.query_with_metadata(gsm8k_task, traces, critiques, uncertainty)

        assert metadata["failure_memory_coalitions_penalised"] == 0, (
            "No pool trace proposes '99', so coalitions_penalised must be 0"
        )
        assert metadata["failure_memory_traces_affected_by_coalition"] == 0


# ---------------------------------------------------------------------------
# 2. Trace vs coalition distinction
# ---------------------------------------------------------------------------


class TestTraceVsCoalitionDistinction:

    def test_coalition_penalty_changes_answer_ranking(self, gsm8k_task, coalition_pool):
        """Before penalty: majority coalition ('42') has higher raw aggregate.
        After storing a wrong_pick for '42', the coalition penalty must flip the
        ranking so that the minority trace ('8') ends up with the lower penalty.

        This validates that coalition-level memory can shift *answer* selection,
        not just *trace* selection within the same answer.
        """
        traces, critiques, uncertainty = coalition_pool
        store = _store_with_wrong_pick_42("(42,)")
        query = FailureMemoryQuery(store, penalty_scale=0.3)

        penalties = query.query_penalties(gsm8k_task, traces, critiques, uncertainty)

        # Correct trace must have strictly lower (better) net penalty
        max_coalition_penalty = max(penalties["t_llm"], penalties["t_llm_warm"])
        assert penalties["t_llm_cot"] < max_coalition_penalty, (
            "Coalition penalty must make the correct minority trace relatively better"
        )

    def test_empty_store_no_change(self, gsm8k_task, coalition_pool):
        """With empty memory, all penalties are zero — rankings unchanged."""
        traces, critiques, uncertainty = coalition_pool
        store = FailureMemoryStore(":memory:")
        query = FailureMemoryQuery(store, penalty_scale=0.3)

        penalties = query.query_penalties(gsm8k_task, traces, critiques, uncertainty)
        assert all(p == 0.0 for p in penalties.values())


# ---------------------------------------------------------------------------
# 3. Coalition-level metadata
# ---------------------------------------------------------------------------


class TestCoalitionMetadata:

    def test_coalitions_penalised_field_present(self, gsm8k_task, coalition_pool):
        traces, critiques, uncertainty = coalition_pool
        store = _store_with_wrong_pick_42("(42,)")
        query = FailureMemoryQuery(store, penalty_scale=0.3)

        _, metadata = query.query_with_metadata(gsm8k_task, traces, critiques, uncertainty)

        assert "failure_memory_coalitions_penalised" in metadata

    def test_coalitions_penalised_count_correct(self, gsm8k_task, coalition_pool):
        """One bad answer signature ('42') matched → coalitions_penalised == 1."""
        traces, critiques, uncertainty = coalition_pool
        store = _store_with_wrong_pick_42("(42,)")
        query = FailureMemoryQuery(store, penalty_scale=0.3)

        _, metadata = query.query_with_metadata(gsm8k_task, traces, critiques, uncertainty)

        assert metadata["failure_memory_coalitions_penalised"] == 1

    def test_traces_affected_by_coalition_correct(self, gsm8k_task, coalition_pool):
        """Two traces propose '42' → traces_affected_by_coalition == 2."""
        traces, critiques, uncertainty = coalition_pool
        store = _store_with_wrong_pick_42("(42,)")
        query = FailureMemoryQuery(store, penalty_scale=0.3)

        _, metadata = query.query_with_metadata(gsm8k_task, traces, critiques, uncertainty)

        assert metadata["failure_memory_traces_affected_by_coalition"] == 2

    def test_bad_answer_signatures_listed(self, gsm8k_task, coalition_pool):
        """The bad answer signature must appear in the metadata."""
        traces, critiques, uncertainty = coalition_pool
        store = _store_with_wrong_pick_42("(42,)")
        query = FailureMemoryQuery(store, penalty_scale=0.3)

        _, metadata = query.query_with_metadata(gsm8k_task, traces, critiques, uncertainty)

        assert "(42,)" in metadata["failure_memory_bad_answer_signatures"]

    def test_zero_coalitions_when_no_match(self, gsm8k_task, coalition_pool):
        """When the stored signature's answer_signature does not appear in the
        current pool, coalitions_penalised must be 0."""
        traces, critiques, uncertainty = coalition_pool
        store = _store_with_wrong_pick_42("(99,)")  # not present in pool
        query = FailureMemoryQuery(store, penalty_scale=0.3)

        _, metadata = query.query_with_metadata(gsm8k_task, traces, critiques, uncertainty)

        assert metadata["failure_memory_coalitions_penalised"] == 0
        assert metadata["failure_memory_traces_affected_by_coalition"] == 0

    def test_no_coalitions_from_empty_store(self, gsm8k_task, coalition_pool):
        """Empty store → all coalition fields must be zero."""
        traces, critiques, uncertainty = coalition_pool
        store = FailureMemoryStore(":memory:")
        query = FailureMemoryQuery(store, penalty_scale=0.3)

        _, metadata = query.query_with_metadata(gsm8k_task, traces, critiques, uncertainty)

        assert metadata["failure_memory_coalitions_penalised"] == 0
        assert metadata["failure_memory_traces_affected_by_coalition"] == 0
        assert metadata["failure_memory_bad_answer_signatures"] == []


# ---------------------------------------------------------------------------
# 4. ARC regression
# ---------------------------------------------------------------------------


class TestCoalitionArcRegression:

    def test_arc_task_unaffected_by_coalition_penalty(self):
        """ARC tasks must not be affected by GSM8K coalition signatures."""
        arc_task = Task(
            task_id="arc_coalition_reg",
            domain=TaskDomain.ARC_LIKE,
            description="Grid transform",
            train=[],
            test_input=[[1, 2]],
            ground_truth=[[1, 2]],
        )
        traces = [
            CandidateTrace(
                trace_id="arc_t1", generator_name="greedy",
                answer=[[1, 2]], reasoning_steps=[],
            ),
        ]
        critiques = [
            CritiqueResult(
                trace_id="arc_t1", consistency_score=0.9,
                rule_coherence_score=1.0, morphology_score=1.0,
                failure_similarity_penalty=0.0,
                invariant_compliance_score=1.0, aggregate_score=0.9,
            ),
        ]
        uncertainty = UncertaintyReport(
            entropy=0.0, margin=1.0, coalition_mass=1.0,
            disagreement_rate=0.0,
            per_trace_quality={"arc_t1": 1.0},
        )

        # Store a GSM8K wrong_pick signature
        store = FailureMemoryStore(":memory:")
        gsm_sig = FailureSignature(
            domain="gsm8k_math",
            task_id="gsm_past",
            failure_type=FailureType.WRONG_PICK,
            answer_signature="(42,)",
            disagreement_rate=0.667,
        )
        store.store(gsm_sig)

        query = FailureMemoryQuery(store, penalty_scale=0.3)
        penalties = query.query_penalties(arc_task, traces, critiques, uncertainty)

        # ARC task must get zero penalties — GSM8K signatures don't cross domains
        assert all(p == 0.0 for p in penalties.values()), (
            "GSM8K coalition signature must not penalise ARC task traces"
        )

    def test_arc_metadata_shows_no_coalition_penalty(self):
        """Coalition metadata fields must all be zero for an ARC task when
        only GSM8K signatures exist in the store."""
        arc_task = Task(
            task_id="arc_meta_reg",
            domain=TaskDomain.ARC_LIKE,
            train=[], test_input=[[0]], ground_truth=[[1]],
        )
        traces = [
            CandidateTrace(
                trace_id="a1", generator_name="greedy",
                answer=[[1]], reasoning_steps=[],
            ),
        ]
        critiques = [
            CritiqueResult(
                trace_id="a1", consistency_score=0.8, rule_coherence_score=1.0,
                morphology_score=1.0, failure_similarity_penalty=0.0,
                invariant_compliance_score=1.0, aggregate_score=0.8,
            ),
        ]
        uncertainty = UncertaintyReport(
            entropy=0.0, margin=1.0, coalition_mass=1.0,
            disagreement_rate=0.0, per_trace_quality={"a1": 1.0},
        )

        store = _store_with_wrong_pick_42("(42,)")
        query = FailureMemoryQuery(store, penalty_scale=0.3)

        _, metadata = query.query_with_metadata(arc_task, traces, critiques, uncertainty)

        assert metadata["failure_memory_coalitions_penalised"] == 0
        assert metadata["failure_memory_traces_affected_by_coalition"] == 0
