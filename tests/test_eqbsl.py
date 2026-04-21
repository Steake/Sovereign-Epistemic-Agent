"""Focused tests for the additive EQBSL coalition-opinion layer."""

from __future__ import annotations

import json

from epistemic_tribunal.config import EQBSLConfig, EqbslSourceTrustConfig, EqbslVerificationConfig, TribunalConfig
from epistemic_tribunal.eqbsl.decision import EqbslDecisionPolicy
from epistemic_tribunal.eqbsl.models import CoalitionOpinion, EvidenceOpinion, VerificationEvidence
from epistemic_tribunal.eqbsl.sources import CoalitionBundle, EqbslSourceBuilder
from epistemic_tribunal.eqbsl.verification import VerificationSourceProvider
from epistemic_tribunal.ledger.writer import LedgerWriter
from epistemic_tribunal.tribunal.aggregator import TribunalAggregator
from epistemic_tribunal.tribunal.scoring import TraceScore
from epistemic_tribunal.tribunal_types import CandidateTrace, CritiqueResult, DecisionKind, UncertaintyReport


def _trace_score(
    trace_id: str,
    generator: str,
    *,
    U: float,
    C: float,
    M: float,
    V: float,
    total: float,
) -> TraceScore:
    return TraceScore(
        trace_id=trace_id,
        generator_name=generator,
        U=U,
        C=C,
        M=M,
        V=V,
        total=total,
    )


def _critique(trace_id: str, *, aggregate: float, penalty: float = 0.0) -> CritiqueResult:
    return CritiqueResult(
        trace_id=trace_id,
        consistency_score=aggregate,
        rule_coherence_score=aggregate,
        morphology_score=aggregate,
        failure_similarity_penalty=penalty,
        invariant_compliance_score=1.0,
        aggregate_score=aggregate,
        violated_invariants=[],
    )


def test_memory_structural_hits_raise_uncertainty_more_than_disbelief(gsm8k_task) -> None:
    builder = EqbslSourceBuilder(EQBSLConfig())
    trace = CandidateTrace(
        trace_id="t1",
        generator_name="llm_cot",
        answer="9",
        reasoning_steps=["Compute eggs sold", "Subtract breakfast eggs"],
        confidence_score=0.8,
    )
    coalition = CoalitionBundle(
        answer_signature="(9,)",
        traces=[trace],
        critiques=[_critique("t1", aggregate=0.8, penalty=0.1)],
        trace_scores=[_trace_score("t1", "llm_cot", U=0.5, C=0.8, M=0.9, V=1.0, total=0.76)],
        representative_trace_id="t1",
        representative_generator="llm_cot",
    )
    opinions = builder.build(
        task=gsm8k_task,
        coalitions=[coalition],
        all_traces=[trace],
        all_critiques=[_critique("t1", aggregate=0.8, penalty=0.1)],
        uncertainty=UncertaintyReport(
            entropy=0.4,
            margin=0.2,
            coalition_mass=0.5,
            disagreement_rate=1.0,
            per_trace_quality={"t1": 0.5},
        ),
        failure_memory_metadata={
            "failure_memory_penalty_scale": 0.3,
            "failure_memory_trace_decomposition": {
                "t1": {
                    "exact_penalty": 0.0,
                    "structural_penalty": 0.3,
                    "n_exact_matches": 0,
                    "n_structural_matches": 2,
                    "top_structural_similarity": 0.8,
                    "final_total_penalty": 0.3,
                }
            },
        },
    )
    memory_opinion = opinions[0].source_opinions["M"].opinion
    assert memory_opinion.uncertainty > memory_opinion.disbelief
    assert memory_opinion.metadata["structural_hits"] == 2


def test_memory_exact_hits_raise_disbelief_strongly(gsm8k_task) -> None:
    builder = EqbslSourceBuilder(EQBSLConfig())
    trace = CandidateTrace(
        trace_id="t1",
        generator_name="llm_cot",
        answer="9",
        reasoning_steps=["Reason"],
        confidence_score=0.8,
    )
    coalition = CoalitionBundle(
        answer_signature="(9,)",
        traces=[trace],
        critiques=[_critique("t1", aggregate=0.8, penalty=0.3)],
        trace_scores=[_trace_score("t1", "llm_cot", U=0.5, C=0.8, M=0.7, V=1.0, total=0.7)],
        representative_trace_id="t1",
        representative_generator="llm_cot",
    )
    opinions = builder.build(
        task=gsm8k_task,
        coalitions=[coalition],
        all_traces=[trace],
        all_critiques=[_critique("t1", aggregate=0.8, penalty=0.3)],
        uncertainty=UncertaintyReport(
            entropy=0.4,
            margin=0.2,
            coalition_mass=0.5,
            disagreement_rate=1.0,
            per_trace_quality={"t1": 0.5},
        ),
        failure_memory_metadata={
            "failure_memory_penalty_scale": 0.3,
            "failure_memory_trace_decomposition": {
                "t1": {
                    "exact_penalty": 0.3,
                    "structural_penalty": 0.0,
                    "n_exact_matches": 2,
                    "n_structural_matches": 0,
                    "top_structural_similarity": 0.0,
                    "final_total_penalty": 0.3,
                }
            },
        },
    )
    memory_opinion = opinions[0].source_opinions["M"].opinion
    assert memory_opinion.disbelief > memory_opinion.uncertainty
    assert memory_opinion.metadata["exact_hits"] == 2


def test_eqbsl_same_answer_coalition_selects_instead_of_spurious_abstain(gsm8k_task) -> None:
    aggregator = TribunalAggregator(
        config=TribunalConfig(fusion_mode="eqbsl"),
        eqbsl_config=EQBSLConfig(enabled=True),
    )
    traces = [
        CandidateTrace(trace_id="t1", generator_name="llm_cot", answer="9", reasoning_steps=["step", "step"]),
        CandidateTrace(trace_id="t2", generator_name="llm", answer="9", reasoning_steps=["step", "step"]),
        CandidateTrace(trace_id="t3", generator_name="greedy", answer="10", reasoning_steps=["guess"]),
    ]
    critiques = [
        _critique("t1", aggregate=0.85),
        _critique("t2", aggregate=0.8),
        _critique("t3", aggregate=0.25),
    ]
    uncertainty = UncertaintyReport(
        entropy=0.5,
        margin=0.34,
        coalition_mass=0.667,
        disagreement_rate=0.667,
        per_trace_quality={"t1": 0.667, "t2": 0.667, "t3": 0.333},
    )
    decision = aggregator.adjudicate(
        gsm8k_task,
        traces,
        critiques,
        uncertainty,
        failure_memory_metadata={},
    )
    assert decision.decision == DecisionKind.SELECT
    assert decision.selected_answer == "9"
    assert decision.metadata["eqbsl"]["same_answer_tie_case"] is True


def test_eqbsl_high_uncertainty_abstains() -> None:
    policy = EqbslDecisionPolicy(TribunalConfig(fusion_mode="eqbsl"), EQBSLConfig())
    coalitions = [
        CoalitionOpinion(
            answer_signature="(9,)",
            coalition_member_trace_ids=["t1"],
            coalition_member_generators=["llm"],
            representative_trace_id="t1",
            representative_generator="llm",
            fused_opinion=EvidenceOpinion(
                belief=0.15,
                disbelief=0.05,
                uncertainty=0.80,
                base_rate=0.25,
                positive_evidence=0.2,
                negative_evidence=0.1,
                prior_weight=2.0,
            ),
        ),
        CoalitionOpinion(
            answer_signature="(10,)",
            coalition_member_trace_ids=["t2"],
            coalition_member_generators=["llm_cot"],
            representative_trace_id="t2",
            representative_generator="llm_cot",
            fused_opinion=EvidenceOpinion(
                belief=0.14,
                disbelief=0.06,
                uncertainty=0.80,
                base_rate=0.25,
                positive_evidence=0.2,
                negative_evidence=0.1,
                prior_weight=2.0,
            ),
        ),
    ]
    result = policy.decide(coalitions)
    assert result.decision == DecisionKind.ABSTAIN
    assert result.reason.reason_code == "abstain_high_uncertainty"


def test_eqbsl_strong_belief_selects() -> None:
    policy = EqbslDecisionPolicy(TribunalConfig(fusion_mode="eqbsl"), EQBSLConfig())
    coalitions = [
        CoalitionOpinion(
            answer_signature="(9,)",
            coalition_member_trace_ids=["t1"],
            coalition_member_generators=["llm_cot"],
            representative_trace_id="t1",
            representative_generator="llm_cot",
            fused_opinion=EvidenceOpinion(
                belief=0.72,
                disbelief=0.08,
                uncertainty=0.20,
                base_rate=0.25,
                positive_evidence=7.2,
                negative_evidence=0.8,
                prior_weight=2.0,
            ),
        ),
        CoalitionOpinion(
            answer_signature="(10,)",
            coalition_member_trace_ids=["t2"],
            coalition_member_generators=["llm"],
            representative_trace_id="t2",
            representative_generator="llm",
            fused_opinion=EvidenceOpinion(
                belief=0.32,
                disbelief=0.28,
                uncertainty=0.40,
                base_rate=0.25,
                positive_evidence=3.2,
                negative_evidence=2.8,
                prior_weight=2.0,
            ),
        ),
    ]
    result = policy.decide(coalitions)
    assert result.decision == DecisionKind.SELECT
    assert result.selected_answer_signature == "(9,)"


def test_eqbsl_verification_contradiction_blocks_select() -> None:
    policy = EqbslDecisionPolicy(
        TribunalConfig(fusion_mode="eqbsl"),
        EQBSLConfig(
            verification=EqbslVerificationConfig(
                enabled=True,
                enable_decision_semantics=True,
                contradiction_abstain_confidence_threshold=0.85,
            )
        ),
    )
    coalitions = [
        CoalitionOpinion(
            answer_signature="(9,)",
            coalition_member_trace_ids=["t1"],
            coalition_member_generators=["llm_cot"],
            representative_trace_id="t1",
            representative_generator="llm_cot",
            source_opinions={
                "verification": {
                    "source_name": "verification",
                    "source_type": "verification",
                    "trust_weight": 0.9,
                    "opinion": EvidenceOpinion.neutral(base_rate=0.25, prior_weight=2.0),
                    "metadata": {"classification": "contradiction", "confidence": 0.9},
                }
            },
            fused_opinion=EvidenceOpinion(
                belief=0.72,
                disbelief=0.08,
                uncertainty=0.20,
                base_rate=0.25,
                positive_evidence=7.2,
                negative_evidence=0.8,
                prior_weight=2.0,
            ),
        ),
        CoalitionOpinion(
            answer_signature="(10,)",
            coalition_member_trace_ids=["t2"],
            coalition_member_generators=["llm"],
            representative_trace_id="t2",
            representative_generator="llm",
            fused_opinion=EvidenceOpinion(
                belief=0.32,
                disbelief=0.28,
                uncertainty=0.40,
                base_rate=0.25,
                positive_evidence=3.2,
                negative_evidence=2.8,
                prior_weight=2.0,
            ),
        ),
    ]
    result = policy.decide(coalitions)
    assert result.decision == DecisionKind.ABSTAIN
    assert result.reason.reason_code == "abstain_verification_contradiction"


def test_eqbsl_verification_support_can_break_low_gap_abstain() -> None:
    policy = EqbslDecisionPolicy(
        TribunalConfig(fusion_mode="eqbsl"),
        EQBSLConfig(
            verification=EqbslVerificationConfig(
                enabled=True,
                enable_decision_semantics=True,
                support_select_confidence_threshold=0.85,
                support_advantage_margin=0.05,
            )
        ),
    )
    coalitions = [
        CoalitionOpinion(
            answer_signature="(9,)",
            coalition_member_trace_ids=["t1"],
            coalition_member_generators=["llm_cot"],
            representative_trace_id="t1",
            representative_generator="llm_cot",
            source_opinions={
                "verification": {
                    "source_name": "verification",
                    "source_type": "verification",
                    "trust_weight": 0.9,
                    "opinion": EvidenceOpinion.neutral(base_rate=0.25, prior_weight=2.0),
                    "metadata": {"classification": "support", "confidence": 0.9},
                }
            },
            fused_opinion=EvidenceOpinion(
                belief=0.635472,
                disbelief=0.114538,
                uncertainty=0.24999,
                base_rate=0.25,
                positive_evidence=6.35472,
                negative_evidence=1.14538,
                prior_weight=2.0,
            ),
        ),
        CoalitionOpinion(
            answer_signature="(10,)",
            coalition_member_trace_ids=["t2"],
            coalition_member_generators=["llm"],
            representative_trace_id="t2",
            representative_generator="llm",
            source_opinions={
                "verification": {
                    "source_name": "verification",
                    "source_type": "verification",
                    "trust_weight": 0.9,
                    "opinion": EvidenceOpinion.neutral(base_rate=0.25, prior_weight=2.0),
                    "metadata": {"classification": "contradiction", "confidence": 0.9},
                }
            },
            fused_opinion=EvidenceOpinion(
                belief=0.620419,
                disbelief=0.186224,
                uncertainty=0.193357,
                base_rate=0.25,
                positive_evidence=6.20419,
                negative_evidence=1.86224,
                prior_weight=2.0,
            ),
        ),
    ]
    result = policy.decide(coalitions)
    assert result.decision == DecisionKind.SELECT
    assert result.reason.reason_code == "select_verification_supported_low_gap"


def test_eqbsl_rationale_rich_minority_can_beat_shallow_majority(gsm8k_task) -> None:
    aggregator = TribunalAggregator(
        config=TribunalConfig(fusion_mode="eqbsl"),
        eqbsl_config=EQBSLConfig(
            enabled=True,
            source_trust=EqbslSourceTrustConfig(u=0.4, c=1.8, m=1.0, v=0.5, generator_trust=0.0),
        ),
    )
    traces = [
        CandidateTrace(trace_id="t1", generator_name="greedy", answer="10", reasoning_steps=[]),
        CandidateTrace(trace_id="t2", generator_name="llm", answer="10", reasoning_steps=["guess"]),
        CandidateTrace(
            trace_id="t3",
            generator_name="llm_cot",
            answer="9",
            reasoning_steps=["compute sold eggs", "compute breakfast eggs", "subtract all"],
        ),
    ]
    critiques = [
        _critique("t1", aggregate=0.25),
        _critique("t2", aggregate=0.30),
        _critique("t3", aggregate=0.95),
    ]
    uncertainty = UncertaintyReport(
        entropy=0.6,
        margin=0.34,
        coalition_mass=0.667,
        disagreement_rate=0.667,
        per_trace_quality={"t1": 0.667, "t2": 0.667, "t3": 0.333},
    )
    decision = aggregator.adjudicate(
        gsm8k_task,
        traces,
        critiques,
        uncertainty,
        failure_memory_metadata={},
    )
    assert decision.decision == DecisionKind.SELECT
    assert decision.selected_answer == "9"


class _StubVerificationProvider(VerificationSourceProvider):
    def __init__(self, config: EQBSLConfig, evidence: VerificationEvidence) -> None:
        super().__init__(config)
        self._evidence = evidence

    def build_source(self, **kwargs):  # type: ignore[override]
        return self._assessment_to_source(self._evidence)


def test_verification_source_adds_typed_source_opinion(gsm8k_task) -> None:
    config = EQBSLConfig(verification=EqbslVerificationConfig(enabled=True, api_key="test-key"))
    builder = EqbslSourceBuilder(
        config,
        verification_provider=_StubVerificationProvider(
            config,
            VerificationEvidence(
                classification="support",
                confidence=0.9,
                rationale="Arithmetic checks out.",
                rationale_tags=["arithmetic_consistent"],
            ),
        ),
    )
    trace = CandidateTrace(
        trace_id="t1",
        generator_name="llm_cot",
        answer="9",
        reasoning_steps=["Compute the remaining eggs."],
        confidence_score=0.8,
    )
    coalition = CoalitionBundle(
        answer_signature="(9,)",
        traces=[trace],
        critiques=[_critique("t1", aggregate=0.8)],
        trace_scores=[_trace_score("t1", "llm_cot", U=0.5, C=0.8, M=0.9, V=1.0, total=0.76)],
        representative_trace_id="t1",
        representative_generator="llm_cot",
    )
    opinions = builder.build(
        task=gsm8k_task,
        coalitions=[coalition],
        all_traces=[trace],
        all_critiques=[_critique("t1", aggregate=0.8)],
        uncertainty=UncertaintyReport(
            entropy=0.4,
            margin=0.2,
            coalition_mass=0.5,
            disagreement_rate=1.0,
            per_trace_quality={"t1": 0.5},
        ),
        failure_memory_metadata={},
    )
    verification_source = opinions[0].source_opinions["verification"]
    assert verification_source.metadata["classification"] == "support"
    assert verification_source.metadata["rationale_tags"] == ["arithmetic_consistent"]
    assert verification_source.opinion.belief > verification_source.opinion.disbelief


def test_verification_source_inconclusive_stays_uncertainty_first() -> None:
    config = EQBSLConfig(verification=EqbslVerificationConfig(enabled=True, api_key="test-key"))
    provider = VerificationSourceProvider(config)
    source = provider._assessment_to_source(  # type: ignore[attr-defined]
        VerificationEvidence(
            classification="inconclusive",
            confidence=0.8,
            rationale="Evidence is mixed.",
            rationale_tags=["mixed_evidence"],
        )
    )
    assert source.opinion.uncertainty == 1.0
    assert source.opinion.belief == 0.0
    assert source.opinion.disbelief == 0.0


def test_verification_source_parses_embedded_json() -> None:
    payload = VerificationSourceProvider._parse_json_object(  # type: ignore[attr-defined]
        "```json\n{\"classification\":\"support\",\"confidence\":0.8,"
        "\"rationale_tags\":[\"check\"],\"rationale\":\"ok\"}\n```"
    )
    assert payload["classification"] == "support"
    assert payload["confidence"] == 0.8


def test_writer_persists_coalition_opinions(in_memory_store) -> None:
    writer = LedgerWriter(in_memory_store)
    writer.write_coalition_opinions(
        run_id="run-1",
        task_id="task-1",
        coalition_rows=[
            {
                "answer_signature": "(9,)",
                "coalition_member_trace_ids": ["t1", "t2"],
                "coalition_member_generators": ["llm", "llm_cot"],
                "representative_trace_id": "t1",
                "representative_generator": "llm",
                "source_opinions": {"U": {"source_name": "U"}},
                "generator_trust_opinion": {"source_name": "generator_trust"},
                "fused_opinion": {
                    "belief": 0.6,
                    "disbelief": 0.1,
                    "uncertainty": 0.3,
                    "base_rate": 0.25,
                },
                "final_expectation": 0.675,
                "base_rate_contribution": 0.075,
                "decision_role": "winner",
                "decision_reason_code": "select_strong_expectation",
                "decision_reason_text": "Selected by EQBSL.",
                "explanation_metadata": {"domain": "gsm8k_math"},
            }
        ],
    )
    rows = in_memory_store.get_coalition_opinions(run_ids=["run-1"])
    assert len(rows) == 1
    assert rows[0]["answer_signature"] == "(9,)"
    assert json.loads(rows[0]["coalition_member_trace_ids_json"]) == ["t1", "t2"]
