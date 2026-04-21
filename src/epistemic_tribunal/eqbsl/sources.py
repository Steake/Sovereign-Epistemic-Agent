"""Source-opinion builders for the EQBSL coalition layer."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any, Optional

from epistemic_tribunal.config import EQBSLConfig
from epistemic_tribunal.eqbsl.adapters import get_eqbsl_adapter
from epistemic_tribunal.eqbsl.models import CoalitionOpinion, EvidenceOpinion, OpinionSource
from epistemic_tribunal.eqbsl.trust import GeneratorTrustEstimator
from epistemic_tribunal.eqbsl.verification import VerificationSourceProvider
from epistemic_tribunal.tribunal.scoring import TraceScore
from epistemic_tribunal.tribunal_types import CandidateTrace, CritiqueResult, Task, UncertaintyReport


@dataclass
class CoalitionBundle:
    answer_signature: str
    traces: list[CandidateTrace]
    critiques: list[CritiqueResult]
    trace_scores: list[TraceScore]
    representative_trace_id: str
    representative_generator: str


class EqbslSourceBuilder:
    """Builds coalition-level source opinions from existing tribunal signals."""

    def __init__(
        self,
        config: EQBSLConfig,
        trust_estimator: Optional[GeneratorTrustEstimator] = None,
        verification_provider: Optional[VerificationSourceProvider] = None,
    ) -> None:
        self._config = config
        self._trust_estimator = trust_estimator or GeneratorTrustEstimator(config)
        self._verification_provider = verification_provider or VerificationSourceProvider(config)

    def build(
        self,
        *,
        task: Task,
        coalitions: list[CoalitionBundle],
        all_traces: list[CandidateTrace],
        all_critiques: list[CritiqueResult],
        uncertainty: UncertaintyReport,
        failure_memory_metadata: Optional[dict[str, Any]] = None,
    ) -> list[CoalitionOpinion]:
        domain_adapter = get_eqbsl_adapter(task.domain)
        all_critic_scores = [c.aggregate_score for c in all_critiques]
        all_flat = len({round(s, 4) for s in all_critic_scores}) <= 1 if all_critic_scores else True

        coalition_opinions: list[CoalitionOpinion] = []
        for coalition in coalitions:
            domain_features = domain_adapter.build_features(
                task=task,
                coalition_traces=coalition.traces,
                coalition_critiques=coalition.critiques,
                all_traces=all_traces,
                all_critiques=all_critiques,
                coalition_answer_signature=coalition.answer_signature,
                failure_memory_metadata=failure_memory_metadata,
            )
            source_opinions = {
                "U": self._build_u_source(coalition, all_traces),
                "C": self._build_c_source(coalition, all_flat),
                "M": self._build_m_source(coalition, failure_memory_metadata),
                "V": self._build_v_source(coalition),
            }
            verification_source = self._verification_provider.build_source(
                task=task,
                coalition_answer_signature=coalition.answer_signature,
                coalition_traces=coalition.traces,
                coalition_critiques=coalition.critiques,
                domain_features=domain_features,
            )
            if verification_source is not None:
                source_opinions["verification"] = verification_source
            generator_trust = self._trust_estimator.estimate(
                coalition_generators=[trace.generator_name for trace in coalition.traces],
                coalition_features=domain_features,
            )
            coalition_opinions.append(
                CoalitionOpinion(
                    answer_signature=coalition.answer_signature,
                    coalition_member_trace_ids=[t.trace_id for t in coalition.traces],
                    coalition_member_generators=[t.generator_name for t in coalition.traces],
                    representative_trace_id=coalition.representative_trace_id,
                    representative_generator=coalition.representative_generator,
                    source_opinions=source_opinions,
                    generator_trust_opinion=generator_trust,
                    fused_opinion=EvidenceOpinion.neutral(
                        base_rate=self._config.default_base_rate,
                        prior_weight=self._config.k,
                        metadata={"pending_fusion": True},
                    ),
                    explanation_metadata={
                        "domain_features": domain_features,
                        "coalition_size": len(coalition.traces),
                        "coalition_mass": round(len(coalition.traces) / max(len(all_traces), 1), 4),
                    },
                )
            )
        return coalition_opinions

    def _build_u_source(self, coalition: CoalitionBundle, all_traces: list[CandidateTrace]) -> OpinionSource:
        coalition_mass = len(coalition.traces) / max(len(all_traces), 1)
        positive = coalition_mass * 4.0
        negative = (1.0 - coalition_mass) * 1.0
        opinion = EvidenceOpinion.from_evidence(
            positive_evidence=positive,
            negative_evidence=negative,
            prior_weight=self._config.k,
            base_rate=self._config.default_base_rate,
            metadata={
                "coalition_mass": round(coalition_mass, 6),
                "coalition_size": len(coalition.traces),
                "pool_size": len(all_traces),
            },
        )
        return OpinionSource(
            source_name="U",
            source_type="support",
            trust_weight=self._config.source_trust.u,
            opinion=opinion,
            metadata={"coalition_mass": round(coalition_mass, 6)},
        )

    def _build_c_source(self, coalition: CoalitionBundle, all_flat: bool) -> OpinionSource:
        avg_c = mean(ts.C for ts in coalition.trace_scores) if coalition.trace_scores else 0.0
        reasoning_present = mean(1.0 if t.reasoning_steps else 0.0 for t in coalition.traces) if coalition.traces else 0.0
        reasoning_depth = min(1.0, mean(len(t.reasoning_steps) for t in coalition.traces) / 4.0) if coalition.traces else 0.0

        positive = (avg_c * 3.0) + (reasoning_present * 1.4) + (reasoning_depth * 1.2)
        negative = max(0.0, 0.6 - avg_c) * 1.5 + (1.0 - reasoning_present) * 0.8

        if all_flat:
            positive *= 0.5
            negative *= 0.5

        opinion = EvidenceOpinion.from_evidence(
            positive_evidence=positive,
            negative_evidence=negative,
            prior_weight=self._config.k,
            base_rate=self._config.default_base_rate,
            metadata={
                "avg_c": round(avg_c, 6),
                "reasoning_present": round(reasoning_present, 6),
                "reasoning_depth": round(reasoning_depth, 6),
                "all_flat": all_flat,
            },
        )
        return OpinionSource(
            source_name="C",
            source_type="critic",
            trust_weight=self._config.source_trust.c,
            opinion=opinion,
            metadata={"all_flat": all_flat},
        )

    def _build_m_source(
        self,
        coalition: CoalitionBundle,
        failure_memory_metadata: Optional[dict[str, Any]],
    ) -> OpinionSource:
        trace_decomp = {}
        penalty_scale = 1.0
        if failure_memory_metadata:
            trace_decomp = failure_memory_metadata.get("failure_memory_trace_decomposition", {})
            penalty_scale = max(float(failure_memory_metadata.get("failure_memory_penalty_scale", 1.0)), 1e-6)

        exact_penalties = []
        structural_penalties = []
        exact_hits = 0
        structural_hits = 0
        for trace in coalition.traces:
            decomp = trace_decomp.get(trace.trace_id, {})
            exact_penalties.append(float(decomp.get("exact_penalty", 0.0)))
            structural_penalties.append(float(decomp.get("structural_penalty", 0.0)))
            exact_hits += int(decomp.get("n_exact_matches", 0))
            structural_hits += int(decomp.get("n_structural_matches", 0))

        exact_norm = (mean(exact_penalties) / penalty_scale) if exact_penalties else 0.0
        structural_norm = (mean(structural_penalties) / penalty_scale) if structural_penalties else 0.0

        negative = (4.0 * exact_norm) + (0.75 * structural_norm)
        opinion = EvidenceOpinion.from_evidence(
            positive_evidence=0.0,
            negative_evidence=max(negative, 0.0),
            prior_weight=self._config.k,
            base_rate=self._config.default_base_rate,
            metadata={
                "exact_norm": round(exact_norm, 6),
                "structural_norm": round(structural_norm, 6),
                "exact_hits": exact_hits,
                "structural_hits": structural_hits,
            },
        )
        return OpinionSource(
            source_name="M",
            source_type="memory",
            trust_weight=self._config.source_trust.m,
            opinion=opinion,
            metadata={
                "exact_hits": exact_hits,
                "structural_hits": structural_hits,
            },
        )

    def _build_v_source(self, coalition: CoalitionBundle) -> OpinionSource:
        avg_v = mean(ts.V for ts in coalition.trace_scores) if coalition.trace_scores else 0.0
        violations = sum(len(c.violated_invariants) for c in coalition.critiques)
        positive = avg_v * 2.5
        negative = (1.0 - avg_v) * 2.0 + min(violations, 3) * 0.4
        opinion = EvidenceOpinion.from_evidence(
            positive_evidence=positive,
            negative_evidence=negative,
            prior_weight=self._config.k,
            base_rate=self._config.default_base_rate,
            metadata={
                "avg_v": round(avg_v, 6),
                "violations": violations,
            },
        )
        return OpinionSource(
            source_name="V",
            source_type="constraint",
            trust_weight=self._config.source_trust.v,
            opinion=opinion,
            metadata={"violations": violations},
        )
