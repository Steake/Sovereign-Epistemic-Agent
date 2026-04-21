"""Fusion engine for EQBSL coalition opinions."""

from __future__ import annotations

from epistemic_tribunal.config import EQBSLConfig
from epistemic_tribunal.eqbsl.models import CoalitionOpinion, EvidenceOpinion, OpinionSource


class EqbslFusionEngine:
    """Fuses source opinions into a single coalition opinion."""

    def __init__(self, config: EQBSLConfig) -> None:
        self._config = config

    def fuse(self, coalition: CoalitionOpinion) -> CoalitionOpinion:
        total_positive = 0.0
        total_negative = 0.0

        all_sources: list[OpinionSource] = list(coalition.source_opinions.values())
        if coalition.generator_trust_opinion is not None:
            all_sources.append(coalition.generator_trust_opinion)

        for source in all_sources:
            total_positive += source.trust_weight * source.opinion.positive_evidence
            total_negative += source.trust_weight * source.opinion.negative_evidence

        coalition.fused_opinion = EvidenceOpinion.from_evidence(
            positive_evidence=total_positive,
            negative_evidence=total_negative,
            prior_weight=self._config.k,
            base_rate=self._config.default_base_rate,
            metadata={
                "source_count": len(all_sources),
                "fused_positive_evidence": round(total_positive, 6),
                "fused_negative_evidence": round(total_negative, 6),
            },
        )
        coalition.final_expectation = round(coalition.fused_opinion.expectation, 6)
        coalition.base_rate_contribution = round(
            coalition.fused_opinion.base_rate_contribution, 6
        )
        return coalition
