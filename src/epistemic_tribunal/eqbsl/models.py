"""Typed models for the EQBSL coalition-opinion layer."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class EvidenceOpinion(BaseModel):
    """A binomial subjective-logic opinion with persisted evidence masses."""

    belief: float = Field(ge=0.0, le=1.0)
    disbelief: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    base_rate: float = Field(default=0.25, ge=0.0, le=0.5)
    positive_evidence: float = Field(default=0.0, ge=0.0)
    negative_evidence: float = Field(default=0.0, ge=0.0)
    prior_weight: float = Field(default=2.0, gt=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_sum(self) -> "EvidenceOpinion":
        total = self.belief + self.disbelief + self.uncertainty
        if abs(total - 1.0) > 1e-6:
            raise ValueError("belief + disbelief + uncertainty must equal 1.")
        return self

    @property
    def expectation(self) -> float:
        return round(self.belief + (self.base_rate * self.uncertainty), 6)

    @property
    def base_rate_contribution(self) -> float:
        return round(self.base_rate * self.uncertainty, 6)

    @classmethod
    def from_evidence(
        cls,
        *,
        positive_evidence: float,
        negative_evidence: float,
        prior_weight: float,
        base_rate: float,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "EvidenceOpinion":
        total = positive_evidence + negative_evidence + prior_weight
        if total <= 0.0:
            total = 1.0
        belief = round(positive_evidence / total, 6)
        disbelief = round(negative_evidence / total, 6)
        uncertainty = round(max(0.0, 1.0 - belief - disbelief), 6)
        return cls(
            belief=belief,
            disbelief=disbelief,
            uncertainty=uncertainty,
            base_rate=base_rate,
            positive_evidence=round(positive_evidence, 6),
            negative_evidence=round(negative_evidence, 6),
            prior_weight=round(prior_weight, 6),
            metadata=metadata or {},
        )

    @classmethod
    def neutral(
        cls,
        *,
        base_rate: float,
        prior_weight: float,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "EvidenceOpinion":
        return cls.from_evidence(
            positive_evidence=0.0,
            negative_evidence=0.0,
            prior_weight=prior_weight,
            base_rate=base_rate,
            metadata=metadata,
        )


class OpinionSource(BaseModel):
    """A named evidence source contributing to a coalition opinion."""

    source_name: str
    source_type: str
    trust_weight: float = Field(default=1.0, ge=0.0)
    opinion: EvidenceOpinion
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerificationEvidence(BaseModel):
    """Typed output from an answer-conditioned verification check."""

    classification: Literal["support", "contradiction", "inconclusive"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(default="")
    rationale_tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DecisionReason(BaseModel):
    """Structured reason for a coalition-level decision."""

    reason_code: str
    reason_text: str
    metrics: dict[str, Any] = Field(default_factory=dict)


class CoalitionOpinion(BaseModel):
    """Opinion state for a coalition of same-answer traces."""

    answer_signature: str
    coalition_member_trace_ids: list[str] = Field(default_factory=list)
    coalition_member_generators: list[str] = Field(default_factory=list)
    representative_trace_id: Optional[str] = None
    representative_generator: Optional[str] = None
    source_opinions: dict[str, OpinionSource] = Field(default_factory=dict)
    generator_trust_opinion: Optional[OpinionSource] = None
    fused_opinion: EvidenceOpinion
    final_expectation: float = Field(default=0.0, ge=0.0, le=1.0)
    base_rate_contribution: float = Field(default=0.0, ge=0.0, le=1.0)
    decision_role: str = Field(default="other")
    decision_reason_code: str = Field(default="")
    decision_reason_text: str = Field(default="")
    explanation_metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _derive_expectation(self) -> "CoalitionOpinion":
        self.final_expectation = round(self.fused_opinion.expectation, 6)
        self.base_rate_contribution = round(
            self.fused_opinion.base_rate_contribution, 6
        )
        return self
