"""Decision policy for the EQBSL coalition layer."""

from __future__ import annotations

from dataclasses import dataclass

from epistemic_tribunal.config import EQBSLConfig, TribunalConfig
from epistemic_tribunal.eqbsl.models import CoalitionOpinion, DecisionReason
from epistemic_tribunal.tribunal_types import DecisionKind


@dataclass
class EqbslDecisionResult:
    decision: DecisionKind
    selected_answer_signature: str | None
    selected_trace_id: str | None
    reason: DecisionReason
    coalitions: list[CoalitionOpinion]
    confidence: float
    top_gap: float
    base_rate_changed_winner: bool


class EqbslDecisionPolicy:
    """Turns fused coalition opinions into tribunal decisions."""

    def __init__(self, tribunal_config: TribunalConfig, eqbsl_config: EQBSLConfig) -> None:
        self._tribunal = tribunal_config
        self._eqbsl = eqbsl_config

    def decide(self, coalitions: list[CoalitionOpinion]) -> EqbslDecisionResult:
        if not coalitions:
            return EqbslDecisionResult(
                decision=DecisionKind.ABSTAIN,
                selected_answer_signature=None,
                selected_trace_id=None,
                reason=DecisionReason(
                    reason_code="abstain_no_coalitions",
                    reason_text="No coalitions were available for EQBSL adjudication.",
                ),
                coalitions=[],
                confidence=0.0,
                top_gap=0.0,
                base_rate_changed_winner=False,
            )

        ranked = sorted(coalitions, key=lambda c: c.final_expectation, reverse=True)
        belief_ranked = sorted(coalitions, key=lambda c: c.fused_opinion.belief, reverse=True)
        base_rate_changed_winner = (
            ranked[0].answer_signature != belief_ranked[0].answer_signature
        )

        top = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None
        gap = round(top.final_expectation - (second.final_expectation if second else 0.0), 6)
        top_verification = self._verification_signal(top)
        second_verification = self._verification_signal(second) if second is not None else None

        for idx, coalition in enumerate(ranked):
            coalition.decision_role = "winner" if idx == 0 else "runner_up" if idx == 1 else "other"

        # Keep the existing diversity-floor protection for single-mind lockout.
        if (
            top.explanation_metadata.get("coalition_mass", 0.0) > self._tribunal.diversity_floor
            and len(set(top.coalition_member_generators)) == 1
        ):
            reason = DecisionReason(
                reason_code="resample_diversity_floor",
                reason_text=(
                    "Top coalition exceeded the diversity floor with support from only one "
                    "generator type."
                ),
                metrics={
                    "coalition_mass": top.explanation_metadata.get("coalition_mass", 0.0),
                    "diversity_floor": self._tribunal.diversity_floor,
                },
            )
            self._tag_reason(top, reason)
            return EqbslDecisionResult(
                decision=DecisionKind.RESAMPLE,
                selected_answer_signature=None,
                selected_trace_id=None,
                reason=reason,
                coalitions=ranked,
                confidence=self._confidence(top, gap),
                top_gap=gap,
                base_rate_changed_winner=base_rate_changed_winner,
            )

        if top.fused_opinion.uncertainty >= self._eqbsl.abstain_uncertainty_threshold:
            reason = DecisionReason(
                reason_code="abstain_high_uncertainty",
                reason_text="Top coalition retained too much unresolved uncertainty.",
                metrics={"uncertainty": top.fused_opinion.uncertainty},
            )
            self._tag_reason(top, reason)
            return EqbslDecisionResult(
                decision=DecisionKind.ABSTAIN,
                selected_answer_signature=None,
                selected_trace_id=None,
                reason=reason,
                coalitions=ranked,
                confidence=self._confidence(top, gap),
                top_gap=gap,
                base_rate_changed_winner=base_rate_changed_winner,
            )

        if (
            self._eqbsl.verification.enable_decision_semantics
            and top.final_expectation >= self._eqbsl.selection_expectation_threshold
            and top.fused_opinion.uncertainty <= self._eqbsl.max_select_uncertainty
            and top_verification is not None
            and top_verification["classification"] == "contradiction"
            and top_verification["confidence"]
            >= self._eqbsl.verification.contradiction_abstain_confidence_threshold
        ):
            reason = DecisionReason(
                reason_code="abstain_verification_contradiction",
                reason_text=(
                    "Top coalition was answer-conditioned contradicted by the verification source, "
                    "so selection was blocked despite otherwise strong expectation."
                ),
                metrics={
                    "expectation": top.final_expectation,
                    "uncertainty": top.fused_opinion.uncertainty,
                    "verification_confidence": top_verification["confidence"],
                },
            )
            self._tag_reason(top, reason)
            return EqbslDecisionResult(
                decision=DecisionKind.ABSTAIN,
                selected_answer_signature=None,
                selected_trace_id=None,
                reason=reason,
                coalitions=ranked,
                confidence=self._confidence(top, gap),
                top_gap=gap,
                base_rate_changed_winner=base_rate_changed_winner,
            )

        if (
            self._eqbsl.verification.enable_decision_semantics
            and top.final_expectation >= self._eqbsl.selection_expectation_threshold
            and top.fused_opinion.uncertainty <= self._eqbsl.max_select_uncertainty
            and gap < self._eqbsl.distinct_answer_gap_threshold
            and top_verification is not None
            and top_verification["classification"] == "support"
            and top_verification["confidence"]
            >= self._eqbsl.verification.support_select_confidence_threshold
            and not self._runner_up_has_comparable_support(
                top_signal=top_verification,
                runner_up_signal=second_verification,
            )
        ):
            reason = DecisionReason(
                reason_code="select_verification_supported_low_gap",
                reason_text=(
                    "Top coalition remained low-gap, but answer-conditioned verification provided "
                    "stronger support than the nearest distinct competitor."
                ),
                metrics={
                    "expectation": top.final_expectation,
                    "uncertainty": top.fused_opinion.uncertainty,
                    "gap": gap,
                    "verification_confidence": top_verification["confidence"],
                    "runner_up_verification_confidence": (
                        second_verification["confidence"] if second_verification is not None else None
                    ),
                },
            )
            self._tag_reason(top, reason)
            return EqbslDecisionResult(
                decision=DecisionKind.SELECT,
                selected_answer_signature=top.answer_signature,
                selected_trace_id=top.representative_trace_id,
                reason=reason,
                coalitions=ranked,
                confidence=self._confidence(top, gap),
                top_gap=gap,
                base_rate_changed_winner=base_rate_changed_winner,
            )

        if (
            top.final_expectation >= self._eqbsl.selection_expectation_threshold
            and top.fused_opinion.uncertainty <= self._eqbsl.max_select_uncertainty
            and gap >= self._eqbsl.distinct_answer_gap_threshold
        ):
            reason = DecisionReason(
                reason_code="select_strong_expectation",
                reason_text="Top coalition cleared the expectation, uncertainty, and gap thresholds.",
                metrics={
                    "expectation": top.final_expectation,
                    "uncertainty": top.fused_opinion.uncertainty,
                    "gap": gap,
                },
            )
            self._tag_reason(top, reason)
            return EqbslDecisionResult(
                decision=DecisionKind.SELECT,
                selected_answer_signature=top.answer_signature,
                selected_trace_id=top.representative_trace_id,
                reason=reason,
                coalitions=ranked,
                confidence=self._confidence(top, gap),
                top_gap=gap,
                base_rate_changed_winner=base_rate_changed_winner,
            )

        if gap < self._eqbsl.distinct_answer_gap_threshold:
            reason = DecisionReason(
                reason_code="abstain_low_gap",
                reason_text="Top coalition did not separate cleanly from the best distinct answer.",
                metrics={"gap": gap},
            )
            self._tag_reason(top, reason)
            return EqbslDecisionResult(
                decision=DecisionKind.ABSTAIN,
                selected_answer_signature=None,
                selected_trace_id=None,
                reason=reason,
                coalitions=ranked,
                confidence=self._confidence(top, gap),
                top_gap=gap,
                base_rate_changed_winner=base_rate_changed_winner,
            )

        if top.final_expectation >= self._eqbsl.resample_expectation_threshold:
            reason = DecisionReason(
                reason_code="resample_borderline_expectation",
                reason_text="Top coalition is plausible but not strong enough for a stable selection.",
                metrics={"expectation": top.final_expectation},
            )
            self._tag_reason(top, reason)
            return EqbslDecisionResult(
                decision=DecisionKind.RESAMPLE,
                selected_answer_signature=None,
                selected_trace_id=None,
                reason=reason,
                coalitions=ranked,
                confidence=self._confidence(top, gap),
                top_gap=gap,
                base_rate_changed_winner=base_rate_changed_winner,
            )

        reason = DecisionReason(
            reason_code="abstain_low_expectation",
            reason_text="No coalition accumulated enough fused belief to justify selection.",
            metrics={"expectation": top.final_expectation},
        )
        self._tag_reason(top, reason)
        return EqbslDecisionResult(
            decision=DecisionKind.ABSTAIN,
            selected_answer_signature=None,
            selected_trace_id=None,
            reason=reason,
            coalitions=ranked,
            confidence=self._confidence(top, gap),
            top_gap=gap,
            base_rate_changed_winner=base_rate_changed_winner,
        )

    @staticmethod
    def _tag_reason(coalition: CoalitionOpinion, reason: DecisionReason) -> None:
        coalition.decision_reason_code = reason.reason_code
        coalition.decision_reason_text = reason.reason_text

    @staticmethod
    def _confidence(top: CoalitionOpinion, gap: float) -> float:
        confidence = top.fused_opinion.belief * (1.0 - top.fused_opinion.uncertainty) + (0.5 * max(gap, 0.0))
        return round(max(0.0, min(1.0, confidence)), 4)

    @staticmethod
    def _verification_signal(coalition: CoalitionOpinion | None) -> dict[str, float | str] | None:
        if coalition is None:
            return None
        source = coalition.source_opinions.get("verification")
        if source is None:
            return None
        metadata = source.metadata or {}
        classification = metadata.get("classification")
        confidence = metadata.get("confidence")
        if classification is None or confidence is None:
            return None
        return {
            "classification": str(classification),
            "confidence": float(confidence),
        }

    def _runner_up_has_comparable_support(
        self,
        *,
        top_signal: dict[str, float | str],
        runner_up_signal: dict[str, float | str] | None,
    ) -> bool:
        if runner_up_signal is None:
            return False
        if runner_up_signal["classification"] != "support":
            return False
        return float(runner_up_signal["confidence"]) >= (
            float(top_signal["confidence"]) - self._eqbsl.verification.support_advantage_margin
        )
