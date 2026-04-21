"""Tribunal aggregator — combines all signals to make the final decision."""

from __future__ import annotations

from typing import Any, Optional

from epistemic_tribunal.config import EQBSLConfig, TribunalConfig
from epistemic_tribunal.domains.factory import get_adapter
from epistemic_tribunal.eqbsl import (
    EqbslDecisionPolicy,
    EqbslFusionEngine,
    EqbslSourceBuilder,
    GeneratorTrustEstimator,
)
from epistemic_tribunal.eqbsl.sources import CoalitionBundle
from epistemic_tribunal.ledger.store import LedgerStore
from epistemic_tribunal.tribunal.scoring import (
    TraceScore,
    compute_trace_score,
    normalise_weights,
)
from epistemic_tribunal.tribunal_types import (
    CandidateTrace,
    CritiqueResult,
    DecisionKind,
    InvariantSet,
    Task,
    TribunalDecision,
    UncertaintyReport,
)
from epistemic_tribunal.utils.logging import get_logger

log = get_logger(__name__)


class TribunalAggregator:
    """Adjudicates between candidate traces and produces a :class:`TribunalDecision`."""

    def __init__(
        self,
        config: Optional[TribunalConfig] = None,
        *,
        eqbsl_config: Optional[EQBSLConfig] = None,
        ledger_store: Optional[LedgerStore] = None,
    ) -> None:
        self._config = config or TribunalConfig()
        self._eqbsl = eqbsl_config or EQBSLConfig()
        self._trust_estimator = GeneratorTrustEstimator(self._eqbsl, store=ledger_store)
        self._source_builder = EqbslSourceBuilder(
            self._eqbsl,
            trust_estimator=self._trust_estimator,
        )
        self._fusion_engine = EqbslFusionEngine(self._eqbsl)
        self._decision_policy = EqbslDecisionPolicy(self._config, self._eqbsl)

    def adjudicate(
        self,
        task: Task,
        traces: list[CandidateTrace],
        critiques: list[CritiqueResult],
        uncertainty: UncertaintyReport,
        invariant_set: Optional[InvariantSet] = None,
        is_single_source_mode: bool = False,
        attempt: int = 0,
        failure_memory_metadata: Optional[dict[str, Any]] = None,
    ) -> TribunalDecision:
        if not traces:
            log.warning("Tribunal invoked with no traces — abstaining.")
            return TribunalDecision(
                task_id=task.task_id,
                decision=DecisionKind.ABSTAIN,
                reasoning="No candidate traces available.",
            )

        trace_scores = self._compute_trace_scores(traces, critiques, uncertainty)
        if not trace_scores:
            return TribunalDecision(
                task_id=task.task_id,
                decision=DecisionKind.ABSTAIN,
                reasoning="All traces were missing critiques.",
            )

        trace_by_id = {trace.trace_id: trace for trace in traces}
        score_map = {ts.trace_id: ts.total for ts in trace_scores}
        forensic_data = self._build_trace_forensic(trace_scores, trace_by_id)

        if self._config.fusion_mode == "eqbsl":
            decision = self._adjudicate_eqbsl(
                task=task,
                traces=traces,
                critiques=critiques,
                trace_scores=trace_scores,
                uncertainty=uncertainty,
                failure_memory_metadata=failure_memory_metadata,
                score_map=score_map,
                forensic_data=forensic_data,
            )
        else:
            decision = self._adjudicate_weighted_sum(
                task=task,
                traces=traces,
                critiques=critiques,
                trace_scores=trace_scores,
                uncertainty=uncertainty,
                is_single_source_mode=is_single_source_mode,
                attempt=attempt,
                score_map=score_map,
                forensic_data=forensic_data,
            )

        return decision

    def _compute_trace_scores(
        self,
        traces: list[CandidateTrace],
        critiques: list[CritiqueResult],
        uncertainty: UncertaintyReport,
    ) -> list[TraceScore]:
        w = self._config.weights
        alpha, beta, gamma, delta = normalise_weights(
            w.uncertainty, w.critic, w.memory, w.invariant
        )
        critique_by_id: dict[str, CritiqueResult] = {c.trace_id: c for c in critiques}
        trace_scores: list[TraceScore] = []
        for trace in traces:
            critique = critique_by_id.get(trace.trace_id)
            if critique is None:
                log.warning("No critique for trace %s — skipping.", trace.trace_id)
                continue
            trace_scores.append(
                compute_trace_score(
                    trace,
                    critique,
                    uncertainty,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    delta=delta,
                )
            )
        trace_scores.sort(key=lambda ts: ts.total, reverse=True)
        return trace_scores

    @staticmethod
    def _build_trace_forensic(
        trace_scores: list[TraceScore],
        trace_by_id: dict[str, CandidateTrace],
    ) -> list[dict[str, Any]]:
        forensic_data = []
        for ts in trace_scores:
            trace = trace_by_id.get(ts.trace_id)
            trace_length = trace.token_count if trace else 0
            finish_reason = trace.metadata.get("finish_reason", "stop") if trace else "stop"
            forensic_data.append(
                {
                    "trace_id": ts.trace_id,
                    "generator": ts.generator_name,
                    "U": ts.U,
                    "C": ts.C,
                    "M": ts.M,
                    "V": ts.V,
                    "total": ts.total,
                    "confidence": round(ts.total, 4),
                    "trace_length": trace_length,
                    "finish_reason": finish_reason,
                }
            )
        return forensic_data

    def _adjudicate_eqbsl(
        self,
        *,
        task: Task,
        traces: list[CandidateTrace],
        critiques: list[CritiqueResult],
        trace_scores: list[TraceScore],
        uncertainty: UncertaintyReport,
        failure_memory_metadata: Optional[dict[str, Any]],
        score_map: dict[str, float],
        forensic_data: list[dict[str, Any]],
    ) -> TribunalDecision:
        adapter = get_adapter(task.domain)
        trace_by_id = {trace.trace_id: trace for trace in traces}
        critique_by_id = {critique.trace_id: critique for critique in critiques}
        score_by_id = {score.trace_id: score for score in trace_scores}

        coalitions: list[CoalitionBundle] = []
        for cluster in adapter.cluster_answers(traces):
            if not cluster:
                continue
            cluster_scores = [
                score_by_id[trace.trace_id]
                for trace in cluster
                if trace.trace_id in score_by_id
            ]
            if not cluster_scores:
                continue
            cluster_scores.sort(key=lambda item: item.total, reverse=True)
            representative = trace_by_id[cluster_scores[0].trace_id]
            coalitions.append(
                CoalitionBundle(
                    answer_signature=str(adapter.get_cluster_key(cluster[0].answer)),
                    traces=cluster,
                    critiques=[
                        critique_by_id[trace.trace_id]
                        for trace in cluster
                        if trace.trace_id in critique_by_id
                    ],
                    trace_scores=cluster_scores,
                    representative_trace_id=representative.trace_id,
                    representative_generator=representative.generator_name,
                )
            )

        coalition_opinions = self._source_builder.build(
            task=task,
            coalitions=coalitions,
            all_traces=traces,
            all_critiques=critiques,
            uncertainty=uncertainty,
            failure_memory_metadata=failure_memory_metadata,
        )
        coalition_opinions = [self._fusion_engine.fuse(coalition) for coalition in coalition_opinions]
        no_memory_ranked = self._rank_coalitions_without_source(coalition_opinions, "M")
        no_verification_ranked = self._rank_coalitions_without_source(coalition_opinions, "verification")
        eqbsl_result = self._decision_policy.decide(coalition_opinions)
        winner = eqbsl_result.coalitions[0] if eqbsl_result.coalitions else None

        exact_memory_changed_winner = False
        verification_changed_winner = False
        if winner is not None and no_memory_ranked:
            winner_memory_meta = winner.source_opinions.get("M", {}).opinion.metadata if winner.source_opinions.get("M") else {}
            alt_winner = no_memory_ranked[0]
            exact_memory_changed_winner = (
                alt_winner.answer_signature != winner.answer_signature
                and winner_memory_meta.get("exact_hits", 0) > 0
            )
        if winner is not None and no_verification_ranked:
            verification_changed_winner = (
                no_verification_ranked[0].answer_signature != winner.answer_signature
            )

        selected_trace_id = eqbsl_result.selected_trace_id
        selected_answer = trace_by_id[selected_trace_id].answer if selected_trace_id else None

        top_two = [
            {
                "answer_signature": coalition.answer_signature,
                "expectation": coalition.final_expectation,
                "belief": coalition.fused_opinion.belief,
                "disbelief": coalition.fused_opinion.disbelief,
                "uncertainty": coalition.fused_opinion.uncertainty,
            }
            for coalition in eqbsl_result.coalitions[:2]
        ]

        eqbsl_summary = {
            "fusion_mode": "eqbsl",
            "winning_coalition_signature": eqbsl_result.selected_answer_signature,
            "top_two_expectations": top_two,
            "task_reason": eqbsl_result.reason.model_dump(mode="json"),
            "base_rate_changed_winner": eqbsl_result.base_rate_changed_winner,
            "exact_memory_changed_winner": exact_memory_changed_winner,
            "verification_changed_winner": verification_changed_winner,
            "same_answer_tie_case": bool(
                winner is not None and len(winner.coalition_member_trace_ids) > 1
            ),
            "structural_memory_uncertainty_without_disbelief": bool(
                winner is not None
                and winner.source_opinions.get("M") is not None
                and winner.source_opinions["M"].opinion.metadata.get("structural_hits", 0) > 0
                and winner.source_opinions["M"].opinion.metadata.get("exact_hits", 0) == 0
                and winner.source_opinions["M"].opinion.uncertainty
                > winner.source_opinions["M"].opinion.disbelief
            ),
            "top_gap": eqbsl_result.top_gap,
            "mean_belief": round(
                sum(c.fused_opinion.belief for c in eqbsl_result.coalitions)
                / max(len(eqbsl_result.coalitions), 1),
                4,
            ),
            "mean_disbelief": round(
                sum(c.fused_opinion.disbelief for c in eqbsl_result.coalitions)
                / max(len(eqbsl_result.coalitions), 1),
                4,
            ),
            "mean_uncertainty": round(
                sum(c.fused_opinion.uncertainty for c in eqbsl_result.coalitions)
                / max(len(eqbsl_result.coalitions), 1),
                4,
            ),
        }
        if winner is not None and winner.source_opinions.get("verification") is not None:
            verification_meta = winner.source_opinions["verification"].metadata
            eqbsl_summary["winner_verification"] = {
                "classification": verification_meta.get("classification"),
                "confidence": verification_meta.get("confidence"),
                "rationale_tags": verification_meta.get("rationale_tags", []),
            }

        reasoning = [
            f"EQBSL decision: {eqbsl_result.decision.value}",
            f"Reason: {eqbsl_result.reason.reason_code}",
            eqbsl_result.reason.reason_text,
        ]
        if top_two:
            reasoning.append(
                "Top coalition expectations: "
                + ", ".join(
                    f"{item['answer_signature']}={item['expectation']:.4f}"
                    for item in top_two
                )
            )

        metadata = {
            "forensic": forensic_data,
            "structural_margin": 0.0,
            "eqbsl": eqbsl_summary,
            "_eqbsl_coalitions": [
                coalition.model_dump(mode="json")
                for coalition in eqbsl_result.coalitions
            ],
        }

        if selected_trace_id is not None:
            selected_trace = trace_by_id[selected_trace_id]
            metadata["candidate_source"] = selected_trace.generator_name
            metadata["trace_length"] = selected_trace.token_count
            metadata["finish_reason"] = selected_trace.metadata.get("finish_reason", "stop")

        return TribunalDecision(
            task_id=task.task_id,
            decision=eqbsl_result.decision,
            selected_trace_id=selected_trace_id,
            selected_answer=selected_answer,
            scores=score_map,
            reasoning="\n".join(reasoning),
            confidence=eqbsl_result.confidence,
            metadata=metadata,
        )

    def _rank_coalitions_without_source(
        self,
        coalitions: list,
        source_name: str,
    ) -> list:
        adjusted = []
        for coalition in coalitions:
            variant = coalition.model_copy(deep=True)
            variant.source_opinions.pop(source_name, None)
            adjusted.append(self._fusion_engine.fuse(variant))
        adjusted.sort(key=lambda item: item.final_expectation, reverse=True)
        return adjusted

    def _adjudicate_weighted_sum(
        self,
        *,
        task: Task,
        traces: list[CandidateTrace],
        critiques: list[CritiqueResult],
        trace_scores: list[TraceScore],
        uncertainty: UncertaintyReport,
        is_single_source_mode: bool,
        attempt: int,
        score_map: dict[str, float],
        forensic_data: list[dict[str, Any]],
    ) -> TribunalDecision:
        trace_by_id = {t.trace_id: t for t in traces}
        critique_by_id: dict[str, CritiqueResult] = {c.trace_id: c for c in critiques}
        best = trace_scores[0]
        best_trace = trace_by_id.get(best.trace_id)

        adapter = get_adapter(task.domain)
        second = None
        if best_trace:
            for ts in trace_scores[1:]:
                other_trace = trace_by_id.get(ts.trace_id)
                if other_trace and not adapter.answers_equal(other_trace.answer, best_trace.answer):
                    second = ts
                    break

        selected_mass = 0.0
        coalition_generator_types: set[str] = set()
        if best_trace is not None:
            selected_traces = [t for t in traces if adapter.answers_equal(t.answer, best_trace.answer)]
            selected_mass = len(selected_traces) / len(traces)
            coalition_generator_types = {t.generator_name for t in selected_traces}
            selected_mass = max(selected_mass, uncertainty.coalition_mass)

        beta = self._config.weights.critic
        delta = self._config.weights.invariant
        best_structural = (beta * best.C) + (delta * best.V)
        second_structural = (beta * second.C + delta * second.V) if second else 0.0
        structural_margin = best_structural - second_structural
        best_critique = critique_by_id.get(best.trace_id)
        violation_count = len(best_critique.violated_invariants) if best_critique else 0

        decision: DecisionKind = DecisionKind.ABSTAIN
        selected_id: Optional[str] = None
        selected_answer: Optional[list[list[int]]] = None
        override_triggered = False
        reasoning_parts: list[str] = [
            f"Top score: {best.total:.4f} (generator={best.generator_name!r})",
            f"Structural Scores: best={best_structural:.4f}, margin={structural_margin:.4f}",
            f"Selection threshold: {self._config.selection_threshold}",
            f"Resample threshold: {self._config.resample_threshold}",
            f"Scores: {{ {', '.join(f'{ts.generator_name}: {ts.total:.4f}' for ts in trace_scores)} }}",
        ]

        ov = self._config.structural_override
        path_b_stats = {
            "met_gate_potential": False,
            "fired": False,
            "failed_V": False,
            "failed_C": False,
            "failed_violations": False,
            "failed_margin": False,
        }

        if ov.enabled:
            path_b_stats["met_gate_potential"] = (len(coalition_generator_types) == 1)
            if best.V < ov.v_threshold:
                path_b_stats["failed_V"] = True
            if best.C < ov.c_threshold:
                path_b_stats["failed_C"] = True
            if violation_count > 0:
                path_b_stats["failed_violations"] = True
            if structural_margin < ov.margin_threshold:
                path_b_stats["failed_margin"] = True

            can_bypass = (
                not path_b_stats["failed_V"]
                and not path_b_stats["failed_C"]
                and not path_b_stats["failed_violations"]
                and not path_b_stats["failed_margin"]
            )
            if can_bypass and len(coalition_generator_types) == 1:
                override_triggered = True
                path_b_stats["fired"] = True
                reasoning_parts.append(
                    "Path B Bypass Triggered: Overriding single-mind lockout due to overwhelming structural evidence."
                )

        if (
            best.total >= self._config.selection_threshold
            and selected_mass > self._config.diversity_floor
            and len(coalition_generator_types) == 1
            and not override_triggered
            and not (is_single_source_mode and attempt > 0)
        ):
            decision = DecisionKind.RESAMPLE
            reasoning_parts.append(
                "Top candidate exceeded the diversity floor with support from only one generator type — requesting resample."
            )
        elif best.total >= self._config.selection_threshold or override_triggered:
            if is_single_source_mode and attempt == 0:
                decision = DecisionKind.RESAMPLE
                reasoning_parts.append(
                    "Single-source mode: forcing resample on first draw to establish a stochastic pool."
                )
            else:
                decision = DecisionKind.SELECT
                selected_id = best.trace_id
                selected_answer = best_trace.answer if best_trace else None
                reasoning_parts.append(f"Selected {best.generator_name!r} (score={best.total:.4f}).")

        margin = 0.0
        if second:
            margin = best.total - second.total
        elif len(trace_scores) > 1:
            margin = 1.0
        else:
            margin = 1.0

        margin_threshold = self._config.guardrail_margin_threshold
        min_coalition_mass = self._config.guardrail_min_coalition_mass

        if decision == DecisionKind.SELECT and not override_triggered:
            if margin_threshold > 0 and margin < margin_threshold:
                decision = DecisionKind.RESAMPLE
                reasoning_parts.append(
                    f"Guardrail: Selection demoted to RESAMPLE due to negligible margin ({margin:.3f} < {margin_threshold:.3f})."
                )
            elif min_coalition_mass > 0 and uncertainty.coalition_mass < min_coalition_mass:
                decision = DecisionKind.RESAMPLE
                reasoning_parts.append(
                    f"Guardrail: Selection demoted to RESAMPLE due to weak coalition support ({uncertainty.coalition_mass:.3f} < {min_coalition_mass:.3f})."
                )

        if decision != DecisionKind.SELECT:
            if best.total >= self._config.resample_threshold:
                decision = DecisionKind.RESAMPLE
                reasoning_parts.append("Best score below selection threshold — requesting resample.")
            else:
                decision = DecisionKind.ABSTAIN
                reasoning_parts.append("Best score below resample threshold — abstaining.")

        raw_confidence = min(1.0, best.total / max(self._config.selection_threshold, 1e-6))
        margin = 0.0
        if len(trace_scores) > 1:
            margin = best.total - trace_scores[1].total
        epistemic_confidence = raw_confidence * (0.5 + 0.5 * margin) * (
            0.5 + 0.5 * uncertainty.coalition_mass
        )
        confidence = max(0.0, min(1.0, epistemic_confidence))
        if override_triggered:
            confidence = min(confidence, ov.confidence_cap)

        metadata = {
            "forensic": forensic_data,
            "path_b_override": override_triggered,
            "path_b_stats": path_b_stats,
            "structural_margin": round(structural_margin, 4),
        }
        if best_trace and decision == DecisionKind.SELECT:
            metadata["candidate_source"] = best_trace.generator_name
            metadata["trace_length"] = best_trace.token_count
            metadata["finish_reason"] = best_trace.metadata.get("finish_reason", "stop")

        return TribunalDecision(
            task_id=task.task_id,
            decision=decision,
            selected_trace_id=selected_id,
            selected_answer=selected_answer,
            scores=score_map,
            reasoning="\n".join(reasoning_parts),
            confidence=round(confidence, 4),
            metadata=metadata,
        )
