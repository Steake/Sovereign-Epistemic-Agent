"""Tribunal aggregator — combines all signals to make the final decision.

The aggregator:
1. Computes weighted scores for all candidate traces.
2. Selects the best-scoring candidate if it exceeds the selection threshold.
3. Returns RESAMPLE if the best score is above the resample threshold.
4. Returns ABSTAIN otherwise.

This is the metacognitive core: it adjudicates between competing accounts
of the task rather than simply accepting the first plausible answer.
"""

from __future__ import annotations

from typing import Optional

from epistemic_tribunal.config import TribunalConfig, TribunalWeights
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
    """Adjudicates between candidate traces and produces a :class:`TribunalDecision`.

    Parameters
    ----------
    config:
        Tribunal configuration (weights and thresholds).
    """

    def __init__(self, config: Optional[TribunalConfig] = None) -> None:
        self._config = config or TribunalConfig()

    def adjudicate(
        self,
        task: Task,
        traces: list[CandidateTrace],
        critiques: list[CritiqueResult],
        uncertainty: UncertaintyReport,
        invariant_set: Optional[InvariantSet] = None,
    ) -> TribunalDecision:
        """Run the tribunal and return a :class:`TribunalDecision`.

        Parameters
        ----------
        task:
            The original task.
        traces:
            All candidate traces.
        critiques:
            One :class:`CritiqueResult` per trace (matched by trace_id).
        uncertainty:
            Uncertainty report computed across the trace pool.
        invariant_set:
            Extracted invariant set for the task (optional).
        """
        if not traces:
            log.warning("Tribunal invoked with no traces — abstaining.")
            return TribunalDecision(
                task_id=task.task_id,
                decision=DecisionKind.ABSTAIN,
                reasoning="No candidate traces available.",
            )

        # Normalise weights
        w = self._config.weights
        alpha, beta, gamma, delta = normalise_weights(
            w.uncertainty, w.critic, w.memory, w.invariant
        )

        # Build critique lookup
        critique_by_id: dict[str, CritiqueResult] = {c.trace_id: c for c in critiques}

        # Score each trace
        trace_scores: list[TraceScore] = []
        for trace in traces:
            critique = critique_by_id.get(trace.trace_id)
            if critique is None:
                log.warning("No critique for trace %s — skipping.", trace.trace_id)
                continue
            ts = compute_trace_score(
                trace, critique, uncertainty,
                alpha=alpha, beta=beta, gamma=gamma, delta=delta,
            )
            trace_scores.append(ts)

        if not trace_scores:
            return TribunalDecision(
                task_id=task.task_id,
                decision=DecisionKind.ABSTAIN,
                reasoning="All traces were missing critiques.",
            )

        # Sort descending by total score
        trace_scores.sort(key=lambda ts: ts.total, reverse=True)
        best = trace_scores[0]
        
        # Look up the winning trace
        trace_by_id = {t.trace_id: t for t in traces}
        best_trace = trace_by_id.get(best.trace_id)
        
        # Find the best trace with a DIFFERENT answer for margin calculation
        second = None
        if best_trace:
            for ts in trace_scores[1:]:
                other_trace = trace_by_id.get(ts.trace_id)
                if other_trace and other_trace.answer != best_trace.answer:
                    second = ts
                    break

        score_map: dict[str, float] = {ts.trace_id: ts.total for ts in trace_scores}

        coalition_generator_types: set[str] = set()
        if best_trace is not None:
            coalition_generator_types = {
                trace.generator_name
                for trace in traces
                if trace.answer == best_trace.answer
            }

        # Path B: Structural Margin calculation
        # Use beta (critic) and delta (invariant) weights to define the structural component
        best_structural = (beta * best.C) + (delta * best.V)
        # If no distinct second candidate, margin is 1.0 (overwhelmingly better)
        second_structural = (beta * second.C + delta * second.V) if second else 0.0
        structural_margin = best_structural - second_structural

        # Extract violation count for the best candidate
        best_critique = critique_by_id.get(best.trace_id)
        violation_count = len(best_critique.violated_invariants) if best_critique else 0

        # Decision logic
        decision: DecisionKind
        selected_id: Optional[str] = None
        selected_answer: Optional[list[list[int]]] = None
        override_triggered = False
        reasoning_parts: list[str] = [
            f"Top score: {best.total:.4f} (generator={best.generator_name!r})",
            f"Structural Scores: best={best_structural:.4f}, margin={structural_margin:.4f}",
            f"Selection threshold: {self._config.selection_threshold}",
            f"Resample threshold: {self._config.resample_threshold}",
            f"Scores: { {ts.generator_name: ts.total for ts in trace_scores} }",
        ]

        # Check for Path B Bypass (Structural Override)
        ov = self._config.structural_override
        if ov.enabled:
            can_bypass = (
                best.V >= ov.v_threshold and
                best.C >= ov.c_threshold and
                violation_count == 0 and
                structural_margin >= ov.margin_threshold
            )
            
            if can_bypass and len(coalition_generator_types) == 1:
                override_triggered = True
                reasoning_parts.append(
                    f"Path B Bypass Triggered: Overriding single-mind lockout due to overwhelming structural evidence "
                    f"(V={best.V:.3f}, C={best.C:.3f}, violations=0, structural_margin={structural_margin:.3f})."
                )
                log.info(
                    "Path B Bypass Triggered for task %s: Structural Margin %.3f >= %.3f",
                    task.task_id, structural_margin, ov.margin_threshold
                )

        if (
            best.total >= self._config.selection_threshold
            and uncertainty.coalition_mass > self._config.diversity_floor
            and len(coalition_generator_types) == 1
            and not override_triggered
        ):
            decision = DecisionKind.RESAMPLE
            reasoning_parts.append(
                "Top candidate exceeded the diversity floor with support from only one "
                f"generator type ({sorted(coalition_generator_types)}) — requesting resample."
            )
            log.info(
                "Resampling task %s because coalition_mass=%.3f exceeded diversity_floor=%.3f "
                "with a single generator type: %s",
                task.task_id,
                uncertainty.coalition_mass,
                self._config.diversity_floor,
                sorted(coalition_generator_types),
            )
        elif best.total >= self._config.selection_threshold or override_triggered:
            decision = DecisionKind.SELECT
            selected_id = best.trace_id
            selected_answer = best_trace.answer if best_trace else None
            reasoning_parts.append(
                f"Selected {best.generator_name!r} (score={best.total:.4f})."
            )

        # FIX B: Discordant Resample Guardrail
        # If we have high disagreement (negligible margin or weak top candidate dominance),
        # force a RESAMPLE instead of a blind SELECT.
        # EXCEPTION: If Path B triggered, we trust the structural override.
        margin = 0.0
        if len(trace_scores) > 1:
            margin = best.total - trace_scores[1].total

        # PROVISIONAL: experimental control margin to prevent false-certainty
        # without collapsing successful runs. Do not canonize.
        margin_threshold = 0.01
        
        if decision == DecisionKind.SELECT and not override_triggered:
            if margin < margin_threshold:
                decision = DecisionKind.RESAMPLE
                reasoning_parts.append(
                    f"Guardrail: Selection demoted to RESAMPLE due to negligible margin ({margin:.3f}) between top candidates."
                )
            elif uncertainty.coalition_mass < 0.4:
                decision = DecisionKind.RESAMPLE
                reasoning_parts.append(
                    f"Guardrail: Selection demoted to RESAMPLE due to weak coalition support ({uncertainty.coalition_mass:.3f})."
                )

        if decision != DecisionKind.SELECT:
            if best.total >= self._config.resample_threshold:
                decision = DecisionKind.RESAMPLE
                reasoning_parts.append(
                    "Best score below selection threshold — requesting resample."
                )
            else:
                decision = DecisionKind.ABSTAIN
                reasoning_parts.append(
                    "Best score below resample threshold — abstaining."
                )

        # FIX A: Confidence Demotion
        # Confidence must not be 1.0 if disagreement is high and margin is negligible.
        raw_confidence = min(1.0, best.total / max(self._config.selection_threshold, 1e-6))
        
        # Penalise confidence by the margin and coalition mass
        margin = 0.0
        if len(trace_scores) > 1:
            margin = best.total - trace_scores[1].total
            
        epistemic_confidence = raw_confidence * (0.5 + 0.5 * margin) * (0.5 + 0.5 * uncertainty.coalition_mass)
        confidence = max(0.0, min(1.0, epistemic_confidence))

        # Path B: Override Confidence Cap
        if override_triggered:
            confidence = min(confidence, ov.confidence_cap)

        # Forensic breakdown for all candidates
        forensic_data = []
        for ts in trace_scores:
            forensic_data.append({
                "generator": ts.generator_name,
                "U": ts.U,
                "C": ts.C,
                "M": ts.M,
                "V": ts.V,
                "total": ts.total,
                "confidence": round(min(1.0, ts.total / max(self._config.selection_threshold, 1e-6)), 4)
            })

        return TribunalDecision(
            task_id=task.task_id,
            decision=decision,
            selected_trace_id=selected_id,
            selected_answer=selected_answer,
            scores=score_map,
            reasoning="\n".join(reasoning_parts),
            confidence=round(confidence, 4),
            metadata={
                "forensic": forensic_data,
                "path_b_override": override_triggered,
                "structural_margin": round(structural_margin, 4)
            }
        )
