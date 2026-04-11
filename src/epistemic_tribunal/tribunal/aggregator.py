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

from epistemic_tribunal.config import TribunalConfig
from epistemic_tribunal.tribunal.scoring import (
    TraceScore,
    compute_trace_score,
    normalise_weights,
)
from epistemic_tribunal.types import (
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

        score_map: dict[str, float] = {ts.trace_id: ts.total for ts in trace_scores}

        # Look up the winning trace
        trace_by_id = {t.trace_id: t for t in traces}
        best_trace = trace_by_id.get(best.trace_id)
        coalition_generator_types: set[str] = set()
        if best_trace is not None:
            coalition_generator_types = {
                trace.generator_name
                for trace in traces
                if trace.answer == best_trace.answer
            }

        # Decision logic
        decision: DecisionKind
        selected_id: Optional[str] = None
        selected_answer: Optional[list[list[int]]] = None
        reasoning_parts: list[str] = [
            f"Top score: {best.total:.4f} (generator={best.generator_name!r})",
            f"Selection threshold: {self._config.selection_threshold}",
            f"Resample threshold: {self._config.resample_threshold}",
            f"Scores: { {ts.generator_name: ts.total for ts in trace_scores} }",
        ]

        if (
            best.total >= self._config.selection_threshold
            and uncertainty.coalition_mass > self._config.diversity_floor
            and len(coalition_generator_types) == 1
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
        elif best.total >= self._config.selection_threshold:
            decision = DecisionKind.SELECT
            selected_id = best.trace_id
            selected_answer = best_trace.answer if best_trace else None
            reasoning_parts.append(
                f"Selected {best.generator_name!r} (score={best.total:.4f})."
            )
        elif best.total >= self._config.resample_threshold:
            decision = DecisionKind.RESAMPLE
            reasoning_parts.append(
                "Best score below selection threshold — requesting resample."
            )
        else:
            decision = DecisionKind.ABSTAIN
            reasoning_parts.append(
                "Best score below resample threshold — abstaining."
            )

        # Confidence = best score normalised relative to selection threshold
        confidence = min(1.0, best.total / max(self._config.selection_threshold, 1e-6))

        return TribunalDecision(
            task_id=task.task_id,
            decision=decision,
            selected_trace_id=selected_id,
            selected_answer=selected_answer,
            scores=score_map,
            reasoning="\n".join(reasoning_parts),
            confidence=round(confidence, 4),
        )
