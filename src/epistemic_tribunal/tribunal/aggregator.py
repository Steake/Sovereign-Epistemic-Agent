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
        *,
        completed_task_count: int = 0,
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

        # Normalise weights — zero out gamma during ledger warmup
        w = self._config.weights
        warmup_threshold = self._config.ledger_warmup_tasks
        effective_memory = w.memory
        if completed_task_count < warmup_threshold:
            effective_memory = 0.0
            log.debug(
                "Ledger warmup active (%d/%d tasks) — gamma set to 0.0.",
                completed_task_count,
                warmup_threshold,
            )

        alpha, beta, gamma, delta = normalise_weights(
            w.uncertainty, w.critic, effective_memory, w.invariant
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

        if best.total >= self._config.selection_threshold:
            # Diversity floor: if coalition mass exceeds the threshold AND
            # only one generator type contributes to the top answer, force
            # RESAMPLE so the tribunal actually adjudicates.
            diversity_floor = self._config.diversity_floor
            if (
                uncertainty.coalition_mass > diversity_floor
                and self._single_generator_type_dominates(
                    best, trace_scores, traces, uncertainty
                )
            ):
                decision = DecisionKind.RESAMPLE
                reasoning_parts.append(
                    f"Diversity floor triggered: coalition_mass="
                    f"{uncertainty.coalition_mass:.3f} > {diversity_floor:.2f} "
                    f"with single generator type dominant — forcing RESAMPLE."
                )
                log.info(
                    "Diversity floor triggered for task %s — forcing RESAMPLE.",
                    task.task_id,
                )
            else:
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

    @staticmethod
    def _single_generator_type_dominates(
        best: TraceScore,
        all_scores: list[TraceScore],
        traces: list[CandidateTrace],
        uncertainty: UncertaintyReport,
    ) -> bool:
        """Return True if the coalition supporting the top answer comes from
        only one generator type (e.g. only ``"llm"``).

        We identify the top answer by finding all traces that share the same
        answer grid as the best-scoring trace, then check if they all have
        the same ``generator_name``.
        """
        # If there is only one generator type in the pool, there is nothing
        # to diversify against — skip the check.
        all_gen_names = {t.generator_name for t in traces}
        if len(all_gen_names) <= 1:
            return False

        trace_by_id = {t.trace_id: t for t in traces}
        best_trace = trace_by_id.get(best.trace_id)
        if best_trace is None:
            return False

        def to_key(grid: list[list[int]]) -> tuple:
            return tuple(tuple(row) for row in grid)

        best_key = to_key(best_trace.answer)

        # Collect generator names of all traces whose answer matches the best
        coalition_gen_names = {
            t.generator_name
            for t in traces
            if to_key(t.answer) == best_key
        }

        return len(coalition_gen_names) == 1
