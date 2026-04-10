"""End-to-end Orchestrator — wires every module together into one pipeline.

Pipeline stages
---------------
1. **Generate** — run all enabled generators to build a candidate pool.
2. **Extract invariants** — infer structural constraints from the task.
3. **Critique** — score each trace with the TraceCritic.
4. **Analyse uncertainty** — compute entropy / margin / coalition signals.
5. **Adjudicate** — tribunal aggregator selects the best trace or abstains.
6. **Evaluate** — compare against ground truth when available.
7. **Persist** — write records to the failure ledger.
8. **Report** — return structured output (ExperimentRun + JSON-serialisable dict).
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Optional

from epistemic_tribunal.config import TribunalSettings, load_config
from epistemic_tribunal.critics.trace_critic import TraceCritic
from epistemic_tribunal.generators.base import build_generators
from epistemic_tribunal.invariants.extractor import InvariantExtractor
from epistemic_tribunal.ledger.store import LedgerStore
from epistemic_tribunal.ledger.writer import LedgerWriter
from epistemic_tribunal.tasks.base import grids_equal
from epistemic_tribunal.tribunal.aggregator import TribunalAggregator
from epistemic_tribunal.types import (
    CandidateTrace,
    CritiqueResult,
    DecisionKind,
    ExperimentRun,
    FailureRecord,
    InvariantSet,
    Task,
    TribunalDecision,
    UncertaintyReport,
)
from epistemic_tribunal.uncertainty.analyzer import UncertaintyAnalyzer
from epistemic_tribunal.utils.logging import get_logger

log = get_logger(__name__)


class Orchestrator:
    """End-to-end tribunal pipeline runner.

    Parameters
    ----------
    config:
        Application settings.  Loaded from default YAML if omitted.
    ledger_store:
        Pre-constructed ledger store.  A new in-memory store is created if
        omitted (useful for tests).
    """

    def __init__(
        self,
        config: Optional[TribunalSettings] = None,
        ledger_store: Optional[LedgerStore] = None,
    ) -> None:
        self._config = config or load_config()
        self._store = ledger_store or LedgerStore(self._config.ledger.path)
        self._writer = LedgerWriter(self._store)

        # Build components
        self._generators = build_generators(
            self._config.generators.enabled,
            seed=self._config.generators.seed,
        )
        self._extractor = InvariantExtractor(
            enabled_checks=self._config.invariants.enabled_checks,
            confidence_threshold=self._config.invariants.confidence_threshold,
        )
        self._critic = TraceCritic(
            consistency_weight=self._config.critic.consistency_weight,
            rule_coherence_weight=self._config.critic.rule_coherence_weight,
            morphology_weight=self._config.critic.morphology_weight,
            failure_similarity_weight=self._config.critic.failure_similarity_weight,
        )
        self._uncertainty = UncertaintyAnalyzer(
            min_coalition_mass=self._config.uncertainty.min_coalition_mass
        )
        self._tribunal = TribunalAggregator(config=self._config.tribunal)
        self._completed_task_count = 0
        self._warmup_logged = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, task: Task) -> ExperimentRun:
        """Execute the full tribunal pipeline for *task*.

        Returns
        -------
        ExperimentRun
            Structured record of the run including the final decision.
        """
        start_time = time.monotonic()
        run_id = str(uuid.uuid4())
        log.info("Starting tribunal run %s for task %s", run_id[:8], task.task_id)

        # Persist task record
        self._writer.write_task(task)

        # 1. Generate candidates
        traces = self._generate(task)
        self._writer.write_traces(task, traces)

        # 2. Extract invariants
        invariant_set = self._extractor.extract(task)
        log.info(
            "Extracted %d invariant(s) for task %s",
            len(invariant_set.invariants), task.task_id
        )

        # 3. Retrieve past failure patterns from ledger
        failure_patterns = self._store.get_failure_patterns(task.task_id)

        # 4. Critique each trace
        critiques = self._critique(task, traces, invariant_set, failure_patterns)

        # 5. Analyse uncertainty
        uncertainty_report = self._uncertainty.analyze(task, traces)
        log.info("Uncertainty: %s", uncertainty_report.notes)

        # 6. Adjudicate (pass task count for ledger warmup logic)
        decision = self._tribunal.adjudicate(
            task, traces, critiques, uncertainty_report, invariant_set,
            completed_task_count=self._completed_task_count,
        )
        log.info("Decision: %s (confidence=%.3f)", decision.decision.value, decision.confidence)
        self._writer.write_decision(decision)

        # 7. Evaluate against ground truth
        ground_truth_match: Optional[bool] = None
        if task.ground_truth is not None and decision.selected_answer is not None:
            ground_truth_match = grids_equal(decision.selected_answer, task.ground_truth)
            log.info("Ground-truth match: %s", ground_truth_match)

        # 8. Write failure records
        self._persist_failures(
            task, traces, critiques, decision, uncertainty_report, ground_truth_match
        )

        duration = time.monotonic() - start_time

        run = ExperimentRun(
            run_id=run_id,
            task_id=task.task_id,
            generator_names=[g.name for g in self._generators],
            decision=decision.decision,
            selected_trace_id=decision.selected_trace_id,
            ground_truth_match=ground_truth_match,
            duration_seconds=round(duration, 4),
            config_snapshot={
                "selection_threshold": self._config.tribunal.selection_threshold,
                "resample_threshold": self._config.tribunal.resample_threshold,
                "generators": self._config.generators.enabled,
            },
        )
        self._writer.write_run(run)

        # Track completed tasks for ledger warmup
        self._completed_task_count += 1
        warmup = self._config.tribunal.ledger_warmup_tasks
        if (
            not self._warmup_logged
            and warmup > 0
            and self._completed_task_count >= warmup
        ):
            self._warmup_logged = True
            log.info(
                "Ledger warmup complete after %d tasks — "
                "memory weight gamma is now active.",
                self._completed_task_count,
            )

        return run

    def run_and_format(self, task: Task) -> dict:
        """Run the pipeline and return a JSON-serialisable result dict."""
        run = self.run(task)
        return {
            "run_id": run.run_id,
            "task_id": run.task_id,
            "decision": run.decision.value,
            "selected_trace_id": run.selected_trace_id,
            "ground_truth_match": run.ground_truth_match,
            "duration_seconds": run.duration_seconds,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate(self, task: Task) -> list[CandidateTrace]:
        traces: list[CandidateTrace] = []
        for gen in self._generators:
            try:
                trace = gen.generate(task)
                traces.append(trace)
                log.debug("Generator %r produced trace %s", gen.name, trace.trace_id[:8])
            except Exception as exc:
                log.error("Generator %r failed: %s", gen.name, exc)
        return traces

    def _critique(
        self,
        task: Task,
        traces: list[CandidateTrace],
        invariant_set: InvariantSet,
        failure_patterns: list[dict],
    ) -> list[CritiqueResult]:
        critiques: list[CritiqueResult] = []
        for trace in traces:
            result = self._critic.critique(
                task, trace, invariant_set, failure_patterns
            )
            critiques.append(result)
        return critiques

    def _persist_failures(
        self,
        task: Task,
        traces: list[CandidateTrace],
        critiques: list[CritiqueResult],
        decision: TribunalDecision,
        uncertainty: UncertaintyReport,
        ground_truth_match: Optional[bool],
    ) -> None:
        """Write failure record and invariant violations when appropriate."""
        # Persist invariant violations for all traces
        critique_by_id = {c.trace_id: c for c in critiques}
        trace_by_id = {t.trace_id: t for t in traces}
        for critique in critiques:
            trace = trace_by_id.get(critique.trace_id)
            if trace and critique.violated_invariants:
                self._writer.write_invariant_violations(task, trace, critique)

        # Write failure record if: wrong, abstained, resampled, or
        # configured to always record
        should_record = (
            ground_truth_match is False
            or decision.decision != DecisionKind.SELECT
            or self._config.ledger.always_record
        )

        if not should_record:
            return

        # Gather all violated invariants across candidates
        all_violated = list(
            {
                inv
                for c in critiques
                for inv in c.violated_invariants
            }
        )

        # Disagreement pattern summary
        unique_answers = len(
            {
                tuple(tuple(row) for row in t.answer)
                for t in traces
            }
        )
        disagreement_pattern = (
            f"{unique_answers}/{len(traces)} unique answers; "
            f"disagreement_rate={uncertainty.disagreement_rate:.3f}"
        )

        # Diagnosis
        if decision.decision == DecisionKind.ABSTAIN:
            diagnosis = "Tribunal abstained — all candidates scored below resample threshold."
        elif decision.decision == DecisionKind.RESAMPLE:
            diagnosis = "Tribunal requested resample — best score below selection threshold."
        elif ground_truth_match is False:
            diagnosis = "Selected candidate did not match ground truth."
        else:
            diagnosis = "Recorded by policy (always_record=true)."

        failure = FailureRecord(
            task_id=task.task_id,
            selected_trace_id=decision.selected_trace_id,
            all_candidate_trace_ids=[t.trace_id for t in traces],
            violated_invariants=all_violated,
            disagreement_pattern=disagreement_pattern,
            diagnosis=diagnosis,
            ground_truth_match=ground_truth_match,
        )
        self._writer.write_failure(failure)
        log.info("Failure record written: %s", failure.failure_id[:8])
