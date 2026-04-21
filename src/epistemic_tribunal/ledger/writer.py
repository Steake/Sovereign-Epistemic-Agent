"""Ledger writer — high-level helper that persists tribunal results.

Converts Pydantic domain objects into ledger records and writes them to the
:class:`LedgerStore`.
"""

from __future__ import annotations

import json
import uuid

from epistemic_tribunal.ledger.models import (
    CoalitionOpinionRecord,
    DecisionRecord,
    ExperimentRunRecord,
    FailureRecordRow,
    InvariantViolationRecord,
    TaskRecord,
    TraceRecord,
)
from epistemic_tribunal.ledger.store import LedgerStore
from epistemic_tribunal.tribunal_types import (
    CandidateTrace,
    CritiqueResult,
    ExperimentRun,
    FailureRecord,
    Task,
    TribunalDecision,
)
from epistemic_tribunal.utils.logging import get_logger

log = get_logger(__name__)


class LedgerWriter:
    """High-level writer that converts domain objects → ledger records.

    Parameters
    ----------
    store:
        Backing :class:`LedgerStore` instance.
    """

    def __init__(self, store: LedgerStore) -> None:
        self._store = store

    # ------------------------------------------------------------------
    # Public write methods
    # ------------------------------------------------------------------

    def write_task(self, task: Task) -> None:
        """Persist a task record."""
        rec = TaskRecord(
            task_id=task.task_id,
            domain=task.domain.value,
            description=task.description,
            train_examples_count=len(task.train),
        )
        self._store.upsert_task(rec)

    def write_traces(self, task: Task, traces: list[CandidateTrace]) -> None:
        """Persist all candidate traces for a task."""
        for trace in traces:
            rec = TraceRecord(
                trace_id=trace.trace_id,
                task_id=task.task_id,
                generator_name=trace.generator_name,
                confidence_score=trace.confidence_score or 0.0,
                answer_json=json.dumps(trace.answer),
                reasoning_steps_json=json.dumps(trace.reasoning_steps),
            )
            self._store.insert_trace(rec)

    def write_decision(self, decision: TribunalDecision) -> None:
        """Persist a tribunal decision."""
        rec = DecisionRecord(
            decision_id=str(uuid.uuid4()),
            task_id=decision.task_id,
            decision=decision.decision.value,
            selected_trace_id=decision.selected_trace_id,
            confidence=decision.confidence,
            reasoning=decision.reasoning,
            scores_json=json.dumps(decision.scores),
        )
        self._store.insert_decision(rec)

    def write_failure(self, failure: FailureRecord) -> None:
        """Persist a failure record."""
        rec = FailureRecordRow(
            failure_id=failure.failure_id,
            task_id=failure.task_id,
            selected_trace_id=failure.selected_trace_id,
            all_candidate_trace_ids_json=json.dumps(failure.all_candidate_trace_ids),
            violated_invariants_json=json.dumps(failure.violated_invariants),
            disagreement_pattern=failure.disagreement_pattern,
            diagnosis=failure.diagnosis,
            notes=failure.notes,
            ground_truth_match=(
                1 if failure.ground_truth_match is True
                else 0 if failure.ground_truth_match is False
                else None
            ),
        )
        self._store.insert_failure(rec)

    def write_invariant_violations(
        self,
        task: Task,
        trace: CandidateTrace,
        critique: CritiqueResult,
    ) -> None:
        """Persist invariant violations found by the critic."""
        for inv_name in critique.violated_invariants:
            rec = InvariantViolationRecord(
                violation_id=str(uuid.uuid4()),
                task_id=task.task_id,
                trace_id=trace.trace_id,
                invariant_name=inv_name,
                note=critique.notes,
            )
            self._store.insert_invariant_violation(rec)

    def write_run(self, run: ExperimentRun) -> None:
        """Persist an experiment run record."""
        rec = ExperimentRunRecord(
            run_id=run.run_id,
            task_id=run.task_id,
            decision=run.decision.value,
            confidence=run.confidence,
            selected_trace_id=run.selected_trace_id,
            ground_truth_match=(
                1 if run.ground_truth_match is True
                else 0 if run.ground_truth_match is False
                else None
            ),
            duration_seconds=run.duration_seconds,
            generator_names_json=json.dumps(run.generator_names),
            config_snapshot_json=json.dumps(run.config_snapshot),
            metadata_json=json.dumps(run.metadata),
        )
        self._store.insert_run(rec)

    def write_coalition_opinions(
        self,
        *,
        run_id: str,
        task_id: str,
        coalition_rows: list[dict],
    ) -> None:
        """Persist EQBSL coalition-opinion forensic rows."""
        for row in coalition_rows:
            rec = CoalitionOpinionRecord(
                run_id=run_id,
                task_id=task_id,
                answer_signature=row["answer_signature"],
                coalition_member_trace_ids_json=json.dumps(row.get("coalition_member_trace_ids", [])),
                coalition_member_generators_json=json.dumps(row.get("coalition_member_generators", [])),
                representative_trace_id=row.get("representative_trace_id"),
                representative_generator=row.get("representative_generator"),
                source_opinions_json=json.dumps(row.get("source_opinions", {})),
                generator_trust_opinion_json=json.dumps(row.get("generator_trust_opinion")),
                fused_opinion_json=json.dumps(row.get("fused_opinion", {})),
                belief=float(row.get("fused_opinion", {}).get("belief", 0.0)),
                disbelief=float(row.get("fused_opinion", {}).get("disbelief", 0.0)),
                uncertainty=float(row.get("fused_opinion", {}).get("uncertainty", 1.0)),
                base_rate=float(row.get("fused_opinion", {}).get("base_rate", 0.25)),
                expectation=float(row.get("final_expectation", 0.0)),
                base_rate_contribution=float(row.get("base_rate_contribution", 0.0)),
                decision_role=row.get("decision_role", "other"),
                decision_reason_code=row.get("decision_reason_code", ""),
                decision_reason_text=row.get("decision_reason_text", ""),
                explanation_metadata_json=json.dumps(row.get("explanation_metadata", {})),
            )
            self._store.insert_coalition_opinion(rec)
