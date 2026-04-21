"""SQLite schema models for the Epistemic Tribunal failure ledger.

We use plain dataclasses here to keep the ledger layer independent of
Pydantic — allowing it to be swapped to DuckDB or another backend easily.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class TaskRecord:
    task_id: str
    domain: str
    description: str
    train_examples_count: int
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class TraceRecord:
    trace_id: str
    task_id: str
    generator_name: str
    confidence_score: float
    answer_json: str          # JSON-serialised answer grid
    reasoning_steps_json: str  # JSON-serialised list of steps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class DecisionRecord:
    decision_id: str
    task_id: str
    decision: str             # select | resample | abstain
    selected_trace_id: str | None
    confidence: float
    reasoning: str
    scores_json: str          # JSON-serialised score dict
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class FailureRecordRow:
    failure_id: str
    task_id: str
    selected_trace_id: str | None
    all_candidate_trace_ids_json: str
    violated_invariants_json: str
    disagreement_pattern: str
    diagnosis: str
    notes: str
    ground_truth_match: int | None  # SQLite stores as 0/1/NULL
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class InvariantViolationRecord:
    violation_id: str
    task_id: str
    trace_id: str
    invariant_name: str
    note: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ExperimentRunRecord:
    run_id: str
    task_id: str
    decision: str
    confidence: float
    selected_trace_id: str | None
    ground_truth_match: int | None
    duration_seconds: float
    generator_names_json: str
    config_snapshot_json: str
    metadata_json: str = "{}"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CoalitionOpinionRecord:
    run_id: str
    task_id: str
    answer_signature: str
    coalition_member_trace_ids_json: str
    coalition_member_generators_json: str
    representative_trace_id: str | None
    representative_generator: str | None
    source_opinions_json: str
    generator_trust_opinion_json: str
    fused_opinion_json: str
    belief: float
    disbelief: float
    uncertainty: float
    base_rate: float
    expectation: float
    base_rate_contribution: float
    decision_role: str
    decision_reason_code: str
    decision_reason_text: str
    explanation_metadata_json: str = "{}"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
