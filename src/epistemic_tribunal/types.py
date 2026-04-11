"""Typed domain models for the Epistemic Tribunal architecture.

All data flowing through the pipeline is represented as Pydantic models.
This module is the single source of truth for shared types.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class DecisionKind(str, Enum):
    """Possible outcomes from the tribunal aggregator."""

    SELECT = "select"
    RESAMPLE = "resample"
    ABSTAIN = "abstain"


class TaskDomain(str, Enum):
    """High-level task domain."""

    ARC_LIKE = "arc_like"
    GENERIC = "generic"


# ---------------------------------------------------------------------------
# Core task models
# ---------------------------------------------------------------------------


class GridExample(BaseModel):
    """A single input/output grid example (ARC-like)."""

    input: list[list[int]] = Field(..., description="2D input grid")
    output: list[list[int]] = Field(..., description="2D output grid")


class Task(BaseModel):
    """A reasoning task to be evaluated by the tribunal."""

    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    domain: TaskDomain = Field(default=TaskDomain.ARC_LIKE)
    description: str = Field(default="", description="Human-readable task description")
    train: list[GridExample] = Field(default_factory=list)
    test_input: list[list[int]] = Field(..., description="Test grid input")
    ground_truth: Optional[list[list[int]]] = Field(
        default=None, description="Expected output (may be absent)"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Candidate reasoning trace
# ---------------------------------------------------------------------------


class CandidateTrace(BaseModel):
    """A single reasoning trace produced by a generator."""

    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    generator_name: str
    answer: list[list[int]] = Field(description="Predicted output grid")
    reasoning_steps: list[str] = Field(default_factory=list)
    raw_trace: str = Field(default="")
    token_count: Optional[int] = None
    confidence_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Generator-reported confidence"
    )
    derived_features: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Invariant models
# ---------------------------------------------------------------------------


class Invariant(BaseModel):
    """A single inferred task-level constraint."""

    name: str
    description: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    notes: str = ""


class InvariantSet(BaseModel):
    """Collection of invariants inferred from a task."""

    task_id: str
    invariants: list[Invariant] = Field(default_factory=list)
    extraction_notes: str = ""

    def by_name(self, name: str) -> Optional[Invariant]:
        """Return the invariant with the given name, or None."""
        for inv in self.invariants:
            if inv.name == name:
                return inv
        return None


# ---------------------------------------------------------------------------
# Critique result
# ---------------------------------------------------------------------------


class CritiqueResult(BaseModel):
    """Scores and notes produced by the TraceCritic for one candidate."""

    trace_id: str
    consistency_score: float = Field(ge=0.0, le=1.0)
    rule_coherence_score: float = Field(ge=0.0, le=1.0)
    morphology_score: float = Field(ge=0.0, le=1.0)
    failure_similarity_penalty: float = Field(ge=0.0, le=1.0)
    invariant_compliance_score: float = Field(ge=0.0, le=1.0)
    aggregate_score: float = Field(ge=0.0, le=1.0)
    violated_invariants: list[str] = Field(default_factory=list)
    notes: str = ""


# ---------------------------------------------------------------------------
# Uncertainty report
# ---------------------------------------------------------------------------


class UncertaintyReport(BaseModel):
    """Uncertainty signals computed across the generator bank."""

    entropy: float = Field(ge=0.0, description="Answer-distribution entropy")
    margin: float = Field(
        ge=0.0, le=1.0, description="Score gap between top-2 candidates"
    )
    coalition_mass: float = Field(
        ge=0.0, le=1.0, description="Fraction of generators agreeing with top candidate"
    )
    disagreement_rate: float = Field(
        ge=0.0, le=1.0, description="Fraction of generator pairs that disagree"
    )
    per_trace_quality: dict[str, float] = Field(
        default_factory=dict, description="Normalised quality score per trace_id"
    )
    notes: str = ""


# ---------------------------------------------------------------------------
# Tribunal decision
# ---------------------------------------------------------------------------


class TribunalDecision(BaseModel):
    """The final decision produced by the tribunal aggregator."""

    task_id: str
    decision: DecisionKind
    selected_trace_id: Optional[str] = None
    selected_answer: Optional[list[list[int]]] = None
    scores: dict[str, float] = Field(
        default_factory=dict, description="Per-trace aggregate scores"
    )
    reasoning: str = ""
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Failure record
# ---------------------------------------------------------------------------


class FailureRecord(BaseModel):
    """Structured failure record written to the persistent ledger."""

    failure_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str
    selected_trace_id: Optional[str] = None
    all_candidate_trace_ids: list[str] = Field(default_factory=list)
    violated_invariants: list[str] = Field(default_factory=list)
    disagreement_pattern: str = ""
    diagnosis: str = ""
    notes: str = ""
    ground_truth_match: Optional[bool] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Experiment run
# ---------------------------------------------------------------------------


class ExperimentRun(BaseModel):
    """Metadata for one end-to-end tribunal run."""

    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str
    generator_names: list[str] = Field(default_factory=list)
    decision: DecisionKind
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    selected_trace_id: Optional[str] = None
    ground_truth_match: Optional[bool] = None
    duration_seconds: float = 0.0
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
