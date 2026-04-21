"""Typed models for the failure-memory protocol.

Two key distinctions:

* ``FailureSignature`` — **retrospective**, written after ground truth is known.
  Uses outcome labels like ``wrong_pick``, ``bad_abstention``, etc.

* ``FailureProbe`` — **online**, built before adjudication from observable-only
  features (coalition shape, rationale presence, trace length, disagreement
  pattern).  Does NOT use ground-truth labels.

* ``FailureMatch`` — a stored signature plus a similarity score returned by
  the memory store during lookup.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class FailureType(str, Enum):
    """Outcome-labelled failure type (retrospective)."""

    WRONG_PICK = "wrong_pick"
    BAD_ABSTENTION = "bad_abstention"
    GOOD_ABSTENTION = "good_abstention"
    CORRECT_SELECT = "correct_select"


# ---------------------------------------------------------------------------
# Retrospective signature (written post-evaluation)
# ---------------------------------------------------------------------------


class FailureSignature(BaseModel):
    """A labelled record of what happened on a tribunal run.

    Written *after* ground truth is known.  Fields that depend on ground truth
    (``failure_type``, ``minority_correct``, ``false_majority``) are filled by
    the extractor only because the outcome is already resolved.
    """

    signature_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    domain: str
    task_id: str
    failure_type: FailureType

    # Answer that was selected / abstained on
    answer_signature: str = ""

    # Pool-level context (observable retrospectively)
    coalition_context: dict[str, Any] = Field(default_factory=dict)
    # Typical keys: majority_size, minority_correct, false_majority,
    #               n_clusters, coalition_mass

    # Trace-quality features of the selected / top candidate
    trace_quality_features: dict[str, Any] = Field(default_factory=dict)
    # Typical keys: rationale_present, reasoning_step_count, trace_length,
    #               finish_reason

    # Critic context
    critic_context: dict[str, Any] = Field(default_factory=dict)
    # Typical keys: aggregate_score, judge_scores

    # Pool-level uncertainty
    disagreement_rate: float = 0.0
    structural_margin: float = 0.0

    # Human-readable summary
    outcome_label: str = ""

    # Domain-specific extension slot
    domain_features: dict[str, Any] = Field(default_factory=dict)

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ---------------------------------------------------------------------------
# Online probe (built before adjudication, observable-only)
# ---------------------------------------------------------------------------


class FailureProbe(BaseModel):
    """Observable pool shape used to query failure memory *before* adjudication.

    Critical invariant: this must NEVER contain ground-truth-derived fields.
    It can only use features visible from the candidate pool, critiques,
    and uncertainty report.
    """

    domain: str

    # Coalition / disagreement shape
    n_candidates: int = 0
    n_clusters: int = 0
    coalition_mass: float = 0.0
    disagreement_rate: float = 0.0

    # Per-candidate observable features
    # Key = trace_id, value = dict of observable features
    candidate_features: dict[str, dict[str, Any]] = Field(default_factory=dict)
    # Each dict has: generator_name, is_majority, rationale_present,
    #                reasoning_step_count, trace_length, finish_reason,
    #                critic_aggregate_score

    # Pool-level observable patterns
    majority_has_rationale: bool = True
    minority_has_rationale: bool = False
    all_critics_flat: bool = False
    structural_margin: float = 0.0


# ---------------------------------------------------------------------------
# Match result from memory lookup
# ---------------------------------------------------------------------------


class FailureMatch(BaseModel):
    """A past failure signature plus its similarity score to the current probe."""

    signature: FailureSignature
    similarity: float = Field(ge=0.0, le=1.0)
    matching_features: list[str] = Field(default_factory=list)
