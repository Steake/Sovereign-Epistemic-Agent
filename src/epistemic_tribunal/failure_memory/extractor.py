"""Post-evaluation failure-signature extractor.

Runs *after* ground truth is known.  Produces a :class:`FailureSignature`
for every tribunal run (not just failures — correct selections are also
stored so the memory can learn positive patterns).

Domain-specific feature extraction is delegated to lightweight adapter
functions so the core logic stays generic.
"""

from __future__ import annotations

from typing import Any, Optional

from epistemic_tribunal.failure_memory.models import FailureSignature, FailureType
from epistemic_tribunal.tribunal_types import (
    CandidateTrace,
    CritiqueResult,
    DecisionKind,
    Task,
    TribunalDecision,
    UncertaintyReport,
)
from epistemic_tribunal.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Domain-specific feature adapters
# ---------------------------------------------------------------------------


def _gsm8k_features(
    task: Task,
    traces: list[CandidateTrace],
    decision: TribunalDecision,
    any_correct: Optional[bool],
) -> dict[str, Any]:
    """Extract GSM8K-specific features for the failure signature."""
    features: dict[str, Any] = {}

    # Check if any candidate has reasoning steps
    has_reasoning = [bool(t.reasoning_steps) for t in traces]
    features["any_trace_has_reasoning"] = any(has_reasoning)
    features["all_traces_have_reasoning"] = all(has_reasoning) if traces else False

    # If selected, check the selected trace's reasoning
    if decision.selected_trace_id:
        selected = next(
            (t for t in traces if t.trace_id == decision.selected_trace_id), None
        )
        if selected:
            features["selected_has_reasoning"] = bool(selected.reasoning_steps)
            features["selected_reasoning_steps"] = len(selected.reasoning_steps)

    return features


def _arc_features(
    task: Task,
    traces: list[CandidateTrace],
    decision: TribunalDecision,
    any_correct: Optional[bool],
) -> dict[str, Any]:
    """Extract ARC-specific features (passthrough for now)."""
    return {
        "grid_domain": True,
    }


_DOMAIN_EXTRACTORS = {
    "gsm8k_math": _gsm8k_features,
    "arc_like": _arc_features,
}


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------


class FailureSignatureExtractor:
    """Extracts a :class:`FailureSignature` from a completed tribunal run.

    This is a **retrospective** operation: it uses ground-truth outcome labels
    to classify the signature.  The resulting signature is stored in persistent
    memory for future *online* lookup via :class:`FailureProbe`.
    """

    def extract(
        self,
        task: Task,
        traces: list[CandidateTrace],
        critiques: list[CritiqueResult],
        decision: TribunalDecision,
        uncertainty: UncertaintyReport,
        ground_truth_match: Optional[bool],
        any_correct: Optional[bool],
    ) -> Optional[FailureSignature]:
        """Build a failure signature from a completed run.

        Returns ``None`` if there is insufficient data (e.g. no ground truth).
        """
        if ground_truth_match is None and any_correct is None:
            return None  # No ground truth → nothing to label

        # ----- Determine failure type -----
        if decision.decision == DecisionKind.SELECT:
            if ground_truth_match is True:
                failure_type = FailureType.CORRECT_SELECT
            else:
                failure_type = FailureType.WRONG_PICK
        else:
            # Abstain or Resample
            if any_correct is True:
                failure_type = FailureType.BAD_ABSTENTION
            else:
                failure_type = FailureType.GOOD_ABSTENTION

        # ----- Coalition context (retrospective — knows ground truth) -----
        from epistemic_tribunal.domains.factory import get_adapter

        adapter = get_adapter(task.domain)
        clusters = adapter.cluster_answers(traces)
        n_clusters = len(clusters)
        cluster_sizes = sorted([len(c) for c in clusters], reverse=True)
        majority_size = cluster_sizes[0] if cluster_sizes else 0

        # Determine if the majority answer is correct
        false_majority = False
        minority_correct = False
        if task.ground_truth is not None and clusters:
            majority_cluster = max(clusters, key=len)
            majority_answer = majority_cluster[0].answer
            majority_is_correct = adapter.answers_equal(
                majority_answer, task.ground_truth
            )
            if not majority_is_correct and majority_size > 1:
                false_majority = True
            # Check if any minority cluster has the correct answer
            for cluster in clusters:
                if len(cluster) < majority_size:
                    if adapter.answers_equal(cluster[0].answer, task.ground_truth):
                        minority_correct = True
                        break

        coalition_context = {
            "majority_size": majority_size,
            "n_clusters": n_clusters,
            "coalition_mass": uncertainty.coalition_mass,
            "false_majority": false_majority,
            "minority_correct": minority_correct,
            "total_candidates": len(traces),
        }

        # ----- Trace quality features (of the selected / top candidate) -----
        selected_trace = None
        if decision.selected_trace_id:
            selected_trace = next(
                (t for t in traces if t.trace_id == decision.selected_trace_id), None
            )

        trace_quality: dict[str, Any] = {}
        if selected_trace:
            trace_quality = {
                "rationale_present": bool(selected_trace.reasoning_steps),
                "reasoning_step_count": len(selected_trace.reasoning_steps),
                "trace_length": selected_trace.token_count or 0,
                "finish_reason": selected_trace.metadata.get("finish_reason", "unknown"),
                "generator_name": selected_trace.generator_name,
            }
        else:
            # Abstention — use the top-scoring candidate from forensic
            forensic = decision.metadata.get("forensic", [])
            if forensic:
                top = forensic[0]
                trace_quality = {
                    "rationale_present": False,  # Unknown without the trace
                    "reasoning_step_count": 0,
                    "trace_length": top.get("trace_length", 0),
                    "finish_reason": top.get("finish_reason", "unknown"),
                    "generator_name": top.get("generator", "unknown"),
                }

        # Check if oracle-best trace has rationale (retrospective)
        oracle_has_rationale = False
        if task.ground_truth is not None:
            for t in traces:
                if adapter.answers_equal(t.answer, task.ground_truth):
                    if t.reasoning_steps:
                        oracle_has_rationale = True
                        break
        trace_quality["oracle_has_rationale"] = oracle_has_rationale

        # ----- Critic context -----
        critique_by_id = {c.trace_id: c for c in critiques}
        critic_ctx: dict[str, Any] = {}
        if selected_trace and selected_trace.trace_id in critique_by_id:
            c = critique_by_id[selected_trace.trace_id]
            critic_ctx["aggregate_score"] = c.aggregate_score
            critic_ctx["consistency_score"] = c.consistency_score
        # Check if all critics are flat (same score)
        if critiques:
            scores = [c.aggregate_score for c in critiques]
            critic_ctx["all_flat"] = len(set(round(s, 3) for s in scores)) == 1

        # ----- Answer signature -----
        answer_sig = ""
        if decision.selected_answer is not None:
            answer_sig = str(adapter.get_cluster_key(decision.selected_answer))

        # ----- Outcome label -----
        outcome_label = failure_type.value
        if false_majority:
            outcome_label += " | false_majority"
        if minority_correct:
            outcome_label += " | minority_correct"

        # ----- Domain-specific features -----
        domain_str = task.domain.value
        domain_extractor = _DOMAIN_EXTRACTORS.get(domain_str)
        domain_features = {}
        if domain_extractor:
            domain_features = domain_extractor(task, traces, decision, any_correct)

        # ----- Parse contamination -----
        parse_issues = any(
            t.metadata.get("parse_failure") or t.metadata.get("truncated")
            for t in traces
        )
        coalition_context["parse_issue_present"] = parse_issues

        sig = FailureSignature(
            domain=domain_str,
            task_id=task.task_id,
            failure_type=failure_type,
            answer_signature=answer_sig,
            coalition_context=coalition_context,
            trace_quality_features=trace_quality,
            critic_context=critic_ctx,
            disagreement_rate=uncertainty.disagreement_rate,
            structural_margin=decision.metadata.get("structural_margin", 0.0),
            outcome_label=outcome_label,
            domain_features=domain_features,
        )

        log.debug(
            "Extracted failure signature %s: type=%s, task=%s",
            sig.signature_id[:8],
            failure_type.value,
            task.task_id,
        )
        return sig
