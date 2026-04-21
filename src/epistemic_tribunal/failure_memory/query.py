"""Pre-adjudication failure-memory query.

Builds a :class:`FailureProbe` from the current pool state (observable-only
features) and queries the :class:`FailureMemoryStore` for similar past
failures.  Returns per-trace penalty scores that flow into the ``M`` weight
channel of the tribunal scoring formula.

Critical invariant: the probe and penalty computation must NEVER use
ground-truth information.  All features are derived from the candidate pool,
critiques, and uncertainty report — information available *before* the
tribunal makes its decision.
"""

from __future__ import annotations

from typing import Any

from epistemic_tribunal.failure_memory.models import FailureProbe
from epistemic_tribunal.failure_memory.store import FailureMemoryStore
from epistemic_tribunal.tribunal_types import (
    CandidateTrace,
    CritiqueResult,
    Task,
    UncertaintyReport,
)
from epistemic_tribunal.utils.logging import get_logger

log = get_logger(__name__)


class FailureMemoryQuery:
    """Pre-adjudication lookup: checks whether the current pool shape
    resembles past failures and returns per-trace penalty scores.

    Parameters
    ----------
    store:
        The :class:`FailureMemoryStore` to query.
    penalty_scale:
        Maximum total penalty (before capping).  Default ``0.3``.
    """

    def __init__(
        self, 
        store: FailureMemoryStore, 
        penalty_scale: float = 0.3,
        pattern_weights: dict[str, float] | None = None,
    ) -> None:
        self._store = store
        self._penalty_scale = penalty_scale
        self._pattern_weights = pattern_weights or {
            "false_majority": 1.5,
            "flat_critics": 0.8,
            "no_rationale": 0.6,
        }

    def build_probe(
        self,
        task: Task,
        traces: list[CandidateTrace],
        critiques: list[CritiqueResult],
        uncertainty: UncertaintyReport,
    ) -> FailureProbe:
        """Build a :class:`FailureProbe` from observable-only features.

        This method must NOT access ``task.ground_truth``.
        """
        from epistemic_tribunal.domains.factory import get_adapter

        adapter = get_adapter(task.domain)
        clusters = adapter.cluster_answers(traces)
        n_clusters = len(clusters)

        # Determine majority cluster
        majority_cluster = max(clusters, key=len) if clusters else []
        majority_ids = {t.trace_id for t in majority_cluster}

        # Per-candidate features (observable only)
        critique_by_id = {c.trace_id: c for c in critiques}
        candidate_features: dict[str, dict[str, Any]] = {}
        for trace in traces:
            cr = critique_by_id.get(trace.trace_id)
            candidate_features[trace.trace_id] = {
                "generator_name": trace.generator_name,
                "is_majority": trace.trace_id in majority_ids,
                "rationale_present": bool(trace.reasoning_steps),
                "reasoning_step_count": len(trace.reasoning_steps),
                "trace_length": trace.token_count or 0,
                "finish_reason": trace.metadata.get("finish_reason", "stop"),
                "critic_aggregate_score": cr.aggregate_score if cr else 0.0,
            }

        # Majority / minority rationale presence
        majority_has_rationale = all(
            candidate_features[tid]["rationale_present"]
            for tid in majority_ids
            if tid in candidate_features
        ) if majority_ids else True

        minority_ids = {t.trace_id for t in traces} - majority_ids
        minority_has_rationale = any(
            candidate_features[tid]["rationale_present"]
            for tid in minority_ids
            if tid in candidate_features
        ) if minority_ids else False

        # Check if all critics are flat
        if critiques:
            scores = [c.aggregate_score for c in critiques]
            all_critics_flat = len(set(round(s, 3) for s in scores)) == 1
        else:
            all_critics_flat = True

        # Structural margin from metadata will be computed during adjudication,
        # but at probe-build time we can approximate from uncertainty margin.
        structural_margin = uncertainty.margin

        return FailureProbe(
            domain=task.domain.value,
            n_candidates=len(traces),
            n_clusters=n_clusters,
            coalition_mass=uncertainty.coalition_mass,
            disagreement_rate=uncertainty.disagreement_rate,
            candidate_features=candidate_features,
            majority_has_rationale=majority_has_rationale,
            minority_has_rationale=minority_has_rationale,
            all_critics_flat=all_critics_flat,
            structural_margin=structural_margin,
        )

    def query_penalties(
        self,
        task: Task,
        traces: list[CandidateTrace],
        critiques: list[CritiqueResult],
        uncertainty: UncertaintyReport,
    ) -> dict[str, float]:
        """Query failure memory and return per-trace penalty scores.

        Returns a dict mapping ``trace_id`` \u2192 penalty (0.0\u20130.8).
        """
        penalties, _ = self._query_internal(task, traces, critiques, uncertainty)
        return penalties

    def query_with_metadata(
        self,
        task: Task,
        traces: list[CandidateTrace],
        critiques: list[CritiqueResult],
        uncertainty: UncertaintyReport,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Like :meth:`query_penalties` but also returns reporting metadata."""
        penalties, internal = self._query_internal(task, traces, critiques, uncertainty)

        exact_penalties = internal["exact_penalties"]
        structural_penalties = internal["structural_penalties"]
        exact_match_counts = internal["exact_match_counts"]
        structural_match_counts = internal["structural_match_counts"]
        top_structural_sims = internal["top_structural_sims"]
        matches = internal["matches"]
        answer_to_trace_ids = internal["answer_to_trace_ids"]

        n_penalised = sum(1 for p in penalties.values() if p > 0)
        bad_answers = {
            str(m.signature.answer_signature)
            for m in matches
            if m.signature.answer_signature and m.signature.failure_type.value == "wrong_pick"
        }
        # Only include bad answers that are actually present in the current pool
        bad_answers = {ans for ans in bad_answers if ans in answer_to_trace_ids}
        traces_affected_by_coalition = sum(
            len(answer_to_trace_ids.get(ans, set()))
            for ans in bad_answers
        )

        metadata = {
            "failure_memory_matches_found": len(matches),
            "failure_memory_candidates_penalised": n_penalised,
            "failure_memory_penalty_scale": self._penalty_scale,
            "failure_memory_top_match_similarity": (
                matches[0].similarity if matches else 0.0
            ),
            "failure_memory_top_match_features": (
                matches[0].matching_features if matches else []
            ),
            "failure_memory_penalties": {
                t.generator_name: penalties[t.trace_id]
                for t in traces
                if penalties[t.trace_id] > 0
            },
            "failure_memory_decomposition": {
                t.generator_name: {
                    "exact_penalty": round(exact_penalties[t.trace_id], 4),
                    "structural_penalty": round(structural_penalties[t.trace_id], 4),
                    "n_exact_matches": exact_match_counts[t.trace_id],
                    "n_structural_matches": structural_match_counts[t.trace_id],
                    "top_structural_similarity": round(top_structural_sims[t.trace_id], 4),
                    "final_total_penalty": penalties[t.trace_id],
                }
                for t in traces
                if penalties[t.trace_id] > 0
            },
            "failure_memory_trace_penalties": {
                t.trace_id: penalties[t.trace_id]
                for t in traces
                if penalties[t.trace_id] > 0
            },
            "failure_memory_trace_decomposition": {
                t.trace_id: {
                    "generator_name": t.generator_name,
                    "exact_penalty": round(exact_penalties[t.trace_id], 4),
                    "structural_penalty": round(structural_penalties[t.trace_id], 4),
                    "n_exact_matches": exact_match_counts[t.trace_id],
                    "n_structural_matches": structural_match_counts[t.trace_id],
                    "top_structural_similarity": round(top_structural_sims[t.trace_id], 4),
                    "final_total_penalty": penalties[t.trace_id],
                }
                for t in traces
                if penalties[t.trace_id] > 0
            },
            "failure_memory_coalitions_penalised": len(bad_answers),
            "failure_memory_traces_affected_by_coalition": traces_affected_by_coalition,
            "failure_memory_bad_answer_signatures": list(bad_answers),
        }
        return penalties, metadata

    def _query_internal(
        self,
        task: Task,
        traces: list[CandidateTrace],
        critiques: list[CritiqueResult],
        uncertainty: UncertaintyReport,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Internal shared query logic implementing Max-over-Sum aggregation.

        Rules:
        - Exact answer-signature matches: SUMMED (multiple failures against one answer stack)
        - Structural-pattern matches: MAXED (repeated warnings about situation shape don't stack)
        """
        probe = self.build_probe(task, traces, critiques, uncertainty)
        matches = self._store.query_similar(probe)

        if not matches:
            return {t.trace_id: 0.0 for t in traces}, {
                "exact_penalties": {},
                "structural_penalties": {},
                "exact_match_counts": {},
                "structural_match_counts": {},
                "top_structural_sims": {},
                "matches": [],
                "probe": probe,
                "answer_to_trace_ids": {},
            }

        from epistemic_tribunal.domains.factory import get_adapter

        adapter = get_adapter(task.domain)
        answer_to_trace_ids: dict[str, set[str]] = {}
        for trace in traces:
            key = str(adapter.get_cluster_key(trace.answer))
            answer_to_trace_ids.setdefault(key, set()).add(trace.trace_id)

        exact_penalties: dict[str, float] = {t.trace_id: 0.0 for t in traces}
        structural_penalties: dict[str, float] = {t.trace_id: 0.0 for t in traces}
        exact_match_counts: dict[str, int] = {t.trace_id: 0 for t in traces}
        structural_match_counts: dict[str, int] = {t.trace_id: 0 for t in traces}
        top_structural_sims: dict[str, float] = {t.trace_id: 0.0 for t in traces}

        for match in matches:
            # ------------------------------------------------------------------
            # Class A: Exact-answer penalty (Summed)
            # ------------------------------------------------------------------
            if (
                match.signature.failure_type.value == "wrong_pick"
                and match.signature.answer_signature
            ):
                bad_answer = str(match.signature.answer_signature)
                affected_ids = answer_to_trace_ids.get(bad_answer, set())
                if affected_ids:
                    p = match.similarity * self._penalty_scale * 0.8
                    for tid in affected_ids:
                        exact_penalties[tid] += p
                        exact_match_counts[tid] += 1

            # ------------------------------------------------------------------
            # Class B: Structural patterns (Maxed)
            # ------------------------------------------------------------------
            pattern_match_val = 0.0
            if "minority_rationale_false_majority_prior" in match.matching_features:
                pattern_match_val = max(
                    pattern_match_val, 
                    match.similarity * self._penalty_scale * self._pattern_weights.get("false_majority", 1.2)
                )
            if (
                "both_critics_flat" in match.matching_features
                and "high_disagreement" in match.matching_features
            ):
                pattern_match_val = max(
                    pattern_match_val, 
                    match.similarity * self._penalty_scale * self._pattern_weights.get("flat_critics", 0.8)
                )
            if "majority_lacks_rationale" in match.matching_features:
                pattern_match_val = max(
                    pattern_match_val, 
                    match.similarity * self._penalty_scale * self._pattern_weights.get("no_rationale", 0.6)
                )

            if pattern_match_val > 0:
                for trace in traces:
                    cf = probe.candidate_features.get(trace.trace_id, {})
                    if cf.get("is_majority"):
                        structural_penalties[trace.trace_id] = max(
                            structural_penalties[trace.trace_id], pattern_match_val
                        )
                        structural_match_counts[trace.trace_id] += 1
                        top_structural_sims[trace.trace_id] = max(
                            top_structural_sims[trace.trace_id], match.similarity
                        )

        penalties: dict[str, float] = {}
        for tid in [t.trace_id for t in traces]:
            # Aggregate buckets and cap at 0.8
            total = exact_penalties[tid] + structural_penalties[tid]
            penalties[tid] = round(min(0.8, total), 4)

        if any(p > 0 for p in penalties.values()):
            log.info(
                "Failure memory penalties applied for task %s: %s",
                task.task_id,
                {
                    t.generator_name: penalties[t.trace_id]
                    for t in traces
                    if penalties[t.trace_id] > 0
                },
            )

        return penalties, {
            "exact_penalties": exact_penalties,
            "structural_penalties": structural_penalties,
            "exact_match_counts": exact_match_counts,
            "structural_match_counts": structural_match_counts,
            "top_structural_sims": top_structural_sims,
            "matches": matches,
            "probe": probe,
            "answer_to_trace_ids": answer_to_trace_ids,
        }
