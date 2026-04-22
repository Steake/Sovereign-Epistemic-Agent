"""Strange Loop Memory — pre-generation constraint builder.

Queries the :class:`FailureMemoryStore` **before** generators run and
produces structured negative guidance that generators can inject into
their prompts.  This is the v1 implementation of the Strange Loop
architecture described in the project roadmap.

The constraint builder intentionally avoids ground-truth leakage:
all constraints are derived from *past* tribunal runs where ground truth
was known at write time, but the *current* run's ground truth is never
consulted.
"""

from __future__ import annotations

from epistemic_tribunal.failure_memory.models import (
    FailureConstraints,
    FailureSignature,
    FailureType,
)
from epistemic_tribunal.failure_memory.store import FailureMemoryStore
from epistemic_tribunal.tribunal_types import Task
from epistemic_tribunal.utils.logging import get_logger

log = get_logger(__name__)


class FailureConstraintBuilder:
    """Builds pre-generation constraints from failure memory.

    Parameters
    ----------
    store:
        The :class:`FailureMemoryStore` to query.
    max_bad_answers:
        Maximum number of bad-answer signatures to inject.
    max_warnings:
        Maximum number of structural warnings to inject.
    min_similarity:
        Minimum match similarity to include a failure signature.
    same_task_boost:
        Similarity multiplier for exact task_id matches.
    """

    def __init__(
        self,
        store: FailureMemoryStore,
        max_bad_answers: int = 5,
        max_warnings: int = 3,
        min_similarity: float = 0.3,
        same_task_boost: float = 1.5,
    ) -> None:
        self._store = store
        self._max_bad_answers = max_bad_answers
        self._max_warnings = max_warnings
        self._min_similarity = min_similarity
        self._same_task_boost = same_task_boost

    def build(self, task: Task, mode: str = "full_memory") -> FailureConstraints:
        """Query failure memory and produce constraints for generators.

        Parameters
        ----------
        task:
            The task about to be solved.  Used to match against prior
            failure signatures by task_id and domain.
        mode:
            "off" | "bad_answers_only" | "warnings_only" | "full_memory"

        Returns
        -------
        FailureConstraints
            Structured negative guidance.  Empty (``has_constraints == False``)
            if no relevant failures are found.
        """
        if mode == "off":
            return FailureConstraints()

        signatures = self._query_relevant_failures(task)
        if not signatures:
            return FailureConstraints()

        bad_answers = self._extract_bad_answers(signatures, task)
        warnings = self._extract_structural_warnings(signatures, task)
        
        if mode == "warnings_only":
            bad_answers = []
        elif mode == "bad_answers_only":
            warnings = []

        source_task_ids = list({sig.task_id for _, sig in signatures})

        # Compute overall constraint strength from the best match quality
        best_similarity = max(sim for sim, _ in signatures)
        constraint_strength = round(min(1.0, best_similarity), 4)

        constraints = FailureConstraints(
            bad_answers=bad_answers[:self._max_bad_answers],
            structural_warnings=warnings[:self._max_warnings],
            source_task_ids=source_task_ids,
            constraint_strength=constraint_strength,
            metadata={
                "n_signatures_queried": len(signatures),
                "best_similarity": best_similarity,
                "same_task_matches": sum(
                    1 for _, sig in signatures if sig.task_id == task.task_id
                ),
            },
        )

        if constraints.has_constraints:
            log.info(
                "Strange Loop (%s): injecting %d bad-answer(s) and %d warning(s) "
                "for task %s (strength=%.3f, from %d prior failure(s))",
                mode,
                len(constraints.bad_answers),
                len(constraints.structural_warnings),
                task.task_id,
                constraint_strength,
                len(signatures),
            )

        return constraints

    # ------------------------------------------------------------------
    # Internal query
    # ------------------------------------------------------------------

    def _query_relevant_failures(
        self, task: Task
    ) -> list[tuple[float, FailureSignature]]:
        """Retrieve and score relevant past failure signatures.

        Returns a list of ``(effective_similarity, signature)`` pairs,
        sorted by effective similarity descending.
        """
        all_sigs = self._store.get_all(domain=task.domain.value)

        # Filter to actual failures (wrong_pick, bad_abstention)
        failure_sigs = [
            sig
            for sig in all_sigs
            if sig.failure_type in (FailureType.WRONG_PICK, FailureType.BAD_ABSTENTION)
        ]

        if not failure_sigs:
            return []

        scored: list[tuple[float, FailureSignature]] = []
        for sig in failure_sigs:
            similarity = self._compute_task_similarity(task, sig)

            # Boost exact task_id matches (replay scenario)
            if sig.task_id == task.task_id:
                similarity = min(1.0, similarity * self._same_task_boost)

            if similarity >= self._min_similarity:
                scored.append((round(similarity, 4), sig))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    @staticmethod
    def _compute_task_similarity(task: Task, sig: FailureSignature) -> float:
        """Compute a lightweight similarity score between the current task
        and a stored failure signature.

        Uses observable task-level features only (domain, task_id pattern).
        """
        score = 0.0
        features_checked = 0

        # 1. Same domain (required — already filtered, but explicit)
        features_checked += 1
        if task.domain.value == sig.domain:
            score += 1.0

        # 2. Exact task_id match (strongest signal)
        features_checked += 1
        if task.task_id == sig.task_id:
            score += 1.0

        # 3. Similar disagreement regime
        features_checked += 1
        if sig.disagreement_rate > 0.3:
            score += 0.5  # Prior failure was contested — relevant context

        # 4. High coalition context (false majority)
        features_checked += 1
        if sig.coalition_context.get("false_majority", False):
            score += 0.5

        return round(score / max(features_checked, 1), 4)

    # ------------------------------------------------------------------
    # Constraint extraction
    # ------------------------------------------------------------------

    def _extract_bad_answers(
        self,
        signatures: list[tuple[float, FailureSignature]],
        task: Task,
    ) -> list[str]:
        """Extract bad-answer signatures, prioritising exact task matches."""
        seen: set[str] = set()
        bad_answers: list[str] = []

        for _, sig in signatures:
            if sig.failure_type != FailureType.WRONG_PICK:
                continue
            ans_sig = sig.answer_signature
            if not ans_sig or ans_sig in seen:
                continue
            seen.add(ans_sig)
            bad_answers.append(ans_sig)

        return bad_answers

    def _extract_structural_warnings(
        self,
        signatures: list[tuple[float, FailureSignature]],
        task: Task,
    ) -> list[str]:
        """Derive natural-language warnings from structural failure patterns."""
        warnings: list[str] = []
        seen_codes: set[str] = set()

        for _, sig in signatures:
            coal = sig.coalition_context

            # Pattern: false majority — the majority answer was wrong
            if coal.get("false_majority") and "false_majority" not in seen_codes:
                seen_codes.add("false_majority")
                warnings.append(
                    "On a similar prior task, the majority/most-common answer "
                    "was WRONG. Do not blindly follow the most obvious pattern."
                )

            # Pattern: minority had the correct answer
            if coal.get("minority_correct") and "minority_correct" not in seen_codes:
                seen_codes.add("minority_correct")
                warnings.append(
                    "On a similar prior task, a less obvious minority answer "
                    "turned out to be correct. Consider alternative interpretations."
                )

            # Pattern: selected trace lacked reasoning
            tq = sig.trace_quality_features
            if (
                sig.failure_type == FailureType.WRONG_PICK
                and not tq.get("rationale_present", True)
                and "no_rationale_selected" not in seen_codes
            ):
                seen_codes.add("no_rationale_selected")
                warnings.append(
                    "On a similar prior task, an answer without explicit "
                    "reasoning was selected and turned out wrong. Show your "
                    "reasoning explicitly."
                )

            # Pattern: parse contamination present
            if (
                coal.get("parse_issue_present")
                and "parse_contamination" not in seen_codes
            ):
                seen_codes.add("parse_contamination")
                warnings.append(
                    "Prior attempts on similar tasks had parse/truncation "
                    "issues. Ensure your output is clean and complete."
                )

        return warnings
