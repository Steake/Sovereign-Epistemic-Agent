"""TraceCritic — scores candidate traces on multiple quality dimensions.

Scoring dimensions
------------------
1. **internal_consistency** — how self-consistent are the reasoning steps?
2. **rule_coherence** — do the stated rules agree with the answer?
3. **morphology_quality** — does the answer look structurally plausible?
4. **failure_similarity_penalty** — how similar is this trace to known bad ones?
5. **invariant_compliance** — how many inferred invariants does it satisfy?

Because we operate without an LLM, all scores are heuristic proxies computed
from structural features of the trace and answer grid.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from openai import OpenAI

from epistemic_tribunal.critics.base import BaseCritic
from epistemic_tribunal.invariants.extractor import InvariantExtractor
from epistemic_tribunal.utils.logging import get_logger
from epistemic_tribunal.tasks.base import (
    grid_shape,
    grid_similarity,
    unique_colours,
)
from epistemic_tribunal.tribunal_types import (
    CandidateTrace,
    CritiqueResult,
    InvariantSet,
    Task,
)


log = get_logger(__name__)


class TraceCritic(BaseCritic):
    """Heuristic trace critic for ARC-like tasks."""

    name = "trace_critic"

    def __init__(
        self,
        consistency_weight: float = 0.30,
        rule_coherence_weight: float = 0.25,
        morphology_weight: float = 0.25,
        failure_similarity_weight: float = 0.20,
        use_llm_judge_for_math: bool = False,
    ) -> None:
        total = consistency_weight + rule_coherence_weight + morphology_weight + failure_similarity_weight
        self._w_cons = consistency_weight / total
        self._w_rule = rule_coherence_weight / total
        self._w_morph = morphology_weight / total
        self._w_fail = failure_similarity_weight / total
        self._use_llm_judge = use_llm_judge_for_math
        self._extractor = InvariantExtractor()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def critique(
        self,
        task: Task,
        trace: CandidateTrace,
        invariant_set: Optional[InvariantSet] = None,
        ledger_failure_patterns: Optional[list[dict]] = None,
    ) -> CritiqueResult:
        """Score *trace* on all quality dimensions."""

        consistency = self._score_consistency(trace)
        rule_coherence = self._score_rule_coherence(task, trace)
        morphology = self._score_morphology(task, trace)
        failure_penalty = self._score_failure_similarity(trace, ledger_failure_patterns or [])

        # Invariant compliance
        violated: list[str] = []
        invariant_score = 1.0
        if invariant_set is not None:
            inv_results = self._extractor.check_candidate(
                task, trace.answer, invariant_set
            )
            total_inv = len(inv_results)
            if total_inv > 0:
                passing = sum(1 for (holds, _, _) in inv_results.values() if holds)
                invariant_score = passing / total_inv
                violated = [
                    name
                    for name, (holds, _, _) in inv_results.items()
                    if not holds
                ]

        # Aggregate: failure_penalty is a *penalty* so subtract
        aggregate = (
            self._w_cons * consistency
            + self._w_rule * rule_coherence
            + self._w_morph * morphology
            - self._w_fail * failure_penalty
            + 0.0 * invariant_score  # invariant added separately in tribunal
        )
        # Include invariant compliance in aggregate
        aggregate = (aggregate * 4 + invariant_score) / 5.0
        aggregate = max(0.0, min(1.0, aggregate))

        # Apply a sharpening transform (exponentiation) to combat critic flatness
        aggregate = aggregate ** 3.0

        notes = (
            f"consistency={consistency:.3f}, "
            f"rule_coherence={rule_coherence:.3f}, "
            f"morphology={morphology:.3f}, "
            f"failure_penalty={failure_penalty:.3f}, "
            f"invariant={invariant_score:.3f}, "
            f"violated={violated}"
        )
        if "llm_judge_final_rule_coherence" in trace.metadata:
            notes += (
                f", llm_judge("
                f"arith={trace.metadata.get('llm_judge_arithmetic_consistency', '?'):.3f}, "
                f"logic={trace.metadata.get('llm_judge_logical_consistency', '?'):.3f}, "
                f"align={trace.metadata.get('llm_judge_answer_trace_alignment', '?'):.3f}, "
                f"coherence={trace.metadata.get('llm_judge_final_rule_coherence', '?'):.3f})"
            )

        return CritiqueResult(
            trace_id=trace.trace_id,
            consistency_score=round(consistency, 4),
            rule_coherence_score=round(rule_coherence, 4),
            morphology_score=round(morphology, 4),
            failure_similarity_penalty=round(failure_penalty, 4),
            invariant_compliance_score=round(invariant_score, 4),
            aggregate_score=round(aggregate, 4),
            violated_invariants=violated,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Sub-scorers
    # ------------------------------------------------------------------

    def _score_consistency(self, trace: CandidateTrace) -> float:
        """Heuristic internal consistency: confidence × step coverage."""
        conf = trace.confidence_score or 0.5
        steps = len(trace.reasoning_steps)
        step_score = min(1.0, steps / 5.0)  # saturates at 5 steps
        return round(0.5 * conf + 0.5 * step_score, 4)

    def _score_rule_coherence(self, task: Task, trace: CandidateTrace) -> float:
        """Does the stated reasoning match what we'd expect from the training data?

        Proxy: similarity between the candidate answer and the closest training output.
        """
        from epistemic_tribunal.tribunal_types import TaskDomain
        if task.domain == TaskDomain.GSM8K_MATH:
            if self._use_llm_judge:
                return self._score_rule_coherence_llm(task, trace)
            return 1.0

        if not task.train:
            return 0.5  # no signal

        similarities = [
            grid_similarity(trace.answer, ex.output)
            for ex in task.train
        ]
        return round(max(similarities), 4)

    def _score_rule_coherence_llm(self, task: Task, trace: CandidateTrace) -> float:
        """Uses an LLM to judge the coherence of a math reasoning trace."""
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            log.warning("DEEPSEEK_API_KEY not set. Cannot run LLM judge. Defaulting to 1.0")
            return 1.0
            
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        
        reasoning_text = "\n".join(trace.reasoning_steps) if trace.reasoning_steps else "No reasoning provided."
        
        prompt = f"""Evaluate the following mathematical reasoning trace and extract the specified scores as strict JSON.
Scores must be floats between 0.0 and 1.0.

Question: {task.test_input}
Reasoning Steps:
{reasoning_text}
Proposed Answer: {trace.answer}

Return ONLY strict JSON matching this exact structure, with no markdown formatting or extra text:
{{
  "arithmetic_consistency": 0.0,
  "logical_consistency": 0.0,
  "answer_trace_alignment": 0.0,
  "final_rule_coherence": 0.0,
  "brief_rationale": "..."
}}
"""
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a strict, objective mathematical reasoning critic. You only output valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content
            if not result_text:
                return 0.5
                
            data = json.loads(result_text)
            
            # Persist all rubric dimensions to metadata for downstream analysis
            if "brief_rationale" in data:
                trace.metadata["llm_judge_rationale"] = data["brief_rationale"]
            for key in ("arithmetic_consistency", "logical_consistency", "answer_trace_alignment", "final_rule_coherence"):
                if key in data:
                    trace.metadata[f"llm_judge_{key}"] = float(data[key])
                
            score = float(data.get("final_rule_coherence", 0.5))
            return round(max(0.0, min(1.0, score)), 4)
            
        except Exception as e:
            log.warning("LLM judge failed: %s. Defaulting to 0.5", str(e))
            return 0.5

    def _score_morphology(self, task: Task, trace: CandidateTrace) -> float:
        """Is the answer grid structurally plausible?

        Checks:
        - Non-empty grid.
        - Shape matches test input (if training consistently preserves shape).
        - Colour range is bounded (0–9).
        """
        from epistemic_tribunal.tribunal_types import TaskDomain
        if task.domain == TaskDomain.GSM8K_MATH:
            if trace.answer is None or not str(trace.answer).strip():
                return 0.0
            
            # If they provided reasoning, verify the answer actually appears in it
            if trace.reasoning_steps:
                full_text = " ".join(trace.reasoning_steps)
                if str(trace.answer) in full_text:
                    return 1.0
                return 0.5  # Answer doesn't appear in their reasoning
            return 1.0

        answer = trace.answer
        if not answer or not answer[0]:
            return 0.0

        score = 0.0
        rows_a, cols_a = grid_shape(answer)
        rows_t, cols_t = grid_shape(task.test_input)

        # Shape check
        if rows_a == rows_t and cols_a == cols_t:
            score += 0.4
        elif rows_a > 0 and cols_a > 0:
            score += 0.2

        # Colour range
        colours = unique_colours(answer)
        if colours.issubset(set(range(10))):
            score += 0.4

        # Non-trivial (not all one colour)
        if len(colours) > 1:
            score += 0.2

        return round(min(1.0, score), 4)

    def _score_failure_similarity(
        self, trace: CandidateTrace, patterns: list[dict]
    ) -> float:
        """Penalty for similarity to known failure patterns.

        Uses generator_name as a simple proxy heuristic.
        """
        if not patterns:
            return 0.0

        penalty = 0.0
        for pattern in patterns:
            # If this generator was involved in a past failure, apply small penalty
            if pattern.get("generator_name") == trace.generator_name:
                penalty += 0.15
            # If diagnosis mentions "adversarial" and this is adversarial
            if "adversarial" in str(pattern.get("diagnosis", "")).lower():
                if trace.generator_name == "adversarial":
                    penalty += 0.25

        return round(min(1.0, penalty), 4)
