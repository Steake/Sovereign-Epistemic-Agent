"""Answer-conditioned verification source for EQBSL."""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from epistemic_tribunal.config import EQBSLConfig
from epistemic_tribunal.eqbsl.models import EvidenceOpinion, OpinionSource, VerificationEvidence
from epistemic_tribunal.tribunal_types import CandidateTrace, CritiqueResult, Task, TaskDomain
from epistemic_tribunal.utils.logging import get_logger

log = get_logger(__name__)

_PLACEHOLDER_REASONING = "LLM provided no explicit reasoning steps."


def _non_placeholder_steps(traces: list[CandidateTrace]) -> list[str]:
    steps: list[str] = []
    for trace in traces:
        for step in trace.reasoning_steps:
            clean = step.strip()
            if clean and clean != _PLACEHOLDER_REASONING:
                steps.append(clean)
    return steps


class VerificationSourceProvider:
    """Builds an answer-conditioned verification opinion for a coalition."""

    def __init__(self, config: EQBSLConfig) -> None:
        self._config = config
        self._client: Any = None

    def enabled(self) -> bool:
        verification = self._config.verification
        return bool(verification.enabled and verification.api_key)

    def build_source(
        self,
        *,
        task: Task,
        coalition_answer_signature: str,
        coalition_traces: list[CandidateTrace],
        coalition_critiques: list[CritiqueResult],
        domain_features: dict[str, Any],
    ) -> Optional[OpinionSource]:
        if not self.enabled():
            return None

        payload = self._request_verification(
            task=task,
            coalition_answer_signature=coalition_answer_signature,
            coalition_traces=coalition_traces,
            coalition_critiques=coalition_critiques,
            domain_features=domain_features,
        )
        if payload is None:
            return self._neutral_source("verification_unavailable")
        return self._assessment_to_source(payload)

    def _neutral_source(self, reason: str) -> OpinionSource:
        opinion = EvidenceOpinion.neutral(
            base_rate=self._config.default_base_rate,
            prior_weight=self._config.k,
            metadata={"verification_status": reason},
        )
        return OpinionSource(
            source_name="verification",
            source_type="verification",
            trust_weight=self._config.source_trust.verification,
            opinion=opinion,
            metadata={"verification_status": reason},
        )

    def _assessment_to_source(self, assessment: VerificationEvidence) -> OpinionSource:
        # Keep the signal moderate so it can break low-gap ties without silently dominating.
        magnitude = round(0.5 + (2.5 * assessment.confidence), 6)
        positive = magnitude if assessment.classification == "support" else 0.0
        negative = magnitude if assessment.classification == "contradiction" else 0.0
        opinion = EvidenceOpinion.from_evidence(
            positive_evidence=positive,
            negative_evidence=negative,
            prior_weight=self._config.k,
            base_rate=self._config.default_base_rate,
            metadata={
                "verification_classification": assessment.classification,
                "verification_confidence": round(assessment.confidence, 6),
                "verification_tags": assessment.rationale_tags,
                "verification_rationale": assessment.rationale,
                **assessment.metadata,
            },
        )
        return OpinionSource(
            source_name="verification",
            source_type="verification",
            trust_weight=self._config.source_trust.verification,
            opinion=opinion,
            metadata=assessment.model_dump(mode="json"),
        )

    def _request_verification(
        self,
        *,
        task: Task,
        coalition_answer_signature: str,
        coalition_traces: list[CandidateTrace],
        coalition_critiques: list[CritiqueResult],
        domain_features: dict[str, Any],
    ) -> Optional[VerificationEvidence]:
        system_prompt, user_prompt = self._build_prompt(
            task=task,
            coalition_answer_signature=coalition_answer_signature,
            coalition_traces=coalition_traces,
            coalition_critiques=coalition_critiques,
            domain_features=domain_features,
        )
        if not system_prompt:
            return None

        try:
            client = self._get_client()
            verification = self._config.verification
            response = client.chat.completions.create(
                model=verification.model_name,
                temperature=verification.temperature,
                max_tokens=verification.max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            text = response.choices[0].message.content
            if not text:
                return None
            data = self._parse_json_object(text)
            classification = str(data.get("classification", "inconclusive")).lower()
            if classification not in {"support", "contradiction", "inconclusive"}:
                classification = "inconclusive"
            return VerificationEvidence(
                classification=classification,
                confidence=float(data.get("confidence", 0.0)),
                rationale=str(data.get("rationale", "")),
                rationale_tags=[str(tag) for tag in data.get("rationale_tags", [])],
                metadata={
                    "raw_domain": task.domain.value,
                    "coalition_answer_signature": coalition_answer_signature,
                },
            )
        except Exception as exc:  # pragma: no cover - defensive logging around external API
            log.warning("Verification source failed for task %s: %s", task.task_id, exc)
            return None

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise json.JSONDecodeError("No JSON object found", cleaned, 0)

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            verification = self._config.verification
            self._client = OpenAI(
                api_key=verification.api_key or "",
                base_url=verification.api_base,
            )
        return self._client

    def _build_prompt(
        self,
        *,
        task: Task,
        coalition_answer_signature: str,
        coalition_traces: list[CandidateTrace],
        coalition_critiques: list[CritiqueResult],
        domain_features: dict[str, Any],
    ) -> tuple[str, str]:
        reasoning_steps = _non_placeholder_steps(coalition_traces)
        critique_scores = [round(crit.aggregate_score, 4) for crit in coalition_critiques]
        reasoning_block = "\n".join(f"- {step}" for step in reasoning_steps[:4]) or "- No usable reasoning steps."

        common_suffix = (
            "Return ONLY JSON with keys:\n"
            '{\n'
            '  "classification": "support" | "contradiction" | "inconclusive",\n'
            '  "confidence": 0.0,\n'
            '  "rationale_tags": ["tag"],\n'
            '  "rationale": "short explanation"\n'
            '}\n'
            "Use 'inconclusive' when the available evidence is too weak to support or contradict the answer.\n"
            "Do not regenerate a new answer.\n"
        )

        if task.domain == TaskDomain.GSM8K_MATH:
            system = (
                "You are a strict answer-conditioned verifier for math solutions. "
                "You check whether a proposed final answer is consistent with the problem statement "
                "and any supplied reasoning sketch. You do not solve from scratch unless needed to detect "
                "a contradiction. Prefer 'inconclusive' over weak support."
            )
            user = (
                f"Question:\n{task.test_input}\n\n"
                f"Proposed answer: {coalition_traces[0].answer}\n"
                f"Coalition answer signature: {coalition_answer_signature}\n"
                f"Coalition size: {len(coalition_traces)}\n"
                f"Critique aggregate scores: {critique_scores}\n"
                f"Domain features: {json.dumps(domain_features, sort_keys=True)}\n"
                "Available reasoning sketch:\n"
                f"{reasoning_block}\n\n"
                "Classify whether the proposed answer is numerically and logically consistent with the question. "
                "Support means the answer looks internally consistent. Contradiction means the answer clearly "
                "conflicts with arithmetic or stated constraints. Inconclusive means you cannot tell reliably.\n\n"
                f"{common_suffix}"
            )
            return system, user

        if task.domain == TaskDomain.ARC_LIKE:
            train_examples = "\n\n".join(
                f"Train {idx} input: {json.dumps(example.input)}\n"
                f"Train {idx} output: {json.dumps(example.output)}"
                for idx, example in enumerate(task.train, start=1)
            )
            system = (
                "You are a strict answer-conditioned verifier for ARC-style grid transformations. "
                "Given train input/output pairs, a test input, and a proposed test output, decide whether the "
                "proposed output is plausibly consistent with the demonstrated transformation. Prefer "
                "'inconclusive' when the train evidence does not clearly support or contradict the proposal."
            )
            user = (
                f"Task: {task.task_id}\n\n"
                f"{train_examples}\n\n"
                f"Test input: {json.dumps(task.test_input)}\n"
                f"Proposed output: {json.dumps(coalition_traces[0].answer)}\n"
                f"Coalition answer signature: {coalition_answer_signature}\n"
                f"Coalition size: {len(coalition_traces)}\n"
                f"Critique aggregate scores: {critique_scores}\n"
                f"Domain features: {json.dumps(domain_features, sort_keys=True)}\n"
                "Available reasoning sketch:\n"
                f"{reasoning_block}\n\n"
                "Support means the proposed output is consistent with a transformation suggested by the train "
                "pairs. Contradiction means the output clearly violates likely transform structure, object "
                "counts, color behavior, or train-pair consistency. Inconclusive means the evidence is mixed "
                "or weak.\n\n"
                f"{common_suffix}"
            )
            return system, user

        return "", ""
