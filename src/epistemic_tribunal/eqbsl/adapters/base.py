"""Base EQBSL adapter protocol and registry."""

from __future__ import annotations

from typing import Any, Protocol

from epistemic_tribunal.tribunal_types import CandidateTrace, CritiqueResult, Task, TaskDomain


class EqbslDomainAdapter(Protocol):
    """Provides domain-specific coalition features for EQBSL."""

    def build_features(
        self,
        task: Task,
        coalition_traces: list[CandidateTrace],
        coalition_critiques: list[CritiqueResult],
        all_traces: list[CandidateTrace],
        all_critiques: list[CritiqueResult],
        coalition_answer_signature: str,
        failure_memory_metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Return domain-specific coalition features."""
        ...


class _NeutralEqbslAdapter:
    def build_features(
        self,
        task: Task,
        coalition_traces: list[CandidateTrace],
        coalition_critiques: list[CritiqueResult],
        all_traces: list[CandidateTrace],
        all_critiques: list[CritiqueResult],
        coalition_answer_signature: str,
        failure_memory_metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "domain": task.domain.value,
            "coalition_size": len(coalition_traces),
            "avg_reasoning_steps": round(
                sum(len(t.reasoning_steps) for t in coalition_traces) / max(len(coalition_traces), 1),
                4,
            ),
        }


from epistemic_tribunal.eqbsl.adapters.arc import ArcEqbslAdapter
from epistemic_tribunal.eqbsl.adapters.gsm8k import Gsm8kEqbslAdapter

_ADAPTERS: dict[TaskDomain, EqbslDomainAdapter] = {
    TaskDomain.GSM8K_MATH: Gsm8kEqbslAdapter(),
    TaskDomain.ARC_LIKE: ArcEqbslAdapter(),
}

_NEUTRAL = _NeutralEqbslAdapter()


def get_eqbsl_adapter(domain: TaskDomain) -> EqbslDomainAdapter:
    return _ADAPTERS.get(domain, _NEUTRAL)
