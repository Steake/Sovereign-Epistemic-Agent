"""ARC-specific EQBSL feature extraction.

This adapter is intentionally conservative in v1 so ARC remains untouched
unless EQBSL is explicitly enabled for it.
"""

from __future__ import annotations

from typing import Any

from epistemic_tribunal.tribunal_types import CandidateTrace, CritiqueResult, Task


class ArcEqbslAdapter:
    """Minimal ARC adapter for neutral EQBSL scaffolding."""

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
            "domain": "arc_like",
            "answer_signature": coalition_answer_signature,
            "coalition_size": len(coalition_traces),
            "avg_invariant_compliance": round(
                sum(c.invariant_compliance_score for c in coalition_critiques)
                / max(len(coalition_critiques), 1),
                4,
            ),
            "reasoning_present_fraction": round(
                sum(1 for t in coalition_traces if t.reasoning_steps) / max(len(coalition_traces), 1),
                4,
            ),
        }
