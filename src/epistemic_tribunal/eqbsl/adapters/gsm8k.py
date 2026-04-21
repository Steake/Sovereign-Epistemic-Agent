"""GSM8K-specific EQBSL feature extraction."""

from __future__ import annotations

from typing import Any

from epistemic_tribunal.tribunal_types import CandidateTrace, CritiqueResult, Task


class Gsm8kEqbslAdapter:
    """Extracts coalition features relevant to GSM8K-style reasoning traces."""

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
        total_traces = max(len(all_traces), 1)
        coalition_size = len(coalition_traces)
        coalition_mass = coalition_size / total_traces
        reasoning_counts = [len(t.reasoning_steps) for t in coalition_traces]
        rationale_present = [bool(t.reasoning_steps) for t in coalition_traces]
        avg_reasoning_steps = sum(reasoning_counts) / max(len(reasoning_counts), 1)

        all_cluster_sizes: dict[str, int] = {}
        from epistemic_tribunal.domains.factory import get_adapter

        adapter = get_adapter(task.domain)
        for trace in all_traces:
            key = str(adapter.get_cluster_key(trace.answer))
            all_cluster_sizes[key] = all_cluster_sizes.get(key, 0) + 1

        max_cluster = max(all_cluster_sizes.values()) if all_cluster_sizes else coalition_size
        is_majority = coalition_size == max_cluster
        rationale_rich_minority = (not is_majority) and (avg_reasoning_steps >= 2.0)
        shallow_majority = is_majority and avg_reasoning_steps <= 1.0

        exact_hits = 0
        structural_hits = 0
        if failure_memory_metadata:
            trace_decomp = failure_memory_metadata.get("failure_memory_trace_decomposition", {})
            for trace in coalition_traces:
                if trace.trace_id in trace_decomp:
                    exact_hits += trace_decomp[trace.trace_id].get("n_exact_matches", 0)
                    structural_hits += trace_decomp[trace.trace_id].get("n_structural_matches", 0)

        return {
            "domain": "gsm8k_math",
            "answer_signature": coalition_answer_signature,
            "coalition_size": coalition_size,
            "coalition_mass": round(coalition_mass, 4),
            "is_majority": is_majority,
            "rationale_present_fraction": round(
                sum(1 for flag in rationale_present if flag) / max(len(rationale_present), 1),
                4,
            ),
            "avg_reasoning_steps": round(avg_reasoning_steps, 4),
            "rationale_rich_minority": rationale_rich_minority,
            "shallow_majority": shallow_majority,
            "exact_memory_hits": exact_hits,
            "structural_memory_hits": structural_hits,
        }
