"""UncertaintyAnalyzer — computes entropy, margin, coalition mass, and disagreement.

When token-level probabilities are unavailable (as in mock mode), we derive
proxy uncertainty signals from inter-trace structural disagreement and the
generator-reported confidence metadata.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Optional

from epistemic_tribunal.tasks.base import grid_similarity, grids_equal, grid_shape
from epistemic_tribunal.tribunal_types import CandidateTrace, Task, UncertaintyReport
from epistemic_tribunal.uncertainty.base import BaseUncertaintyAnalyzer


class UncertaintyAnalyzer(BaseUncertaintyAnalyzer):
    """Heuristic uncertainty analyzer for ARC-like generator pools."""

    def __init__(self, min_coalition_mass: float = 0.6) -> None:
        self.min_coalition_mass = min_coalition_mass

    def analyze(
        self,
        task: Task,
        traces: list[CandidateTrace],
    ) -> UncertaintyReport:
        """Compute uncertainty signals for a pool of candidate traces.

        Parameters
        ----------
        task:
            The task being evaluated (used for context).
        traces:
            All candidate traces in the generator pool.
        """
        if not traces:
            return UncertaintyReport(
                entropy=0.0,
                margin=0.0,
                coalition_mass=0.0,
                disagreement_rate=1.0,
                notes="No traces to analyse.",
            )

        # ----------------------------------------------------------------
        # 1. Disagreement rate — fraction of unique pairs that disagree
        # ----------------------------------------------------------------
        from epistemic_tribunal.domains.factory import get_adapter
        adapter = get_adapter(task.domain)
        
        disagreement_rate = adapter.compute_disagreement(traces)

        # ----------------------------------------------------------------
        # 2. Cluster answers (by equality) to compute a distribution
        # ----------------------------------------------------------------
        cluster_counts: Counter = Counter(adapter.get_cluster_key(t.answer) for t in traces)
        total = sum(cluster_counts.values())
        probs = [c / total for c in cluster_counts.values()]


        # ----------------------------------------------------------------
        # 3. Entropy
        # ----------------------------------------------------------------
        # Use 0*log(0)=0 convention; clamp to avoid floating-point negatives.
        entropy = -sum(p * math.log(p) for p in probs if p > 0.0)
        entropy = max(0.0, entropy)
        # Normalise to [0, 1] by max possible entropy (uniform over n clusters)
        n_clusters = max(len(cluster_counts), 1)
        max_entropy = math.log(n_clusters) if n_clusters > 1 else 1.0
        entropy_normalised = min(1.0, entropy / max(max_entropy, 1e-9))

        # ----------------------------------------------------------------
        # 4. Margin — gap between top-2 clusters
        # ----------------------------------------------------------------
        sorted_probs = sorted(probs, reverse=True)
        margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) >= 2 else 1.0

        # ----------------------------------------------------------------
        # 5. Coalition mass — fraction of generators supporting top answer
        # ----------------------------------------------------------------
        top_key = cluster_counts.most_common(1)[0][0]
        coalition_mass = cluster_counts[top_key] / total

        # ----------------------------------------------------------------
        # 6. Per-trace quality score — candidate-specific by answer cluster
        #
        # Each trace receives the fraction of pool generators that produced
        # the same answer as it.  This makes U candidate-discriminating:
        #   - majority-coalition trace (2/3 agree): quality = 2/3 ≈ 0.667
        #   - minority trace (1/3):                 quality = 1/3 ≈ 0.333
        #
        # This ensures structural_margin > 0 on every contested task where
        # generators disagree but a plurality answer exists.
        # ----------------------------------------------------------------
        per_trace_quality: dict[str, float] = {}
        for trace in traces:
            trace_cluster = adapter.get_cluster_key(trace.answer)
            cluster_fraction = cluster_counts[trace_cluster] / total
            per_trace_quality[trace.trace_id] = round(cluster_fraction, 4)

        notes = (
            f"n_traces={len(traces)}, "
            f"clusters={len(cluster_counts)}, "
            f"entropy={entropy:.3f}, "
            f"margin={margin:.3f}, "
            f"coalition_mass={coalition_mass:.3f}, "
            f"disagreement_rate={disagreement_rate:.3f}"
        )

        return UncertaintyReport(
            entropy=round(entropy_normalised, 4),
            margin=round(margin, 4),
            coalition_mass=round(coalition_mass, 4),
            disagreement_rate=round(disagreement_rate, 4),
            per_trace_quality=per_trace_quality,
            notes=notes,
        )
