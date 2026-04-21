"""Generator-context trust estimation for EQBSL."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

from epistemic_tribunal.config import EQBSLConfig
from epistemic_tribunal.eqbsl.models import EvidenceOpinion, OpinionSource
from epistemic_tribunal.ledger.store import LedgerStore


class GeneratorTrustEstimator:
    """Builds a coalition-level generator trust opinion from ledger history."""

    def __init__(self, config: EQBSLConfig, store: Optional[LedgerStore] = None) -> None:
        self._config = config
        self._store = store

    def estimate(
        self,
        *,
        coalition_generators: list[str],
        coalition_features: dict[str, Any],
    ) -> OpinionSource:
        if not self._config.enable_generator_trust:
            return OpinionSource(
                source_name="generator_trust",
                source_type="trust",
                trust_weight=0.0,
                opinion=EvidenceOpinion.neutral(
                    base_rate=self._config.default_base_rate,
                    prior_weight=self._config.k,
                    metadata={"disabled": True},
                ),
                metadata={"disabled": True},
            )

        historical = self._collect_historical_stats(coalition_generators)
        avg_positive = 0.0
        avg_negative = 0.0
        if coalition_generators:
            avg_positive = sum(stat["positive"] for stat in historical.values()) / len(coalition_generators)
            avg_negative = sum(stat["negative"] for stat in historical.values()) / len(coalition_generators)

        # Coalition-context modifiers: rationale-rich minorities receive a small
        # boost; shallow majorities receive suspicion. These are explicit and logged.
        if coalition_features.get("rationale_rich_minority"):
            avg_positive += 0.8
        if coalition_features.get("shallow_majority"):
            avg_negative += 0.8

        opinion = EvidenceOpinion.from_evidence(
            positive_evidence=max(avg_positive, 0.0),
            negative_evidence=max(avg_negative, 0.0),
            prior_weight=self._config.k,
            base_rate=self._config.default_base_rate,
            metadata={
                "historical_stats": historical,
                "contextual_rationale_rich_minority": bool(
                    coalition_features.get("rationale_rich_minority")
                ),
                "contextual_shallow_majority": bool(coalition_features.get("shallow_majority")),
            },
        )
        return OpinionSource(
            source_name="generator_trust",
            source_type="trust",
            trust_weight=self._config.source_trust.generator_trust,
            opinion=opinion,
            metadata={"historical_stats": historical},
        )

    def _collect_historical_stats(self, coalition_generators: list[str]) -> dict[str, dict[str, float]]:
        stats: dict[str, dict[str, float]] = {
            name: {"positive": 0.0, "negative": 0.0, "seen": 0.0}
            for name in coalition_generators
        }
        if self._store is None or not coalition_generators:
            return stats

        runs = self._store.get_experiment_runs()
        per_generator_entries: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in runs:
            metadata_raw = row.get("metadata_json") or "{}"
            try:
                import json

                metadata = json.loads(metadata_raw)
            except Exception:
                metadata = {}
            for entry in metadata.get("generator_outcomes", []):
                name = entry.get("generator_name")
                if name in stats:
                    per_generator_entries[name].append(entry)

        for name, entries in per_generator_entries.items():
            for entry in entries:
                stats[name]["seen"] += 1.0
                if entry.get("is_majority") and entry.get("is_correct") is False:
                    stats[name]["negative"] += 1.4
                if (not entry.get("is_majority")) and entry.get("is_correct") is True:
                    stats[name]["positive"] += 1.4
                if entry.get("rationale_present") and (not entry.get("is_majority")) and entry.get("is_correct") is True:
                    stats[name]["positive"] += 0.8
                if (not entry.get("rationale_present")) and entry.get("is_majority") and entry.get("is_correct") is False:
                    stats[name]["negative"] += 0.8
                if entry.get("was_selected") and entry.get("ground_truth_match") is True:
                    stats[name]["positive"] += 0.6
                if entry.get("was_selected") and entry.get("ground_truth_match") is False:
                    stats[name]["negative"] += 0.6

        return stats
