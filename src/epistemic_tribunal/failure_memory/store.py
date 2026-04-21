"""SQLite-backed persistent store for failure signatures.

Extends the existing ledger database with a ``failure_signatures`` table.
Provides structured-field similarity matching for v1 (no vector embeddings).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from epistemic_tribunal.failure_memory.models import (
    FailureMatch,
    FailureProbe,
    FailureSignature,
    FailureType,
)
from epistemic_tribunal.utils.logging import get_logger

log = get_logger(__name__)

_FAILURE_MEMORY_DDL = """
CREATE TABLE IF NOT EXISTS failure_signatures (
    signature_id TEXT PRIMARY KEY,
    domain TEXT NOT NULL,
    task_id TEXT NOT NULL,
    failure_type TEXT NOT NULL,
    answer_signature TEXT DEFAULT '',
    coalition_context_json TEXT DEFAULT '{}',
    trace_quality_json TEXT DEFAULT '{}',
    critic_context_json TEXT DEFAULT '{}',
    disagreement_rate REAL DEFAULT 0.0,
    structural_margin REAL DEFAULT 0.0,
    outcome_label TEXT DEFAULT '',
    domain_features_json TEXT DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_fs_domain ON failure_signatures(domain);
CREATE INDEX IF NOT EXISTS idx_fs_failure_type ON failure_signatures(failure_type);
CREATE INDEX IF NOT EXISTS idx_fs_task_id ON failure_signatures(task_id);
"""


class FailureMemoryStore:
    """Persistent SQLite store for failure signatures.

    Can share an existing SQLite connection (e.g. the main ledger) or
    create its own database file.

    Parameters
    ----------
    path:
        Path to the SQLite database.  Use ``\":memory:\"`` for tests.
    conn:
        Optional existing ``sqlite3.Connection`` to reuse (e.g. from
        :class:`LedgerStore`).  If provided, *path* is ignored.
    """

    def __init__(
        self,
        path: str | Path = ":memory:",
        conn: Optional[sqlite3.Connection] = None,
    ) -> None:
        if conn is not None:
            self._conn = conn
            self._owns_conn = False
        else:
            self.path = str(path)
            if self.path != ":memory:":
                Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._owns_conn = True
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(_FAILURE_MEMORY_DDL)
        self._conn.commit()

    def close(self) -> None:
        if self._owns_conn:
            self._conn.close()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def store(self, sig: FailureSignature) -> None:
        """Persist a failure signature."""
        self._conn.execute(
            """INSERT OR REPLACE INTO failure_signatures
               (signature_id, domain, task_id, failure_type,
                answer_signature, coalition_context_json, trace_quality_json,
                critic_context_json, disagreement_rate, structural_margin,
                outcome_label, domain_features_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                sig.signature_id,
                sig.domain,
                sig.task_id,
                sig.failure_type.value,
                sig.answer_signature,
                json.dumps(sig.coalition_context),
                json.dumps(sig.trace_quality_features),
                json.dumps(sig.critic_context),
                sig.disagreement_rate,
                sig.structural_margin,
                sig.outcome_label,
                json.dumps(sig.domain_features),
                sig.created_at.isoformat(),
            ),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query_similar(
        self,
        probe: FailureProbe,
        limit: int = 20,
    ) -> list[FailureMatch]:
        """Find past failure signatures similar to *probe*.

        Similarity is computed from structured-field matching (v1).
        Only returns signatures from past *failures* (wrong_pick and
        bad_abstention), not correct selections.

        Parameters
        ----------
        probe:
            Observable pool shape built before adjudication.
        limit:
            Maximum number of matches to return.
        """
        # Fetch candidate rows — filter to same domain and only failure types
        rows = self._conn.execute(
            """SELECT * FROM failure_signatures
               WHERE domain = ?
                 AND failure_type IN (?, ?)
               ORDER BY created_at DESC
               LIMIT ?""",
            (
                probe.domain,
                FailureType.WRONG_PICK.value,
                FailureType.BAD_ABSTENTION.value,
                limit * 5,  # Fetch extra for post-filter scoring
            ),
        ).fetchall()

        if not rows:
            return []

        matches: list[FailureMatch] = []
        for row in rows:
            sig = self._row_to_signature(row)
            similarity, matching = self._compute_similarity(probe, sig)
            if similarity > 0.0:
                matches.append(
                    FailureMatch(
                        signature=sig,
                        similarity=similarity,
                        matching_features=matching,
                    )
                )

        # Sort by similarity descending, take top N
        matches.sort(key=lambda m: m.similarity, reverse=True)
        return matches[:limit]

    def get_all(self, domain: Optional[str] = None) -> list[FailureSignature]:
        """Return all stored signatures, optionally filtered by domain."""
        if domain:
            rows = self._conn.execute(
                "SELECT * FROM failure_signatures WHERE domain = ? ORDER BY created_at ASC",
                (domain,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM failure_signatures ORDER BY created_at ASC"
            ).fetchall()
        return [self._row_to_signature(r) for r in rows]

    def get_stats(self) -> dict[str, Any]:
        """Return summary statistics for the failure-memory store."""
        total = self._conn.execute(
            "SELECT COUNT(*) as c FROM failure_signatures"
        ).fetchone()["c"]

        by_type = self._conn.execute(
            "SELECT failure_type, COUNT(*) as c FROM failure_signatures GROUP BY failure_type"
        ).fetchall()

        by_domain = self._conn.execute(
            "SELECT domain, COUNT(*) as c FROM failure_signatures GROUP BY domain"
        ).fetchall()

        return {
            "total_signatures": total,
            "by_type": {r["failure_type"]: r["c"] for r in by_type},
            "by_domain": {r["domain"]: r["c"] for r in by_domain},
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _row_to_signature(self, row: sqlite3.Row) -> FailureSignature:
        return FailureSignature(
            signature_id=row["signature_id"],
            domain=row["domain"],
            task_id=row["task_id"],
            failure_type=FailureType(row["failure_type"]),
            answer_signature=row["answer_signature"],
            coalition_context=json.loads(row["coalition_context_json"]),
            trace_quality_features=json.loads(row["trace_quality_json"]),
            critic_context=json.loads(row["critic_context_json"]),
            disagreement_rate=row["disagreement_rate"],
            structural_margin=row["structural_margin"],
            outcome_label=row["outcome_label"],
            domain_features=json.loads(row["domain_features_json"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    @staticmethod
    def _compute_similarity(
        probe: FailureProbe, sig: FailureSignature
    ) -> tuple[float, list[str]]:
        """Compute structured-field similarity between a probe and a stored signature.

        Uses observable-only features from the probe (no ground-truth leakage).
        Returns (score, list_of_matching_feature_names).
        """
        features_checked = 0
        features_matched = 0
        matched_names: list[str] = []

        coal = sig.coalition_context
        tq = sig.trace_quality_features

        # 1. Disagreement pattern: both have high disagreement?
        features_checked += 1
        if probe.disagreement_rate > 0.3 and sig.disagreement_rate > 0.3:
            features_matched += 1
            matched_names.append("high_disagreement")

        # 2. Coalition shape: similar n_clusters?
        features_checked += 1
        if probe.n_clusters == coal.get("n_clusters", 0):
            features_matched += 1
            matched_names.append("same_n_clusters")

        # 3. Coalition mass similarity (within 0.15)
        features_checked += 1
        if abs(probe.coalition_mass - coal.get("coalition_mass", 0)) < 0.15:
            features_matched += 1
            matched_names.append("similar_coalition_mass")

        # 4. Structural margin similarity (both near zero?)
        features_checked += 1
        if probe.structural_margin < 0.05 and sig.structural_margin < 0.05:
            features_matched += 1
            matched_names.append("both_low_margin")

        # 5. Critic flatness pattern
        features_checked += 1
        sig_critics_flat = sig.critic_context.get("all_flat", False)
        if probe.all_critics_flat and sig_critics_flat:
            features_matched += 1
            matched_names.append("both_critics_flat")

        # 6. Majority-has-rationale mismatch pattern
        # If the prior failure had a false majority that lacked rationale,
        # and the current probe also shows majority without rationale...
        features_checked += 1
        sig_majority_no_rationale = not tq.get("rationale_present", True)
        if not probe.majority_has_rationale and sig_majority_no_rationale:
            features_matched += 1
            matched_names.append("majority_lacks_rationale")

        # 7. Minority-has-rationale pattern (current probe)
        # If the prior failure had a minority with rationale that was correct
        # (false_majority pattern), and current probe shows minority with rationale
        features_checked += 1
        if probe.minority_has_rationale and coal.get("false_majority", False):
            features_matched += 1
            matched_names.append("minority_rationale_false_majority_prior")

        # 8. Parse contamination
        features_checked += 1
        probe_has_parse = any(
            cf.get("finish_reason") in ("length", "error")
            for cf in probe.candidate_features.values()
        )
        sig_has_parse = coal.get("parse_issue_present", False)
        if probe_has_parse and sig_has_parse:
            features_matched += 1
            matched_names.append("both_parse_contaminated")

        similarity = features_matched / max(features_checked, 1)
        return round(similarity, 4), matched_names
