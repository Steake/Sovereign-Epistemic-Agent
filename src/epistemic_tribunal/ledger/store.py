"""SQLite-backed ledger store for the Epistemic Tribunal.

The store handles all DDL/DML operations.  It is intentionally kept as a
thin wrapper around stdlib ``sqlite3`` so there is no ORM dependency.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Optional

from epistemic_tribunal.ledger.models import (
    CoalitionOpinionRecord,
    DecisionRecord,
    ExperimentRunRecord,
    FailureRecordRow,
    InvariantViolationRecord,
    TaskRecord,
    TraceRecord,
)
from epistemic_tribunal.utils.logging import get_logger

log = get_logger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    domain TEXT,
    description TEXT,
    train_examples_count INTEGER,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS traces (
    trace_id TEXT PRIMARY KEY,
    task_id TEXT,
    generator_name TEXT,
    confidence_score REAL,
    answer_json TEXT,
    reasoning_steps_json TEXT,
    created_at TEXT,
    FOREIGN KEY (task_id) REFERENCES tasks(task_id)
);

CREATE TABLE IF NOT EXISTS decisions (
    decision_id TEXT PRIMARY KEY,
    task_id TEXT,
    decision TEXT,
    selected_trace_id TEXT,
    confidence REAL,
    reasoning TEXT,
    scores_json TEXT,
    created_at TEXT,
    FOREIGN KEY (task_id) REFERENCES tasks(task_id)
);

CREATE TABLE IF NOT EXISTS failures (
    failure_id TEXT PRIMARY KEY,
    task_id TEXT,
    selected_trace_id TEXT,
    all_candidate_trace_ids_json TEXT,
    violated_invariants_json TEXT,
    disagreement_pattern TEXT,
    diagnosis TEXT,
    notes TEXT,
    ground_truth_match INTEGER,
    created_at TEXT,
    FOREIGN KEY (task_id) REFERENCES tasks(task_id)
);

CREATE TABLE IF NOT EXISTS invariant_violations (
    violation_id TEXT PRIMARY KEY,
    task_id TEXT,
    trace_id TEXT,
    invariant_name TEXT,
    note TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS experiment_runs (
    run_id TEXT PRIMARY KEY,
    task_id TEXT,
    decision TEXT,
    confidence REAL DEFAULT 0.0,
    selected_trace_id TEXT,
    ground_truth_match INTEGER,
    duration_seconds REAL,
    generator_names_json TEXT,
    config_snapshot_json TEXT,
    metadata_json TEXT DEFAULT '{}',
    created_at TEXT,
    FOREIGN KEY (task_id) REFERENCES tasks(task_id)
);

CREATE TABLE IF NOT EXISTS coalition_opinions (
    run_id TEXT,
    task_id TEXT,
    answer_signature TEXT,
    coalition_member_trace_ids_json TEXT DEFAULT '[]',
    coalition_member_generators_json TEXT DEFAULT '[]',
    representative_trace_id TEXT,
    representative_generator TEXT,
    source_opinions_json TEXT DEFAULT '{}',
    generator_trust_opinion_json TEXT DEFAULT '{}',
    fused_opinion_json TEXT DEFAULT '{}',
    belief REAL DEFAULT 0.0,
    disbelief REAL DEFAULT 0.0,
    uncertainty REAL DEFAULT 1.0,
    base_rate REAL DEFAULT 0.25,
    expectation REAL DEFAULT 0.0,
    base_rate_contribution REAL DEFAULT 0.0,
    decision_role TEXT DEFAULT 'other',
    decision_reason_code TEXT DEFAULT '',
    decision_reason_text TEXT DEFAULT '',
    explanation_metadata_json TEXT DEFAULT '{}',
    created_at TEXT,
    PRIMARY KEY (run_id, answer_signature),
    FOREIGN KEY (task_id) REFERENCES tasks(task_id),
    FOREIGN KEY (run_id) REFERENCES experiment_runs(run_id)
);
"""


class LedgerStore:
    """Persistent SQLite ledger for failure records and experiment runs.

    Parameters
    ----------
    path:
        Path to the SQLite database file.  Use ``":memory:"`` for tests.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(_DDL)
        self._ensure_experiment_run_columns()
        self._conn.commit()

    def _ensure_experiment_run_columns(self) -> None:
        columns = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(experiment_runs)").fetchall()
        }
        if "confidence" not in columns:
            self._conn.execute(
                "ALTER TABLE experiment_runs ADD COLUMN confidence REAL DEFAULT 0.0"
            )
        if "metadata_json" not in columns:
            self._conn.execute(
                "ALTER TABLE experiment_runs ADD COLUMN metadata_json TEXT DEFAULT '{}'"
            )

    def close(self) -> None:
        self._conn.close()

    def checkpoint(self, destination: str | Path) -> None:
        """Write a SQLite backup snapshot to *destination*."""
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        backup_conn = sqlite3.connect(str(destination))
        try:
            self._conn.backup(backup_conn)
        finally:
            backup_conn.close()

    # ------------------------------------------------------------------
    # Task
    # ------------------------------------------------------------------

    def upsert_task(self, rec: TaskRecord) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO tasks
               (task_id, domain, description, train_examples_count, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                rec.task_id,
                rec.domain,
                rec.description,
                rec.train_examples_count,
                rec.created_at.isoformat(),
            ),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Trace
    # ------------------------------------------------------------------

    def insert_trace(self, rec: TraceRecord) -> None:
        self._conn.execute(
            """INSERT OR IGNORE INTO traces
               (trace_id, task_id, generator_name, confidence_score,
                answer_json, reasoning_steps_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                rec.trace_id,
                rec.task_id,
                rec.generator_name,
                rec.confidence_score,
                rec.answer_json,
                rec.reasoning_steps_json,
                rec.created_at.isoformat(),
            ),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    def insert_decision(self, rec: DecisionRecord) -> None:
        self._conn.execute(
            """INSERT OR IGNORE INTO decisions
               (decision_id, task_id, decision, selected_trace_id,
                confidence, reasoning, scores_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                rec.decision_id,
                rec.task_id,
                rec.decision,
                rec.selected_trace_id,
                rec.confidence,
                rec.reasoning,
                rec.scores_json,
                rec.created_at.isoformat(),
            ),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Failure record
    # ------------------------------------------------------------------

    def insert_failure(self, rec: FailureRecordRow) -> None:
        self._conn.execute(
            """INSERT OR IGNORE INTO failures
               (failure_id, task_id, selected_trace_id,
                all_candidate_trace_ids_json, violated_invariants_json,
                disagreement_pattern, diagnosis, notes,
                ground_truth_match, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                rec.failure_id,
                rec.task_id,
                rec.selected_trace_id,
                rec.all_candidate_trace_ids_json,
                rec.violated_invariants_json,
                rec.disagreement_pattern,
                rec.diagnosis,
                rec.notes,
                rec.ground_truth_match,
                rec.created_at.isoformat(),
            ),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Invariant violation
    # ------------------------------------------------------------------

    def insert_invariant_violation(self, rec: InvariantViolationRecord) -> None:
        self._conn.execute(
            """INSERT OR IGNORE INTO invariant_violations
               (violation_id, task_id, trace_id, invariant_name, note, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                rec.violation_id,
                rec.task_id,
                rec.trace_id,
                rec.invariant_name,
                rec.note,
                rec.created_at.isoformat(),
            ),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Experiment run
    # ------------------------------------------------------------------

    def insert_run(self, rec: ExperimentRunRecord) -> None:
        self._conn.execute(
            """INSERT OR IGNORE INTO experiment_runs
               (run_id, task_id, decision, confidence, selected_trace_id,
                ground_truth_match, duration_seconds,
                generator_names_json, config_snapshot_json, metadata_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                rec.run_id,
                rec.task_id,
                rec.decision,
                rec.confidence,
                rec.selected_trace_id,
                rec.ground_truth_match,
                rec.duration_seconds,
                rec.generator_names_json,
                rec.config_snapshot_json,
                rec.metadata_json,
                rec.created_at.isoformat(),
            ),
        )
        self._conn.commit()

    def insert_coalition_opinion(self, rec: CoalitionOpinionRecord) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO coalition_opinions
               (run_id, task_id, answer_signature, coalition_member_trace_ids_json,
                coalition_member_generators_json, representative_trace_id,
                representative_generator, source_opinions_json,
                generator_trust_opinion_json, fused_opinion_json, belief, disbelief,
                uncertainty, base_rate, expectation, base_rate_contribution,
                decision_role, decision_reason_code, decision_reason_text,
                explanation_metadata_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                rec.run_id,
                rec.task_id,
                rec.answer_signature,
                rec.coalition_member_trace_ids_json,
                rec.coalition_member_generators_json,
                rec.representative_trace_id,
                rec.representative_generator,
                rec.source_opinions_json,
                rec.generator_trust_opinion_json,
                rec.fused_opinion_json,
                rec.belief,
                rec.disbelief,
                rec.uncertainty,
                rec.base_rate,
                rec.expectation,
                rec.base_rate_contribution,
                rec.decision_role,
                rec.decision_reason_code,
                rec.decision_reason_text,
                rec.explanation_metadata_json,
                rec.created_at.isoformat(),
            ),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_failure_patterns(self, task_id: Optional[str] = None) -> list[dict[str, Any]]:
        """Return failure records as dicts, optionally filtered by task."""
        if task_id:
            rows = self._conn.execute(
                "SELECT * FROM failures WHERE task_id = ? ORDER BY created_at DESC",
                (task_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM failures ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self) -> dict[str, Any]:
        """Return high-level statistics from the ledger."""
        stats: dict[str, Any] = {}
        for table in (
            "tasks", "traces", "decisions", "failures",
            "invariant_violations", "experiment_runs", "coalition_opinions"
        ):
            count = self._conn.execute(
                f"SELECT COUNT(*) as c FROM {table}"  # noqa: S608
            ).fetchone()["c"]
            stats[table] = count

        # Decision breakdown
        rows = self._conn.execute(
            "SELECT decision, COUNT(*) as c FROM decisions GROUP BY decision"
        ).fetchall()
        stats["decisions_breakdown"] = {r["decision"]: r["c"] for r in rows}

        return stats

    def get_task_summary(self, task_id: str) -> dict[str, Any]:
        """Return a summary of all records for *task_id*."""
        task_row = self._conn.execute(
            "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
        ).fetchone()
        if not task_row:
            return {"error": f"Task {task_id!r} not found in ledger."}

        failures = self._conn.execute(
            "SELECT * FROM failures WHERE task_id = ?", (task_id,)
        ).fetchall()
        decisions = self._conn.execute(
            "SELECT * FROM decisions WHERE task_id = ?", (task_id,)
        ).fetchall()
        runs = self._conn.execute(
            "SELECT * FROM experiment_runs WHERE task_id = ?", (task_id,)
        ).fetchall()
        coalition_opinions = self._conn.execute(
            "SELECT * FROM coalition_opinions WHERE task_id = ? ORDER BY created_at ASC",
            (task_id,),
        ).fetchall()

        return {
            "task": dict(task_row),
            "failures": [dict(r) for r in failures],
            "decisions": [dict(r) for r in decisions],
            "runs": [dict(r) for r in runs],
            "coalition_opinions": [dict(r) for r in coalition_opinions],
        }

    def get_experiment_runs(
        self, task_ids: Optional[list[str] | set[str]] = None
    ) -> list[dict[str, Any]]:
        """Return experiment run rows as dictionaries."""
        if task_ids is None:
            rows = self._conn.execute(
                "SELECT * FROM experiment_runs ORDER BY created_at ASC"
            ).fetchall()
        elif not task_ids:
            # Empty collection means "no tasks to look up" — return nothing
            # rather than falling through to the unfiltered query.
            return []
        else:
            placeholders = ",".join("?" for _ in task_ids)
            query = (
                "SELECT * FROM experiment_runs "
                f"WHERE task_id IN ({placeholders}) ORDER BY created_at ASC"
            )
            rows = self._conn.execute(query, tuple(task_ids)).fetchall()
        return [dict(r) for r in rows]

    def get_coalition_opinions(
        self,
        *,
        run_ids: Optional[list[str] | set[str]] = None,
        task_ids: Optional[list[str] | set[str]] = None,
    ) -> list[dict[str, Any]]:
        """Return persisted coalition-opinion rows."""
        if run_ids is not None:
            if not run_ids:
                return []
            placeholders = ",".join("?" for _ in run_ids)
            query = (
                "SELECT * FROM coalition_opinions "
                f"WHERE run_id IN ({placeholders}) ORDER BY created_at ASC"
            )
            rows = self._conn.execute(query, tuple(run_ids)).fetchall()
            return [dict(r) for r in rows]
        if task_ids is not None:
            if not task_ids:
                return []
            placeholders = ",".join("?" for _ in task_ids)
            query = (
                "SELECT * FROM coalition_opinions "
                f"WHERE task_id IN ({placeholders}) ORDER BY created_at ASC"
            )
            rows = self._conn.execute(query, tuple(task_ids)).fetchall()
            return [dict(r) for r in rows]
        rows = self._conn.execute(
            "SELECT * FROM coalition_opinions ORDER BY created_at ASC"
        ).fetchall()
        return [dict(r) for r in rows]
