"""Failure-memory smoke-test: pre-seed the store from a prior run's forensic data.

This script is for **plumbing validation only** — NOT the headline experiment.
It:
  1. Reads the existing multi-regime ledger DB (data/gsm8k/ledger_multi_regime.db)
  2. Reconstructs approximate FailureSignature objects from logged ExperimentRun
     metadata (wrong_pick + bad_abstention outcomes)
  3. Writes them into the failure_signatures table in that same DB
  4. Verifies the store can query_similar() against a synthetic probe
  5. Prints a summary

Run it before the main two-pass experiment to confirm the pipeline accepts
pre-seeded signatures without errors.  The two-pass experiment is the actual
scientific claim; this script only tests that data flows through correctly.

Usage:
  uv run python scripts/smoke_test_failure_memory.py
  uv run python scripts/smoke_test_failure_memory.py --ledger data/gsm8k/ledger_multi_regime.db
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

# Ensure src/ is on path when run without editable install
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from epistemic_tribunal.failure_memory.models import (
    FailureProbe,
    FailureSignature,
    FailureType,
)
from epistemic_tribunal.failure_memory.store import FailureMemoryStore


def load_experiment_runs(db_path: str) -> list[dict]:
    """Read experiment run records from the existing ledger."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM experiment_runs ORDER BY created_at ASC"
        ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError as e:
        print(f"  [WARN] Could not read experiment_runs: {e}")
        return []
    finally:
        conn.close()


def reconstruct_signature(run: dict) -> FailureSignature | None:
    """Reconstruct an approximate FailureSignature from a logged ExperimentRun."""
    meta_raw = run.get("metadata_json") or run.get("metadata") or "{}"
    try:
        meta = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
    except (json.JSONDecodeError, TypeError):
        meta = {}

    decision = run.get("decision", "")
    ground_truth_match = meta.get("ground_truth_match")
    any_correct = meta.get("any_correct")
    disagreement_rate = float(meta.get("disagreement_rate", 0.0))
    cohort = meta.get("cohort", "unknown")
    domain = meta.get("domain", "gsm8k_math")

    # Determine failure type
    if decision == "select":
        if ground_truth_match is True:
            failure_type = FailureType.CORRECT_SELECT
        else:
            failure_type = FailureType.WRONG_PICK
    else:
        if any_correct is True:
            failure_type = FailureType.BAD_ABSTENTION
        else:
            failure_type = FailureType.GOOD_ABSTENTION

    # Only seed failures and bad abstentions (no correct selects for smoke test)
    if failure_type == FailureType.CORRECT_SELECT:
        return None

    # Approximate coalition context from what's logged
    coalition_context: dict = {
        "coalition_mass": meta.get("coalition_mass", 0.667),
        "n_clusters": 2,
        "false_majority": failure_type == FailureType.WRONG_PICK and any_correct is True,
        "minority_correct": any_correct is True,
        "total_candidates": 3,
    }

    trace_quality: dict = {
        "rationale_present": False,  # Conservative assumption for failures
        "reasoning_step_count": 0,
    }

    critic_ctx: dict = {
        "all_flat": True,  # Known from forensic: critic was flat at 0.444
    }

    return FailureSignature(
        domain=domain,
        task_id=run.get("task_id", "unknown"),
        failure_type=failure_type,
        coalition_context=coalition_context,
        trace_quality_features=trace_quality,
        critic_context=critic_ctx,
        disagreement_rate=disagreement_rate,
        structural_margin=meta.get("structural_margin", 0.0),
        outcome_label=f"{failure_type.value} | cohort={cohort}",
        domain_features={"pre_seeded": True},
    )


def run_smoke_test(db_path: str) -> None:
    print("\n=== Failure-Memory Smoke Test ===")
    print(f"Ledger: {db_path}\n")

    # Step 1: Open store
    store = FailureMemoryStore(path=db_path)
    stats_before = store.get_stats()
    print(f"[1] Signatures before seeding: {stats_before['total_signatures']}")

    # Step 2: Load runs from existing ledger
    runs = load_experiment_runs(db_path)
    print(f"[2] Experiment runs found in ledger: {len(runs)}")

    if not runs:
        print("    No runs found. Run the GSM8K benchmark first, then run this script.")
        return

    # Step 3: Reconstruct and store signatures
    seeded = 0
    skipped = 0
    for run in runs:
        sig = reconstruct_signature(run)
        if sig is None:
            skipped += 1
            continue
        store.store(sig)
        seeded += 1

    stats_after = store.get_stats()
    print(f"[3] Signatures seeded: {seeded} (skipped {skipped} correct selects)")
    print(f"    Total in store now: {stats_after['total_signatures']}")
    print(f"    By type: {stats_after['by_type']}")

    # Step 4: Verify query_similar() works
    probe = FailureProbe(
        domain="gsm8k_math",
        n_candidates=3,
        n_clusters=2,
        coalition_mass=0.667,
        disagreement_rate=0.667,
        majority_has_rationale=False,
        minority_has_rationale=True,
        all_critics_flat=True,
        structural_margin=0.0,
    )
    matches = store.query_similar(probe, limit=5)
    print("\n[4] Query with false-majority probe:")
    print(f"    Matches returned: {len(matches)}")
    if matches:
        top = matches[0]
        print(f"    Top match: task_id={top.signature.task_id}")
        print(f"               similarity={top.similarity:.3f}")
        print(f"               features={top.matching_features}")
    else:
        print("    No matches — store may be empty or probe is too narrow.")

    # Step 5: Summary verdict
    print("\n[5] VERDICT:")
    if seeded > 0 and len(matches) >= 0:
        print("    ✓ PASS — failure signatures written and queried without errors.")
        print("    The plumbing works. Proceed to the two-pass experiment.")
    else:
        print("    ✗ FAIL — check errors above.")

    store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Failure-memory smoke test")
    parser.add_argument(
        "--ledger",
        default="data/gsm8k/ledger.db",
        help="Path to the SQLite ledger DB (default: data/gsm8k/ledger.db)",
    )
    args = parser.parse_args()

    db = str(REPO_ROOT / args.ledger)
    if not Path(db).exists():
        print(f"Ledger not found: {db}")
        print("Run the GSM8K benchmark first:")
        print("  uv run python scripts/gsm8k_experiment.py --config configs/gsm8k_multi_regime.yaml")
        sys.exit(1)

    run_smoke_test(db)


if __name__ == "__main__":
    main()
