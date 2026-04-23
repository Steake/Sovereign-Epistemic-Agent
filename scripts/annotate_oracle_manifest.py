#!/usr/bin/env python3
"""Annotate Oracle Manifest

Generates a Strange Loop Benchmark compatible JSON manifest from a baseline ledger.

Usage:
    python scripts/annotate_oracle_manifest.py <baseline_ledger_path> <output_json_path>
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path so we can import internal modules easily
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from epistemic_tribunal.ledger.store import LedgerStore


from typing import Any

def parse_json_field(raw: Any, default: dict) -> dict:
    if raw is None:
        return default
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return default


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Strange Loop oracle manifest.")
    parser.add_argument("baseline_ledger_path", help="Path to the baseline SQLite ledger.")
    parser.add_argument("output_json_path", help="Path to output the annotated JSON manifest.")
    args = parser.parse_args()

    ledger_path = Path(args.baseline_ledger_path)
    out_path = Path(args.output_json_path)

    if not ledger_path.exists():
        print(f"Error: Ledger file {ledger_path} does not exist.")
        sys.exit(1)

    store = LedgerStore(ledger_path)
    runs = store.get_experiment_runs()
    store.close()

    if not runs:
        print(f"Error: No experiment runs found in {ledger_path}")
        sys.exit(1)

    manifest_entries = []

    for run_row in runs:
        task_id = run_row.get("task_id")
        if not task_id:
            continue
            
        metadata = parse_json_field(run_row.get("metadata_json"), {})
        
        any_correct = metadata.get("any_correct", False)
        greedy_correct = metadata.get("greedy_correct", False)
        
        # Cohort assignment logic
        if any_correct and greedy_correct:
            cohort = "control_trivial"
            recoverability_status = "exact_candidate_present"
        elif any_correct and not greedy_correct:
            cohort = "same_task_recoverable"
            recoverability_status = "exact_candidate_present"
        else:
            cohort = "same_task_unrecoverable"
            recoverability_status = "no_viable_candidate_present"
            
        entry = {
            "task_id": task_id,
            "cohort": cohort,
            "recoverability_status": recoverability_status,
            "processing_confounded": False,  # Defaulting to false for v1
            "annotation_notes": "auto-generated from baseline ledger pool stats"
        }
        manifest_entries.append(entry)

    out_path.write_text(json.dumps(manifest_entries, indent=2) + "\n")
    print(f"Successfully wrote {len(manifest_entries)} annotations to {out_path}")

    # Print summary
    from collections import Counter
    counts = Counter(e["cohort"] for e in manifest_entries)
    for cohort, count in counts.items():
        print(f"  - {cohort}: {count}")


if __name__ == "__main__":
    main()
