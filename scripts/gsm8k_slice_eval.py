#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from epistemic_tribunal.config import load_config
from epistemic_tribunal.domains.factory import get_adapter
from epistemic_tribunal.orchestrator import Orchestrator
from epistemic_tribunal.tasks.gsm8k import load_tasks_from_jsonl
from epistemic_tribunal.ledger.store import LedgerStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a live GSM8K slice and report pool/oracle stats.")
    parser.add_argument("--config", required=True, help="Config path.")
    parser.add_argument("--data", required=True, help="GSM8K JSONL path.")
    parser.add_argument(
        "--task-ids",
        required=True,
        help="Comma-separated task ids to run in order.",
    )
    parser.add_argument("--out", required=True, help="Directory for results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted = [task_id.strip() for task_id in args.task_ids.split(",") if task_id.strip()]
    tasks = [task for task in load_tasks_from_jsonl(args.data) if task.task_id in wanted]
    order = {task_id: idx for idx, task_id in enumerate(wanted)}
    tasks.sort(key=lambda task: order[task.task_id])

    config = load_config(args.config)
    config.ledger.path = str(out_dir / "ledger.db")
    store = LedgerStore(config.ledger.path)
    orchestrator = Orchestrator(config=config, ledger_store=store)

    runs: list[dict[str, Any]] = []
    for task in tasks:
        run = orchestrator.run(task)
        adapter = get_adapter(task.domain)
        task_rows = store._conn.execute(  # noqa: SLF001
            "SELECT * FROM traces WHERE task_id = ? ORDER BY created_at ASC",
            (task.task_id,),
        ).fetchall()
        answers_by_signature: dict[str, list[dict[str, Any]]] = defaultdict(list)
        trace_rows_by_id: dict[str, dict[str, Any]] = {}
        for row in task_rows:
            answer = json.loads(row["answer_json"])
            trace_rows_by_id[row["trace_id"]] = {
                "generator_name": row["generator_name"],
                "answer": answer,
                "trace_id": row["trace_id"],
            }
            signature = str(adapter.normalize_answer(answer))
            answers_by_signature[signature].append(
                trace_rows_by_id[row["trace_id"]]
            )

        correct_signature = (
            str(adapter.normalize_answer(task.ground_truth)) if task.ground_truth is not None else None
        )
        correct_entries = answers_by_signature.get(correct_signature or "", [])
        selected_entry = trace_rows_by_id.get(run.selected_trace_id or "")
        selected_answer = selected_entry["answer"] if selected_entry is not None else None
        selected_signature = (
            str(adapter.normalize_answer(selected_answer))
            if selected_answer is not None
            else None
        )
        novel_correct_generators = sorted({entry["generator_name"] for entry in correct_entries})

        runs.append(
            {
                "task_id": task.task_id,
                "ground_truth": task.ground_truth,
                "decision": run.decision.value,
                "selected_answer": selected_answer,
                "ground_truth_match": run.ground_truth_match,
                "candidate_pool": answers_by_signature,
                "correct_in_pool": bool(correct_entries),
                "correct_support": len(correct_entries),
                "correct_generators": novel_correct_generators,
                "selected_support": len(answers_by_signature.get(selected_signature or "", []))
                if selected_signature is not None
                else 0,
                "selected_signature": selected_signature,
                "reasoning": run.metadata.get("reasoning", ""),
            }
        )

    total = len(runs)
    correct_in_pool_count = sum(1 for run in runs if run["correct_in_pool"])
    wrong_picks = sum(1 for run in runs if run["decision"] == "select" and run["ground_truth_match"] is False)
    bad_abstentions = sum(
        1 for run in runs if run["decision"] != "select" and run["correct_in_pool"]
    )

    summary = {
        "config": args.config,
        "data": args.data,
        "task_ids": wanted,
        "metrics": {
            "n": total,
            "correct_in_pool_rate": round(correct_in_pool_count / total, 4) if total else 0.0,
            "wrong_picks": wrong_picks,
            "bad_abstentions": bad_abstentions,
            "select_count": sum(1 for run in runs if run["decision"] == "select"),
        },
        "runs": runs,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
