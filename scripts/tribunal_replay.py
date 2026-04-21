"""
tribunal_replay.py — Replay tribunal logic against stored traces without re-running the LLM.

Usage:
    # Single-pass replay (no memory):
    uv run python scripts/tribunal_replay.py \\
        --source data/gsm8k/results/v4/baseline/ledger.db \\
        --config configs/gsm8k_memory_amplified.yaml \\
        --limit 26

    # Two-pass replay (seeds memory from Pass 1, applies it in Pass 2):
    uv run python scripts/tribunal_replay.py \\
        --source data/gsm8k/results/v4/baseline/ledger.db \\
        --config configs/gsm8k_memory_amplified.yaml \\
        --limit 26 \\
        --two-pass \\
        --out data/gsm8k/results/v4/replay/results.json

What this does vs. the live benchmark:
    - LLM generation:    SKIPPED  (uses stored traces from --source ledger)
    - Critic scoring:    RE-RUN   (fresh, uses current critic implementation)
    - Uncertainty:       RE-RUN   (fresh, uses current analyzer)
    - Memory query:      RE-RUN   (from replay memory ledger, seeded by pass 1)
    - Aggregator:        RE-RUN   (uses current config/weights/guardrails)
    - Ground truth eval: RE-RUN   (checks against stored task ground_truth)

This makes tribunal logic changes testable in seconds, not minutes.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import argparse
import shutil
import tempfile
from pathlib import Path
from typing import Optional

# Ensure src/ is on path when run from project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from epistemic_tribunal.config import load_config
from epistemic_tribunal.ledger.store import LedgerStore
from epistemic_tribunal.tribunal.aggregator import TribunalAggregator
from epistemic_tribunal.tribunal_types import (
    CandidateTrace,
    CritiqueResult,
    Task,
    TribunalDecision,
)
from epistemic_tribunal.uncertainty.analyzer import UncertaintyAnalyzer
from epistemic_tribunal.failure_memory.store import FailureMemoryStore
from epistemic_tribunal.failure_memory.query import FailureMemoryQuery
from epistemic_tribunal.failure_memory.extractor import FailureSignatureExtractor
from epistemic_tribunal.critics.trace_critic import TraceCritic
from epistemic_tribunal.invariants.extractor import InvariantExtractor
from epistemic_tribunal.domains.factory import get_adapter
from epistemic_tribunal.tasks.gsm8k import load_tasks_from_jsonl


# ─── Ledger loading ──────────────────────────────────────────────────────────

def load_traces_from_ledger(db_path: str, limit: Optional[int] = None) -> dict[str, list[CandidateTrace]]:
    """Load all stored traces grouped by task_id."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    task_rows = conn.execute("SELECT task_id FROM tasks ORDER BY rowid").fetchall()
    task_ids = [r["task_id"] for r in task_rows]
    if limit:
        task_ids = task_ids[:limit]

    traces_by_task: dict[str, list[CandidateTrace]] = {}
    for tid in task_ids:
        rows = conn.execute(
            "SELECT * FROM traces WHERE task_id = ? ORDER BY created_at",
            (tid,)
        ).fetchall()
        traces = []
        for r in rows:
            traces.append(CandidateTrace(
                trace_id=r["trace_id"],
                generator_name=r["generator_name"],
                answer=json.loads(r["answer_json"]),
                reasoning_steps=json.loads(r["reasoning_steps_json"]),
                confidence_score=r["confidence_score"],
            ))
        if traces:
            traces_by_task[tid] = traces

    conn.close()
    return traces_by_task


def filter_task_payloads(
    tasks_by_id: dict[str, Task],
    traces_by_task: dict[str, list[CandidateTrace]],
    task_ids: Optional[list[str]] = None,
) -> tuple[dict[str, Task], dict[str, list[CandidateTrace]], list[str]]:
    """Filter replay payloads down to an explicit ordered task-id slice."""
    ordered_ids = [task_id for task_id in tasks_by_id.keys() if task_id in traces_by_task]
    if task_ids:
        wanted = set(task_ids)
        ordered_ids = [task_id for task_id in ordered_ids if task_id in wanted]

    return (
        {task_id: tasks_by_id[task_id] for task_id in ordered_ids},
        {task_id: traces_by_task[task_id] for task_id in ordered_ids},
        ordered_ids,
    )


def load_tasks_from_ledger(db_path: str, gsm8k_path: str, limit: Optional[int] = None) -> dict[str, Task]:
    """Load Task objects. Ledger has task_id + domain; ground_truth comes from the JSONL."""
    gsm8k_tasks = load_tasks_from_jsonl(gsm8k_path)
    task_map = {t.task_id: t for t in gsm8k_tasks}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT task_id FROM tasks ORDER BY rowid").fetchall()
    conn.close()

    ordered_ids = [r["task_id"] for r in rows]
    if limit:
        ordered_ids = ordered_ids[:limit]

    return {tid: task_map[tid] for tid in ordered_ids if tid in task_map}


# ─── Single-task replay ──────────────────────────────────────────────────────

def replay_task(
    task: Task,
    traces: list[CandidateTrace],
    aggregator: TribunalAggregator,
    analyzer: UncertaintyAnalyzer,
    critic: TraceCritic,
    invariant_extractor: InvariantExtractor,
    fm_query: Optional[FailureMemoryQuery],
) -> tuple[TribunalDecision, dict]:
    """Re-run the full tribunal stack (minus LLM generation) for one task."""
    # 1. Extract invariants (always fresh)
    invariant_set = invariant_extractor.extract(task)

    # 2. Critique each trace (fresh, using current critic)
    failure_patterns = []  # no legacy failure patterns in replay mode
    critiques: list[CritiqueResult] = [
        critic.critique(task, trace, invariant_set, failure_patterns)
        for trace in traces
    ]

    # 3. Uncertainty analysis (fresh)
    uncertainty = analyzer.analyze(task, traces)

    # 4. Memory query (if memory is active)
    fm_metadata: dict = {}
    if fm_query is not None:
        penalties, fm_metadata = fm_query.query_with_metadata(task, traces, critiques, uncertainty)
        for critique in critiques:
            critique.failure_similarity_penalty = penalties.get(critique.trace_id, 0.0)

    # 5. Adjudicate
    decision = aggregator.adjudicate(
        task,
        traces,
        critiques,
        uncertainty,
        failure_memory_metadata=fm_metadata,
    )

    # 6. Evaluate against ground truth
    adapter = get_adapter(task.domain)
    gt_match: Optional[bool] = None
    if decision.selected_answer is not None and task.ground_truth is not None:
        gt_match = adapter.answers_equal(decision.selected_answer, task.ground_truth)

    any_correct = any(
        adapter.answers_equal(t.answer, task.ground_truth)
        for t in traces
    ) if task.ground_truth is not None else None

    result = {
        "task_id": task.task_id,
        "decision": decision.decision.value,
        "confidence": round(decision.confidence, 4),
        "gt_match": gt_match,
        "any_correct": any_correct,
        "selected_answer": decision.selected_answer,
        "fm_metadata": fm_metadata,
        "forensic": decision.metadata.get("forensic", []),
        "eqbsl": decision.metadata.get("eqbsl", {}),
        "selected_trace_id": decision.selected_trace_id,
        "run_reason": decision.reasoning,
    }
    return decision, result


# ─── Summary stats ────────────────────────────────────────────────────────────

def print_stats(results: list[dict], label: str) -> dict:
    total = len(results)
    selected = [r for r in results if r["decision"] == "select"]
    abstained = [r for r in results if r["decision"] == "abstain"]
    correct = [r for r in selected if r["gt_match"] is True]
    wrong = [r for r in selected if r["gt_match"] is False]
    coverage = len(selected) / total if total else 0
    precision = len(correct) / len(selected) if selected else 0

    [r for r in results if r.get("any_correct") is True and
                 len(set()) > 0]  # approximate: tasks where disagreement occurred
    # Better: contested = tasks where not all traces agree
    # We track this via any_correct + wrong picks

    print(f"\n{'='*55}")
    print(f"  {label}  (n={total})")
    print(f"{'='*55}")
    print(f"  Coverage:           {coverage:.1%}  ({len(selected)}/{total} selected)")
    print(f"  Selective Accuracy: {precision:.1%}  ({len(correct)}/{len(selected)} correct)")
    print(f"  Wrong Picks:        {len(wrong)}")
    print(f"  Abstentions:        {len(abstained)}")
    print(f"  Error Rate:         {len(wrong)/total:.1%}")

    return {
        "label": label, "total": total, "selected": len(selected),
        "correct": len(correct), "wrong": len(wrong), "abstained": len(abstained),
        "coverage": coverage, "precision": precision,
    }


def _config_only_differs_in_eqbsl(base_config_path: str, compare_config_path: str) -> bool:
    base = load_config(base_config_path).model_dump(mode="python")
    compare = load_config(compare_config_path).model_dump(mode="python")

    base["tribunal"].pop("fusion_mode", None)
    compare["tribunal"].pop("fusion_mode", None)
    base.pop("eqbsl", None)
    compare.pop("eqbsl", None)
    return base == compare


def _build_aggregator(config_path: str, source_ledger: str) -> TribunalAggregator:
    config = load_config(config_path)
    trust_store = LedgerStore(source_ledger)
    return TribunalAggregator(
        config.tribunal,
        eqbsl_config=config.eqbsl,
        ledger_store=trust_store,
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Replay tribunal logic against stored traces.")
    parser.add_argument("--source", required=True, help="Source ledger DB with stored traces.")
    parser.add_argument("--config", default="configs/gsm8k_memory_amplified.yaml")
    parser.add_argument("--gsm8k", default="data/gsm8k/test.jsonl", help="Path to GSM8K JSONL for ground truth.")
    parser.add_argument("--limit", type=int, default=None, help="Max tasks to replay.")
    parser.add_argument(
        "--task-ids",
        default=None,
        help="Optional comma-separated task-id slice to replay.",
    )
    parser.add_argument("--two-pass", action="store_true", help="Run two passes: seed memory in pass 1, apply in pass 2.")
    parser.add_argument(
        "--compare-config",
        default=None,
        help="Optional second config for paired pass-2 decision-layer comparison.",
    )
    parser.add_argument("--llm-judge", action="store_true", help="Enable LLM math judge (slow, calls API).")
    parser.add_argument("--out", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    config = load_config(args.config)
    aggregator = _build_aggregator(args.config, args.source)
    analyzer = UncertaintyAnalyzer()
    critic = TraceCritic(
        consistency_weight=0.30,
        rule_coherence_weight=0.25,
        morphology_weight=0.25,
        failure_similarity_weight=0.20,
        use_llm_judge_for_math=args.llm_judge,
    )
    invariant_extractor = InvariantExtractor()

    print(f"Loading traces from {args.source}...")
    traces_by_task = load_traces_from_ledger(args.source, limit=args.limit)
    tasks = load_tasks_from_ledger(args.source, args.gsm8k, limit=args.limit)
    task_ids_arg = [task_id.strip() for task_id in args.task_ids.split(",")] if args.task_ids else None
    tasks, traces_by_task, task_ids = filter_task_payloads(tasks, traces_by_task, task_ids_arg)

    print(f"Loaded {len(task_ids)} tasks with traces. Running replay...")

    if args.two_pass:
        tmp_db = tempfile.mktemp(suffix="_replay_memory.db")
        try:
            # ── Pass 1: no memory, seed store
            fm_store_p1 = FailureMemoryStore(tmp_db)
            extractor = FailureSignatureExtractor()
            pass1_results = []

            print("\n--- Pass 1 (no memory) ---")
            for i, tid in enumerate(task_ids):
                task = tasks[tid]
                traces = traces_by_task.get(tid, [])
                if not traces:
                    continue
                print(f"[{i+1}/{len(task_ids)}] Replaying {tid}...", end="\r", flush=True)
                decision, result = replay_task(
                    task, traces, aggregator, analyzer, critic,
                    invariant_extractor, fm_query=None
                )
                pass1_results.append(result)

                # Seed failure memory
                any_correct = result["any_correct"]
                gt_match = result["gt_match"]
                sig = extractor.extract(
                    task, traces,
                    [CritiqueResult(
                        trace_id=t.trace_id,
                        consistency_score=0.0, rule_coherence_score=0.0,
                        morphology_score=0.0, failure_similarity_penalty=0.0,
                        invariant_compliance_score=1.0, aggregate_score=0.0,
                        violated_invariants=[],
                    ) for t in traces],
                    decision,
                    analyzer.analyze(task, traces),
                    gt_match,
                    any_correct,
                )
                if sig is not None:
                    fm_store_p1.store(sig)

            p1_stats = print_stats(pass1_results, "Pass 1 — No Memory")

            # ── Pass 2: memory active
            if args.compare_config:
                if not _config_only_differs_in_eqbsl(args.config, args.compare_config):
                    raise ValueError(
                        "Paired compare requires configs to differ only in tribunal.fusion_mode and the eqbsl block."
                    )

                weighted_db = tempfile.mktemp(suffix="_weighted_memory.db")
                compare_db = tempfile.mktemp(suffix="_compare_memory.db")
                shutil.copyfile(tmp_db, weighted_db)
                shutil.copyfile(tmp_db, compare_db)

                weighted_store = FailureMemoryStore(weighted_db)
                compare_store = FailureMemoryStore(compare_db)
                fm_query_weighted = FailureMemoryQuery(
                    weighted_store,
                    penalty_scale=config.failure_memory.penalty_scale,
                )
                compare_config = load_config(args.compare_config)
                fm_query_compare = FailureMemoryQuery(
                    compare_store,
                    penalty_scale=compare_config.failure_memory.penalty_scale,
                )

                weighted_results = []
                compare_results = []
                compare_aggregator = _build_aggregator(args.compare_config, args.source)

                print("\n--- Pass 2A (baseline config) ---")
                for i, tid in enumerate(task_ids):
                    task = tasks[tid]
                    traces = traces_by_task.get(tid, [])
                    if not traces:
                        continue
                    print(f"[{i+1}/{len(task_ids)}] Replaying {tid}...", end="\r", flush=True)
                    _, result = replay_task(
                        task, traces, aggregator, analyzer, critic,
                        invariant_extractor, fm_query=fm_query_weighted
                    )
                    weighted_results.append(result)

                print("\n--- Pass 2B (compare config) ---")
                for i, tid in enumerate(task_ids):
                    task = tasks[tid]
                    traces = traces_by_task.get(tid, [])
                    if not traces:
                        continue
                    print(f"[{i+1}/{len(task_ids)}] Replaying {tid}...", end="\r", flush=True)
                    _, result = replay_task(
                        task, traces, compare_aggregator, analyzer, critic,
                        invariant_extractor, fm_query=fm_query_compare
                    )
                    compare_results.append(result)

                weighted_stats = print_stats(weighted_results, "Pass 2A — Baseline Decision Layer")
                compare_stats = print_stats(compare_results, "Pass 2B — Compare Decision Layer")

                weighted_task_map = {r["task_id"]: r for r in weighted_results}
                compare_task_map = {r["task_id"]: r for r in compare_results}
                diffs = []
                for tid in task_ids:
                    left = weighted_task_map.get(tid)
                    right = compare_task_map.get(tid)
                    if left is None or right is None:
                        continue
                    if (
                        left["decision"] != right["decision"]
                        or left["selected_answer"] != right["selected_answer"]
                    ):
                        diffs.append(
                            {
                                "task_id": tid,
                                "baseline_decision": left["decision"],
                                "compare_decision": right["decision"],
                                "baseline_selected_answer": left["selected_answer"],
                                "compare_selected_answer": right["selected_answer"],
                                "baseline_eqbsl": left.get("eqbsl", {}),
                                "compare_eqbsl": right.get("eqbsl", {}),
                            }
                        )

                comparison_report = {
                    "shared_trace_ids": {
                        tid: [trace.trace_id for trace in traces_by_task.get(tid, [])]
                        for tid in task_ids
                    },
                    "shared_seed_signature_count": fm_store_p1.get_stats()["total_signatures"],
                    "configs_only_differ_in_eqbsl": True,
                    "diffs": diffs,
                    "baseline_config": args.config,
                    "compare_config": args.compare_config,
                }

                output = {
                    "pass1": p1_stats,
                    "paired_pass2": {
                        "baseline": weighted_stats,
                        "compare": compare_stats,
                    },
                    "pass1_results": pass1_results,
                    "paired_results": {
                        "baseline": weighted_results,
                        "compare": compare_results,
                    },
                    "comparison_proof": comparison_report,
                }
                weighted_store.close()
                compare_store.close()
                try:
                    import os

                    os.unlink(weighted_db)
                    os.unlink(compare_db)
                except OSError:
                    pass
            else:
                fm_query_p2 = FailureMemoryQuery(
                    fm_store_p1,
                    penalty_scale=config.failure_memory.penalty_scale,
                )
                pass2_results = []

                print("\n--- Pass 2 (memory active) ---")
                for i, tid in enumerate(task_ids):
                    task = tasks[tid]
                    traces = traces_by_task.get(tid, [])
                    if not traces:
                        continue
                    print(f"[{i+1}/{len(task_ids)}] Replaying {tid}...", end="\r", flush=True)
                    decision, result = replay_task(
                        task, traces, aggregator, analyzer, critic,
                        invariant_extractor, fm_query=fm_query_p2
                    )
                    pass2_results.append(result)

                p2_stats = print_stats(pass2_results, "Pass 2 — Memory Active")

                print(f"\n{'─'*65}")
                print("  Trajectories: tasks that changed between passes")
                print(f"{'─'*65}")
                p1_map = {r["task_id"]: r for r in pass1_results}
                changed = 0
                for r2 in pass2_results:
                    tid = r2["task_id"]
                    r1 = p1_map.get(tid)
                    if not r1:
                        continue
                    if r1["gt_match"] != r2["gt_match"] or r1["decision"] != r2["decision"]:
                        changed += 1
                        g1 = "✓" if r1["gt_match"] else "✗" if r1["gt_match"] is False else "?"
                        g2 = "✓" if r2["gt_match"] else "✗" if r2["gt_match"] is False else "?"
                        status = "FIXED" if r2["gt_match"] and not r1["gt_match"] else "CHANGED"
                        mem = r2["fm_metadata"].get("failure_memory_candidates_penalised", 0)
                        coal = r2["fm_metadata"].get("failure_memory_coalitions_penalised", 0)
                        print(f"  {tid}: {r1['decision']}({g1}) → {r2['decision']}({g2})  [{status}]  "
                              f"traces_penalised={mem} coalitions={coal}")
                if changed == 0:
                    print("  (no changes between passes)")

                output = {"pass1": p1_stats, "pass2": p2_stats,
                          "pass1_results": pass1_results, "pass2_results": pass2_results}
        finally:
            try:
                import os

                os.unlink(tmp_db)
            except OSError:
                pass

    else:
        # Single pass, no memory
        results = []
        print("\n--- Single Pass (no memory) ---")
        for tid in task_ids:
            task = tasks[tid]
            traces = traces_by_task.get(tid, [])
            if not traces:
                continue
            _, result = replay_task(
                task, traces, aggregator, analyzer, critic,
                invariant_extractor, fm_query=None
            )
            results.append(result)
        stats = print_stats(results, "Single Pass")
        output = {"stats": stats, "results": results}

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
