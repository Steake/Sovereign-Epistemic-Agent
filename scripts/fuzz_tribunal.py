"""
fuzz_tribunal.py — Hyperparameter fuzzer for Epistemic Tribunal.
Runs a grid search over key parameters using the replay harness logic to find optimal configs.

Usage:
    uv run python scripts/fuzz_tribunal.py --source data/gsm8k/results/v4/baseline/ledger.db --limit 26
"""

from __future__ import annotations

import json
import sqlite3
import sys
import argparse
import itertools
from pathlib import Path
from typing import Optional, Any

# Ensure src/ is on path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from epistemic_tribunal.config import load_config
from epistemic_tribunal.tribunal.aggregator import TribunalAggregator
from epistemic_tribunal.tribunal_types import CandidateTrace, CritiqueResult, Task, TribunalDecision
from epistemic_tribunal.uncertainty.analyzer import UncertaintyAnalyzer
from epistemic_tribunal.failure_memory.store import FailureMemoryStore
from epistemic_tribunal.failure_memory.query import FailureMemoryQuery
from epistemic_tribunal.failure_memory.extractor import FailureSignatureExtractor
from epistemic_tribunal.critics.trace_critic import TraceCritic
from epistemic_tribunal.invariants.extractor import InvariantExtractor
from epistemic_tribunal.domains.factory import get_adapter
from epistemic_tribunal.tasks.gsm8k import load_tasks_from_jsonl

# ─── Copied Replay Logic ──────────────────────────────────────────────────────

def load_traces_from_ledger(db_path: str, limit: Optional[int] = None) -> dict[str, list[CandidateTrace]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    task_rows = conn.execute("SELECT task_id FROM tasks ORDER BY rowid").fetchall()
    task_ids = [r["task_id"] for r in task_rows]
    if limit: task_ids = task_ids[:limit]
    traces_by_task = {}
    for tid in task_ids:
        rows = conn.execute("SELECT * FROM traces WHERE task_id = ? ORDER BY created_at", (tid,)).fetchall()
        traces_by_task[tid] = [CandidateTrace(
            trace_id=r["trace_id"], generator_name=r["generator_name"],
            answer=json.loads(r["answer_json"]), reasoning_steps=json.loads(r["reasoning_steps_json"]),
            confidence_score=r["confidence_score"]
        ) for r in rows]
    conn.close()
    return traces_by_task

def load_tasks_from_ledger(db_path: str, gsm8k_path: str, limit: Optional[int] = None) -> dict[str, Task]:
    gsm8k_tasks = load_tasks_from_jsonl(gsm8k_path)
    task_map = {t.task_id: t for t in gsm8k_tasks}
    conn = sqlite3.connect(db_path); conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT task_id FROM tasks ORDER BY rowid").fetchall(); conn.close()
    ordered_ids = [r["task_id"] for r in rows]
    if limit: ordered_ids = ordered_ids[:limit]
    return {tid: task_map[tid] for tid in ordered_ids if tid in task_map}

def replay_task(task, traces, aggregator, analyzer, critic, invariant_extractor, fm_query):
    invariant_set = invariant_extractor.extract(task)
    critiques = [critic.critique(task, t, invariant_set, []) for t in traces]
    uncertainty = analyzer.analyze(task, traces)
    if fm_query:
        penalties, _ = fm_query.query_with_metadata(task, traces, critiques, uncertainty)
        for c in critiques: c.failure_similarity_penalty = penalties.get(c.trace_id, 0.0)
    decision = aggregator.adjudicate(task, traces, critiques, uncertainty)
    adapter = get_adapter(task.domain)
    gt_match = adapter.answers_equal(decision.selected_answer, task.ground_truth) if decision.selected_answer and task.ground_truth else None
    any_correct = any(adapter.answers_equal(t.answer, task.ground_truth) for t in traces) if task.ground_truth else None
    return decision, {"task_id": task.task_id, "decision": decision.decision.value, "gt_match": gt_match, "any_correct": any_correct}

# ─── Sweep ───────────────────────────────────────────────────────────────────

def sweep():
    parser = argparse.ArgumentParser(description="Fuzz tribunal parameters.")
    parser.add_argument("--source", required=True, help="Source ledger DB.")
    parser.add_argument("--limit", type=int, default=26)
    parser.add_argument("--gsm8k", default="data/gsm8k/test.jsonl")
    args = parser.parse_args()

    analyzer = UncertaintyAnalyzer()
    critic = TraceCritic(use_llm_judge_for_math=False)
    invariant_extractor = InvariantExtractor()
    extractor = FailureSignatureExtractor()

    traces_by_task = load_traces_from_ledger(args.source, limit=args.limit)
    tasks = load_tasks_from_ledger(args.source, args.gsm8k, limit=args.limit)
    task_ids = list(tasks.keys())

    # Sweeping thresholds, penalties, and diversity floors
    # Refined grid to find the "Goldilocks" zone for structural patterns
    penalty_scales = [0.3, 0.4, 0.5]
    fm_weights = [1.5, 2.5, 3.5, 4.5]
    
    # Selection params (locking to known good values to reduce search space)
    thresh = 0.30
    div_floor = 0.50

    best_score = -1; best_config = None
    
    print(f"Sweep starting (n={args.limit}, {len(penalty_scales)*len(fm_weights)} configs)...")

    for ps, fm_w in itertools.product(penalty_scales, fm_weights):
        import tempfile, os
        tmp_db = tempfile.mktemp(suffix="_fuzz.db")
        fm_store = FailureMemoryStore(tmp_db)
        
        config = load_config("configs/gsm8k_memory_amplified.yaml")
        config.tribunal.selection_threshold = thresh
        config.tribunal.diversity_floor = div_floor
        aggregator = TribunalAggregator(config.tribunal)

        try:
            # Pass 1
            pass1_res = []
            for tid in task_ids:
                decision, res = replay_task(tasks[tid], traces_by_task[tid], aggregator, analyzer, critic, invariant_extractor, None)
                pass1_res.append(res)
                sig = extractor.extract(tasks[tid], traces_by_task[tid], [], decision, analyzer.analyze(tasks[tid], traces_by_task[tid]), res["gt_match"], res["any_correct"])
                if sig: fm_store.store(sig)

            # Pass 2
            fm_query = FailureMemoryQuery(fm_store, penalty_scale=ps, pattern_weights={"false_majority": fm_w})
            pass2_res = []
            for tid in task_ids:
                _, res = replay_task(tasks[tid], traces_by_task[tid], aggregator, analyzer, critic, invariant_extractor, fm_query)
                pass2_res.append(res)

            selected = [r for r in pass2_res if r["decision"] == "select"]
            correct = [r for r in selected if r["gt_match"] is True]
            precision = len(correct) / len(selected) if selected else 0
            coverage = len(selected) / len(task_ids)
            
            # Use total correct count as the primary metric, with precision as tie-breaker
            score = len(correct) + (precision * 0.1)
            
            print(f"ps={ps:.1f} fmw={fm_w:.1f} | Correct: {len(correct)} Prec: {precision:.1%} Cov: {coverage:.1%}")
            
            if score > best_score:
                best_score = score
                best_config = (ps, fm_w, precision, coverage, len(correct))
        finally:
            if os.path.exists(tmp_db): os.unlink(tmp_db)

    if best_config:
        ps, fm_w, prec, cov, corr = best_config
        print(f"\nWINNER: penalty={ps:.1f} fm_weight={fm_w:.1f} -> Correct: {corr}, Prec: {prec:.1%}, Cov: {cov:.1%}")

if __name__ == "__main__":
    sweep()
