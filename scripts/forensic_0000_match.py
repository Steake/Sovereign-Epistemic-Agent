"""
forensic_0000_match.py — Find out which task's failure test_0000 is "blindly" matching.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Optional

# Ensure src/ is on path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from epistemic_tribunal.config import load_config
from epistemic_tribunal.tribunal.aggregator import TribunalAggregator
from epistemic_tribunal.tribunal_types import CandidateTrace, Task
from epistemic_tribunal.uncertainty.analyzer import UncertaintyAnalyzer
from epistemic_tribunal.failure_memory.store import FailureMemoryStore
from epistemic_tribunal.failure_memory.query import FailureMemoryQuery
from epistemic_tribunal.failure_memory.extractor import FailureSignatureExtractor
from epistemic_tribunal.critics.trace_critic import TraceCritic
from epistemic_tribunal.invariants.extractor import InvariantExtractor
from epistemic_tribunal.domains.factory import get_adapter
from epistemic_tribunal.tasks.gsm8k import load_tasks_from_jsonl

# ─── Replay Logic ────────────────────────────────────────────────────────────

def load_traces_from_ledger(db_path: str, limit: Optional[int] = None) -> dict[str, list[CandidateTrace]]:
    conn = sqlite3.connect(db_path); conn.row_factory = sqlite3.Row
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
    conn.close(); return traces_by_task

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
        penalties, meta = fm_query.query_with_metadata(task, traces, critiques, uncertainty)
        for c in critiques: c.failure_similarity_penalty = penalties.get(c.trace_id, 0.0)
        return penalties, meta
    decision = aggregator.adjudicate(task, traces, critiques, uncertainty)
    adapter = get_adapter(task.domain)
    gt_match = adapter.answers_equal(decision.selected_answer, task.ground_truth) if decision.selected_answer and task.ground_truth else None
    any_correct = any(adapter.answers_equal(t.answer, task.ground_truth) for t in traces) if task.ground_truth else None
    return decision, {"task_id": task.task_id, "decision": decision.decision.value, "gt_match": gt_match, "any_correct": any_correct}

# ─── Audit ───────────────────────────────────────────────────────────────────

def audit():
    source_db = "data/gsm8k/results/v4/baseline/ledger.db"
    gsm8k_path = "data/gsm8k/test.jsonl"
    config = load_config("configs/gsm8k_memory_amplified.yaml")
    
    analyzer = UncertaintyAnalyzer()
    critic = TraceCritic(use_llm_judge_for_math=False)
    invariant_extractor = InvariantExtractor()
    extractor = FailureSignatureExtractor()
    aggregator = TribunalAggregator(config.tribunal)
    
    traces_by_task = load_traces_from_ledger(source_db, limit=26)
    tasks = load_tasks_from_ledger(source_db, gsm8k_path, limit=26)
    task_ids = list(tasks.keys())
    
    import tempfile
    import os
    tmp_db = tempfile.mktemp(suffix="_forensic_match.db")
    fm_store = FailureMemoryStore(tmp_db)
    
    # Pass 1: Populate memory with ALL failures from the 26-task set
    print("Populating memory from Pass 1...")
    for tid in task_ids:
        task = tasks[tid]
        traces = traces_by_task[tid]
        decision, res = replay_task(task, traces, aggregator, analyzer, critic, invariant_extractor, None)
        sig = extractor.extract(task, traces, [], decision, analyzer.analyze(task, traces), res["gt_match"], res["any_correct"])
        if sig and sig.failure_type == "wrong_pick": # Only store failures
            fm_store.store(sig)
            # print(f"  Stored failure for {tid}")

    # Pass 2: Query for target task
    import sys
    target_id = sys.argv[1] if len(sys.argv) > 1 else "test_0000"
    if target_id not in tasks:
        print(f"Task {target_id} not found.")
        return

    fm_query = FailureMemoryQuery(fm_store, penalty_scale=0.375)
    target_task = tasks[target_id]
    target_traces = traces_by_task[target_id]
    
    print(f"\nQUERYING MEMORY FOR {target_task.task_id}...")
    penalties, meta = replay_task(target_task, target_traces, aggregator, analyzer, critic, invariant_extractor, fm_query)
    
    print(f"Total Penalties: {penalties}")
    print(f"Decomposition: {json.dumps(meta.get('failure_memory_decomposition'), indent=2)}")
    
    probe = fm_query.build_probe(target_task, target_traces, [], analyzer.analyze(target_task, target_traces))
    matches = fm_store.query_similar(probe)
    
    print(f"\nRAW MATCHES FROM STORE (Total: {len(matches)}):")
    for match in matches:
        sig = match.signature
        print(f"  - Source Task: {sig.task_id}")
        print(f"    Similarity:  {match.similarity:.4f}")
        print(f"    Sig Answer:  {sig.answer_signature}")
        print(f"    Sig Type:    {sig.failure_type}")
        print(f"    Outcome:     {sig.outcome_label}")
        print(f"    Features:    {match.matching_features}")

    if os.path.exists(tmp_db): os.unlink(tmp_db)

    if os.path.exists(tmp_db): os.unlink(tmp_db)

if __name__ == "__main__":
    audit()
