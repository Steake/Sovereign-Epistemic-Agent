"""
forensic_0000.py — Deep dive into the test_0000 regression.
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

# ─── Logic ───────────────────────────────────────────────────────────────────

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
        penalties, _ = fm_query.query_with_metadata(task, traces, critiques, uncertainty)
        for c in critiques: c.failure_similarity_penalty = penalties.get(c.trace_id, 0.0)
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
    config.tribunal.selection_threshold = 0.30
    config.tribunal.diversity_floor = 0.50
    penalty_scale = 0.2
    
    analyzer = UncertaintyAnalyzer()
    critic = TraceCritic(use_llm_judge_for_math=False)
    invariant_extractor = InvariantExtractor()
    extractor = FailureSignatureExtractor()
    
    traces_by_task = load_traces_from_ledger(source_db, limit=26)
    tasks = load_tasks_from_ledger(source_db, gsm8k_path, limit=26)
    
    # Task to audit
    import sys
    task_id = sys.argv[1] if len(sys.argv) > 1 else "test_0000"
    if task_id not in tasks:
        print(f"Task {task_id} not found.")
        return
    
    task = tasks[task_id]
    traces = traces_by_task[task_id]
    
    print(f"AUDITING TASK: {task.task_id}")
    print(f"Ground Truth: {task.ground_truth}")
    for t in traces:
        print(f"  Trace {t.trace_id} ({t.generator_name}): Answer={t.answer}")

    aggregator = TribunalAggregator(config.tribunal)
    decision1, res1 = replay_task(task, traces, aggregator, analyzer, critic, invariant_extractor, None)
    
    print("\nPASS 1 RESULT:")
    print(f"  Decision: {decision1.decision.value}")
    print(f"  Selected: {decision1.selected_trace_id} (Answer: {decision1.selected_answer})")
    print(f"  Confidence: {decision1.confidence:.4f}")
    print(f"  GT Match: {res1['gt_match']}")
    
    import tempfile
    import os
    tmp_db = tempfile.mktemp(suffix="_0000.db")
    fm_store = FailureMemoryStore(tmp_db)
    sig = extractor.extract(task, traces, [], decision1, analyzer.analyze(task, traces), res1["gt_match"], res1["any_correct"])
    if sig:
        print("\nStored Signature in Memory:")
        print(f"  Signature Answer: {sig.answer_signature}")
        print(f"  Outcome: {sig.failure_type}")
        fm_store.store(sig)
    
    fm_query = FailureMemoryQuery(fm_store, penalty_scale=penalty_scale)
    decision2, res2 = replay_task(task, traces, aggregator, analyzer, critic, invariant_extractor, fm_query)
    
    print("\nPASS 2 RESULT (with memory):")
    print(f"  Decision: {decision2.decision.value}")
    print(f"  Selected: {decision2.selected_trace_id} (Answer: {decision2.selected_answer})")
    print(f"  Confidence: {decision2.confidence:.4f}")
    print(f"  GT Match: {res2['gt_match']}")
    
    forensic = decision2.metadata.get("forensic", [])
    print("\nForensic Breakdown (Pass 2):")
    for f in forensic:
        print(f"  Trace {f['generator']}: Total={f['total']:.4f}")
        print(f"    U: {f['U']:.4f} (Uncertainty)")
        print(f"    C: {f['C']:.4f} (Critique)")
        print(f"    M: {f['M']:.4f} (Memory Penalty)")
        print(f"    V: {f['V']:.4f} (Visual/Structural)")
        
        # New decomposition metadata
        fm_meta = decision2.metadata.get("failure_memory_decomposition", {}).get(f['generator'], {})
        if fm_meta:
            print("    --- Memory Decomposition ---")
            print(f"    Exact Penalty: {fm_meta.get('exact_penalty', 0.0):.4f} (Matches: {fm_meta.get('n_exact_matches', 0)})")
            print(f"    Structural Penalty: {fm_meta.get('structural_penalty', 0.0):.4f} (Matches: {fm_meta.get('n_structural_matches', 0)})")
            print(f"    Top Structural Similarity: {fm_meta.get('top_structural_similarity', 0.0):.4f}")
    
    if os.path.exists(tmp_db): os.unlink(tmp_db)

if __name__ == "__main__":
    audit()
