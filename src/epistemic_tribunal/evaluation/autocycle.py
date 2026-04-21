"""Autonomous cycle helpers for tribunal and EQBSL evaluation.

These helpers are intentionally narrow:
- use replay first
- build compact blocker slices
- validate policy changes on synthetic EQBSL fixtures
- append one structured log record per cycle
"""

from __future__ import annotations

import json
import sqlite3
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Optional

from epistemic_tribunal.config import load_config
from epistemic_tribunal.domains.factory import get_adapter
from epistemic_tribunal.eqbsl.decision import EqbslDecisionPolicy
from epistemic_tribunal.eqbsl.models import CoalitionOpinion, EvidenceOpinion
from epistemic_tribunal.ledger.store import LedgerStore
from epistemic_tribunal.tribunal.aggregator import TribunalAggregator
from epistemic_tribunal.tribunal_types import (
    CandidateTrace,
    CritiqueResult,
    DecisionKind,
    ExperimentRun,
    Task,
    TribunalDecision,
)
from epistemic_tribunal.evaluation.metrics import summary_report
from epistemic_tribunal.uncertainty.analyzer import UncertaintyAnalyzer
from epistemic_tribunal.critics.trace_critic import TraceCritic
from epistemic_tribunal.invariants.extractor import InvariantExtractor
from epistemic_tribunal.failure_memory.extractor import FailureSignatureExtractor
from epistemic_tribunal.failure_memory.query import FailureMemoryQuery
from epistemic_tribunal.failure_memory.store import FailureMemoryStore
from epistemic_tribunal.tasks.arc_like import load_task_from_file
from epistemic_tribunal.tasks.gsm8k import load_tasks_from_jsonl


AUTOCYCLE_PRIORITY: list[str] = [
    "unrecoverable_wrong_select",
    "bad_abstention",
    "same_answer_tie_nonselect",
    "low_gap_nonselect",
]


def _ordered_task_ids_from_ledger(db_path: str, limit: Optional[int] = None) -> list[str]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT task_id FROM tasks ORDER BY rowid").fetchall()
    conn.close()
    ordered_ids = [row["task_id"] for row in rows]
    if limit:
        ordered_ids = ordered_ids[:limit]
    return ordered_ids


def load_traces_from_ledger(db_path: str, limit: Optional[int] = None) -> dict[str, list[CandidateTrace]]:
    """Load stored traces grouped by task_id."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    task_rows = conn.execute("SELECT task_id FROM tasks ORDER BY rowid").fetchall()
    task_ids = [row["task_id"] for row in task_rows]
    if limit:
        task_ids = task_ids[:limit]

    traces_by_task: dict[str, list[CandidateTrace]] = {}
    for task_id in task_ids:
        rows = conn.execute(
            "SELECT * FROM traces WHERE task_id = ? ORDER BY created_at",
            (task_id,),
        ).fetchall()
        traces_by_task[task_id] = [
            CandidateTrace(
                trace_id=row["trace_id"],
                generator_name=row["generator_name"],
                answer=json.loads(row["answer_json"]),
                reasoning_steps=json.loads(row["reasoning_steps_json"]),
                confidence_score=row["confidence_score"],
            )
            for row in rows
        ]
    conn.close()
    return traces_by_task


def load_tasks_from_ledger(
    db_path: str,
    gsm8k_path: str,
    limit: Optional[int] = None,
) -> dict[str, Task]:
    """Load replay tasks using GSM8K ground truth plus ledger ordering."""
    gsm8k_tasks = load_tasks_from_jsonl(gsm8k_path)
    task_map = {task.task_id: task for task in gsm8k_tasks}

    ordered_ids = _ordered_task_ids_from_ledger(db_path, limit=limit)
    return {task_id: task_map[task_id] for task_id in ordered_ids if task_id in task_map}


def load_arc_tasks_from_dataset(
    db_path: str,
    dataset_path: str,
    limit: Optional[int] = None,
) -> dict[str, Task]:
    """Load ARC-like replay tasks using task IDs stored in the ledger."""
    dataset_root = Path(dataset_path)
    ordered_ids = _ordered_task_ids_from_ledger(db_path, limit=limit)
    tasks_by_id: dict[str, Task] = {}
    for task_id in ordered_ids:
        task_path = dataset_root / f"{task_id}.json"
        if not task_path.exists():
            continue
        tasks_by_id[task_id] = load_task_from_file(task_path)
    return tasks_by_id


def load_replay_tasks(
    *,
    db_path: str,
    gsm8k_path: Optional[str] = None,
    arc_dataset_path: Optional[str] = None,
    limit: Optional[int] = None,
) -> dict[str, Task]:
    """Load replay tasks for either GSM8K or ARC-like ledgers."""
    if gsm8k_path:
        return load_tasks_from_ledger(db_path, gsm8k_path, limit=limit)
    if arc_dataset_path:
        return load_arc_tasks_from_dataset(db_path, arc_dataset_path, limit=limit)
    raise ValueError("Either gsm8k_path or arc_dataset_path must be supplied for replay.")


def replay_task(
    task: Task,
    traces: list[CandidateTrace],
    aggregator: TribunalAggregator,
    analyzer: UncertaintyAnalyzer,
    critic: TraceCritic,
    invariant_extractor: InvariantExtractor,
    fm_query: Optional[FailureMemoryQuery],
) -> tuple[TribunalDecision, dict[str, Any]]:
    """Re-run tribunal logic for one stored-trace task."""
    invariant_set = invariant_extractor.extract(task)
    critiques = [
        critic.critique(task, trace, invariant_set, [])
        for trace in traces
    ]
    uncertainty = analyzer.analyze(task, traces)

    fm_metadata: dict[str, Any] = {}
    if fm_query is not None:
        penalties, fm_metadata = fm_query.query_with_metadata(task, traces, critiques, uncertainty)
        for critique in critiques:
            critique.failure_similarity_penalty = penalties.get(critique.trace_id, 0.0)

    decision = aggregator.adjudicate(
        task,
        traces,
        critiques,
        uncertainty,
        failure_memory_metadata=fm_metadata,
    )

    adapter = get_adapter(task.domain)
    gt_match: Optional[bool] = None
    if decision.selected_answer is not None and task.ground_truth is not None:
        gt_match = adapter.answers_equal(decision.selected_answer, task.ground_truth)

    any_correct = (
        any(adapter.answers_equal(trace.answer, task.ground_truth) for trace in traces)
        if task.ground_truth is not None
        else None
    )
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
        "coalitions": decision.metadata.get("_eqbsl_coalitions", []),
        "selected_trace_id": decision.selected_trace_id,
        "run_reason": decision.reasoning,
    }
    return decision, result


def append_cycle_log(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def _make_experiment_run(
    result: dict[str, Any],
    task: Any,
    traces: list[CandidateTrace],
) -> ExperimentRun:
    return ExperimentRun(
        run_id=f"autocycle:{result['task_id']}",
        task_id=result["task_id"],
        generator_names=[trace.generator_name for trace in traces],
        decision=DecisionKind(result["decision"]),
        confidence=result["confidence"],
        selected_trace_id=result["selected_trace_id"],
        ground_truth_match=result["gt_match"],
        duration_seconds=0.0,
        config_snapshot={},
        metadata={
            "cohort": classify_task_cohort(task, traces),
            "any_correct": result["any_correct"],
            "eqbsl": result.get("eqbsl", {}),
            "greedy_correct": result.get("greedy_correct", False),
        },
    )


def classify_task_cohort(task: Any, traces: list[CandidateTrace]) -> str:
    adapter = get_adapter(task.domain)
    any_correct = any(adapter.answers_equal(trace.answer, task.ground_truth) for trace in traces)
    if not any_correct:
        return "contested-unrecoverable"

    normalised = [str(adapter.normalize_answer(trace.answer)) for trace in traces]
    if len(set(normalised)) == 1:
        return "control-trivial"
    return "contested-recoverable"


def _build_replay_environment(
    *,
    source_ledger: str,
    gsm8k_path: Optional[str],
    arc_dataset_path: Optional[str],
    limit: Optional[int],
    task_ids: Optional[Iterable[str]] = None,
) -> tuple[dict[str, Any], dict[str, list[CandidateTrace]], list[str]]:
    traces_by_task = load_traces_from_ledger(source_ledger, limit=limit)
    tasks_by_id = load_replay_tasks(
        db_path=source_ledger,
        gsm8k_path=gsm8k_path,
        arc_dataset_path=arc_dataset_path,
        limit=limit,
    )

    ordered_ids = [task_id for task_id in tasks_by_id.keys() if task_id in traces_by_task]
    if task_ids is not None:
        wanted = set(task_ids)
        ordered_ids = [task_id for task_id in ordered_ids if task_id in wanted]

    tasks_by_id = {task_id: tasks_by_id[task_id] for task_id in ordered_ids}
    traces_by_task = {task_id: traces_by_task[task_id] for task_id in ordered_ids}
    return tasks_by_id, traces_by_task, ordered_ids


def build_seed_signatures(
    *,
    source_ledger: str,
    gsm8k_path: Optional[str] = None,
    arc_dataset_path: Optional[str] = None,
    config_path: str,
    limit: Optional[int] = None,
    task_ids: Optional[Iterable[str]] = None,
    fusion_mode_override: Optional[str] = None,
) -> dict[str, Any]:
    """Build one shared replay-memory seed set from a single pass-1 config."""
    tasks_by_id, traces_by_task, ordered_ids = _build_replay_environment(
        source_ledger=source_ledger,
        gsm8k_path=gsm8k_path,
        arc_dataset_path=arc_dataset_path,
        limit=limit,
        task_ids=task_ids,
    )
    config = load_config(config_path)
    if fusion_mode_override is not None:
        config.tribunal.fusion_mode = fusion_mode_override
        config.eqbsl.enabled = fusion_mode_override == "eqbsl"
    analyzer = UncertaintyAnalyzer()
    critic = TraceCritic(
        consistency_weight=0.30,
        rule_coherence_weight=0.25,
        morphology_weight=0.25,
        failure_similarity_weight=0.20,
        use_llm_judge_for_math=False,
    )
    invariant_extractor = InvariantExtractor()
    aggregator = TribunalAggregator(
        config.tribunal,
        eqbsl_config=config.eqbsl,
        ledger_store=LedgerStore(source_ledger),
    )
    extractor = FailureSignatureExtractor()
    signatures = []
    for task_id in ordered_ids:
        task = tasks_by_id[task_id]
        traces = traces_by_task[task_id]
        decision, result = replay_task(
            task,
            traces,
            aggregator,
            analyzer,
            critic,
            invariant_extractor,
            fm_query=None,
        )
        signature = extractor.extract(
            task,
            traces,
            [
                CritiqueResult(
                    trace_id=trace.trace_id,
                    consistency_score=0.0,
                    rule_coherence_score=0.0,
                    morphology_score=0.0,
                    failure_similarity_penalty=0.0,
                    invariant_compliance_score=1.0,
                    aggregate_score=0.0,
                    violated_invariants=[],
                )
                for trace in traces
            ],
            decision,
            analyzer.analyze(task, traces),
            result["gt_match"],
            result["any_correct"],
        )
        if signature is not None:
            signatures.append(signature)
    return {"task_ids": ordered_ids, "signatures": signatures}


def run_replay_evaluation(
    *,
    source_ledger: str,
    gsm8k_path: Optional[str] = None,
    arc_dataset_path: Optional[str] = None,
    config_path: str,
    limit: Optional[int] = None,
    task_ids: Optional[Iterable[str]] = None,
    seed_signatures: Optional[list[Any]] = None,
    fusion_mode_override: Optional[str] = None,
) -> dict[str, Any]:
    tasks_by_id, traces_by_task, ordered_ids = _build_replay_environment(
        source_ledger=source_ledger,
        gsm8k_path=gsm8k_path,
        arc_dataset_path=arc_dataset_path,
        limit=limit,
        task_ids=task_ids,
    )

    config = load_config(config_path)
    if fusion_mode_override is not None:
        config.tribunal.fusion_mode = fusion_mode_override
        config.eqbsl.enabled = fusion_mode_override == "eqbsl"
    analyzer = UncertaintyAnalyzer()
    critic = TraceCritic(
        consistency_weight=0.30,
        rule_coherence_weight=0.25,
        morphology_weight=0.25,
        failure_similarity_weight=0.20,
        use_llm_judge_for_math=False,
    )
    invariant_extractor = InvariantExtractor()
    aggregator = TribunalAggregator(
        config.tribunal,
        eqbsl_config=config.eqbsl,
        ledger_store=LedgerStore(source_ledger),
    )

    # Seed a replay-local failure-memory ledger from one shared pass-1 signature set.
    replay_db = Path("/tmp") / f"autocycle_{Path(config_path).stem}_{sqlite3.connect(':memory:').execute('select hex(randomblob(4))').fetchone()[0]}.db"
    fm_store = FailureMemoryStore(str(replay_db))
    if seed_signatures is None:
        seed_payload = build_seed_signatures(
            source_ledger=source_ledger,
            gsm8k_path=gsm8k_path,
            arc_dataset_path=arc_dataset_path,
            config_path=config_path,
            limit=limit,
            task_ids=ordered_ids,
            fusion_mode_override=fusion_mode_override,
        )
        seed_signatures = seed_payload["signatures"]
    for signature in seed_signatures:
        fm_store.store(signature)
    fm_query = FailureMemoryQuery(
        fm_store,
        penalty_scale=config.failure_memory.penalty_scale,
    )

    results: list[dict[str, Any]] = []
    experiment_runs: list[ExperimentRun] = []
    for task_id in ordered_ids:
        task = tasks_by_id[task_id]
        traces = traces_by_task[task_id]
        _, result = replay_task(
            task,
            traces,
            aggregator,
            analyzer,
            critic,
            invariant_extractor,
            fm_query=fm_query,
        )
        adapter = get_adapter(task.domain)
        greedy_trace = next((trace for trace in traces if trace.generator_name == "llm"), None)
        greedy_correct = (
            adapter.answers_equal(greedy_trace.answer, task.ground_truth)
            if greedy_trace is not None
            else False
        )
        result["greedy_correct"] = greedy_correct
        result["cohort"] = classify_task_cohort(task, traces)
        results.append(result)
        experiment_runs.append(_make_experiment_run(result, task, traces))

    metrics = summary_report(experiment_runs)
    try:
        fm_store.close()
        replay_db.unlink(missing_ok=True)
    except OSError:
        pass

    return {
        "config_path": config_path,
        "task_ids": ordered_ids,
        "results": results,
        "metrics": metrics,
    }


def detect_dominant_blocker(results: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = {
        "unrecoverable_wrong_select": [],
        "bad_abstention": [],
        "same_answer_tie_nonselect": [],
        "low_gap_nonselect": [],
    }

    for result in results:
        eqbsl = result.get("eqbsl", {})
        reason = (eqbsl.get("task_reason") or {}).get("reason_code")
        same_answer = bool(eqbsl.get("same_answer_tie_case"))
        if (
            result["decision"] == "select"
            and result["gt_match"] is False
            and result["any_correct"] is False
        ):
            buckets["unrecoverable_wrong_select"].append(result)
        if result["decision"] != "select" and result["any_correct"] is True:
            buckets["bad_abstention"].append(result)
        if result["decision"] != "select" and same_answer:
            buckets["same_answer_tie_nonselect"].append(result)
        if result["decision"] != "select" and reason == "abstain_low_gap":
            buckets["low_gap_nonselect"].append(result)

    for blocker in AUTOCYCLE_PRIORITY:
        if buckets[blocker]:
            return {
                "blocker": blocker,
                "count": len(buckets[blocker]),
                "task_ids": [item["task_id"] for item in buckets[blocker]],
            }

    return {"blocker": "none", "count": 0, "task_ids": []}


def build_forensic_summary(results: list[dict[str, Any]], task_ids: Iterable[str]) -> list[dict[str, Any]]:
    wanted = set(task_ids)
    summary = []
    for result in results:
        if result["task_id"] not in wanted:
            continue
        eqbsl = result.get("eqbsl", {})
        summary.append(
            {
                "task_id": result["task_id"],
                "decision": result["decision"],
                "ground_truth_match": result["gt_match"],
                "any_correct": result["any_correct"],
                "selected_answer": result["selected_answer"],
                "task_reason": eqbsl.get("task_reason", {}),
                "top_two_expectations": eqbsl.get("top_two_expectations", []),
                "top_gap": eqbsl.get("top_gap"),
                "same_answer_tie_case": eqbsl.get("same_answer_tie_case", False),
                "run_reason": result.get("run_reason"),
            }
        )
    return summary


def generated_eqbsl_fixtures(
    failure_classes: Optional[set[str]] = None,
) -> list[dict[str, Any]]:
    def coalition(
        *,
        signature: str,
        belief: float,
        disbelief: float,
        uncertainty: float,
        base_rate: float = 0.25,
        generators: Optional[list[str]] = None,
        trace_ids: Optional[list[str]] = None,
        coalition_mass: float = 0.5,
        verification_classification: str | None = None,
        verification_confidence: float | None = None,
    ) -> CoalitionOpinion:
        source_opinions = {}
        if verification_classification is not None and verification_confidence is not None:
            source_opinions["verification"] = {
                "source_name": "verification",
                "source_type": "verification",
                "trust_weight": 0.9,
                "opinion": EvidenceOpinion.neutral(
                    base_rate=base_rate,
                    prior_weight=2.0,
                    metadata={"fixture": True},
                ),
                "metadata": {
                    "classification": verification_classification,
                    "confidence": verification_confidence,
                    "rationale_tags": ["fixture"],
                },
            }
        return CoalitionOpinion(
            answer_signature=signature,
            coalition_member_trace_ids=trace_ids or [signature],
            coalition_member_generators=generators or ["llm_cot"],
            representative_trace_id=(trace_ids or [signature])[0],
            representative_generator=(generators or ["llm_cot"])[0],
            source_opinions=source_opinions,
            fused_opinion=EvidenceOpinion(
                belief=belief,
                disbelief=disbelief,
                uncertainty=uncertainty,
                base_rate=base_rate,
                positive_evidence=belief * 10.0,
                negative_evidence=disbelief * 10.0,
                prior_weight=2.0,
            ),
            explanation_metadata={"coalition_mass": coalition_mass},
        )

    fixtures = [
        {
            "fixture_id": "same_answer_tie_strong_select",
            "failure_class": "same_answer_tie_issue",
            "domains": ["gsm8k", "arc_like"],
            "expected_decision": "select",
            "coalitions": [
                coalition(
                    signature="(9,)",
                    belief=0.72,
                    disbelief=0.08,
                    uncertainty=0.20,
                    generators=["llm_cot", "llm"],
                    trace_ids=["t1", "t2"],
                    coalition_mass=0.667,
                ),
                coalition(
                    signature="(10,)",
                    belief=0.32,
                    disbelief=0.28,
                    uncertainty=0.40,
                    generators=["llm_warm"],
                    trace_ids=["t3"],
                    coalition_mass=0.333,
                ),
            ],
        },
        {
            "fixture_id": "high_uncertainty_abstain",
            "failure_class": "over_conservative_abstention",
            "domains": ["gsm8k", "arc_like"],
            "expected_decision": "abstain",
            "coalitions": [
                coalition(signature="(9,)", belief=0.15, disbelief=0.05, uncertainty=0.80),
                coalition(signature="(10,)", belief=0.14, disbelief=0.06, uncertainty=0.80),
            ],
        },
        {
            "fixture_id": "memory_heavy_low_gap_abstain",
            "failure_class": "low_gap_unrecoverable_case",
            "domains": ["gsm8k", "arc_like"],
            "expected_decision": "abstain",
            "coalitions": [
                coalition(signature="(12,)", belief=0.635472, disbelief=0.114538, uncertainty=0.24999),
                coalition(signature="(18,)", belief=0.620419, disbelief=0.186224, uncertainty=0.193357),
            ],
        },
        {
            "fixture_id": "recoverable_low_gap_should_select",
            "failure_class": "low_gap_recoverable_case",
            "domains": ["gsm8k", "arc_like"],
            "expected_decision": "select",
            "coalitions": [
                coalition(signature="(366,)", belief=0.635472, disbelief=0.114538, uncertainty=0.24999),
                coalition(signature="(228,)", belief=0.58895, disbelief=0.144395, uncertainty=0.266655),
            ],
        },
        {
            "fixture_id": "single_generator_diversity_floor_resample",
            "failure_class": "under_conservative_wrong_select",
            "domains": ["gsm8k", "arc_like"],
            "expected_decision": "resample",
            "coalitions": [
                coalition(
                    signature="(7,)",
                    belief=0.70,
                    disbelief=0.10,
                    uncertainty=0.20,
                    generators=["llm", "llm", "llm"],
                    trace_ids=["a", "b", "c"],
                    coalition_mass=0.95,
                ),
                coalition(signature="(8,)", belief=0.20, disbelief=0.40, uncertainty=0.40),
            ],
        },
        {
            "fixture_id": "verification_supported_low_gap_select",
            "failure_class": "contradiction_pool_geometry_case",
            "domains": ["gsm8k", "arc_like"],
            "expected_decision": "select",
            "coalitions": [
                coalition(
                    signature="(42,)",
                    belief=0.62,
                    disbelief=0.17,
                    uncertainty=0.21,
                    verification_classification="support",
                    verification_confidence=0.9,
                ),
                coalition(
                    signature="(43,)",
                    belief=0.60,
                    disbelief=0.18,
                    uncertainty=0.22,
                    verification_classification="contradiction",
                    verification_confidence=0.9,
                ),
            ],
        },
        {
            "fixture_id": "verification_contradicted_high_margin_abstain",
            "failure_class": "contradiction_pool_geometry_case",
            "domains": ["gsm8k", "arc_like"],
            "expected_decision": "abstain",
            "coalitions": [
                coalition(
                    signature="(99,)",
                    belief=0.72,
                    disbelief=0.08,
                    uncertainty=0.20,
                    coalition_mass=0.95,
                    generators=["llm", "greedy", "diverse"],
                    trace_ids=["v1", "v2", "v3"],
                    verification_classification="contradiction",
                    verification_confidence=0.95,
                ),
                coalition(
                    signature="(98,)",
                    belief=0.22,
                    disbelief=0.42,
                    uncertainty=0.36,
                    verification_classification="inconclusive",
                    verification_confidence=0.0,
                ),
            ],
        },
    ]
    if failure_classes is None:
        return fixtures
    return [fixture for fixture in fixtures if fixture["failure_class"] in failure_classes]


def evaluate_generated_fixtures(
    config_path: str,
    *,
    failure_classes: Optional[set[str]] = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    policy = EqbslDecisionPolicy(config.tribunal, config.eqbsl)
    rows = []
    for fixture in generated_eqbsl_fixtures(failure_classes=failure_classes):
        coalitions = deepcopy(fixture["coalitions"])
        result = policy.decide(coalitions)
        actual = result.decision.value
        rows.append(
            {
                "fixture_id": fixture["fixture_id"],
                "failure_class": fixture["failure_class"],
                "expected_decision": fixture["expected_decision"],
                "actual_decision": actual,
                "passed": actual == fixture["expected_decision"],
                "reason_code": result.reason.reason_code,
                "top_gap": result.top_gap,
            }
        )

    return {
        "config_path": config_path,
        "fixtures": rows,
        "passed": sum(1 for row in rows if row["passed"]),
        "failed": sum(1 for row in rows if not row["passed"]),
    }


def decide_cycle_outcome(
    *,
    current_full: dict[str, Any],
    candidate_full: dict[str, Any],
    current_subset: dict[str, Any],
    candidate_subset: dict[str, Any],
    current_generated: dict[str, Any],
    candidate_generated: dict[str, Any],
) -> dict[str, Any]:
    current_wrong = current_full["metrics"]["wrong_pick_count"]
    candidate_wrong = candidate_full["metrics"]["wrong_pick_count"]
    current_bad_abst = current_full["metrics"]["abstention_metrics"]["bad_abstentions"]
    candidate_bad_abst = candidate_full["metrics"]["abstention_metrics"]["bad_abstentions"]
    current_generated_failures = current_generated["failed"]
    candidate_generated_failures = candidate_generated["failed"]

    if (
        candidate_wrong < current_wrong
        and candidate_bad_abst <= current_bad_abst
        and candidate_generated_failures <= current_generated_failures
    ):
        return {
            "decision": "accept",
            "reason": "Candidate reduces wrong picks without worsening abstention quality or generated-fixture safety.",
        }

    if (
        candidate_wrong <= current_wrong
        and candidate_bad_abst < current_bad_abst
        and candidate_generated_failures <= current_generated_failures
    ):
        return {
            "decision": "accept",
            "reason": "Candidate reduces bad abstentions while holding wrong picks and fixture safety.",
        }

    if candidate_wrong < current_wrong and candidate_bad_abst > current_bad_abst:
        return {
            "decision": "reject",
            "reason": "Candidate buys safety only by reintroducing recoverable abstentions.",
        }

    if candidate_wrong > current_wrong and candidate_bad_abst <= current_bad_abst:
        return {
            "decision": "reject",
            "reason": "Candidate reduces abstention conservatism but gives back wrong-pick safety.",
        }

    if candidate_generated_failures > current_generated_failures:
        return {
            "decision": "reject",
            "reason": "Candidate creates a generated-fixture regression before live validation.",
        }

    subset_delta = (
        candidate_subset["metrics"]["wrong_pick_count"] - current_subset["metrics"]["wrong_pick_count"]
    )
    if subset_delta == 0:
        return {
            "decision": "park",
            "reason": "Candidate does not separate cleanly on the focused blocker slice.",
        }

    return {
        "decision": "park",
        "reason": "Candidate changes the slice, but the tradeoff is not strong enough for promotion.",
    }
