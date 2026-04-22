"""End-to-end Orchestrator — wires every module together into one pipeline.

Pipeline stages
---------------
1. **Generate** — run all enabled generators to build a candidate pool.
2. **Extract invariants** — infer structural constraints from the task.
3. **Critique** — score each trace with the TraceCritic.
4. **Analyse uncertainty** — compute entropy / margin / coalition signals.
5. **Adjudicate** — tribunal aggregator selects the best trace or abstains.
6. **Evaluate** — compare against ground truth when available.
7. **Persist** — write records to the failure ledger.
8. **Report** — return structured output (ExperimentRun + JSON-serialisable dict).
"""

from __future__ import annotations


import time
import uuid
from typing import Any, Callable, Optional

from epistemic_tribunal.config import TribunalSettings, load_config
from epistemic_tribunal.critics.trace_critic import TraceCritic
from epistemic_tribunal.failure_memory.constraint_builder import FailureConstraintBuilder
from epistemic_tribunal.failure_memory.extractor import FailureSignatureExtractor
from epistemic_tribunal.failure_memory.models import FailureConstraints
from epistemic_tribunal.failure_memory.query import FailureMemoryQuery
from epistemic_tribunal.failure_memory.store import FailureMemoryStore
from epistemic_tribunal.generators.base import build_generators
from epistemic_tribunal.invariants.extractor import InvariantExtractor
from epistemic_tribunal.ledger.store import LedgerStore
from epistemic_tribunal.ledger.writer import LedgerWriter

from epistemic_tribunal.tribunal.aggregator import TribunalAggregator
from epistemic_tribunal.tribunal_types import (
    CandidateTrace,
    CritiqueResult,
    DecisionKind,
    ExperimentRun,
    FailureRecord,
    InvariantSet,
    Task,
    TribunalDecision,
    UncertaintyReport,
)
from epistemic_tribunal.uncertainty.analyzer import UncertaintyAnalyzer
from epistemic_tribunal.utils.logging import get_logger

log = get_logger(__name__)


class Orchestrator:
    """End-to-end tribunal pipeline runner.

    Parameters
    ----------
    config:
        Application settings.  Loaded from default YAML if omitted.
    ledger_store:
        Pre-constructed ledger store.  A new in-memory store is created if
        omitted (useful for tests).
    """

    def __init__(
        self,
        config: Optional[TribunalSettings] = None,
        ledger_store: Optional[LedgerStore] = None,
    ) -> None:
        self._config = config or load_config()
        self._store = ledger_store or LedgerStore(self._config.ledger.path)
        self._writer = LedgerWriter(self._store)

        # Build components
        generator_configs: dict[str, dict[str, Any]] = {}

        # 1. Apply global LLM config block if LLM is enabled
        if "llm" in self._config.generators.enabled:
            generator_configs["llm"] = self._config.generators.llm.model_dump()

        # 2. Merge specific per-generator config overrides from YAML
        generator_configs.update(self._config.generators.configs)

        self._generators = build_generators(
            self._config.generators.enabled,
            seed=self._config.generators.seed,
            generator_configs=generator_configs,
        )
        self._extractor = InvariantExtractor(
            enabled_checks=self._config.invariants.enabled_checks,
            confidence_threshold=self._config.invariants.confidence_threshold,
        )
        self._critic = TraceCritic(
            consistency_weight=self._config.critic.consistency_weight,
            rule_coherence_weight=self._config.critic.rule_coherence_weight,
            morphology_weight=self._config.critic.morphology_weight,
            failure_similarity_weight=self._config.critic.failure_similarity_weight,
            use_llm_judge_for_math=getattr(self._config.critic, "use_llm_judge_for_math", False),
        )
        self._uncertainty = UncertaintyAnalyzer(
            min_coalition_mass=self._config.uncertainty.min_coalition_mass
        )
        self._tribunal = TribunalAggregator(
            config=self._config.tribunal,
            eqbsl_config=self._config.eqbsl,
            ledger_store=self._store,
        )

        # Failure-memory layer
        fm_cfg = self._config.failure_memory
        self._failure_memory_store = FailureMemoryStore(
            path=self._config.ledger.path
        )
        self._failure_extractor = FailureSignatureExtractor()
        self._failure_query = FailureMemoryQuery(
            store=self._failure_memory_store,
            penalty_scale=fm_cfg.penalty_scale,
        ) if fm_cfg.enabled else None
        self._failure_memory_enabled = fm_cfg.enabled

        # Strange Loop memory v1 — pre-generation constraint injection
        sl_cfg = self._config.strange_loop
        self._strange_loop_enabled = sl_cfg.enabled and fm_cfg.enabled
        self._constraint_builder: Optional[FailureConstraintBuilder] = None
        if self._strange_loop_enabled:
            self._constraint_builder = FailureConstraintBuilder(
                store=self._failure_memory_store,
                max_bad_answers=sl_cfg.max_bad_answers,
                max_warnings=sl_cfg.max_warnings,
                min_similarity=sl_cfg.min_similarity,
                same_task_boost=sl_cfg.same_task_boost,
            )
            log.info("Strange Loop memory v1 enabled (max_bad=%d, max_warn=%d)",
                     sl_cfg.max_bad_answers, sl_cfg.max_warnings)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def store(self) -> LedgerStore:
        """Public accessor for the underlying :class:`LedgerStore`."""
        return self._store

    def run(
        self, 
        task: Task, 
        on_token: Optional[Callable[[str, str], None]] = None
    ) -> ExperimentRun:
        """Execute the full tribunal pipeline for *task*."""
        start_time = time.monotonic()
        run_id = str(uuid.uuid4())
        completed_runs_before = self._store.get_stats()["experiment_runs"]
        warmup_threshold = self._config.tribunal.ledger_warmup_tasks
        log.info("Starting tribunal run %s for task %s", run_id[:8], task.task_id)

        # Persist task record
        self._writer.write_task(task)

        # 2. Extract invariants (only needed once)
        invariant_set = self._extractor.extract(task)
        log.info(
            "Extracted %d invariant(s) for task %s",
            len(invariant_set.invariants), task.task_id
        )

        # 3. Retrieve past failure patterns from ledger
        failure_patterns = self._store.get_failure_patterns(task.task_id)

        all_traces: list[CandidateTrace] = []
        all_critiques: list[CritiqueResult] = []
        
        is_single_source_mode = getattr(self._config.tribunal, "single_source_resampling_mode", False)
        max_attempts = self._config.tribunal.max_resample_attempts if is_single_source_mode else 0
        schedule = getattr(self._config.tribunal, "resample_temperature_schedule", [0.1, 0.4, 0.7])
        
        decision = None
        uncertainty_report = None
        
        # We loop max_attempts + 1 times (attempt 0 is the initial run)
        for attempt in range(max_attempts + 1):
            is_resample = attempt > 0
            
            if is_resample:
                temp = schedule[min(attempt, len(schedule) - 1)]
                for g in self._generators:
                    if hasattr(g, "temperature"):
                        g.temperature = temp
                log.info("Resampling task %s (attempt %d) with temperature %.2f", task.task_id, attempt, temp)
                
            # 1. Generate candidates (with Strange Loop constraints if enabled)
            failure_constraints: Optional[FailureConstraints] = None
            if (
                self._strange_loop_enabled
                and self._constraint_builder is not None
                and completed_runs_before >= warmup_threshold
            ):
                failure_constraints = self._constraint_builder.build(task)
                if failure_constraints.has_constraints:
                    log.info(
                        "Strange Loop: injecting constraints into generation "
                        "(bad_answers=%d, warnings=%d, strength=%.3f)",
                        len(failure_constraints.bad_answers),
                        len(failure_constraints.structural_warnings),
                        failure_constraints.constraint_strength,
                    )

            new_traces = self._generate(
                task, on_token=on_token,
                failure_constraints=failure_constraints,
            )
            
            # Inject explicit metadata
            for t in new_traces:
                t.metadata["resample_attempt"] = attempt
                t.metadata["source_mode"] = "resample" if is_resample else "initial"
                t.metadata["generator_family"] = t.generator_name.split("_")[0]
                gen_temp = next((getattr(g, "temperature", None) for g in self._generators if g.name == t.generator_name), None)
                if gen_temp is not None:
                    t.metadata["temperature"] = gen_temp
                    
            self._writer.write_traces(task, new_traces)
            all_traces.extend(new_traces)

            # 4. Critique new traces
            new_critiques = self._critique(task, new_traces, invariant_set, failure_patterns)
            all_critiques.extend(new_critiques)

            # 5. Analyse uncertainty over ALL traces
            uncertainty_report = self._uncertainty.analyze(task, all_traces)

            # 5b. Failure-memory lookup — inject per-trace penalties into M channel
            fm_metadata: dict[str, Any] = {}
            if self._failure_memory_enabled and self._failure_query is not None:
                fm_penalties, fm_metadata = self._failure_query.query_with_metadata(
                    task, all_traces, all_critiques, uncertainty_report
                )
                # Inject penalties into critique objects for scoring via M weight
                for critique in all_critiques:
                    old_penalty = critique.failure_similarity_penalty
                    mem_penalty = fm_penalties.get(critique.trace_id, 0.0)
                    # Take the max of existing (trivial) penalty and memory penalty
                    critique.failure_similarity_penalty = max(old_penalty, mem_penalty)
                if fm_metadata.get("failure_memory_candidates_penalised", 0) > 0:
                    log.info(
                        "Failure memory injected penalties for task %s: %s",
                        task.task_id,
                        fm_metadata.get("failure_memory_penalties", {}),
                    )
            
            # 6. Adjudication
            adjudication_strategy = self._config.tribunal.adjudication_strategy
            if adjudication_strategy == "greedy":
                greedy_traces = [t for t in all_traces if t.generator_name == "greedy"]
                if not greedy_traces:
                    raise ValueError("Adjudication strategy set to 'greedy', but no trace from 'greedy' generator found.")
                selected_trace = greedy_traces[0]
                decision = TribunalDecision(
                    task_id=task.task_id,
                    decision=DecisionKind.SELECT,
                    selected_trace_id=selected_trace.trace_id,
                    selected_answer=selected_trace.answer,
                    confidence=selected_trace.confidence_score or 1.0,
                    reasoning="Bypassing tribunal: Forced greedy selection.",
                    metadata={
                        "forced_greedy": True,
                        "arm_name": self._config.metadata.get("arm_name", "greedy_baseline"),
                        "candidate_count": len(all_traces)
                    }
                )
                log.info("Adjudication strategy is 'greedy': Forced selection of greedy trace.")
                break
            else:
                tribunal = self._tribunal
                if warmup_threshold > 0 and completed_runs_before < warmup_threshold:
                    warmup_config = self._config.tribunal.model_copy(deep=True)
                    warmup_config.weights.memory = 0.0
                    tribunal = TribunalAggregator(
                        config=warmup_config,
                        eqbsl_config=self._config.eqbsl,
                        ledger_store=self._store,
                    )
                    
                decision = tribunal.adjudicate(
                    task, all_traces, all_critiques, uncertainty_report, invariant_set,
                    is_single_source_mode=is_single_source_mode,
                    attempt=attempt,
                    failure_memory_metadata=fm_metadata,
                )
                
            if decision.decision in (DecisionKind.SELECT, DecisionKind.ABSTAIN):
                break
                
        # End of resample loop
        if decision.decision == DecisionKind.RESAMPLE:
            decision.decision = DecisionKind.ABSTAIN
            decision.reasoning += f" Coerced to ABSTAIN after exhausting {max_attempts} resample attempts."
            log.info("Coerced decision to ABSTAIN (exhausted %d resample attempts)", max_attempts)

        log.info("Final Decision: %s (confidence=%.3f)", decision.decision.value, decision.confidence)
        self._writer.write_decision(decision)

        # 7. Evaluate against ground truth and determine cohort
        ground_truth_match: Optional[bool] = None
        any_correct: Optional[bool] = None
        
        if task.ground_truth is not None:
            from epistemic_tribunal.domains.factory import get_adapter
            adapter = get_adapter(task.domain)
            any_correct = any(adapter.answers_equal(t.answer, task.ground_truth) for t in all_traces)
            
            greedy_traces = [t for t in all_traces if t.generator_name == "llm"]
            if greedy_traces:
                greedy_correct = adapter.answers_equal(greedy_traces[0].answer, task.ground_truth)
            else:
                greedy_correct = False

            decision.metadata["any_correct"] = any_correct
            decision.metadata["greedy_correct"] = greedy_correct

            if decision.selected_answer is not None:
                ground_truth_match = adapter.answers_equal(decision.selected_answer, task.ground_truth)
                decision.metadata["ground_truth_match"] = ground_truth_match

            cluster_sizes: dict[str, int] = {}
            for trace in all_traces:
                key = str(adapter.get_cluster_key(trace.answer))
                cluster_sizes[key] = cluster_sizes.get(key, 0) + 1
            max_cluster_size = max(cluster_sizes.values()) if cluster_sizes else 0
            selected_trace_id = decision.selected_trace_id
            generator_outcomes = []
            for trace in all_traces:
                answer_signature = str(adapter.get_cluster_key(trace.answer))
                coalition_size = cluster_sizes.get(answer_signature, 1)
                is_majority = coalition_size == max_cluster_size
                is_correct = adapter.answers_equal(trace.answer, task.ground_truth)
                generator_outcomes.append(
                    {
                        "trace_id": trace.trace_id,
                        "generator_name": trace.generator_name,
                        "answer_signature": answer_signature,
                        "coalition_size": coalition_size,
                        "coalition_mass": round(coalition_size / max(len(all_traces), 1), 4),
                        "is_majority": is_majority,
                        "rationale_present": bool(trace.reasoning_steps),
                        "reasoning_step_count": len(trace.reasoning_steps),
                        "is_correct": is_correct,
                        "was_selected": trace.trace_id == selected_trace_id,
                        "ground_truth_match": ground_truth_match,
                    }
                )
            decision.metadata["generator_outcomes"] = generator_outcomes
        else:
            decision.metadata["any_correct"] = None
            decision.metadata["greedy_correct"] = None

        is_contested = uncertainty_report.disagreement_rate > 0
        if not is_contested:
            cohort = "control-trivial"
        elif any_correct is True:
            cohort = "contested-recoverable"
        elif any_correct is False:
            cohort = "contested-unrecoverable"
        else:
            cohort = "unknown"

        decision.metadata["cohort"] = cohort
        decision.metadata["any_correct"] = any_correct
        decision.metadata["disagreement_rate"] = uncertainty_report.disagreement_rate

        # Attach failure-memory metadata
        if fm_metadata:
            decision.metadata["failure_memory"] = fm_metadata

        # Attach Strange Loop metadata for observability
        if failure_constraints is not None and failure_constraints.has_constraints:
            decision.metadata["strange_loop"] = {
                "constraints_injected": True,
                "n_bad_answers_injected": len(failure_constraints.bad_answers),
                "n_structural_warnings_injected": len(failure_constraints.structural_warnings),
                "constraint_strength": failure_constraints.constraint_strength,
                "source_task_ids": failure_constraints.source_task_ids,
                "metadata": failure_constraints.metadata,
            }

        eqbsl_coalitions = decision.metadata.pop("_eqbsl_coalitions", [])
        if eqbsl_coalitions:
            self._writer.write_coalition_opinions(
                run_id=run_id,
                task_id=task.task_id,
                coalition_rows=eqbsl_coalitions,
            )
            if "eqbsl" in decision.metadata:
                decision.metadata["eqbsl"]["persisted_coalition_count"] = len(eqbsl_coalitions)

        # 8. Write failure records
        self._persist_failures(
            task, all_traces, all_critiques, decision, uncertainty_report, ground_truth_match
        )

        # 8b. Extract and store failure signature (retrospective)
        if self._failure_memory_enabled:
            sig = self._failure_extractor.extract(
                task, all_traces, all_critiques, decision,
                uncertainty_report, ground_truth_match, any_correct,
            )
            if sig is not None:
                # Track whether memory changed the selection
                sig.domain_features["selection_changed_by_memory"] = (
                    fm_metadata.get("failure_memory_candidates_penalised", 0) > 0
                )
                self._failure_memory_store.store(sig)
                log.debug(
                    "Stored failure signature %s (type=%s) for task %s",
                    sig.signature_id[:8], sig.failure_type.value, task.task_id,
                )

        duration = time.monotonic() - start_time
        decision.metadata["generation_stats"] = getattr(self, "last_generation_stats", {})

        # Aggregate LLM judge rubric scores from all traces (average per dimension)
        judge_keys = (
            "llm_judge_arithmetic_consistency",
            "llm_judge_logical_consistency",
            "llm_judge_answer_trace_alignment",
            "llm_judge_final_rule_coherence",
        )
        judge_agg: dict[str, list[float]] = {k: [] for k in judge_keys}
        judge_rationale = ""
        for t in all_traces:
            for k in judge_keys:
                if k in t.metadata:
                    judge_agg[k].append(float(t.metadata[k]))
            if not judge_rationale and "llm_judge_rationale" in t.metadata:
                judge_rationale = str(t.metadata["llm_judge_rationale"])
        if any(judge_agg[k] for k in judge_keys):
            decision.metadata["judge_rubric"] = {
                k.replace("llm_judge_", ""): round(sum(v) / len(v), 4)
                for k, v in judge_agg.items() if v
            }
            if judge_rationale:
                decision.metadata["judge_rubric"]["rationale"] = judge_rationale

        budget = next((getattr(g, "max_new_tokens", None) for g in self._generators if hasattr(g, "max_new_tokens")), None)
        if budget is not None:
            decision.metadata["budget"] = budget
            
        if self._config.ledger.path:
            from pathlib import Path
            decision.metadata["arm_name"] = Path(self._config.ledger.path).stem
        else:
            decision.metadata["arm_name"] = "unknown"


        run = ExperimentRun(
            run_id=run_id,
            task_id=task.task_id,
            generator_names=[g.name for g in self._generators],
            decision=decision.decision,
            confidence=decision.confidence,
            selected_trace_id=decision.selected_trace_id,
            ground_truth_match=ground_truth_match,
            duration_seconds=round(duration, 4),
            config_snapshot={
                "selection_threshold": self._config.tribunal.selection_threshold,
                "resample_threshold": self._config.tribunal.resample_threshold,
                "generators": self._config.generators.enabled,
                "fusion_mode": self._config.tribunal.fusion_mode,
                "eqbsl_base_rate": self._config.eqbsl.default_base_rate,
            },
            metadata=decision.metadata
        )
        self._writer.write_run(run)

        completed_runs_after = completed_runs_before + 1
        if (
            warmup_threshold > 0
            and completed_runs_before < warmup_threshold <= completed_runs_after
        ):
            log.info(
                "Failure-ledger warmup complete after %d task(s); memory weighting is now active.",
                warmup_threshold,
            )
        return run

    def run_and_format(
        self, 
        task: Task, 
        on_token: Optional[Callable[[str, str], None]] = None
    ) -> dict:
        """Run the pipeline and return a JSON-serialisable result dict."""
        run = self.run(task, on_token=on_token)
        return {
            "run_id": run.run_id,
            "task_id": run.task_id,
            "decision": run.decision.value,
            "selected_trace_id": run.selected_trace_id,
            "ground_truth_match": run.ground_truth_match,
            "duration_seconds": run.duration_seconds,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate(
        self,
        task: Task,
        on_token: Optional[Callable[[str, str], None]] = None,
        failure_constraints: Optional[FailureConstraints] = None,
    ) -> list[CandidateTrace]:
        traces: list[CandidateTrace] = []
        self.last_generation_stats = {
            "truncation_count": 0,
            "json_not_found_count": 0,
            "json_invalid_count": 0,
            "grid_shape_invalid_count": 0,
            "reasoning_bleed_count": 0,
            "parse_failure_count": 0  # legacy fallback
        }
        for gen in self._generators:
            try:
                if on_token:
                    on_token("generator_start", gen.name)
                trace = gen.generate(
                    task, on_token=on_token,
                    failure_constraints=failure_constraints,
                )
                traces.append(trace)
                log.debug("Generator %r produced trace %s", gen.name, trace.trace_id[:8])
            except Exception as exc:
                exc_str = str(exc)
                if "[length]" in exc_str:
                    self.last_generation_stats["truncation_count"] += 1
                elif "[json_not_found]" in exc_str:
                    self.last_generation_stats["json_not_found_count"] += 1
                elif "[json_invalid]" in exc_str:
                    self.last_generation_stats["json_invalid_count"] += 1
                elif "[grid_shape_invalid]" in exc_str:
                    self.last_generation_stats["grid_shape_invalid_count"] += 1
                elif "[reasoning_bleed]" in exc_str:
                    self.last_generation_stats["reasoning_bleed_count"] += 1
                else:
                    self.last_generation_stats["parse_failure_count"] += 1
                    
                log.error("Generator %r failed: %s", gen.name, exc)
        return traces

    def _critique(
        self,
        task: Task,
        traces: list[CandidateTrace],
        invariant_set: InvariantSet,
        failure_patterns: list[dict],
    ) -> list[CritiqueResult]:
        critiques: list[CritiqueResult] = []
        for trace in traces:
            result = self._critic.critique(
                task, trace, invariant_set, failure_patterns
            )
            critiques.append(result)
        return critiques

    def _persist_failures(
        self,
        task: Task,
        traces: list[CandidateTrace],
        critiques: list[CritiqueResult],
        decision: TribunalDecision,
        uncertainty: UncertaintyReport,
        ground_truth_match: Optional[bool],
    ) -> None:
        """Write failure record and invariant violations when appropriate."""
        # Persist invariant violations for all traces

        trace_by_id = {t.trace_id: t for t in traces}
        for critique in critiques:
            trace = trace_by_id.get(critique.trace_id)
            if trace and critique.violated_invariants:
                self._writer.write_invariant_violations(task, trace, critique)

        # Write failure record if: wrong, abstained, resampled, or
        # configured to always record
        should_record = (
            ground_truth_match is False
            or decision.decision != DecisionKind.SELECT
            or self._config.ledger.always_record
        )

        if not should_record:
            return

        # Gather all violated invariants across candidates
        all_violated = list(
            {
                inv
                for c in critiques
                for inv in c.violated_invariants
            }
        )

        # Disagreement pattern summary
        from epistemic_tribunal.domains.factory import get_adapter
        adapter = get_adapter(task.domain)
        unique_answers = len(
            {
                adapter.get_cluster_key(t.answer)
                for t in traces
            }
        )
        disagreement_pattern = (
            f"{unique_answers}/{len(traces)} unique answers; "
            f"disagreement_rate={uncertainty.disagreement_rate:.3f}"
        )

        # Diagnosis
        if decision.decision == DecisionKind.ABSTAIN:
            diagnosis = "Tribunal abstained — all candidates scored below resample threshold."
        elif decision.decision == DecisionKind.RESAMPLE:
            diagnosis = "Tribunal requested resample — best score below selection threshold."
        elif ground_truth_match is False:
            diagnosis = "Selected candidate did not match ground truth."
        else:
            diagnosis = "Recorded by policy (always_record=true)."

        failure = FailureRecord(
            task_id=task.task_id,
            selected_trace_id=decision.selected_trace_id,
            all_candidate_trace_ids=[t.trace_id for t in traces],
            violated_invariants=all_violated,
            disagreement_pattern=disagreement_pattern,
            diagnosis=diagnosis,
            ground_truth_match=ground_truth_match,
        )
        self._writer.write_failure(failure)
        log.info("Failure record written: %s", failure.failure_id[:8])
