"""Advanced GSM8K tribunal benchmark tests.

Covers:
- Cohort annotation loading and metadata injection
- Multi-regime generator registration (llm_warm, llm_concise, llm_cot)
- Reasoning trace extraction for math tasks
- Math-domain critic scoring (morphology, rule coherence)
- Oracle / tribunal-usefulness metrics
- Multi-regime disagreement signal on contested-recoverable tasks
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from epistemic_tribunal.critics.trace_critic import TraceCritic
from epistemic_tribunal.evaluation.metrics import (
    best_in_pool_accuracy,
    greedy_accuracy,
    overall_accuracy,
    resolved_accuracy,
    summary_report,
)
from epistemic_tribunal.generators.base import build_generators
from epistemic_tribunal.generators.llm import (
    LLMConciseGenerator,
    LLMSelfCheckGenerator,
    LLMVerifyGenerator,
    LLMWarmGenerator,
)
from epistemic_tribunal.generators.llm_cot import CoTLLMGenerator
from epistemic_tribunal.tasks.gsm8k import (
    extract_math_answer,
    load_task_from_dict,
    load_tasks_from_jsonl,
)
from epistemic_tribunal.tribunal_types import (
    CandidateTrace,
    DecisionKind,
    ExperimentRun,
    Task,
    TaskDomain,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def math_task() -> Task:
    """Simple GSM8K task with known answer."""
    return Task(
        task_id="gsm8k_test_001",
        domain=TaskDomain.GSM8K_MATH,
        description="GSM8K Math Word Problem",
        train=[],
        test_input="Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and sells 4 each day at the farmers market. How many eggs does she have left at the end of each day?",
        ground_truth="9",
    )


@pytest.fixture()
def math_task_contested() -> Task:
    """A 'contested-recoverable' annotated task — tribunal should help here."""
    return Task(
        task_id="gsm8k_contested_001",
        domain=TaskDomain.GSM8K_MATH,
        description="GSM8K Math Word Problem",
        train=[],
        test_input=(
            "John has 3 boxes. Each box has 4 rows of 3 items. "
            "He gives away half of the total. How many items remain?"
        ),
        ground_truth="18",
        metadata={
            "cohort": "contested-recoverable",
            "contestability_index": 0.8,
            "recoverability_index": 0.9,
        },
    )


@pytest.fixture()
def critic() -> TraceCritic:
    return TraceCritic(use_llm_judge_for_math=False)


# ---------------------------------------------------------------------------
# 1. GSM8K task loading and cohort annotation
# ---------------------------------------------------------------------------


class TestGSM8KTaskLoading:
    def test_extract_math_answer_with_hash(self) -> None:
        raw = "Step 1... Step 2... #### 42"
        assert extract_math_answer(raw) == "42"

    def test_extract_math_answer_multidigit(self) -> None:
        raw = "Some working here.\n#### 1234"
        assert extract_math_answer(raw) == "1234"

    def test_extract_math_answer_missing(self) -> None:
        assert extract_math_answer("No hash in this text") is None

    def test_load_task_from_dict_basic(self) -> None:
        data = {
            "question": "How many apples does she have?",
            "answer": "She has 3 apples. #### 3",
        }
        task = load_task_from_dict(data)
        assert task.domain == TaskDomain.GSM8K_MATH
        assert task.test_input == "How many apples does she have?"
        assert task.ground_truth == "3"

    def test_load_task_from_dict_preserves_cohort_metadata(self) -> None:
        data = {
            "question": "Test question?",
            "answer": "#### 5",
            "task_id": "test_001",
            "cohort": "contested-recoverable",
            "contestability_index": 0.8,
            "recoverability_index": 0.9,
            "structural_separability": 0.5,
            "plausible_hypotheses": 2,
            "recoverability_status": "recoverable",
        }
        task = load_task_from_dict(data)
        assert task.metadata["cohort"] == "contested-recoverable"
        assert task.metadata["contestability_index"] == 0.8
        assert task.metadata["recoverability_index"] == 0.9
        assert task.metadata["recoverability_status"] == "recoverable"

    def test_load_tasks_from_jsonl(self, tmp_path: Path) -> None:
        jsonl_path = tmp_path / "test.jsonl"
        lines = [
            json.dumps({
                "question": f"Question {i}?",
                "answer": f"#### {i * 3}",
                "cohort": "control-trivial",
            })
            for i in range(5)
        ]
        jsonl_path.write_text("\n".join(lines))
        tasks = load_tasks_from_jsonl(jsonl_path)
        assert len(tasks) == 5
        for i, task in enumerate(tasks):
            assert task.domain == TaskDomain.GSM8K_MATH
            assert task.ground_truth == str(i * 3)

    def test_load_tasks_assigns_ids(self, tmp_path: Path) -> None:
        jsonl_path = tmp_path / "tasks.jsonl"
        jsonl_path.write_text(
            json.dumps({"question": "Q?", "answer": "#### 1"}) + "\n"
        )
        tasks = load_tasks_from_jsonl(jsonl_path)
        assert tasks[0].task_id == "tasks_0000"

    def test_load_tasks_respects_explicit_task_id(self, tmp_path: Path) -> None:
        jsonl_path = tmp_path / "tasks.jsonl"
        jsonl_path.write_text(
            json.dumps({"task_id": "my_custom_id", "question": "Q?", "answer": "#### 1"})
        )
        tasks = load_tasks_from_jsonl(jsonl_path)
        assert tasks[0].task_id == "my_custom_id"


# ---------------------------------------------------------------------------
# 2. Multi-regime generator registry
# ---------------------------------------------------------------------------


class TestMultiRegimeGeneratorRegistry:
    """Generator classes are registered and instantiable without an API."""

    def test_llm_warm_generator_is_in_registry(self) -> None:
        gens = build_generators(["llm_warm"], seed=0, generator_configs={
            "llm_warm": {"model_name": "stub", "api_key": "k", "api_base": "http://x"}
        })
        assert len(gens) == 1
        assert isinstance(gens[0], LLMWarmGenerator)

    def test_llm_concise_generator_is_in_registry(self) -> None:
        gens = build_generators(["llm_concise"], seed=0, generator_configs={
            "llm_concise": {"model_name": "stub", "api_key": "k", "api_base": "http://x"}
        })
        assert len(gens) == 1
        assert isinstance(gens[0], LLMConciseGenerator)

    def test_llm_cot_generator_is_in_registry(self) -> None:
        gens = build_generators(["llm_cot"], seed=0, generator_configs={
            "llm_cot": {"model_name": "stub", "api_key": "k", "api_base": "http://x"}
        })
        assert len(gens) == 1
        assert isinstance(gens[0], CoTLLMGenerator)

    def test_llm_verify_generator_is_in_registry(self) -> None:
        gens = build_generators(["llm_verify"], seed=0, generator_configs={
            "llm_verify": {"model_name": "stub", "api_key": "k", "api_base": "http://x"}
        })
        assert len(gens) == 1
        assert isinstance(gens[0], LLMVerifyGenerator)

    def test_llm_selfcheck_generator_is_in_registry(self) -> None:
        gens = build_generators(["llm_selfcheck"], seed=0, generator_configs={
            "llm_selfcheck": {"model_name": "stub", "api_key": "k", "api_base": "http://x"}
        })
        assert len(gens) == 1
        assert isinstance(gens[0], LLMSelfCheckGenerator)

    def test_multi_regime_build(self) -> None:
        cfg = {"model_name": "stub", "api_key": "k", "api_base": "http://x"}
        gens = build_generators(
            ["llm", "llm_cot", "llm_warm", "llm_verify"],
            seed=42,
            generator_configs={"llm": cfg, "llm_cot": cfg, "llm_warm": cfg, "llm_verify": cfg},
        )
        assert len(gens) == 4
        names = [g.name for g in gens]
        assert "llm" in names
        assert "llm_cot" in names
        assert "llm_warm" in names
        assert "llm_verify" in names

    def test_llm_warm_default_temperature(self) -> None:
        gens = build_generators(["llm_warm"], seed=0, generator_configs={
            "llm_warm": {"model_name": "stub", "api_key": "k", "api_base": "http://x"}
        })
        assert gens[0].temperature == 0.7  # type: ignore[attr-defined]

    def test_llm_warm_temperature_overridable(self) -> None:
        gens = build_generators(["llm_warm"], seed=0, generator_configs={
            "llm_warm": {"model_name": "stub", "api_key": "k", "api_base": "http://x",
                         "temperature": 0.9}
        })
        assert gens[0].temperature == 0.9  # type: ignore[attr-defined]

    def test_unknown_generator_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown generator"):
            build_generators(["does_not_exist"])


# ---------------------------------------------------------------------------
# 3. Reasoning trace extraction for math tasks
# ---------------------------------------------------------------------------


class TestMathReasoningExtraction:
    """The LLMGenerator._extract_reasoning_steps must surface think blocks."""

    def _make_generator(self) -> LLMWarmGenerator:
        return LLMWarmGenerator(
            seed=0, model_name="stub", api_key="k", api_base="http://x"
        )

    def test_extracts_think_block(self) -> None:
        gen = self._make_generator()
        response = "<think>\nStep 1: 16 - 3 = 13\nStep 2: 13 - 4 = 9\n</think>\n{\"answer\": \"9\"}"
        steps = gen._extract_reasoning_steps(response)
        assert any("16" in s for s in steps)
        assert any("13" in s for s in steps)

    def test_returns_fallback_on_empty_response(self) -> None:
        gen = self._make_generator()
        steps = gen._extract_reasoning_steps("")
        assert isinstance(steps, list)
        assert len(steps) >= 1  # fallback message

    def test_extracts_json_reasoning_steps(self) -> None:
        gen = self._make_generator()
        response = json.dumps({
            "answer": "9",
            "reasoning_steps": ["Step 1: 16-3=13", "Step 2: 13-4=9"]
        })
        steps = gen._extract_reasoning_steps(response)
        assert any("16" in s for s in steps)

    def test_math_prompt_builds_correctly(self, math_task: Task) -> None:
        gen = self._make_generator()
        prompt, schema = gen._build_math_prompt(math_task)
        assert "answer" in schema.get("properties", {})
        assert "math word problem" in prompt.lower() or "question" in prompt.lower()

    def test_cot_math_prompt_no_schema_constraint(self, math_task: Task) -> None:
        gen = CoTLLMGenerator(
            seed=0, model_name="stub", api_key="k", api_base="http://x"
        )
        prompt, schema = gen._build_math_prompt(math_task)
        # CoT returns empty schema so the API doesn't enforce JSON-only output
        assert schema == {}
        assert "step-by-step" in prompt.lower() or "reasoning" in prompt.lower()

    def test_concise_math_prompt_forbids_reasoning(self, math_task: Task) -> None:
        gen = LLMConciseGenerator(
            seed=0, model_name="stub", api_key="k", api_base="http://x"
        )
        prompt, _ = gen._build_math_prompt(math_task)
        assert "DO NOT EXPLAIN" in prompt or "NO" in prompt.upper()


# ---------------------------------------------------------------------------
# 4. Math-domain critic scoring
# ---------------------------------------------------------------------------


class TestMathCriticScoring:
    def _make_trace(
        self,
        answer: str | int | float = "9",
        reasoning_steps: list[str] | None = None,
    ) -> CandidateTrace:
        return CandidateTrace(
            generator_name="llm",
            answer=answer,
            reasoning_steps=reasoning_steps or [],
            confidence_score=1.0,
        )

    def test_morphology_non_empty_answer(self, critic: TraceCritic, math_task: Task) -> None:
        trace = self._make_trace("9")
        score = critic._score_morphology(math_task, trace)
        assert score > 0.0

    def test_morphology_empty_answer_penalised(self, critic: TraceCritic, math_task: Task) -> None:
        trace = self._make_trace("")
        score = critic._score_morphology(math_task, trace)
        assert score == 0.0

    def test_morphology_answer_in_reasoning_bonus(self, critic: TraceCritic, math_task: Task) -> None:
        trace = self._make_trace("9", reasoning_steps=["So the answer is 9 eggs."])
        score_with = critic._score_morphology(math_task, trace)
        trace_without = self._make_trace("9", reasoning_steps=["No number here."])
        score_without = critic._score_morphology(math_task, trace_without)
        # Having the answer in the reasoning trace should score >= the case without
        assert score_with >= score_without

    def test_rule_coherence_math_no_judge(self, critic: TraceCritic, math_task: Task) -> None:
        trace = self._make_trace("9")
        score = critic._score_rule_coherence(math_task, trace)
        assert score == 1.0  # default without judge

    def test_consistency_increases_with_steps(self, critic: TraceCritic, math_task: Task) -> None:
        trace_bare = self._make_trace("9", [])
        trace_rich = self._make_trace("9", ["Step1", "Step2", "Step3", "Step4", "Step5"])
        s_bare = critic._score_consistency(trace_bare)
        s_rich = critic._score_consistency(trace_rich)
        assert s_rich > s_bare

    def test_full_critique_returns_valid_aggregate(
        self, critic: TraceCritic, math_task: Task
    ) -> None:
        trace = self._make_trace("9", ["16 - 3 - 4 = 9"])
        result = critic.critique(math_task, trace)
        assert 0.0 <= result.aggregate_score <= 1.0
        assert result.trace_id == trace.trace_id


# ---------------------------------------------------------------------------
# 5. Oracle / tribunal-usefulness metrics
# ---------------------------------------------------------------------------


def _make_run(
    decision: DecisionKind = DecisionKind.SELECT,
    ground_truth_match: bool | None = True,
    any_correct: bool | None = True,
    cohort: str = "control-trivial",
    greedy_correct: bool | None = None,
) -> ExperimentRun:
    from datetime import datetime, timezone
    meta: dict = {"any_correct": any_correct, "cohort": cohort}
    if greedy_correct is not None:
        meta["greedy_correct"] = greedy_correct
    return ExperimentRun(
        run_id="r1",
        task_id="t1",
        generator_names=["llm"],
        decision=decision,
        confidence=0.5,
        selected_trace_id=None,
        ground_truth_match=ground_truth_match,
        duration_seconds=1.0,
        config_snapshot={},
        metadata=meta,
    )


class TestOracleMetrics:
    def test_best_in_pool_accuracy_all_correct(self) -> None:
        runs = [_make_run(any_correct=True) for _ in range(4)]
        assert best_in_pool_accuracy(runs) == 1.0

    def test_best_in_pool_accuracy_none_correct(self) -> None:
        runs = [_make_run(any_correct=False) for _ in range(4)]
        assert best_in_pool_accuracy(runs) == 0.0

    def test_greedy_accuracy(self) -> None:
        runs = [
            _make_run(greedy_correct=True),
            _make_run(greedy_correct=False),
            _make_run(greedy_correct=True),
        ]
        assert abs(greedy_accuracy(runs) - 2 / 3) < 1e-6

    def test_greedy_accuracy_no_data(self) -> None:
        runs = [_make_run(greedy_correct=None) for _ in range(3)]
        assert greedy_accuracy(runs) == 0.0

    def test_overall_accuracy(self) -> None:
        runs = [
            _make_run(ground_truth_match=True),
            _make_run(ground_truth_match=False),
            _make_run(ground_truth_match=True),
        ]
        assert abs(overall_accuracy(runs) - 2 / 3) < 1e-6

    def test_resolved_accuracy_only_selects(self) -> None:
        runs = [
            _make_run(decision=DecisionKind.SELECT, ground_truth_match=True),
            _make_run(decision=DecisionKind.ABSTAIN, ground_truth_match=False),  # not counted
            _make_run(decision=DecisionKind.SELECT, ground_truth_match=False),
        ]
        # 1 correct out of 2 selected
        assert abs(resolved_accuracy(runs) - 0.5) < 1e-6

    def test_summary_report_structure(self) -> None:
        runs = [_make_run() for _ in range(10)]
        report = summary_report(runs)
        for key in ("total_runs", "overall_accuracy", "selective_accuracy",
                    "coverage", "wrong_pick_count", "cohort_metrics",
                    "abstention_metrics", "diagnostics", "tribunal_usefulness"):
            assert key in report, f"Missing key: {key}"

    def test_summary_report_tribunal_usefulness(self) -> None:
        runs = [_make_run(greedy_correct=True) for _ in range(5)]
        report = summary_report(runs)
        tu = report["tribunal_usefulness"]
        assert "best_candidate_in_pool_accuracy" in tu
        assert "greedy_accuracy" in tu
        assert "tribunal_lift_over_greedy" in tu

    def test_summary_report_cohort_stratification(self) -> None:
        runs = [
            _make_run(cohort="control-trivial", ground_truth_match=True),
            _make_run(cohort="contested-recoverable", ground_truth_match=False),
            _make_run(cohort="contested-unrecoverable", decision=DecisionKind.ABSTAIN,
                      ground_truth_match=None),
        ]
        report = summary_report(runs)
        cohorts = report["cohort_metrics"]
        assert "control-trivial" in cohorts
        assert "contested-recoverable" in cohorts

    def test_abstention_metrics_good_bad(self) -> None:
        runs = [
            # Good abstention: no candidate was correct
            _make_run(decision=DecisionKind.ABSTAIN, ground_truth_match=None,
                      any_correct=False),
            # Bad abstention: a correct candidate existed but was passed over
            _make_run(decision=DecisionKind.ABSTAIN, ground_truth_match=None,
                      any_correct=True),
        ]
        report = summary_report(runs)
        abst = report["abstention_metrics"]
        assert abst["good_abstentions"] == 1
        assert abst["bad_abstentions"] == 1


# ---------------------------------------------------------------------------
# 6. Multi-regime disagreement signal
# ---------------------------------------------------------------------------


class TestMultiRegimeDisagreementSignal:
    """Verify that disagreement across generator names drives meaningful signal."""

    def _make_run_with_generators(
        self,
        generators: list[str],
        disagreement_rate: float = 0.0,
        decision: DecisionKind = DecisionKind.SELECT,
        ground_truth_match: bool | None = True,
        cohort: str = "contested-recoverable",
    ) -> ExperimentRun:
        from datetime import datetime, timezone
        return ExperimentRun(
            run_id="r1",
            task_id="t1",
            generator_names=generators,
            decision=decision,
            confidence=0.5,
            selected_trace_id=None,
            ground_truth_match=ground_truth_match,
            duration_seconds=1.0,
            config_snapshot={},
            metadata={
                "cohort": cohort,
                "disagreement_rate": disagreement_rate,
                "any_correct": True,
            },
        )

    def test_contested_runs_with_high_disagreement(self) -> None:
        """High disagreement on contested task → tribunal should abstain or treat carefully."""
        run = self._make_run_with_generators(
            generators=["llm", "llm_cot", "llm_warm"],
            disagreement_rate=0.667,  # 2/3 disagree
            decision=DecisionKind.ABSTAIN,
        )
        assert run.metadata["disagreement_rate"] > 0.5
        assert run.decision == DecisionKind.ABSTAIN

    def test_contested_cohort_in_report_separates(self) -> None:
        """The report separates contested-recoverable from control-trivial."""
        runs = [
            self._make_run_with_generators(
                ["llm", "llm_cot", "llm_warm"],
                disagreement_rate=0.667,
                decision=DecisionKind.ABSTAIN,
                ground_truth_match=None,
                cohort="contested-recoverable",
            ),
            self._make_run_with_generators(
                ["llm", "llm_cot", "llm_warm"],
                disagreement_rate=0.0,
                decision=DecisionKind.SELECT,
                ground_truth_match=True,
                cohort="control-trivial",
            ),
        ]
        report = summary_report(runs)
        cohorts = report["cohort_metrics"]
        assert "contested-recoverable" in cohorts
        assert "control-trivial" in cohorts
        # Control-trivial had a selection; contested-recoverable had an abstention
        assert cohorts["control-trivial"]["selective_acc"] == 1.0
        assert cohorts["contested-recoverable"]["abstain_rate"] == 1.0

    def test_generator_plurality_name_tracking(self) -> None:
        """ExperimentRun preserves all generator names from a multi-regime run."""
        run = self._make_run_with_generators(
            generators=["llm", "llm_cot", "llm_warm"]
        )
        assert set(run.generator_names) == {"llm", "llm_cot", "llm_warm"}
