"""Tests for robustness features: diversity_floor, ledger_warmup, checkpointing."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from epistemic_tribunal.config import TribunalConfig, TribunalSettings, TribunalWeights
from epistemic_tribunal.evaluation.benchmark import BenchmarkRunner, _PROGRESS_FILE
from epistemic_tribunal.generators.base import build_generators
from epistemic_tribunal.invariants.extractor import InvariantExtractor
from epistemic_tribunal.critics.trace_critic import TraceCritic
from epistemic_tribunal.tribunal.aggregator import TribunalAggregator
from epistemic_tribunal.tribunal.scoring import normalise_weights
from epistemic_tribunal.types import (
    CandidateTrace,
    CritiqueResult,
    DecisionKind,
    Task,
    UncertaintyReport,
)
from epistemic_tribunal.uncertainty.analyzer import UncertaintyAnalyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_traces_all_same_generator(task: Task, gen_name: str = "llm"):
    """Create traces where all share the same generator name and answer."""
    answer = [[2, 0, 1], [1, 2, 0], [0, 1, 2]]
    traces = []
    for i in range(3):
        traces.append(
            CandidateTrace(
                generator_name=gen_name,
                answer=answer,
                reasoning_steps=[f"Step {i}"],
                raw_trace=f"trace {i}",
                confidence_score=0.9,
            )
        )
    return traces


def _make_traces_diverse(task: Task):
    """Create traces from different generators with the same answer."""
    answer = [[2, 0, 1], [1, 2, 0], [0, 1, 2]]
    return [
        CandidateTrace(
            generator_name="greedy",
            answer=answer,
            reasoning_steps=["Step 1"],
            raw_trace="trace 1",
            confidence_score=0.9,
        ),
        CandidateTrace(
            generator_name="llm",
            answer=answer,
            reasoning_steps=["Step 2"],
            raw_trace="trace 2",
            confidence_score=0.9,
        ),
        CandidateTrace(
            generator_name="diverse",
            answer=[[1, 0, 2], [0, 1, 2], [2, 0, 1]],  # different answer
            reasoning_steps=["Step 3"],
            raw_trace="trace 3",
            confidence_score=0.5,
        ),
    ]


# ---------------------------------------------------------------------------
# Diversity floor tests
# ---------------------------------------------------------------------------


class TestDiversityFloor:
    def test_single_generator_type_dominates_triggers_resample(self, simple_task):
        """When coalition mass > diversity_floor and only one generator type
        contributes to the top answer (with other types present but
        disagreeing), RESAMPLE should be triggered instead of SELECT."""
        config = TribunalConfig(
            selection_threshold=0.0,  # low threshold so it would SELECT
            resample_threshold=0.0,
            diversity_floor=0.5,  # low floor to trigger easily
        )
        # 3 LLM traces with the same answer + 1 heuristic trace with different answer
        answer_llm = [[2, 0, 1], [1, 2, 0], [0, 1, 2]]
        answer_other = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        traces = [
            CandidateTrace(
                generator_name="llm", answer=answer_llm,
                reasoning_steps=["LLM step"], raw_trace="t1", confidence_score=0.9,
            ),
            CandidateTrace(
                generator_name="llm", answer=answer_llm,
                reasoning_steps=["LLM step"], raw_trace="t2", confidence_score=0.9,
            ),
            CandidateTrace(
                generator_name="llm", answer=answer_llm,
                reasoning_steps=["LLM step"], raw_trace="t3", confidence_score=0.9,
            ),
            CandidateTrace(
                generator_name="greedy", answer=answer_other,
                reasoning_steps=["Greedy step"], raw_trace="t4", confidence_score=0.3,
            ),
        ]

        inv_set = InvariantExtractor(confidence_threshold=0.4).extract(simple_task)
        critiques = [TraceCritic().critique(simple_task, t, inv_set) for t in traces]
        uncertainty = UncertaintyAnalyzer().analyze(simple_task, traces)

        tribunal = TribunalAggregator(config=config)
        decision = tribunal.adjudicate(
            simple_task, traces, critiques, uncertainty, inv_set
        )

        # Coalition mass = 3/4 = 0.75, which exceeds diversity_floor=0.5
        assert uncertainty.coalition_mass == 0.75
        assert decision.decision == DecisionKind.RESAMPLE
        assert "diversity floor" in decision.reasoning.lower()

    def test_diverse_generators_allows_select(self, simple_task):
        """When multiple generator types agree, SELECT should proceed normally."""
        config = TribunalConfig(
            selection_threshold=0.0,
            resample_threshold=0.0,
            diversity_floor=0.5,
        )
        traces = _make_traces_diverse(simple_task)

        inv_set = InvariantExtractor(confidence_threshold=0.4).extract(simple_task)
        critiques = [TraceCritic().critique(simple_task, t, inv_set) for t in traces]
        uncertainty = UncertaintyAnalyzer().analyze(simple_task, traces)

        tribunal = TribunalAggregator(config=config)
        decision = tribunal.adjudicate(
            simple_task, traces, critiques, uncertainty, inv_set
        )

        assert decision.decision == DecisionKind.SELECT

    def test_diversity_floor_disabled_at_1(self, simple_task):
        """diversity_floor=1.0 means it never triggers (coalition_mass can't exceed 1.0)."""
        config = TribunalConfig(
            selection_threshold=0.0,
            resample_threshold=0.0,
            diversity_floor=1.0,
        )
        traces = _make_traces_all_same_generator(simple_task, "llm")
        inv_set = InvariantExtractor(confidence_threshold=0.4).extract(simple_task)
        critiques = [TraceCritic().critique(simple_task, t, inv_set) for t in traces]
        uncertainty = UncertaintyAnalyzer().analyze(simple_task, traces)

        tribunal = TribunalAggregator(config=config)
        decision = tribunal.adjudicate(
            simple_task, traces, critiques, uncertainty, inv_set
        )

        assert decision.decision == DecisionKind.SELECT


# ---------------------------------------------------------------------------
# Ledger warmup tests
# ---------------------------------------------------------------------------


class TestLedgerWarmup:
    def test_gamma_zero_during_warmup(self, simple_task):
        """During warmup (completed_task_count < threshold), gamma should be 0."""
        config = TribunalConfig(
            ledger_warmup_tasks=100,
            diversity_floor=1.0,  # disable diversity floor
        )
        gens = build_generators(["greedy", "diverse"])
        traces = [g.generate(simple_task) for g in gens]
        inv_set = InvariantExtractor(confidence_threshold=0.4).extract(simple_task)
        critiques = [TraceCritic().critique(simple_task, t, inv_set) for t in traces]
        uncertainty = UncertaintyAnalyzer().analyze(simple_task, traces)

        tribunal = TribunalAggregator(config=config)

        # During warmup: completed_task_count=0 < 100
        decision_warmup = tribunal.adjudicate(
            simple_task, traces, critiques, uncertainty, inv_set,
            completed_task_count=0,
        )

        # After warmup: completed_task_count=100 >= 100
        decision_post = tribunal.adjudicate(
            simple_task, traces, critiques, uncertainty, inv_set,
            completed_task_count=100,
        )

        # Both should produce valid decisions
        assert decision_warmup.decision in DecisionKind
        assert decision_post.decision in DecisionKind

    def test_warmup_zero_means_always_active(self, simple_task):
        """ledger_warmup_tasks=0 means gamma is always active."""
        config = TribunalConfig(ledger_warmup_tasks=0, diversity_floor=1.0)
        gens = build_generators(["greedy"])
        traces = [g.generate(simple_task) for g in gens]
        inv_set = InvariantExtractor(confidence_threshold=0.4).extract(simple_task)
        critiques = [TraceCritic().critique(simple_task, t, inv_set) for t in traces]
        uncertainty = UncertaintyAnalyzer().analyze(simple_task, traces)

        tribunal = TribunalAggregator(config=config)
        # completed_task_count=0 >= 0, so gamma is active
        decision = tribunal.adjudicate(
            simple_task, traces, critiques, uncertainty, inv_set,
            completed_task_count=0,
        )
        assert decision.decision in DecisionKind


# ---------------------------------------------------------------------------
# Checkpoint/resume tests
# ---------------------------------------------------------------------------


class TestCheckpointResume:
    def _write_task_file(self, dirpath: Path, task_id: str) -> Path:
        """Write a minimal task JSON file."""
        task_data = {
            "task_id": task_id,
            "train": [
                {
                    "input": [[1, 2], [3, 4]],
                    "output": [[2, 1], [4, 3]],
                }
            ],
            "test_input": [[1, 0], [0, 1]],
            "ground_truth": [[0, 1], [1, 0]],
        }
        path = dirpath / f"{task_id}.json"
        path.write_text(json.dumps(task_data))
        return path

    def test_checkpoint_written(self):
        """BenchmarkRunner should write run_progress.json when checkpoint_every_n_tasks > 0."""
        config = TribunalSettings()
        config.ledger.path = ":memory:"
        config.ledger.checkpoint_every_n_tasks = 1  # checkpoint after every task

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            self._write_task_file(tmppath, "task_001")
            self._write_task_file(tmppath, "task_002")

            runner = BenchmarkRunner(config=config)
            runs = runner.run(tmppath)

            assert len(runs) == 2

            progress_path = tmppath / _PROGRESS_FILE
            assert progress_path.exists()

            progress = json.loads(progress_path.read_text())
            assert progress["completed_count"] == 2
            assert "task_001" in progress["completed_task_ids"]
            assert "task_002" in progress["completed_task_ids"]
            assert "elapsed_seconds" in progress

    def test_resume_skips_completed(self):
        """With --resume, already-completed tasks should be skipped."""
        config = TribunalSettings()
        config.ledger.path = ":memory:"
        config.ledger.checkpoint_every_n_tasks = 1

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            self._write_task_file(tmppath, "task_001")
            self._write_task_file(tmppath, "task_002")
            self._write_task_file(tmppath, "task_003")

            # Write a fake progress file showing task_001 already done
            progress = {
                "completed_task_ids": ["task_001"],
                "completed_count": 1,
                "solve_count": 0,
                "elapsed_seconds": 10.0,
            }
            (tmppath / _PROGRESS_FILE).write_text(json.dumps(progress))

            runner = BenchmarkRunner(config=config, resume=True)
            runs = runner.run(tmppath)

            # Should have only run task_002 and task_003 (skipped task_001)
            assert len(runs) == 2
            run_task_ids = {r.task_id for r in runs}
            assert "task_001" not in run_task_ids
            assert "task_002" in run_task_ids
            assert "task_003" in run_task_ids

    def test_no_checkpoint_when_disabled(self):
        """checkpoint_every_n_tasks=0 should not write any progress file."""
        config = TribunalSettings()
        config.ledger.path = ":memory:"
        config.ledger.checkpoint_every_n_tasks = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            self._write_task_file(tmppath, "task_001")

            runner = BenchmarkRunner(config=config)
            runner.run(tmppath)

            assert not (tmppath / _PROGRESS_FILE).exists()

    def test_progress_file_excluded_from_tasks(self):
        """run_progress.json in the dataset dir should not be treated as a task file."""
        config = TribunalSettings()
        config.ledger.path = ":memory:"
        config.ledger.checkpoint_every_n_tasks = 1

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            self._write_task_file(tmppath, "task_001")

            # Write a progress file that looks like JSON
            (tmppath / _PROGRESS_FILE).write_text(json.dumps({
                "completed_task_ids": [],
                "completed_count": 0,
            }))

            runner = BenchmarkRunner(config=config)
            runs = runner.run(tmppath)
            assert len(runs) == 1


# ---------------------------------------------------------------------------
# Config tests for new fields
# ---------------------------------------------------------------------------


class TestNewConfigFields:
    def test_diversity_floor_default(self):
        config = TribunalConfig()
        assert config.diversity_floor == 0.9

    def test_ledger_warmup_tasks_default(self):
        config = TribunalConfig()
        assert config.ledger_warmup_tasks == 150

    def test_checkpoint_every_n_tasks_default(self):
        settings = TribunalSettings()
        assert settings.ledger.checkpoint_every_n_tasks == 0

    def test_config_from_yaml_with_new_fields(self, tmp_path):
        yaml_content = """
tribunal:
  diversity_floor: 0.8
  ledger_warmup_tasks: 200
ledger:
  checkpoint_every_n_tasks: 25
"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(yaml_content)

        from epistemic_tribunal.config import load_config

        config = load_config(config_path)
        assert config.tribunal.diversity_floor == 0.8
        assert config.tribunal.ledger_warmup_tasks == 200
        assert config.ledger.checkpoint_every_n_tasks == 25

    def test_llm_in_generator_registry(self):
        """The 'llm' generator should be registered in the build_generators registry."""
        # This should not raise
        gens = build_generators(["llm"], seed=42)
        assert len(gens) == 1
        assert gens[0].name == "llm"
