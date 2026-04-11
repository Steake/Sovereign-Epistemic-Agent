"""Tests for the generator bank."""

from __future__ import annotations

import pytest

from epistemic_tribunal.generators.adversarial import AdversarialGenerator
from epistemic_tribunal.generators.base import build_generators
from epistemic_tribunal.generators.diverse import DiverseGenerator
from epistemic_tribunal.generators.greedy import GreedyGenerator
from epistemic_tribunal.generators.llm import LLMGenerator
from epistemic_tribunal.generators.minimal import MinimalDescriptionGenerator
from epistemic_tribunal.generators.rule_first import RuleFirstGenerator
from epistemic_tribunal.tasks.base import grid_shape
from epistemic_tribunal.types import CandidateTrace, Task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_trace(trace: CandidateTrace, task: Task, gen_name: str) -> None:
    """Assert a trace has the minimum required structure."""
    assert trace.generator_name == gen_name
    assert trace.trace_id != ""
    assert len(trace.answer) > 0
    rows, cols = grid_shape(task.test_input)
    assert grid_shape(trace.answer) == (rows, cols), (
        f"Generator {gen_name!r} produced wrong shape"
    )
    assert 0.0 <= (trace.confidence_score or 0) <= 1.0
    assert isinstance(trace.reasoning_steps, list)
    assert len(trace.reasoning_steps) >= 1


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_greedy_generator(simple_task: Task) -> None:
    gen = GreedyGenerator(seed=42)
    trace = gen.generate(simple_task)
    _check_trace(trace, simple_task, "greedy")
    assert "colour_mapping" in trace.derived_features


def test_diverse_generator(simple_task: Task) -> None:
    gen = DiverseGenerator(seed=42)
    trace = gen.generate(simple_task)
    _check_trace(trace, simple_task, "diverse")
    assert "perturbations" in trace.derived_features


def test_adversarial_generator(simple_task: Task) -> None:
    gen = AdversarialGenerator(seed=42)
    trace = gen.generate(simple_task)
    _check_trace(trace, simple_task, "adversarial")
    assert trace.confidence_score is not None
    assert trace.confidence_score < 0.35  # adversarial should be low confidence


def test_rule_first_generator(simple_task: Task) -> None:
    gen = RuleFirstGenerator(seed=42)
    trace = gen.generate(simple_task)
    _check_trace(trace, simple_task, "rule_first")
    assert "selected_rule" in trace.derived_features
    assert "rule_fit" in trace.derived_features


def test_rule_first_identity_task(identity_task: Task) -> None:
    """Rule-first should select 'copy' for an identity task."""
    gen = RuleFirstGenerator(seed=42)
    trace = gen.generate(identity_task)
    assert trace.derived_features["selected_rule"] == "copy"
    assert trace.derived_features["rule_fit"] == pytest.approx(1.0)


def test_minimal_description_generator(simple_task: Task) -> None:
    gen = MinimalDescriptionGenerator(seed=42)
    trace = gen.generate(simple_task)
    _check_trace(trace, simple_task, "minimal_description")
    assert "mdl_score" in trace.derived_features


def test_build_generators_all() -> None:
    gens = build_generators(
        ["greedy", "diverse", "adversarial", "rule_first", "minimal_description"],
        seed=7,
    )
    assert len(gens) == 5


def test_build_generators_with_llm_config() -> None:
    gens = build_generators(
        ["llm"],
        seed=7,
        generator_configs={"llm": {"model_name": "test-model", "max_new_tokens": 128}},
    )
    assert len(gens) == 1
    assert gens[0].name == "llm"
    assert isinstance(gens[0], LLMGenerator)
    assert gens[0].model_name == "test-model"


def test_build_generators_subset() -> None:
    gens = build_generators(["greedy", "diverse"], seed=1)
    names = [g.name for g in gens]
    assert names == ["greedy", "diverse"]


def test_build_generators_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown generator"):
        build_generators(["unknown_gen"])


def test_generators_deterministic(simple_task: Task) -> None:
    """Same seed → same answer."""
    gen_a = GreedyGenerator(seed=42)
    gen_b = GreedyGenerator(seed=42)
    trace_a = gen_a.generate(simple_task)
    trace_b = gen_b.generate(simple_task)
    assert trace_a.answer == trace_b.answer


def test_generators_different_seeds(simple_task: Task) -> None:
    """Diverse generator with different seeds should (usually) differ."""
    gen_a = DiverseGenerator(seed=1)
    gen_b = DiverseGenerator(seed=9999)
    trace_a = gen_a.generate(simple_task)
    trace_b = gen_b.generate(simple_task)
    # Not guaranteed to differ for every task, but almost always will
    # Just check both are valid
    assert trace_a.answer is not None
    assert trace_b.answer is not None


def test_no_training_data(simple_task: Task) -> None:
    """Generators should not crash on a task with no training examples."""
    from copy import deepcopy
    task = deepcopy(simple_task)
    task.train = []
    for GenClass in [GreedyGenerator, DiverseGenerator, AdversarialGenerator,
                     RuleFirstGenerator, MinimalDescriptionGenerator]:
        gen = GenClass(seed=42)
        trace = gen.generate(task)
        assert trace.answer is not None
        assert len(trace.answer) == len(task.test_input)


def test_llm_parse_response_rejects_trailing_prose(simple_task: Task) -> None:
    gen = LLMGenerator(model_name="test-model")
    response = (
        '<think>\nspot the swap\n</think>\n'
        '{"answer": [["2", "0", "1"], ["1", "2", "0"], ["0", "1", "2"]], '
        '"reasoning_steps": ["swap 1 and 2"], "confidence": 0.82}'
        "\nExtra prose after the closing brace."
    )
    answer, _steps, confidence = gen._parse_response(response, expected_shape=(3, 3))
    assert answer is None
    assert confidence == 0.0


def test_llm_parse_response_allows_think_after_json(simple_task: Task) -> None:
    gen = LLMGenerator(model_name="test-model")
    response = (
        '{"answer": [["2", "0", "1"], ["1", "2", "0"], ["0", "1", "2"]], '
        '"reasoning_steps": ["swap 1 and 2"], "confidence": 0.82}'
        "\n<think>post-hoc thought</think>"
    )
    answer, steps, confidence = gen._parse_response(response, expected_shape=(3, 3))
    assert answer == simple_task.ground_truth
    assert "swap 1 and 2" in steps
    assert confidence == pytest.approx(0.82)


def test_llm_parse_response_rejects_wrong_shape(simple_task: Task) -> None:
    gen = LLMGenerator(model_name="test-model")
    response = '{"answer": [[1, 2], [3, 4]], "reasoning_steps": ["short"], "confidence": 0.7}'
    answer, steps, confidence = gen._parse_response(response, expected_shape=(3, 3))
    assert answer is None
    assert steps == ["short"]
    assert confidence == 0.0


def test_llm_parse_response_rejects_reasoning_bleed(simple_task: Task) -> None:
    gen = LLMGenerator(model_name="test-model")
    response = (
        '{"answer": [["2 because swap 1 and 2", 0, 1], [1, 2, 0], [0, 1, 2]], '
        '"reasoning_steps": ["because swap 1 and 2"], "confidence": 0.7}'
    )
    answer, _steps, confidence = gen._parse_response(response, expected_shape=(3, 3))
    assert answer is None
    assert confidence == 0.0


def test_llm_generator_defaults_bfloat16() -> None:
    gen = LLMGenerator(model_name="test-model")
    assert gen.torch_dtype == "bfloat16"


def test_llm_generator_custom_dtype() -> None:
    gen = LLMGenerator(model_name="test-model", torch_dtype="float16")
    assert gen.torch_dtype == "float16"


def test_llm_generator_default_attn_auto() -> None:
    gen = LLMGenerator(model_name="test-model")
    assert gen.attn_implementation == "auto"


def test_llm_resolve_attn_explicit() -> None:
    gen = LLMGenerator(model_name="test-model", attn_implementation="sdpa")
    assert gen._resolve_attn_implementation() == "sdpa"


def test_llm_resolve_attn_explicit_eager() -> None:
    gen = LLMGenerator(model_name="test-model", attn_implementation="eager")
    assert gen._resolve_attn_implementation() == "eager"


def test_llm_resolve_attn_auto_no_flash(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without flash-attn installed, auto should fall back to sdpa."""
    import importlib.util

    gen = LLMGenerator(model_name="test-model", attn_implementation="auto")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert gen._resolve_attn_implementation() == "sdpa"


def test_llm_resolve_attn_auto_with_flash(monkeypatch: pytest.MonkeyPatch) -> None:
    """With flash-attn installed, auto should select flash_attention_2."""
    import importlib.util

    gen = LLMGenerator(model_name="test-model", attn_implementation="auto")
    fake_spec = importlib.util.spec_from_loader("flash_attn", loader=None)
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: fake_spec if name == "flash_attn" else None,
    )
    assert gen._resolve_attn_implementation() == "flash_attention_2"
