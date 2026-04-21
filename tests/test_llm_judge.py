"""Tests for the LLM-as-a-judge mechanism in TraceCritic (GSM8K domain)."""

from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch

from epistemic_tribunal.critics.trace_critic import TraceCritic
from epistemic_tribunal.tribunal_types import CandidateTrace, Task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gsm8k_trace(answer: str = "9", steps: list[str] | None = None) -> CandidateTrace:
    """Build a minimal GSM8K candidate trace."""
    return CandidateTrace(
        generator_name="llm",
        answer=answer,
        reasoning_steps=steps or [
            "Eggs laid per day: 16",
            "Eggs eaten: 3",
            "Eggs sold: 4",
            "Eggs left: 16 - 3 - 4 = 9",
        ],
        confidence_score=0.9,
    )


def _judge_response(
    arithmetic: float = 0.95,
    logical: float = 0.90,
    alignment: float = 0.92,
    coherence: float = 0.93,
    rationale: str = "Correct arithmetic and trace alignment.",
) -> MagicMock:
    """Build a mock OpenAI response object returning a structured JSON rubric."""
    payload = json.dumps({
        "arithmetic_consistency": arithmetic,
        "logical_consistency": logical,
        "answer_trace_alignment": alignment,
        "final_rule_coherence": coherence,
        "brief_rationale": rationale,
    })
    message = MagicMock()
    message.content = payload
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# Flag-disabled behaviour (default path)
# ---------------------------------------------------------------------------


def test_llm_judge_disabled_returns_one(gsm8k_task: Task) -> None:
    """With use_llm_judge_for_math=False, rule coherence should be 1.0 for GSM8K."""
    critic = TraceCritic(use_llm_judge_for_math=False)
    trace = _gsm8k_trace()
    result = critic.critique(gsm8k_task, trace)
    assert result.rule_coherence_score == 1.0


def test_llm_judge_disabled_no_api_calls(gsm8k_task: Task) -> None:
    """With the flag off, no HTTP calls to the LLM should be made."""
    critic = TraceCritic(use_llm_judge_for_math=False)
    trace = _gsm8k_trace()
    with patch("openai.OpenAI") as mock_client:
        critic.critique(gsm8k_task, trace)
        mock_client.assert_not_called()


# ---------------------------------------------------------------------------
# Flag-enabled: happy path
# ---------------------------------------------------------------------------


def test_llm_judge_enabled_uses_final_rule_coherence(gsm8k_task: Task) -> None:
    """When the judge is enabled, rule_coherence_score must reflect final_rule_coherence."""
    critic = TraceCritic(use_llm_judge_for_math=True)
    trace = _gsm8k_trace()
    mock_response = _judge_response(coherence=0.85)

    with patch("epistemic_tribunal.critics.trace_critic.OpenAI") as mock_openai_cls:
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
            instance = mock_openai_cls.return_value
            instance.chat.completions.create.return_value = mock_response
            result = critic.critique(gsm8k_task, trace)

    assert result.rule_coherence_score == pytest.approx(0.85, abs=1e-4)


def test_llm_judge_persists_all_dimensions_to_metadata(gsm8k_task: Task) -> None:
    """All four rubric dimensions and rationale must be stored in trace.metadata."""
    critic = TraceCritic(use_llm_judge_for_math=True)
    trace = _gsm8k_trace()
    mock_response = _judge_response(
        arithmetic=0.8,
        logical=0.7,
        alignment=0.9,
        coherence=0.85,
        rationale="Good reasoning.",
    )

    with patch("epistemic_tribunal.critics.trace_critic.OpenAI") as mock_openai_cls:
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
            instance = mock_openai_cls.return_value
            instance.chat.completions.create.return_value = mock_response
            critic.critique(gsm8k_task, trace)

    assert trace.metadata["llm_judge_arithmetic_consistency"] == pytest.approx(0.8)
    assert trace.metadata["llm_judge_logical_consistency"] == pytest.approx(0.7)
    assert trace.metadata["llm_judge_answer_trace_alignment"] == pytest.approx(0.9)
    assert trace.metadata["llm_judge_final_rule_coherence"] == pytest.approx(0.85)
    assert trace.metadata["llm_judge_rationale"] == "Good reasoning."


def test_llm_judge_notes_include_all_dimensions(gsm8k_task: Task) -> None:
    """The CritiqueResult.notes must include the llm_judge breakdown."""
    critic = TraceCritic(use_llm_judge_for_math=True)
    trace = _gsm8k_trace()
    mock_response = _judge_response(coherence=0.91)

    with patch("epistemic_tribunal.critics.trace_critic.OpenAI") as mock_openai_cls:
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
            instance = mock_openai_cls.return_value
            instance.chat.completions.create.return_value = mock_response
            result = critic.critique(gsm8k_task, trace)

    assert "llm_judge(" in result.notes


def test_llm_judge_scores_bounded(gsm8k_task: Task) -> None:
    """Rule coherence score must remain in [0.0, 1.0] even with extreme judge outputs."""
    critic = TraceCritic(use_llm_judge_for_math=True)
    trace = _gsm8k_trace()
    # Judge returns values outside [0, 1]
    payload = json.dumps({
        "arithmetic_consistency": 1.5,
        "logical_consistency": -0.2,
        "answer_trace_alignment": 0.5,
        "final_rule_coherence": 1.9,
        "brief_rationale": "Extreme values test.",
    })
    message = MagicMock()
    message.content = payload
    choice = MagicMock()
    choice.message = message
    mock_response = MagicMock()
    mock_response.choices = [choice]

    with patch("epistemic_tribunal.critics.trace_critic.OpenAI") as mock_openai_cls:
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
            instance = mock_openai_cls.return_value
            instance.chat.completions.create.return_value = mock_response
            result = critic.critique(gsm8k_task, trace)

    assert 0.0 <= result.rule_coherence_score <= 1.0
    assert 0.0 <= result.aggregate_score <= 1.0


# ---------------------------------------------------------------------------
# Flag-enabled: degraded / failure paths
# ---------------------------------------------------------------------------


def test_llm_judge_missing_api_key_defaults_to_one(gsm8k_task: Task) -> None:
    """If DEEPSEEK_API_KEY is absent, the judge must safely default to 1.0."""
    critic = TraceCritic(use_llm_judge_for_math=True)
    trace = _gsm8k_trace()
    with patch.dict("os.environ", {}, clear=True):
        # Remove the key if present
        import os
        os.environ.pop("DEEPSEEK_API_KEY", None)
        result = critic.critique(gsm8k_task, trace)

    assert result.rule_coherence_score == 1.0


def test_llm_judge_api_failure_defaults_to_half(gsm8k_task: Task) -> None:
    """If the API call raises an exception, rule_coherence must fall back to 0.5."""
    critic = TraceCritic(use_llm_judge_for_math=True)
    trace = _gsm8k_trace()

    with patch("epistemic_tribunal.critics.trace_critic.OpenAI") as mock_openai_cls:
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
            instance = mock_openai_cls.return_value
            instance.chat.completions.create.side_effect = RuntimeError("Network error")
            result = critic.critique(gsm8k_task, trace)

    assert result.rule_coherence_score == pytest.approx(0.5)


def test_llm_judge_empty_response_defaults_to_half(gsm8k_task: Task) -> None:
    """If the LLM returns an empty content string, fall back to 0.5."""
    critic = TraceCritic(use_llm_judge_for_math=True)
    trace = _gsm8k_trace()

    message = MagicMock()
    message.content = ""
    choice = MagicMock()
    choice.message = message
    mock_response = MagicMock()
    mock_response.choices = [choice]

    with patch("epistemic_tribunal.critics.trace_critic.OpenAI") as mock_openai_cls:
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
            instance = mock_openai_cls.return_value
            instance.chat.completions.create.return_value = mock_response
            result = critic.critique(gsm8k_task, trace)

    assert result.rule_coherence_score == pytest.approx(0.5)


def test_llm_judge_malformed_json_defaults_to_half(gsm8k_task: Task) -> None:
    """Malformed JSON in the response must not crash — fall back to 0.5."""
    critic = TraceCritic(use_llm_judge_for_math=True)
    trace = _gsm8k_trace()

    message = MagicMock()
    message.content = "NOT VALID JSON {{{}"
    choice = MagicMock()
    choice.message = message
    mock_response = MagicMock()
    mock_response.choices = [choice]

    with patch("epistemic_tribunal.critics.trace_critic.OpenAI") as mock_openai_cls:
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
            instance = mock_openai_cls.return_value
            instance.chat.completions.create.return_value = mock_response
            result = critic.critique(gsm8k_task, trace)

    assert result.rule_coherence_score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Domain isolation: ARC tasks must never trigger the LLM judge
# ---------------------------------------------------------------------------


def test_llm_judge_never_fires_for_arc_tasks(simple_task: Task) -> None:
    """Even with the flag on, ARC tasks must not invoke the LLM judge."""
    critic = TraceCritic(use_llm_judge_for_math=True)
    trace = CandidateTrace(
        generator_name="greedy",
        answer=[[2, 0, 1], [1, 2, 0], [0, 1, 2]],
        reasoning_steps=["Swap colours."],
        confidence_score=0.8,
    )
    with patch("epistemic_tribunal.critics.trace_critic.OpenAI") as mock_openai_cls:
        critic.critique(simple_task, trace)
        mock_openai_cls.assert_not_called()
