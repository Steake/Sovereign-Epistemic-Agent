"""Tests for the LLM generator and its response parsing/validation logic."""

from __future__ import annotations

import json

import pytest

from epistemic_tribunal.generators.llm import (
    LLMGenerator,
    _clean_answer_grid,
    _extract_int_from_cell,
    _extract_json_object,
    _has_reasoning_bleed,
    _strip_think_blocks,
)
from epistemic_tribunal.types import Task


# ---------------------------------------------------------------------------
# _strip_think_blocks
# ---------------------------------------------------------------------------


class TestStripThinkBlocks:
    def test_removes_think_block(self):
        text = '<think>Some reasoning here</think>{"answer": [[1]]}'
        assert _strip_think_blocks(text) == '{"answer": [[1]]}'

    def test_removes_multiline_think_block(self):
        text = '<think>\nLine 1\nLine 2\n</think>\n{"answer": [[1]]}'
        result = _strip_think_blocks(text)
        assert "<think>" not in result
        assert '{"answer": [[1]]}' in result

    def test_removes_multiple_think_blocks(self):
        text = '<think>A</think>middle<think>B</think>end'
        result = _strip_think_blocks(text)
        assert result == "middleend"

    def test_no_think_blocks(self):
        text = '{"answer": [[1, 2], [3, 4]]}'
        assert _strip_think_blocks(text) == text


# ---------------------------------------------------------------------------
# _extract_json_object
# ---------------------------------------------------------------------------


class TestExtractJsonObject:
    def test_clean_json(self):
        text = '{"answer": [[1, 2]], "confidence": 0.8}'
        result = _extract_json_object(text)
        assert result is not None
        assert result["confidence"] == 0.8

    def test_json_with_trailing_text(self):
        text = '{"answer": [[1]]} and some trailing reasoning bleed'
        result = _extract_json_object(text)
        assert result is not None
        assert result["answer"] == [[1]]

    def test_json_in_markdown_fence(self):
        text = '```json\n{"answer": [[1, 2]], "confidence": 0.9}\n```'
        result = _extract_json_object(text)
        assert result is not None
        assert result["answer"] == [[1, 2]]

    def test_json_with_leading_prose(self):
        text = 'Here is my answer:\n{"answer": [[3, 4]], "confidence": 0.7}'
        result = _extract_json_object(text)
        assert result is not None
        assert result["answer"] == [[3, 4]]

    def test_no_json(self):
        text = "There is no JSON here at all."
        assert _extract_json_object(text) is None

    def test_malformed_json(self):
        text = '{"answer": [[1, 2], [3, 4'
        assert _extract_json_object(text) is None

    def test_nested_braces(self):
        text = '{"answer": [[1]], "meta": {"key": "val"}}'
        result = _extract_json_object(text)
        assert result is not None
        assert result["meta"] == {"key": "val"}


# ---------------------------------------------------------------------------
# _extract_int_from_cell
# ---------------------------------------------------------------------------


class TestExtractIntFromCell:
    def test_int_value(self):
        assert _extract_int_from_cell(3) == 3

    def test_float_whole(self):
        assert _extract_int_from_cell(3.0) == 3

    def test_float_fractional(self):
        assert _extract_int_from_cell(3.5) is None

    def test_string_digit(self):
        assert _extract_int_from_cell("3") == 3

    def test_string_with_text(self):
        assert _extract_int_from_cell("3 (blue)") == 3

    def test_string_whitespace(self):
        assert _extract_int_from_cell("  5  ") == 5

    def test_pure_text(self):
        assert _extract_int_from_cell("blue") is None

    def test_none(self):
        assert _extract_int_from_cell(None) is None

    def test_list(self):
        assert _extract_int_from_cell([1]) is None


# ---------------------------------------------------------------------------
# _clean_answer_grid
# ---------------------------------------------------------------------------


class TestCleanAnswerGrid:
    def test_clean_int_grid(self):
        raw = [[1, 2], [3, 4]]
        assert _clean_answer_grid(raw) == [[1, 2], [3, 4]]

    def test_string_cells(self):
        raw = [["1", "2"], ["3", "4"]]
        assert _clean_answer_grid(raw) == [[1, 2], [3, 4]]

    def test_polluted_cells(self):
        raw = [["1 (red)", "2 (blue)"], [3, 4]]
        assert _clean_answer_grid(raw) == [[1, 2], [3, 4]]

    def test_non_list_row(self):
        raw = [[1, 2], "not a list"]
        assert _clean_answer_grid(raw) is None

    def test_unconvertible_cell(self):
        raw = [[1, "hello"], [3, 4]]
        assert _clean_answer_grid(raw) is None


# ---------------------------------------------------------------------------
# _has_reasoning_bleed
# ---------------------------------------------------------------------------


class TestHasReasoningBleed:
    def test_clean_response(self):
        parsed = {
            "answer": [[1, 2], [3, 4]],
            "reasoning_steps": ["Applied colour swap rule."],
        }
        assert _has_reasoning_bleed(parsed) is False

    def test_reasoning_in_answer_cells(self):
        parsed = {
            "answer": [["1 colour mapping applied", "2"], ["3", "4"]],
            "reasoning_steps": ["Applied colour mapping to grid."],
        }
        assert _has_reasoning_bleed(parsed) is True

    def test_no_reasoning_steps(self):
        parsed = {"answer": [[1, 2]], "reasoning_steps": []}
        assert _has_reasoning_bleed(parsed) is False

    def test_int_cells_no_bleed(self):
        parsed = {
            "answer": [[1, 2], [3, 4]],
            "reasoning_steps": ["Very detailed reasoning about colour transformations."],
        }
        assert _has_reasoning_bleed(parsed) is False


# ---------------------------------------------------------------------------
# LLMGenerator._parse_response
# ---------------------------------------------------------------------------


class TestParseResponse:
    def setup_method(self):
        self.gen = LLMGenerator(seed=42, api_base="http://test", api_key="k", model="m")

    def test_valid_response(self):
        raw = json.dumps({
            "answer": [[1, 2, 3], [4, 5, 6]],
            "confidence": 0.85,
            "reasoning_steps": ["Step 1", "Step 2"],
        })
        answer, conf, steps = self.gen._parse_response(raw, expected_shape=(2, 3))
        assert answer == [[1, 2, 3], [4, 5, 6]]
        assert conf == 0.85
        assert "Step 1" in steps

    def test_shape_mismatch_returns_none(self):
        raw = json.dumps({
            "answer": [[1, 2], [3, 4]],
            "confidence": 0.8,
        })
        answer, conf, steps = self.gen._parse_response(raw, expected_shape=(3, 3))
        assert answer is None
        assert conf == 0.0
        assert any("shape mismatch" in s.lower() for s in steps)

    def test_values_outside_0_9_returns_none(self):
        raw = json.dumps({
            "answer": [[1, 10], [3, 4]],
            "confidence": 0.8,
        })
        answer, conf, steps = self.gen._parse_response(raw, expected_shape=(2, 2))
        assert answer is None
        assert conf == 0.0

    def test_think_block_stripped(self):
        raw = '<think>Let me think...</think>' + json.dumps({
            "answer": [[1, 2], [3, 4]],
            "confidence": 0.7,
        })
        answer, conf, steps = self.gen._parse_response(raw, expected_shape=(2, 2))
        assert answer == [[1, 2], [3, 4]]

    def test_prose_wrapped_json(self):
        raw = 'Here is my answer:\n' + json.dumps({
            "answer": [[0, 1], [2, 3]],
            "confidence": 0.6,
        }) + '\nI hope that helps!'
        answer, conf, steps = self.gen._parse_response(raw, expected_shape=(2, 2))
        assert answer == [[0, 1], [2, 3]]

    def test_no_json_returns_none(self):
        raw = "I cannot solve this task, sorry."
        answer, conf, steps = self.gen._parse_response(raw, expected_shape=(2, 2))
        assert answer is None
        assert conf == 0.0

    def test_string_cell_values_cleaned(self):
        raw = json.dumps({
            "answer": [["1", "2"], ["3", "4"]],
            "confidence": 0.9,
        })
        answer, conf, steps = self.gen._parse_response(raw, expected_shape=(2, 2))
        assert answer == [[1, 2], [3, 4]]

    def test_malformed_brackets_returns_none(self):
        raw = '{"answer": [[1, 2], [3, 4'
        answer, conf, steps = self.gen._parse_response(raw, expected_shape=(2, 2))
        assert answer is None
        assert conf == 0.0


# ---------------------------------------------------------------------------
# LLMGenerator.generate (with mocked _call_llm)
# ---------------------------------------------------------------------------


class TestLLMGeneratorGenerate:
    def test_successful_generate(self, simple_task: Task, monkeypatch):
        gen = LLMGenerator(seed=42, api_base="http://test", api_key="k", model="m")
        response = json.dumps({
            "answer": [[2, 0, 1], [1, 2, 0], [0, 1, 2]],
            "confidence": 0.85,
            "reasoning_steps": ["Identified swap pattern."],
        })
        monkeypatch.setattr(gen, "_call_llm", lambda prompt: response)

        trace = gen.generate(simple_task)
        assert trace.generator_name == "llm"
        assert trace.answer == [[2, 0, 1], [1, 2, 0], [0, 1, 2]]
        assert trace.confidence_score == 0.85

    def test_failed_parse_returns_zero_grid(self, simple_task: Task, monkeypatch):
        gen = LLMGenerator(seed=42, api_base="http://test", api_key="k", model="m")
        monkeypatch.setattr(gen, "_call_llm", lambda prompt: "garbage response")

        trace = gen.generate(simple_task)
        assert trace.generator_name == "llm"
        assert trace.confidence_score == 0.0
        # Zero grid with correct shape
        assert len(trace.answer) == 3
        assert all(len(row) == 3 for row in trace.answer)
        assert all(cell == 0 for row in trace.answer for cell in row)

    def test_wrong_shape_returns_zero_grid(self, simple_task: Task, monkeypatch):
        gen = LLMGenerator(seed=42, api_base="http://test", api_key="k", model="m")
        response = json.dumps({
            "answer": [[1, 2], [3, 4]],  # 2x2 instead of 3x3
            "confidence": 0.9,
        })
        monkeypatch.setattr(gen, "_call_llm", lambda prompt: response)

        trace = gen.generate(simple_task)
        assert trace.confidence_score == 0.0
        assert len(trace.answer) == 3
        assert all(len(row) == 3 for row in trace.answer)
