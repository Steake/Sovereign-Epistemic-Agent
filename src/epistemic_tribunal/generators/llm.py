"""LLM-backed generator — calls an external language model via OpenAI-compatible API.

Handles common LLM failure modes:
- JSON extraction from prose-wrapped responses
- <think> block bleed into answer fields
- Grid shape validation against expected test input dimensions
- Non-integer token stripping from grid cells
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

from epistemic_tribunal.generators.base import BaseGenerator
from epistemic_tribunal.tasks.base import grid_shape
from epistemic_tribunal.types import CandidateTrace, Task
from epistemic_tribunal.utils.logging import get_logger

log = get_logger(__name__)


class LLMGenerator(BaseGenerator):
    """Generator that delegates to an external LLM via an OpenAI-compatible API.

    Parameters
    ----------
    seed:
        RNG seed (passed to base class, not used for LLM calls).
    api_base:
        API base URL.  Falls back to ``LLM_API_BASE`` env var.
    api_key:
        API key.  Falls back to ``LLM_API_KEY`` env var.
    model:
        Model identifier.  Falls back to ``LLM_MODEL`` env var.
    temperature:
        Sampling temperature.
    max_tokens:
        Maximum tokens to generate.
    """

    name = "llm"

    def __init__(
        self,
        seed: int = 42,
        *,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> None:
        super().__init__(seed=seed, **kwargs)
        self.api_base = api_base or os.environ.get("LLM_API_BASE", "")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.model = model or os.environ.get("LLM_MODEL", "")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, task: Task) -> CandidateTrace:
        """Call the LLM and parse the response into a CandidateTrace."""
        prompt = self._build_prompt(task)
        raw_response = self._call_llm(prompt)

        expected_shape = grid_shape(task.test_input)
        answer, confidence, reasoning_steps = self._parse_response(
            raw_response, expected_shape=expected_shape
        )

        if answer is None:
            # Validation failed — return a zero grid with confidence 0.0
            rows, cols = expected_shape
            answer = [[0] * cols for _ in range(rows)]
            confidence = 0.0
            reasoning_steps.append(
                "PARSE_FAILURE: LLM response could not be validated; "
                "returning zero grid with confidence 0.0."
            )
            log.warning(
                "LLM generator parse failure for task %s; returning zero grid.",
                task.task_id,
            )

        return CandidateTrace(
            generator_name=self.name,
            answer=answer,
            reasoning_steps=reasoning_steps,
            raw_trace=raw_response,
            token_count=len(raw_response.split()),
            confidence_score=confidence,
            derived_features={"source": "llm", "model": self.model},
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self, task: Task) -> str:
        """Build a structured prompt asking the LLM to solve the task."""
        parts = [
            "You are solving an ARC-like grid transformation task.",
            "Analyse the training examples and predict the output grid for the test input.",
            "",
        ]

        for i, example in enumerate(task.train, 1):
            parts.append(f"Training Example {i}:")
            parts.append(f"  Input:  {example.input}")
            parts.append(f"  Output: {example.output}")
            parts.append("")

        parts.append(f"Test Input: {task.test_input}")
        parts.append("")
        parts.append(
            "Respond with a JSON object containing exactly these fields:\n"
            '  "answer": a 2D list of integers (the predicted output grid)\n'
            '  "confidence": a float between 0.0 and 1.0\n'
            '  "reasoning_steps": a list of strings describing your reasoning\n'
            "\n"
            "Do NOT include any text outside the JSON object. "
            "Do NOT embed reasoning inside the answer grid."
        )
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # LLM API call
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API and return the raw response text.

        Uses the ``openai`` client library with a custom base URL so it
        works with RunPod, vLLM, and other OpenAI-compatible endpoints.
        """
        try:
            from openai import OpenAI  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "The 'openai' package is required for LLMGenerator. "
                "Install it with: pip install openai"
            ) from exc

        client = OpenAI(base_url=self.api_base, api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""

    # ------------------------------------------------------------------
    # Response parsing with validation
    # ------------------------------------------------------------------

    def _parse_response(
        self,
        raw: str,
        *,
        expected_shape: tuple[int, int],
    ) -> tuple[Optional[list[list[int]]], float, list[str]]:
        """Parse and validate the LLM response.

        Returns
        -------
        (answer, confidence, reasoning_steps)
            If validation fails, answer is ``None`` and confidence is ``0.0``.
        """
        reasoning_steps: list[str] = []

        # Strip <think>...</think> blocks (model chain-of-thought)
        cleaned = _strip_think_blocks(raw)

        # Extract JSON object from potentially prose-wrapped response
        parsed = _extract_json_object(cleaned)
        if parsed is None:
            reasoning_steps.append("JSON extraction failed from LLM response.")
            log.warning("Failed to extract JSON from LLM response.")
            return None, 0.0, reasoning_steps

        # Extract fields
        answer_raw = parsed.get("answer")
        confidence_raw = parsed.get("confidence", 0.5)
        steps_raw = parsed.get("reasoning_steps", [])

        if isinstance(steps_raw, list):
            reasoning_steps.extend(str(s) for s in steps_raw)

        # Validate confidence
        try:
            confidence = float(confidence_raw)
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.5

        # Validate answer is a 2D list
        if not isinstance(answer_raw, list):
            reasoning_steps.append("Answer field is not a list.")
            log.warning("LLM answer is not a list.")
            return None, 0.0, reasoning_steps

        # Clean and validate the answer grid
        answer = _clean_answer_grid(answer_raw)
        if answer is None:
            reasoning_steps.append(
                "Answer grid contains non-convertible values after cleaning."
            )
            log.warning("LLM answer grid contains non-convertible values.")
            return None, 0.0, reasoning_steps

        # Validate all values are 0-9
        for r, row in enumerate(answer):
            for c, val in enumerate(row):
                if not (0 <= val <= 9):
                    reasoning_steps.append(
                        f"Answer cell [{r}][{c}] value {val} outside 0-9 range."
                    )
                    log.warning(
                        "LLM answer cell [%d][%d] = %d outside valid range 0-9.",
                        r, c, val,
                    )
                    return None, 0.0, reasoning_steps

        # Validate grid dimensions match expected shape
        actual_shape = grid_shape(answer)
        if actual_shape != expected_shape:
            reasoning_steps.append(
                f"Grid shape mismatch: expected {expected_shape}, "
                f"got {actual_shape}."
            )
            log.warning(
                "LLM answer grid shape mismatch: expected %s, got %s.",
                expected_shape,
                actual_shape,
            )
            return None, 0.0, reasoning_steps

        # Check for reasoning text bleeding into the answer grid
        if _has_reasoning_bleed(parsed):
            reasoning_steps.append(
                "Detected reasoning text embedded in the answer grid."
            )
            log.warning("Reasoning text detected inside LLM answer grid.")
            return None, 0.0, reasoning_steps

        return answer, confidence, reasoning_steps


# ---------------------------------------------------------------------------
# Module-level helper functions
# ---------------------------------------------------------------------------


def _strip_think_blocks(text: str) -> str:
    """Remove ``<think>...</think>`` blocks from LLM output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_json_object(text: str) -> Optional[dict]:
    """Extract the first valid JSON object from *text*.

    Handles cases where the model wraps the JSON in markdown fences or
    bleeds reasoning text after the closing brace.
    """
    # Try direct parse first
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Try to find JSON within markdown code fences
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        try:
            obj = json.loads(fence_match.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # Find the first { and try to match a balanced JSON object
    start = text.find("{")
    if start == -1:
        return None

    # Walk through to find matching brace (handles nested braces)
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    pass
                break

    return None


def _clean_answer_grid(
    raw_grid: list,
) -> Optional[list[list[int]]]:
    """Clean a raw answer grid by stripping non-integer tokens from cells.

    Handles cases where the model embeds reasoning text or extra tokens
    inside cell values (e.g. ``"3 (blue)"`` → ``3``).

    Returns ``None`` if any cell cannot be converted to an integer.
    """
    cleaned: list[list[int]] = []
    for row in raw_grid:
        if not isinstance(row, list):
            return None
        cleaned_row: list[int] = []
        for cell in row:
            val = _extract_int_from_cell(cell)
            if val is None:
                return None
            cleaned_row.append(val)
        cleaned.append(cleaned_row)
    return cleaned


def _extract_int_from_cell(cell: Any) -> Optional[int]:
    """Extract an integer value from a potentially polluted cell value.

    Handles: ``3``, ``"3"``, ``"3 (blue)"``, ``" 3 "`` etc.
    Returns ``None`` if no integer can be extracted.
    """
    if isinstance(cell, int):
        return cell
    if isinstance(cell, float):
        if cell == int(cell):
            return int(cell)
        return None
    if isinstance(cell, str):
        # Strip whitespace and try direct conversion
        cell = cell.strip()
        try:
            return int(cell)
        except ValueError:
            pass
        # Extract leading integer (e.g. "3 (blue)" → 3)
        match = re.match(r"^(\d+)", cell)
        if match:
            return int(match.group(1))
        return None
    return None


def _has_reasoning_bleed(parsed: dict) -> bool:
    """Check if reasoning_steps text appears inside the answer grid.

    This catches a common failure mode where the model includes
    chain-of-thought text inside answer cell values.
    """
    steps = parsed.get("reasoning_steps", [])
    if not isinstance(steps, list) or not steps:
        return False

    answer = parsed.get("answer", [])
    if not isinstance(answer, list):
        return False

    # Build a set of reasoning keywords (words >= 4 chars from reasoning)
    reasoning_text = " ".join(str(s) for s in steps).lower()
    reasoning_words = {
        w for w in reasoning_text.split() if len(w) >= 4 and w.isalpha()
    }

    if not reasoning_words:
        return False

    # Check if any cell in the answer grid contains reasoning words
    for row in answer:
        if not isinstance(row, list):
            continue
        for cell in row:
            if isinstance(cell, str):
                cell_words = {
                    w for w in cell.lower().split() if len(w) >= 4 and w.isalpha()
                }
                if cell_words & reasoning_words:
                    return True

    return False
