"""LLM-backed generator with strict output validation for ARC-like tasks."""

from __future__ import annotations

import ast
import json
import re
import time
from typing import Any, Optional, Callable

from epistemic_tribunal.failure_memory.models import FailureConstraints
from epistemic_tribunal.generators.base import BaseGenerator
from epistemic_tribunal.tasks.base import grid_shape
from epistemic_tribunal.tribunal_types import CandidateTrace, Task
from epistemic_tribunal.utils.logging import get_logger

log = get_logger(__name__)



class LLMGenerator(BaseGenerator):
    """Generate a candidate trace with a text-generation model."""

    name = "llm"

    def __init__(
        self,
        seed: int = 42,
        model_name: str = "deepseek-reasoner",
        max_new_tokens: int = 8192,
        temperature: float = 0.1,
        system_prompt: Optional[str] = None,
        prompt_style: str = "standard",
        include_think: bool = True,
        top_p: float = 0.95,
        trust_remote_code: bool = False,
        device: Optional[str] = None,
        torch_dtype: str = "bfloat16",
        attn_implementation: str = "auto",
        use_json_schema: bool = True,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        # Allow per-instance name override (e.g. 'llm_warm') so that two LLM
        # generators can be registered with distinct names for coalition scoring.
        instance_name = kwargs.pop("name", None)
        super().__init__(seed=seed, **kwargs)
        if instance_name:
            self.name = instance_name  # type: ignore[assignment]
        self.model_name = kwargs.pop("model", model_name)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.prompt_style = prompt_style
        self.include_think = include_think
        self.top_p = top_p
        self.trust_remote_code = trust_remote_code
        self.device = device
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation
        self.use_json_schema = use_json_schema
        self.api_base = api_base
        self.api_key = api_key
        
        self._pipeline: Any = None
        self._client: Optional[Any] = None

    def generate(
        self,
        task: Task,
        on_token: Optional[Callable[[str, str], None]] = None,
        failure_constraints: Optional[FailureConstraints] = None,
    ) -> CandidateTrace:
        from epistemic_tribunal.tribunal_types import TaskDomain
        
        is_math = task.domain == TaskDomain.GSM8K_MATH
        expected_shape = grid_shape(task.test_input) if not is_math else None
        
        if is_math:
            prompt, schema = self._build_math_prompt(task, failure_constraints)
        else:
            prompt, schema = self._build_prompt(task, expected_shape, failure_constraints)
        
        response, finish_reason = self._complete(prompt, schema, on_token=on_token)
        self.last_finish_reason = finish_reason
        log.info("LLM Generation complete. Length: %d characters", len(response))
        log.debug("Raw Response: %s...", response[:200])
        
        if finish_reason == "length":
            raise ValueError(f"[length] LLM hit max_tokens={self.max_new_tokens}")

        reasoning_steps: list[str] = []
        if is_math:
            # Always extract reasoning for math tasks — parses <think> blocks, JSON
            # reasoning_steps fields, and pre-JSON prose. Populates the judge rubric.
            reasoning_steps = self._extract_reasoning_steps(response)

        confidence = 1.0      # Hardcoded default since unprompted

        payload = self._extract_payload(response, finish_reason=finish_reason)
        if payload is None:
            raise ValueError("[json_not_found] LLM response contained no parsable JSON object.")

        if "answer" not in payload:
            raise ValueError("[json_invalid] JSON object parsed but missing 'answer' key.")

        answer_raw = payload.get("answer")
        
        if is_math:
            # We don't validate shape or bleed for math in the same way
            answer = answer_raw
        else:
            if self._contains_reasoning_text(answer_raw, reasoning_steps):
                raise ValueError("[reasoning_bleed] Rejected LLM answer because reasoning text bled into the answer grid.")
                
            answer = self._validate_answer(answer_raw, expected_shape=expected_shape)
            if answer is None:
                # Attempt shape-clamp rescue before hard-failing:
                clamped = self._clamp_to_shape(answer_raw, expected_shape)
                if clamped is not None:
                    log.info(
                        "Shape-clamp rescue succeeded for task %s: clamped to %dx%d.",
                        task.task_id, *expected_shape
                    )
                    answer = clamped
                else:
                    raise ValueError(
                        f"[grid_shape_invalid] LLM response grid did not match "
                        f"exact {expected_shape[0]}x{expected_shape[1]} dimensionality "
                        "or contained invalid tokens (clamp rescue also failed)."
                    )

        return CandidateTrace(
            generator_name=self.name,
            answer=answer,
            reasoning_steps=reasoning_steps,
            raw_trace=response,
            token_count=len(response.split()),
            confidence_score=confidence,
            derived_features={
                "model_name": self.model_name,
                "expected_shape": expected_shape,
                "finish_reason": finish_reason,
            },
            metadata={
                "model": self.model_name,
                "finish_reason": finish_reason,
                "raw_response": response
            }
        )

    def _build_math_prompt(
        self, task: Task,
        failure_constraints: Optional[FailureConstraints] = None,
    ) -> tuple[str, dict]:
        constraint_block = self._build_constraint_block(failure_constraints)
        main_prompt = (
            "Solve the following math word problem. Show your reasoning, then provide the final scalar answer.\n"
            "Return ONLY a JSON object containing your final numerical answer.\n"
            "Schema: {\"answer\": \"123.45\"}\n"
            "You may return it as a string, integer, or float, but it must be under the key 'answer'."
        )
        prompt = (
            f"{constraint_block}"
            f"{main_prompt}\n\n"
            f"Question:\n{task.test_input}\n\n"
            "JSON answer:"
        )
        schema = {
            "type": "object",
            "properties": {
                "answer": {
                    "type": ["string", "number", "integer"]
                }
            },
            "required": ["answer"],
            "additionalProperties": False
        }
        return prompt, schema

    def _build_prompt(
        self, task: Task, expected_shape: tuple[int, int],
        failure_constraints: Optional[FailureConstraints] = None,
    ) -> tuple[str, dict]:
        H, W = expected_shape
        train_examples = []
        for idx, example in enumerate(task.train, start=1):
            train_examples.append(
                f"Example {idx}\n"
                f"Input: {json.dumps(example.input)}\n"
                f"Output: {json.dumps(example.output)}"
            )

        # Build an explicit template showing the exact target shape
        template_row = ["?"] * W
        template_grid = [template_row for _ in range(H)]
        template_str = json.dumps(template_grid)

        main_prompt = (
            f"=== SHAPE REQUIREMENT (HARD CONSTRAINT) ===\n"
            f"Your answer grid MUST be EXACTLY {H} rows × {W} columns.\n"
            f"Required shape template (replace ? with integers 0-9):\n"
            f"{template_str}\n\n"
            "=== RESPONSE FORMAT ===\n"
            "Return ONLY a JSON object. No prose. No markdown. No explanation.\n"
            'Schema: {"answer": [[int, ...], ...]}\n'
            f"• Exactly {H} rows.\n"
            f"• Exactly {W} integers per row.\n"
            "• Every integer must be in range 0–9.\n"
            "• Do NOT include extra keys, confidence, or reasoning_steps.\n"
            f"• WRONG examples: fewer rows, more columns, flat list. RIGHT: {H}×{W} nested list."
        )

        constraint_block = self._build_constraint_block(failure_constraints)

        train_block = "\n".join(train_examples) if train_examples else "None"
        prompt = (
            f"{constraint_block}"
            f"{main_prompt}\n\n"
            f"Task: {task.task_id}\n"
            f"Train:\n{train_block}\n\n"
            f"Test Input ({len(task.test_input)}×{len(task.test_input[0]) if task.test_input else 0} grid):\n"
            f"{json.dumps(task.test_input)}\n\n"
            f"JSON answer (must be {H}×{W}):"
        )

        # Sprint B: Brutally simple, completely flat schema
        schema = {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "array",
                    "minItems": H,
                    "maxItems": H,
                    "items": {
                        "type": "array",
                        "minItems": W,
                        "maxItems": W,
                        "items": {"type": "integer", "minimum": 0, "maximum": 9}
                    }
                },
            },
            "required": ["answer"],
            "additionalProperties": False
        }
        return prompt, schema

    @staticmethod
    def _build_constraint_block(
        constraints: Optional[FailureConstraints],
    ) -> str:
        """Format failure constraints as a prompt-injectable text block.

        Returns an empty string when there are no constraints to inject,
        so callers can unconditionally prepend the result.
        """
        if constraints is None or not constraints.has_constraints:
            return ""

        parts: list[str] = ["=== FAILURE MEMORY (AVOID THESE) ==="]

        if constraints.bad_answers:
            parts.append(
                "The following answers have been tried on this or similar "
                "tasks and were WRONG. Do NOT reproduce them:"
            )
            for ans in constraints.bad_answers:
                parts.append(f"  ✗ {ans}")

        if constraints.structural_warnings:
            parts.append("")
            parts.append("Structural warnings from prior failures:")
            for warning in constraints.structural_warnings:
                parts.append(f"  ⚠ {warning}")

        parts.append(
            "\nUse this information to avoid repeating known mistakes.\n"
            "=== END FAILURE MEMORY ===\n\n"
        )
        return "\n".join(parts)

    def _load_pipeline(self) -> Any:
        """Loads the remote OpenAI-compatible client endpoint."""
        if hasattr(self, "_pipeline") and self._pipeline is not None:
            return self._pipeline
        
        # Hard requirements for Kaggle cloud endpoint
        if not self.api_base:
            self.api_base = "https://api.deepseek.com"
            
        from openai import OpenAI
        self._pipeline = OpenAI(base_url=self.api_base, api_key=self.api_key or "sk-no-key-required")
        log.info("Initialized remote API client at %s", self.api_base)
            
        return self._pipeline

    def _complete(
        self, 
        prompt: str, 
        schema: dict, 
        on_token: Optional[Callable[[str, str], None]] = None
    ) -> tuple[str, str]:
        self._load_pipeline()

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        api_kwargs = {}
        if schema:
            if self.use_json_schema:
                api_kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "arc_answer",
                        "schema": schema
                    }
                }
            else:
                api_kwargs["response_format"] = {"type": "json_object"}

        max_retries = 5
        backoff = 2.0
        
        for attempt in range(max_retries):
            try:
                # If no callback, run non-streaming (faster for non-reasoning models)
                if not on_token:
                    response = self._pipeline.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_new_tokens,
                        top_p=self.top_p,
                        stream=False,
                        **api_kwargs
                    )
                    message = response.choices[0].message
                    text = message.content or ""
                    reasoning = getattr(message, "reasoning_content", None)
                    if reasoning:
                        text = f"<think>\n{reasoning}\n</think>\n{text}"
                    
                    return text, response.choices[0].finish_reason

                # Streaming mode
                response_stream = self._pipeline.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_new_tokens,
                    top_p=self.top_p,
                    stream=True,
                    **api_kwargs
                )

                full_content = []
                full_reasoning = []
                finish_reason = "stop"

                for chunk in response_stream:
                    if not chunk.choices:
                        continue
                    
                    delta = chunk.choices[0].delta
                    reasoning = getattr(delta, "reasoning_content", None)
                    if reasoning:
                        full_reasoning.append(reasoning)
                        on_token("reasoning", reasoning)
                    
                    content = delta.content
                    if content:
                        full_content.append(content)
                        on_token("content", content)
                    
                    if chunk.choices[0].finish_reason:
                        finish_reason = chunk.choices[0].finish_reason

                text_content = "".join(full_content)
                if full_reasoning:
                    text_reasoning = "".join(full_reasoning)
                    text = f"<think>\n{text_reasoning}\n</think>\n{text_content}"
                else:
                    text = text_content

                return text, finish_reason

            except Exception as e:
                if attempt == max_retries - 1:
                    log.error("API generation failed after %d attempts: %s", max_retries, e)
                    raise ValueError(f"API failure: {e}")
                
                wait_time = backoff ** (attempt + 1)
                log.warning("API connection failed (attempt %d/%d). Retrying in %.1fs... Error: %s", 
                            attempt + 1, max_retries, wait_time, e)
                # Reset full results for the next attempt if streaming already started
                time.sleep(wait_time)
                
        return "", "error"

    def _parse_response(
        self,
        response: str,
        *,
        expected_shape: tuple[int, int],
        finish_reason: str = "stop"
    ) -> tuple[Optional[list[list[int]]], list[str], float]:
        reasoning_steps = self._extract_reasoning_steps(response)
        payload = self._extract_payload(response, finish_reason=finish_reason)
        if payload is None:
            log.warning("LLM response contained no JSON object with an answer field.")
            return None, reasoning_steps, 0.0

        answer_raw = payload.get("answer")
        if self._contains_reasoning_text(answer_raw, reasoning_steps):
            log.warning("Rejected LLM answer because reasoning text bled into the answer grid.")
            return None, reasoning_steps, 0.0

        answer = self._validate_answer(answer_raw, expected_shape=expected_shape)
        if answer is None:
            return None, reasoning_steps, 0.0

        confidence = self._parse_confidence(payload.get("confidence"))
        return answer, reasoning_steps, confidence

    def _extract_reasoning_steps(self, response: str) -> list[str]:
        think_blocks = re.findall(r"<think>(.*?)</think>", response, flags=re.DOTALL | re.IGNORECASE)
        steps: list[str] = []
        for block in think_blocks:
            for line in block.splitlines():
                line = line.strip(" -*\t")
                if line:
                    steps.append(line)

        payload = self._extract_payload(response)
        if payload is not None and isinstance(payload.get("reasoning_steps"), list):
            for step in payload["reasoning_steps"]:
                if step is None:
                    continue
                text = str(step).strip()
                if text:
                    steps.append(text)

        return steps or ["LLM provided no explicit reasoning steps."]

    def _extract_payload(self, response: str, finish_reason: str = "stop") -> Optional[dict[str, Any]]:
        # First try to find a JSON markdown block
        md_match = re.search(r"```json\s*(.*?)\s*```", response, flags=re.DOTALL | re.IGNORECASE)
        if md_match:
            candidate = md_match.group(1).strip()
            parsed = self._parse_json_like(candidate)
            if isinstance(parsed, dict) and "answer" in parsed:
                return parsed

        # Search the region AFTER </think> first (preferred)
        post_think = re.split(r"</think>", response, flags=re.IGNORECASE)
        search_regions = [post_think[-1]] + post_think[:-1] if len(post_think) > 1 else [response]

        for region in search_regions:
            candidate = self._extract_balanced_json(region)
            if candidate is None:
                continue
            parsed = self._parse_json_like(candidate)
            if isinstance(parsed, dict) and "answer" in parsed:
                return parsed

        # STRICT SALVAGE for truncation
        if finish_reason == "length":
            start = response.find("{")
            if start != -1:
                base_text = response[start:].rstrip()
                # Strict subset brace completion.
                # Only succeeds if the grid rows and cells already parsed are complete.
                for suffix in ["}", "]}", "]]}", "]}]}"]:
                    candidate = base_text + suffix
                    parsed = self._parse_json_like(candidate)
                    if isinstance(parsed, dict) and "answer" in parsed:
                        log.info("Successfully salvaged truncated JSON using precise bracket subset rule.")
                        return parsed

        return None

    def _extract_balanced_json(self, text: str) -> Optional[str]:
        start = text.find("{")
        while start != -1:
            depth = 0
            in_string = False
            escape = False
            for idx in range(start, len(text)):
                char = text[idx]
                if in_string:
                    if escape:
                        escape = False
                    elif char == "\\":
                        escape = True
                    elif char == '"':
                        in_string = False
                    continue
                if char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        block = text[start : idx + 1]
                        if '"answer"' in block or "'answer'" in block:
                            return block
                        break
            start = text.find("{", start + 1)
        return None

    def _parse_json_like(self, text: str) -> Optional[Any]:
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(text)
            except (ValueError, SyntaxError, TypeError):
                continue
        return None

    def _validate_answer(
        self,
        answer_raw: Any,
        *,
        expected_shape: tuple[int, int],
    ) -> Optional[list[list[int]]]:
        if not isinstance(answer_raw, list):
            log.warning("Rejected LLM answer because it is not a 2D list.")
            return None

        answer: list[list[int]] = []
        for row in answer_raw:
            if not isinstance(row, list):
                log.warning("Rejected LLM answer because a row was not a list.")
                return None

            parsed_row: list[int] = []
            for cell in row:
                parsed_cell = self._parse_cell_value(cell)
                if parsed_cell is None or not 0 <= parsed_cell <= 9:
                    log.warning("Rejected LLM answer because a cell was outside the 0-9 range.")
                    return None
                parsed_row.append(parsed_cell)
            answer.append(parsed_row)

        actual_shape = grid_shape(answer)
        if actual_shape != expected_shape:
            log.warning(
                "Rejected LLM answer due to shape mismatch: expected %s, got %s.",
                expected_shape,
                actual_shape,
            )
            return None

        if any(len(row) != expected_shape[1] for row in answer):
            log.warning(
                "Rejected LLM answer due to ragged rows: expected %d columns in every row.",
                expected_shape[1],
            )
            return None

        return answer

    def _clamp_to_shape(
        self,
        answer_raw: Any,
        expected_shape: tuple[int, int],
    ) -> Optional[list[list[int]]]:
        """Best-effort shape rescue: crop / pad a near-miss grid to exact dimensions.

        Only operates on 2D lists whose cells all parse to 0-9 integers.
        Pads with the modal cell value (most common colour) from the partial grid.
        Returns None if the grid is fundamentally unparseable.
        """
        if not isinstance(answer_raw, list):
            return None

        H, W = expected_shape

        # Validate and flatten all cells first
        parsed_rows: list[list[int]] = []
        all_values: list[int] = []
        for row in answer_raw:
            if not isinstance(row, list):
                return None
            parsed_row: list[int] = []
            for cell in row:
                val = self._parse_cell_value(cell)
                if val is None or not 0 <= val <= 9:
                    return None
                parsed_row.append(val)
                all_values.append(val)
            parsed_rows.append(parsed_row)

        if not parsed_rows:
            return None

        # Choose pad value: modal colour in the partial grid
        from collections import Counter
        pad_value = Counter(all_values).most_common(1)[0][0] if all_values else 0

        # Clamp / pad rows
        clamped: list[list[int]] = []
        for row in parsed_rows[:H]:
            # Crop or pad columns
            if len(row) >= W:
                clamped.append(row[:W])
            else:
                clamped.append(row + [pad_value] * (W - len(row)))

        # Pad missing rows with a full pad-value row
        while len(clamped) < H:
            clamped.append([pad_value] * W)

        # Final shape sanity check
        if len(clamped) == H and all(len(r) == W for r in clamped):
            log.debug("_clamp_to_shape: produced %dx%d grid (pad_value=%d).", H, W, pad_value)
            return clamped

        return None

    def _parse_cell_value(self, value: Any) -> Optional[int]:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)

        text = str(value).strip()
        match = re.search(r"-?\d+", text)
        if match is None:
            return None
        return int(match.group(0))

    def _contains_reasoning_text(
        self,
        answer_raw: Any,
        reasoning_steps: list[str],
    ) -> bool:
        flattened = self._flatten_answer_payload(answer_raw).lower()
        if "<think>" in flattened or "</think>" in flattened:
            return True
            
        # Hard red flag: alphabet characters inside the grid payload indicates bleed
        if re.search(r'[a-z]', flattened):
            log.warning("Alphabet bleed detected in grid payload: %s", flattened[:100])
            return True
            
        for step in reasoning_steps:
            normalised_step = " ".join(step.lower().split())
            if len(normalised_step) >= 8 and normalised_step in flattened:
                return True
        return False

    def _flatten_answer_payload(self, value: Any) -> str:
        if isinstance(value, list):
            return " ".join(self._flatten_answer_payload(item) for item in value)
        if value is None:
            return ""
        return str(value)

    def _parse_confidence(self, value: Any) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, round(confidence, 4)))


class OpenAIGenerator(LLMGenerator):
    """Bridge for remote Cloud API servers."""

    name: str = "openai"

    def __init__(
        self,
        model: str,
        api_base: str = "http://localhost:8000/v1",
        api_key: str = "sk-no-key-required",
        max_tokens: int = 4096,
        temperature: float = 0.6,
        top_p: float = 0.95,
        reasoning_parser: str = "deepseek",
        system_prompt: Optional[str] = None,
        prompt_style: str = "standard",
        include_think: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model,
            temperature=temperature,
            max_new_tokens=max_tokens,
            top_p=top_p,
            system_prompt=system_prompt,
            prompt_style=prompt_style,
            include_think=include_think,
            **kwargs
        )
        self.model_name = model
        self.api_base = api_base
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.reasoning_parser = reasoning_parser
        self._client: Optional[Any] = None

    def _load_pipeline(self) -> Any:
        # Lazy load openai client to avoid dependency on dev pods
        from openai import OpenAI

        if self._client is None:
            self._client = OpenAI(base_url=self.api_base, api_key=self.api_key)
        return self._client

    def _complete(
        self, 
        prompt: str, 
        schema: dict, 
        on_token: Optional[Callable[[str, str], None]] = None
    ) -> tuple[str, str]:
        client = self._load_pipeline()
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        max_retries = 5
        backoff = 2.0
        
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    stream=False 
                )
                return response.choices[0].message.content or "", response.choices[0].finish_reason
            except Exception as e:
                if attempt == max_retries - 1:
                    log.error("API generation failed after %d attempts: %s", max_retries, e)
                    raise
                
                wait_time = backoff ** (attempt + 1)
                log.warning("API connection failed (attempt %d/%d). Retrying in %.1fs... Error: %s", 
                            attempt + 1, max_retries, wait_time, e)
                time.sleep(wait_time)
        
        return "", "error"


class LLMWarmGenerator(LLMGenerator):
    """Generates an LLM trace with a higher temperature for diversity."""
    name = "llm_warm"

    def __init__(self, seed: int = 42, **kwargs: Any) -> None:
        kwargs["temperature"] = kwargs.get("temperature", 0.7)
        super().__init__(seed=seed, **kwargs)


class LLMConciseGenerator(LLMGenerator):
    """Generates a concise answer trace, demanding zero reasoning."""
    name = "llm_concise"

    def _build_math_prompt(
        self, task: Task,
        failure_constraints: Optional[FailureConstraints] = None,
    ) -> tuple[str, dict]:
        constraint_block = self._build_constraint_block(failure_constraints)
        main_prompt = (
            "Solve the following math word problem. DO NOT EXPLAIN. DO NOT SHOW REASONING.\n"
            "Return ONLY a JSON object containing your final numerical answer.\n"
            "Schema: {\"answer\": \"123.45\"}\n"
            "You may return it as a string, integer, or float, but it must be under the key 'answer'."
        )
        prompt = (
            f"{constraint_block}"
            f"{main_prompt}\n\n"
            f"Question:\n{task.test_input}\n\n"
            "JSON answer:"
        )
        schema = {
            "type": "object",
            "properties": {
                "answer": {
                    "type": ["string", "number", "integer"]
                }
            },
            "required": ["answer"],
            "additionalProperties": False
        }
        return prompt, schema


class LLMVerifyGenerator(LLMGenerator):
    """Generates a verified answer trace with an explicit draft/check/revise contract."""

    name = "llm_verify"

    def __init__(self, seed: int = 42, **kwargs: Any) -> None:
        kwargs["temperature"] = kwargs.get("temperature", 0.0)
        super().__init__(seed=seed, **kwargs)

    def _build_math_prompt(
        self, task: Task,
        failure_constraints: Optional[FailureConstraints] = None,
    ) -> tuple[str, dict]:
        constraint_block = self._build_constraint_block(failure_constraints)
        main_prompt = (
            "Solve the following math word problem.\n"
            "Use this exact workflow:\n"
            "1. Compute a draft answer.\n"
            "2. Verify the arithmetic and units with a second pass.\n"
            "3. If the verification disagrees, revise the draft before finalising.\n"
            "4. After verification, emit ONE JSON markdown block with the final answer.\n"
            '{"answer": 123}\n'
            "Do not emit multiple candidate answers. The JSON block must be the final thing in the response."
        )
        prompt = (
            f"{constraint_block}"
            f"{main_prompt}\n\n"
            f"Question:\n{task.test_input}\n\n"
            "Verified reasoning and JSON output:"
        )
        return prompt, {}


class LLMSelfCheckGenerator(LLMGenerator):
    """Runs one draft answer followed by one lightweight verification pass."""

    name = "llm_selfcheck"

    def __init__(self, seed: int = 42, **kwargs: Any) -> None:
        kwargs["temperature"] = kwargs.get("temperature", 0.1)
        super().__init__(seed=seed, **kwargs)

    def generate(
        self,
        task: Task,
        on_token: Optional[Callable[[str, str], None]] = None,
        failure_constraints: Optional[FailureConstraints] = None,
    ) -> CandidateTrace:
        from epistemic_tribunal.tribunal_types import TaskDomain

        if task.domain != TaskDomain.GSM8K_MATH:
            return super().generate(task, on_token=on_token)

        draft_prompt, draft_schema = LLMGenerator._build_math_prompt(self, task)
        draft_response, draft_finish = self._complete(draft_prompt, draft_schema, on_token=None)
        if draft_finish == "length":
            raise ValueError(f"[length] LLM self-check draft hit max_tokens={self.max_new_tokens}")

        draft_payload = self._extract_payload(draft_response, finish_reason=draft_finish)
        draft_answer = draft_payload.get("answer") if isinstance(draft_payload, dict) else None

        verify_prompt = (
            "Re-check the proposed answer to the math word problem.\n"
            "If the proposed answer is wrong, correct it.\n"
            "Return ONLY a JSON object with the final numerical answer.\n"
            'Schema: {"answer": 123}\n\n'
            f"Question:\n{task.test_input}\n\n"
            f"Proposed answer to verify: {draft_answer}\n\n"
            "Final JSON answer:"
        )
        verify_schema = {
            "type": "object",
            "properties": {"answer": {"type": ["string", "number", "integer"]}},
            "required": ["answer"],
            "additionalProperties": False,
        }
        final_response, final_finish = self._complete(verify_prompt, verify_schema, on_token=on_token)
        if final_finish == "length":
            raise ValueError(f"[length] LLM self-check verify hit max_tokens={self.max_new_tokens}")

        reasoning_steps = self._extract_reasoning_steps(draft_response) + self._extract_reasoning_steps(final_response)
        payload = self._extract_payload(final_response, finish_reason=final_finish)
        if payload is None or "answer" not in payload:
            raise ValueError("[json_not_found] LLM self-check verify response contained no parsable answer JSON.")

        answer = payload.get("answer")
        return CandidateTrace(
            generator_name=self.name,
            answer=answer,
            reasoning_steps=reasoning_steps,
            raw_trace=f"{draft_response}\n\n<self_check>\n{final_response}\n</self_check>",
            token_count=len((draft_response + ' ' + final_response).split()),
            confidence_score=1.0,
            derived_features={
                "model_name": self.model_name,
                "self_check": True,
                "draft_finish_reason": draft_finish,
                "verify_finish_reason": final_finish,
            },
            metadata={
                "model": self.model_name,
                "finish_reason": final_finish,
                "draft_answer": draft_answer,
                "draft_response": draft_response,
                "verify_response": final_response,
            },
        )
