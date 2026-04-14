"""LLM-backed generator with strict output validation for ARC-like tasks."""

from __future__ import annotations

import ast
import importlib.util
import json
import re
from typing import Any, Optional

from epistemic_tribunal.generators.base import BaseGenerator
from epistemic_tribunal.tasks.base import grid_shape
from epistemic_tribunal.tribunal_types import CandidateTrace, Task
from epistemic_tribunal.utils.logging import get_logger

log = get_logger(__name__)

_GLOBAL_OAI_CLIENT = None

class LLMGenerator(BaseGenerator):
    """Generate a candidate trace with a text-generation model."""

    name = "llm"

    def __init__(
        self,
        seed: int = 42,
        model_name: str = "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2",
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
        **kwargs: Any,
    ) -> None:
        super().__init__(seed=seed, **kwargs)
        self.model_name = model_name
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
        self._pipeline: Any = None

    def generate(self, task: Task) -> CandidateTrace:
        response, finish_reason = self._complete(self._build_prompt(task))
        self.last_finish_reason = finish_reason
        log.info("LLM Generation complete. Length: %d characters", len(response))
        log.debug("Raw Response: %s...", response[:200])
        expected_shape = grid_shape(task.test_input)
        answer, reasoning_steps, confidence = self._parse_response(
            response, expected_shape=expected_shape, finish_reason=finish_reason
        )
        if answer is None:
            raise ValueError(
                f"LLM response did not contain a valid {expected_shape[0]}x{expected_shape[1]} grid."
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
                "model_name": self.model_name,
                "finish_reason": finish_reason,
            },
        )

    def _build_prompt(self, task: Task) -> str:
        train_examples = []
        for idx, example in enumerate(task.train, start=1):
            train_examples.append(
                f"Example {idx}\n"
                f"Input: {json.dumps(example.input)}\n"
                f"Output: {json.dumps(example.output)}"
            )

        instructions = ""
        if self.prompt_style == "rule":
            instructions = "Identify the transformation rule first, then apply it exactly."
        elif self.prompt_style == "adversarial":
            instructions = "Consider the non-obvious transformation. Only output JSON."

        main_prompt = (
            "Solve the ARC task. Output ONLY the JSON object below — no prose, no markdown, no <think> tags.\n"
            'Schema: {"answer": [[int, ...], ...], "reasoning_steps": ["step"], "confidence": 0.0}\n'
            "The answer must be a 2-D list of integers matching the expected output grid dimensions.\n"
        )

        if instructions:
            main_prompt = f"{instructions}\n\n{main_prompt}"

        train_block = "\n".join(train_examples) if train_examples else "None"
        return (
            f"{main_prompt}\n"
            f"Task: {task.task_id}\n"
            f"Train:\n{train_block}\n"
            f"Test Input: {json.dumps(task.test_input)}\n"
            "JSON answer:"
        )

    def _load_vllm_pipeline(self) -> Any:
        global _GLOBAL_OAI_CLIENT
        if _GLOBAL_OAI_CLIENT is not None:
            return _GLOBAL_OAI_CLIENT, None

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai client is required.") from exc

        import os
        base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
        log.info("Connecting to vLLM background API server for inference at %s", base_url)

        client = OpenAI(
            base_url=base_url,
            api_key="empty",
        )

        _GLOBAL_OAI_CLIENT = client
        return client, None

    def _complete(self, prompt: str) -> tuple[str, str]:
        if self._pipeline is None:
            self._pipeline, _ = self._load_vllm_pipeline()

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._pipeline.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
                top_p=self.top_p,
            )
            message = response.choices[0].message
            text = message.content or ""
            finish_reason = response.choices[0].finish_reason

            # llama-server may prefix content with '</think>' even when
            # --reasoning off is set; strip it so the JSON parser finds
            # the object cleanly.
            if text.lstrip().startswith("</think>"):
                text = text.lstrip()[len("</think>"):].lstrip()

            if finish_reason == "length":
                log.warning(
                    "LLM response truncated at max_tokens=%d — answer may be incomplete.",
                    self.max_new_tokens,
                )

            snippet = text[:200].replace("\n", " ")
            log.info(
                "LLM Generation complete. Length: %d chars. finish=%s. Snippet: %s...",
                len(text), finish_reason, snippet,
            )
            return text, finish_reason

        except Exception as e:
            log.error("vLLM API generation failed. Is the vllm server running? Error: %s", e)
            raise ValueError(f"vLLM API failure: {e}")

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
    """Bridge for OpenAI-compatible API servers (llama.cpp, vLLM, etc)."""

    name: str = "openai"

    def __init__(
        self,
        model: str,
        api_base: str = "http://localhost:8000/v1",
        api_key: str = "sk-no-key-required",
        max_tokens: int = 4096,
        temperature: float = 0.6,
        top_p: float = 0.95,
        reasoning_parser: str = "qwen3",
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

    def _complete(self, prompt: str) -> str:
        client = self._load_pipeline()
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return response.choices[0].message.content or ""
