"""LLM-backed generator with strict output validation for ARC-like tasks."""

from __future__ import annotations

import ast
import importlib.util
import json
import re
from typing import Any, Optional

from epistemic_tribunal.generators.base import BaseGenerator
from epistemic_tribunal.tasks.base import grid_shape
from epistemic_tribunal.types import CandidateTrace, Task
from epistemic_tribunal.utils.logging import get_logger

log = get_logger(__name__)


class LLMGenerator(BaseGenerator):
    """Generate a candidate trace with a text-generation model."""

    name = "llm"

    def __init__(
        self,
        seed: int = 42,
        model_name: str = "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2",
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
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
        self.top_p = top_p
        self.trust_remote_code = trust_remote_code
        self.device = device
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation
        self._pipeline: Any = None

    def generate(self, task: Task) -> CandidateTrace:
        response = self._complete(self._build_prompt(task))
        expected_shape = grid_shape(task.test_input)
        answer, reasoning_steps, confidence = self._parse_response(
            response, expected_shape=expected_shape
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
            },
            metadata={"model_name": self.model_name},
        )

    def _build_prompt(self, task: Task) -> str:
        train_examples = []
        for idx, example in enumerate(task.train, start=1):
            train_examples.append(
                f"Example {idx}\n"
                f"Input: {json.dumps(example.input)}\n"
                f"Output: {json.dumps(example.output)}"
            )

        return (
            "Solve the ARC-style task and return JSON only.\n"
            "Use this schema exactly:\n"
            '{"answer": [[0]], "reasoning_steps": ["short step"], "confidence": 0.0}\n'
            "Put hidden reasoning only inside <think>...</think> if you emit it at all.\n"
            "Do not include prose outside the JSON object.\n\n"
            f"Task ID: {task.task_id}\n"
            f"Description: {task.description}\n"
            f"Train:\n{chr(10).join(train_examples) if train_examples else 'None'}\n"
            f"Test Input: {json.dumps(task.test_input)}\n"
            f"Expected answer shape: {grid_shape(task.test_input)}\n"
        )

    def _complete(self, prompt: str) -> str:
        if self._pipeline is None:
            self._pipeline = self._load_pipeline()

        outputs = self._pipeline(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=self.temperature > 0.0,
            return_full_text=False,
        )
        if not outputs:
            raise ValueError("LLM pipeline returned no output.")
        return str(outputs[0]["generated_text"])

    def _load_pipeline(self) -> Any:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError(
                "transformers>=4.30 is required to use the llm generator. "
                "Install the optional dependencies with `pip install .[llm]`."
            ) from exc

        try:
            import torch  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "A supported runtime backend is required to use the llm generator. "
                "Install the optional dependencies with `pip install .[llm]` "
                "(or install `torch` manually)."
            ) from exc

        _dtype_map: dict[str, Any] = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        resolved_dtype: Any = _dtype_map.get(self.torch_dtype, self.torch_dtype)
        resolved_attn = self._resolve_attn_implementation()

        has_accelerate = importlib.util.find_spec("accelerate") is not None
        if self.device is not None:
            device_label = self.device
        elif has_accelerate:
            device_label = "auto"
        else:
            device_label = "cpu"

        pipe_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "tokenizer": self.model_name,
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": resolved_dtype,
            "model_kwargs": {"attn_implementation": resolved_attn},
        }
        if self.device is not None:
            pipe_kwargs["device"] = self.device
        elif has_accelerate:
            pipe_kwargs["device_map"] = "auto"
        else:
            log.warning(
                "accelerate is not installed; LLM pipeline will run on CPU. "
                "Install accelerate>=0.20 for automatic device placement."
            )

        log.info(
            "Loading LLM pipeline: model=%s dtype=%s attn=%s device=%s",
            self.model_name,
            self.torch_dtype,
            resolved_attn,
            device_label,
        )
        return pipeline("text-generation", **pipe_kwargs)

    def _resolve_attn_implementation(self) -> str:
        """Resolve the attention implementation to use.

        ``"auto"`` selects ``"flash_attention_2"`` when the ``flash-attn``
        package is installed (fastest on H200/B100), otherwise falls back to
        ``"sdpa"`` (PyTorch 2.0+ native scaled dot-product attention).
        """
        if self.attn_implementation != "auto":
            return self.attn_implementation
        if importlib.util.find_spec("flash_attn") is not None:
            log.info("flash-attn detected; using flash_attention_2 for maximum throughput.")
            return "flash_attention_2"
        log.info(
            "flash-attn not found; using 'sdpa' (PyTorch scaled dot-product attention). "
            "Install flash-attn>=2.0 for maximum throughput on H200/B100."
        )
        return "sdpa"

    def _parse_response(
        self,
        response: str,
        *,
        expected_shape: tuple[int, int],
    ) -> tuple[Optional[list[list[int]]], list[str], float]:
        reasoning_steps = self._extract_reasoning_steps(response)
        payload = self._extract_payload(response)
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

    def _extract_payload(self, response: str) -> Optional[dict[str, Any]]:
        candidate = self._extract_balanced_json(response)
        if candidate is None:
            return None

        parsed = self._parse_json_like(candidate)
        if isinstance(parsed, dict) and "answer" in parsed:
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
                            trailing = text[idx + 1 :]
                            trailing_clean = re.sub(
                                r"<think>.*?</think>",
                                "",
                                trailing,
                                flags=re.DOTALL | re.IGNORECASE,
                            ).strip()
                            if trailing_clean:
                                log.warning(
                                    "LLM response contained trailing prose after JSON "
                                    "object; rejecting."
                                )
                                return None
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
