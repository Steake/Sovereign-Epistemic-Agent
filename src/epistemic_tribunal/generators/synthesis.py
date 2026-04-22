from __future__ import annotations

import re
from typing import Any, Optional, Callable

from epistemic_tribunal.failure_memory.models import FailureConstraints
from epistemic_tribunal.generators.llm import LLMGenerator
from epistemic_tribunal.tribunal_types import CandidateTrace, Task
from epistemic_tribunal.utils.execution import execute_transformation, get_sandbox_docs
from epistemic_tribunal.utils.logging import get_logger
from epistemic_tribunal.tasks.base import grids_equal

log = get_logger(__name__)

class ProgramSynthesisGenerator(LLMGenerator):
    """Generate a candidate by synthesising and executing a transformation program."""

    name = "synthesis"

    def __init__(self, **kwargs: Any) -> None:
        # Default to include think blocks as reasoning for synthesis is critical
        kwargs.setdefault("include_think", True)
        super().__init__(**kwargs)

    def generate(
        self,
        task: Task,
        on_token: Optional[Callable[[str, str], None]] = None,
        failure_constraints: Optional[FailureConstraints] = None,
    ) -> CandidateTrace:
        """Synthesise a program, verify against train, and execute on test."""
        prompt, _ = self._build_synthesis_prompt(task)
        
        # 1. Generate code via LLM
        response, finish_reason = self._complete(prompt, schema={}, on_token=on_token)
        
        # 2. Extract code block
        code = self._extract_code(response)
        if not code:
            raise ValueError("[code_not_found] LLM failed to provide a ```python block.")

        # 3. Internal Validation (Training Examples)
        failures = []
        train_results = []
        for idx, example in enumerate(task.train):
            try:
                predicted = execute_transformation(code, example.input)
                is_correct = grids_equal(predicted, example.output)
                train_results.append(is_correct)
                if not is_correct:
                    failures.append({
                        "id": idx,
                        "input": example.input,
                        "expected": example.output,
                        "actual": predicted
                    })
            except Exception as e:
                log.warning("Synthesis failed on train example %d: %s", idx, e)
                train_results.append(False)
                failures.append({
                    "id": idx,
                    "input": example.input,
                    "expected": example.output,
                    "error": str(e)
                })

        # 4. Attempt Refinement if any train example failed
        if failures:
            log.info("Synthesis failed training verification on %d examples. Refining...", len(failures))
            code = self._refine_code(task, code, failures, on_token=on_token)

        # 5. Final Execution on Test Input
        try:
            answer = execute_transformation(code, task.test_input)
        except Exception as e:
            raise RuntimeError(f"[execution_failure] Synthesised code crashed on test input: {e}")

        # 6. Build Trace
        # We store the synthesized code in reasoning_steps for the dashboard
        return CandidateTrace(
            generator_name=self.name,
            answer=answer,
            reasoning_steps=[f"### Synthesised Code\n```python\n{code}\n```"],
            raw_trace=response,
            token_count=len(response.split()),
            confidence_score=1.0 if all(train_results) else 0.5,
            metadata={
                "code": code,
                "train_passed_count": sum(train_results),
                "train_total": len(task.train),
                "finish_reason": finish_reason
            }
        )

    def _build_synthesis_prompt(self, task: Task) -> tuple[str, dict]:
        """Specific prompter for code synthesis."""
        train_blocks = []
        for i, ex in enumerate(task.train):
            inp_vis = self._format_grid(ex.input)
            out_vis = self._format_grid(ex.output)
            train_blocks.append(
                f"Example {i}:\n"
                f"Input:\n{inp_vis}\n"
                f"Output:\n{out_vis}"
            )
        
        prompt = (
            "You are an expert ARC (Abstraction and Reasoning Corpus) solver. "
            "Write a Python function `solve(input_grid)` that transforms the input to the output.\n\n"
            "STRATEGY:\n"
            "1. Analyze the examples and describe the logic in words first.\n"
            "2. Identify objects, colors, and spatial transformations (rotation, reflection, repetition).\n"
            "3. Synthesize the final Python function.\n\n"
            "RULES:\n"
            "- Use `list[list[int]]` for grids.\n"
            "- Output ONLY the code inside a ```python block.\n"
            "- Your function MUST pass all training examples.\n\n"
            "Common Utilities Available in Sandbox:\n"
            f"{get_sandbox_docs()}\n\n"
            f"Task {task.task_id}:\n"
            "\n".join(train_blocks) + "\n\n"
            f"Test Input Query:\n{self._format_grid(task.test_input)}\n"
            "Synthesize the solution function:"
        )
        return prompt, {}

    def _format_grid(self, grid: list[list[int]]) -> str:
        """Create a compact string representation for the LLM."""
        rows = []
        for r in grid:
            rows.append(" ".join(f"{c}" for c in r))
        return "\n".join(rows)

    def _extract_code(self, response: str) -> Optional[str]:
        md_match = re.search(r"```python\s*(.*?)\s*```", response, flags=re.DOTALL | re.IGNORECASE)
        if md_match:
            return md_match.group(1).strip()
        return None

    def _refine_code(self, task: Task, old_code: str, failures: list[dict], on_token: Optional[Callable[[str, str], None]] = None) -> str:
        """Attempt to fix code based on specific training failures."""
        reports = []
        for f in failures:
            if "error" in f:
                reports.append(f"Example {f['id']}: Runtime Error: {f['error']}")
            else:
                reports.append(
                    f"Example {f['id']}: Incorrect Output.\n"
                    f" Expected:\n{self._format_grid(f['expected'])}\n"
                    f" Actual:\n{self._format_grid(f['actual'])}"
                )

        prompt = (
            "The following code failed verification:\n"
            "```python\n" + old_code + "\n```\n\n"
            "FEEDBACK:\n"
            + "\n".join(reports) + "\n\n"
            "Instructions:\n"
            "1. Analyze the gap between Expected and Actual results.\n"
            "2. Correct the transformation logic.\n"
            "3. Provide the full corrected function in a ```python block."
        )
        new_response, _ = self._complete(prompt, schema={}, on_token=on_token)
        new_code = self._extract_code(new_response)
        return new_code if new_code else old_code
