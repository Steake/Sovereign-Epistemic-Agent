import ast
import json
import multiprocessing
import re
from typing import Any, Optional, Callable

from epistemic_tribunal.generators.llm import LLMGenerator
from epistemic_tribunal.tribunal_types import CandidateTrace, Task
from epistemic_tribunal.tasks.base import grid_shape
from epistemic_tribunal.utils.logging import get_logger

log = get_logger(__name__)

def _sandbox_runner(code_str: str, input_grid: list[list[int]], result_queue: multiprocessing.Queue):
    """Executes the untrusted code in a minimal sandbox and puts the result in the queue."""
    # Restricted environment
    safe_globals = {
        "__builtins__": {
            "range": range, "len": len, "enumerate": enumerate, 
            "int": int, "list": list, "tuple": tuple, "set": set, "dict": dict,
            "max": max, "min": min, "sum": sum, "abs": abs,
            "bool": bool, "all": all, "any": any
        }
    }
    safe_locals = {}
    
    try:
        exec(code_str, safe_globals, safe_locals)
        if "transform" not in safe_locals:
            result_queue.put(("error", "[codegen_invalid] Python code did not define a 'transform' function."))
            return
            
        transform_func = safe_locals["transform"]
        output_grid = transform_func(input_grid)
        result_queue.put(("success", output_grid))
    except Exception as e:
        result_queue.put(("error", f"[codegen_exec_error] Execution failed: {str(e)}"))


class CodeGenLLMGenerator(LLMGenerator):
    """Generate an ARC grid computationally by writing and executing Python code.
    
    Forces the LLM to think in absolute logical algorithms rather than visual 
    pattern completion. The LLM writes `def transform(grid):`, which we isolate,
    compile, and execute against the task instances to yield the output grid.
    """
    
    name = "llm_codegen"

    def _build_prompt(self, task: Task, expected_shape: tuple[int, int]) -> tuple[str, dict]:
        H, W = expected_shape
        train_examples = []
        for idx, example in enumerate(task.train, start=1):
            train_examples.append(
                f"Example {idx}\n"
                f"Input: {json.dumps(example.input)}\n"
                f"Output: {json.dumps(example.output)}"
            )

        template_row = ["?"] * W
        template_grid = [template_row for _ in range(H)]
        template_str = json.dumps(template_grid)

        main_prompt = (
            f"=== SHAPE REQUIREMENT (HARD CONSTRAINT) ===\n"
            f"The final grid returned by your function MUST be EXACTLY {H} rows × {W} columns.\n"
            f"Required shape template (replace ? with integers 0-9):\n"
            f"{template_str}\n\n"
            "=== RESPONSE FORMAT ===\n"
            "You are an expert abstract algorithmic Python coder.\n"
            "You MUST write a Python function named `transform` that takes a 2D integer list `grid` and returns the transformed 2D integer list.\n"
            "1. Enclose your Python code in a ```python ... ``` markdown block.\n"
            "2. Ensure the code compiles cleanly and uses no external libraries (only standard python builtins like lists, loops, etc).\n"
            "3. The returned value MUST be a nested list of integers (0-9).\n"
            "4. Reason about the geometric constraints before writing the code, using <think>...</think> tags if you wish.\n"
        )

        train_block = "\n".join(train_examples) if train_examples else "None"
        prompt = (
            f"{main_prompt}\n\n"
            f"Task: {task.task_id}\n"
            f"Train:\n{train_block}\n\n"
            f"Test Input ({len(task.test_input)}×{len(task.test_input[0]) if task.test_input else 0} grid):\n"
            f"{json.dumps(task.test_input)}\n\n"
            f"Provide your Python code:"
        )

        return prompt, {}

    def _extract_payload(self, response: str, finish_reason: str = "stop") -> Optional[dict[str, Any]]:
        # Redefine payload extraction to pull out the python code, instead of JSON
        # Then, execute it and serialize the answer back into the expectation format.
        
        # 1. Extract Python code
        code_match = re.search(r"```python\s*(.*?)\s*```", response, flags=re.DOTALL | re.IGNORECASE)
        if not code_match:
            # Fallback block search if they forgot the 'python' tag
            code_match = re.search(r"```\s*(def transform.*?)\s*```", response, flags=re.DOTALL | re.IGNORECASE)
            
        if not code_match:
            log.warning("No python code block found in LLM response.")
            return None
            
        code_str = code_match.group(1).strip()
        log.info("Extracted Python code payload (%d chars).", len(code_str))
        
        # Note: at this phase we don't have the task input natively available in extract_payload 
        # unless we pass it. To cleanly integrate into LLMGenerator's flow, we'll override generate() directly.
        pass

    def generate(
        self, 
        task: Task, 
        on_token: Optional[Callable[[str, str], None]] = None
    ) -> CandidateTrace:
        expected_shape = grid_shape(task.test_input)
        prompt, schema = self._build_prompt(task, expected_shape)
        
        response, finish_reason = self._complete(prompt, schema, on_token=on_token)
        self.last_finish_reason = finish_reason
        
        if finish_reason == "length":
            raise ValueError(f"[length] LLM hit max_tokens={self.max_new_tokens}")

        reasoning_steps = self._extract_reasoning_steps(response)
        
        # Extract the python code
        code_match = re.search(r"```python\s*(.*?)\s*```", response, flags=re.DOTALL | re.IGNORECASE)
        if not code_match:
            code_match = re.search(r"```\s*(def transform.*?)\s*```", response, flags=re.DOTALL | re.IGNORECASE)
        
        if not code_match:
            raise ValueError("[codegen_missing] LLM response contained no Python markdown block.")
            
        code_str = code_match.group(1).strip()
        
        # Sandbox execution
        answer_raw = self._execute_sandbox(code_str, task.test_input)
        
        answer = self._validate_answer(answer_raw, expected_shape=expected_shape)
        if answer is None:
            # Try shape-clamp rescue
            clamped = self._clamp_to_shape(answer_raw, expected_shape)
            if clamped is not None:
                log.info(
                    "Shape-clamp rescue succeeded for codegen task %s: clamped to %dx%d.",
                    task.task_id, *expected_shape
                )
                answer = clamped
            else:
                raise ValueError(
                    f"[grid_shape_invalid] Generated Python produced a grid that did not match "
                    f"exact {expected_shape} dimensionality, or contained invalid tokens."
                )

        return CandidateTrace(
            generator_name=self.name,
            answer=answer,
            reasoning_steps=reasoning_steps,
            raw_trace=response,
            token_count=len(response.split()),
            confidence_score=1.0,
            derived_features={
                "model_name": self.model_name,
                "expected_shape": expected_shape,
                "finish_reason": finish_reason,
            },
            metadata={
                "model": self.model_name,
                "finish_reason": finish_reason,
                "raw_response": response,
                "code": code_str
            }
        )
        
    def _execute_sandbox(self, code_str: str, input_grid: list[list[int]]) -> Any:
        queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=_sandbox_runner, 
            args=(code_str, input_grid, queue)
        )
        process.start()
        
        # 5 second timeout to prevent infinite loops (e.g. while True:)
        process.join(timeout=5.0)
        
        if process.is_alive():
            process.terminate()
            process.join()
            raise ValueError("[codegen_timeout] Python execution exceeded 5 second timeout.")
            
        if not queue.empty():
            status, result = queue.get()
            if status == "error":
                raise ValueError(result)
            return result
        else:
            raise ValueError("[codegen_crash] Python execution crashed silently without return.")
