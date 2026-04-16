import json
from typing import Optional

from epistemic_tribunal.generators.llm import LLMGenerator
from epistemic_tribunal.tribunal_types import Task

class CoTLLMGenerator(LLMGenerator):
    """Generate a candidate trace with explicit Chain-of-Thought reasoning.
    
    This generator shifts the cognitive mode by demanding that the LLM explicitly
    reasons about geometric transformations, counts, colors, and spatial logic
    *before* emitting the final JSON grid.
    """
    
    name = "llm_cot"

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

        # In CoT, we explicitly DONT ask for 'Return ONLY JSON', we mandate reasoning.
        main_prompt = (
            f"=== SHAPE REQUIREMENT (HARD CONSTRAINT) ===\n"
            f"Your answer grid MUST be EXACTLY {H} rows × {W} columns.\n"
            f"Required shape template (replace ? with integers 0-9):\n"
            f"{template_str}\n\n"
            "=== RESPONSE FORMAT ===\n"
            "1. You MUST first reason explicitly about the transformations you observe.\n"
            "Write your reasoning inside <think>...</think> XML tags.\n"
            "Analyze colors, objects, spatial movements, and scaling rules.\n\n"
            "2. After your <think> block, emit a JSON markdown block with your answer.\n"
            'Schema: {"answer": [[int, ...], ...]}\n'
            f"• Exactly {H} rows.\n"
            f"• Exactly {W} integers per row.\n"
            "• Every integer must be in range 0–9.\n"
            "• Provide ONLY ONE JSON markdown block. Do not place reasoning inside the JSON.\n"
        )

        train_block = "\n".join(train_examples) if train_examples else "None"
        prompt = (
            f"{main_prompt}\n\n"
            f"Task: {task.task_id}\n"
            f"Train:\n{train_block}\n\n"
            f"Test Input ({len(task.test_input)}×{len(task.test_input[0]) if task.test_input else 0} grid):\n"
            f"{json.dumps(task.test_input)}\n\n"
            f"Reasoning and JSON output:"
        )

        # For CoT, we don't force JSON mode on the API layer because we want the <think> text first.
        # We pass an empty schema dict which the base class respects by simply passing the prompt as-is.
        return prompt, {}
