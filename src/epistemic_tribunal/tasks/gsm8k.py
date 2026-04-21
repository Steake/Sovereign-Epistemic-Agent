from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional
import uuid

from epistemic_tribunal.tribunal_types import Task, TaskDomain
from epistemic_tribunal.utils.logging import get_logger

log = get_logger(__name__)


def extract_math_answer(text: str) -> Optional[str]:
    """Extract the final numerical answer from a GSM8K target string.
    
    GSM8K target strings typically end with '#### <answer>'.
    """
    if "####" in text:
        return text.split("####")[-1].strip()
    return None


def load_task_from_dict(data: dict[str, Any], source_path: Optional[Path] = None) -> Task:
    """Parse a GSM8K json dict into a :class:`Task` instance."""
    # GSM8K format: {"question": "...", "answer": "..."}
    
    question = data.get("question")
    if not question:
        raise ValueError("GSM8K task missing 'question' field")
        
    raw_answer = data.get("answer")
    ground_truth = None
    if raw_answer:
        extracted = extract_math_answer(raw_answer)
        if extracted is not None:
            ground_truth = extracted
        else:
            ground_truth = raw_answer

    task_id = data.get("task_id")
    if not task_id:
        task_id = str(uuid.uuid4())[:8]

    metadata = {"raw_answer": raw_answer} if raw_answer else {}
    for key in ["cohort", "contestability_index", "recoverability_index", "structural_separability", "plausible_hypotheses", "recoverability_status"]:
        if key in data:
            metadata[key] = data[key]

    return Task(
        task_id=task_id,
        domain=TaskDomain.GSM8K_MATH,
        description="GSM8K Math Word Problem",
        train=[],  # GSM8K is typically zero-shot or few-shot in the prompt, not provided as grid pairs
        test_input=question,
        ground_truth=ground_truth,
        metadata=metadata
    )


def load_tasks_from_jsonl(path: Path | str) -> list[Task]:
    """Load multiple GSM8K tasks from a JSONL file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Task file not found: {path}")

    tasks = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            # Assign a task ID if missing
            if "task_id" not in data:
                data["task_id"] = f"{path.stem}_{i:04d}"
            tasks.append(load_task_from_dict(data, source_path=path))
    return tasks
