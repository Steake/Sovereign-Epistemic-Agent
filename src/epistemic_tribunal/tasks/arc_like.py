"""ARC-like task loading and validation utilities.

Tasks are stored as JSON files matching the ARC challenge format extended
with optional fields (description, ground_truth).

JSON schema::

    {
        "task_id": "unique_string",          // optional; inferred from filename
        "description": "...",                 // optional
        "train": [
            {"input": [[...]], "output": [[...]]},
            ...
        ],
        "test": [
            {"input": [[...]]}               // ARC standard: list of test inputs
        ],
        "ground_truth": [[...]]              // optional; expected output for test[0]
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from epistemic_tribunal.tribunal_types import GridExample, Task, TaskDomain
from epistemic_tribunal.utils.logging import get_logger

log = get_logger(__name__)


def load_task_from_file(path: Path | str) -> Task:
    """Load a :class:`Task` from a JSON file.

    Supports both the canonical ARC format and the extended format used by
    the Epistemic Tribunal benchmark.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Task file not found: {path}")

    with open(path) as fh:
        data: dict[str, Any] = json.load(fh)

    return _parse_task_dict(data, source_path=path)


def load_task_from_dict(data: dict[str, Any]) -> Task:
    """Load a :class:`Task` from a raw dictionary."""
    return _parse_task_dict(data)


def _parse_task_dict(data: dict[str, Any], source_path: Optional[Path] = None) -> Task:
    """Parse a raw dict into a :class:`Task` instance."""
    task_id: str = data.get("task_id") or (
        source_path.stem if source_path else "unknown"
    )
    description: str = data.get("description", "")

    # Train examples
    train_raw = data.get("train", [])
    train: list[GridExample] = [
        GridExample(input=ex["input"], output=ex["output"]) for ex in train_raw
    ]

    # Test input — support both {"test": [{"input": ...}]} and {"test_input": ...}
    test_input: list[list[int]]
    if "test_input" in data:
        test_input = data["test_input"]
    elif "test" in data and data["test"]:
        test_input = data["test"][0]["input"]
    else:
        raise ValueError(f"Task {task_id!r} has no test input")

    # Ground truth — support standard, test-nested, or 'golden' manifest formats
    ground_truth: Optional[list[list[int]]] = data.get("ground_truth")
    if ground_truth is None:
        if "test" in data and data["test"] and "output" in data["test"][0]:
            ground_truth = data["test"][0]["output"]
        elif "golden" in data and "expected_test_outputs" in data["golden"]:
            ground_truth = data["golden"]["expected_test_outputs"][0]

    metadata: dict[str, Any] = data.get("metadata", {})

    return Task(
        task_id=task_id,
        domain=TaskDomain.ARC_LIKE,
        description=description,
        train=train,
        test_input=test_input,
        ground_truth=ground_truth,
        metadata=metadata,
    )


def task_summary(task: Task) -> str:
    """Return a short human-readable summary of a task."""
    rows, cols = len(task.test_input), len(task.test_input[0]) if task.test_input else 0
    return (
        f"Task {task.task_id!r} | "
        f"domain={task.domain.value} | "
        f"train_examples={len(task.train)} | "
        f"test_grid={rows}x{cols}"
    )
