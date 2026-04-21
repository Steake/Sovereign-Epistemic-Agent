"""Annotation and oracle metadata loaders for the Tribunal Usefulness Benchmark.

Supports JSON array and JSONL formats.  Validation is strict:
- duplicate task_id raises ValueError
- invalid field values raise ValidationError propagated as ValueError
- unknown cohort or status values raise ValueError

Both loaders return plain dicts keyed by task_id for O(1) lookup.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from pydantic import ValidationError

from epistemic_tribunal.evaluation.benchmark_spec import (
    TaskBenchmarkAnnotation,
    TaskOracleMetadata,
)


def _parse_records(raw: str) -> list[dict]:
    """Parse a raw string as either a JSON array or JSONL (one object per line)."""
    stripped = raw.strip()
    if stripped.startswith("["):
        return json.loads(stripped)
    # JSONL
    records = []
    for lineno, line in enumerate(stripped.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSONL parse error at line {lineno}: {exc}") from exc
    return records


def load_annotations(
    path: Union[str, Path],
) -> dict[str, TaskBenchmarkAnnotation]:
    """Load task benchmark annotations from a JSON array or JSONL file.

    Parameters
    ----------
    path:
        Path to the annotation file.

    Returns
    -------
    dict[str, TaskBenchmarkAnnotation]
        Mapping from task_id to its annotation.

    Raises
    ------
    ValueError
        On duplicate task_id, unknown cohort/status values, or out-of-range scores.
    FileNotFoundError
        If *path* does not exist.
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    records = _parse_records(raw)

    result: dict[str, TaskBenchmarkAnnotation] = {}
    for idx, record in enumerate(records):
        try:
            annotation = TaskBenchmarkAnnotation.model_validate(record)
        except ValidationError as exc:
            raise ValueError(
                f"Annotation record #{idx} failed validation: {exc}"
            ) from exc

        if annotation.task_id in result:
            raise ValueError(
                f"Duplicate task_id {annotation.task_id!r} in annotation file {path}"
            )
        result[annotation.task_id] = annotation

    return result


def load_oracle_metadata(
    path: Union[str, Path],
) -> dict[str, TaskOracleMetadata]:
    """Load oracle metadata from a JSON array or JSONL file.

    Parameters
    ----------
    path:
        Path to the oracle metadata file.

    Returns
    -------
    dict[str, TaskOracleMetadata]
        Mapping from task_id to its oracle record.

    Raises
    ------
    ValueError
        On duplicate task_id or schema violations.
    FileNotFoundError
        If *path* does not exist.
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    records = _parse_records(raw)

    result: dict[str, TaskOracleMetadata] = {}
    for idx, record in enumerate(records):
        try:
            meta = TaskOracleMetadata.model_validate(record)
        except ValidationError as exc:
            raise ValueError(
                f"Oracle record #{idx} failed validation: {exc}"
            ) from exc

        if meta.task_id in result:
            raise ValueError(
                f"Duplicate task_id {meta.task_id!r} in oracle file {path}"
            )
        result[meta.task_id] = meta

    return result
