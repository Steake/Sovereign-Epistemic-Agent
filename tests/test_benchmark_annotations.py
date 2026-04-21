"""Tests for benchmark_annotations loaders.

Covers:
- Valid JSON array loads cleanly
- Valid JSONL loads cleanly
- Duplicate task_id raises ValueError
- Invalid cohort raises ValueError
- Invalid CI range raises ValueError
- Oracle metadata loads cleanly
- Oracle duplicate task_id raises ValueError
- Missing file raises FileNotFoundError
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from epistemic_tribunal.evaluation.benchmark_annotations import (
    load_annotations,
    load_oracle_metadata,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ann(task_id: str, **overrides) -> dict:
    base = {
        "task_id": task_id,
        "cohort": "contested_recoverable",
        "contestability_index": 2,
        "recoverability_index": 3,
        "structural_separability": 1,
        "plausible_hypotheses": ["h1"],
        "recoverability_status": "exact_candidate_present",
    }
    base.update(overrides)
    return base


def _oracle(task_id: str) -> dict:
    return {
        "task_id": task_id,
        "oracle_exact_candidate_present": True,
        "oracle_structurally_defensible_candidate_present": True,
    }


# ---------------------------------------------------------------------------
# load_annotations — JSON array
# ---------------------------------------------------------------------------


def test_load_annotations_json_array(tmp_path: Path) -> None:
    data = [_ann("t1"), _ann("t2")]
    p = tmp_path / "anns.json"
    p.write_text(json.dumps(data))

    result = load_annotations(p)
    assert set(result.keys()) == {"t1", "t2"}
    assert result["t1"].contestability_index == 2


def test_load_annotations_jsonl(tmp_path: Path) -> None:
    lines = "\n".join(json.dumps(_ann(f"t{i}")) for i in range(3))
    p = tmp_path / "anns.jsonl"
    p.write_text(lines)

    result = load_annotations(p)
    assert len(result) == 3


def test_load_annotations_empty_lines_ignored(tmp_path: Path) -> None:
    lines = "\n".join([json.dumps(_ann("t1")), "", "  ", json.dumps(_ann("t2"))])
    p = tmp_path / "anns.jsonl"
    p.write_text(lines)
    result = load_annotations(p)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# load_annotations — failures
# ---------------------------------------------------------------------------


def test_duplicate_task_id_raises(tmp_path: Path) -> None:
    data = [_ann("t1"), _ann("t1")]  # duplicate
    p = tmp_path / "dup.json"
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Duplicate task_id"):
        load_annotations(p)


def test_invalid_cohort_raises(tmp_path: Path) -> None:
    data = [_ann("t1", cohort="not_a_cohort")]
    p = tmp_path / "bad_cohort.json"
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="failed validation"):
        load_annotations(p)


def test_invalid_ci_range_raises(tmp_path: Path) -> None:
    data = [_ann("t1", contestability_index=99)]
    p = tmp_path / "bad_ci.json"
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="failed validation"):
        load_annotations(p)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_annotations(tmp_path / "nonexistent.json")


# ---------------------------------------------------------------------------
# load_oracle_metadata
# ---------------------------------------------------------------------------


def test_load_oracle_json_array(tmp_path: Path) -> None:
    data = [_oracle("t1"), _oracle("t2")]
    p = tmp_path / "oracle.json"
    p.write_text(json.dumps(data))
    result = load_oracle_metadata(p)
    assert "t1" in result
    assert result["t1"].oracle_exact_candidate_present is True


def test_oracle_duplicate_raises(tmp_path: Path) -> None:
    data = [_oracle("t1"), _oracle("t1")]
    p = tmp_path / "dup_oracle.json"
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Duplicate task_id"):
        load_oracle_metadata(p)


def test_oracle_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_oracle_metadata(Path("/nonexistent/oracle.json"))
