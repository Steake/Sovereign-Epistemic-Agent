"""Shared pytest fixtures and helpers for the Epistemic Tribunal test suite."""

from __future__ import annotations

import pytest

from epistemic_tribunal.config import TribunalSettings
from epistemic_tribunal.ledger.store import LedgerStore
from epistemic_tribunal.types import GridExample, Task, TaskDomain


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_task() -> Task:
    """A minimal 3x3 colour-swap task with ground truth."""
    return Task(
        task_id="test_task_001",
        domain=TaskDomain.ARC_LIKE,
        description="Swap colours 1 and 2",
        train=[
            GridExample(
                input=[[1, 2, 0], [0, 1, 2], [2, 0, 1]],
                output=[[2, 1, 0], [0, 2, 1], [1, 0, 2]],
            ),
            GridExample(
                input=[[1, 1, 0], [2, 0, 2], [0, 1, 2]],
                output=[[2, 2, 0], [1, 0, 1], [0, 2, 1]],
            ),
        ],
        test_input=[[1, 0, 2], [2, 1, 0], [0, 2, 1]],
        ground_truth=[[2, 0, 1], [1, 2, 0], [0, 1, 2]],
    )


@pytest.fixture()
def identity_task() -> Task:
    """A task where the output is identical to the input."""
    return Task(
        task_id="test_task_identity",
        domain=TaskDomain.ARC_LIKE,
        description="Identity transform",
        train=[
            GridExample(
                input=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                output=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            ),
        ],
        test_input=[[2, 0, 2], [0, 2, 0], [2, 0, 2]],
        ground_truth=[[2, 0, 2], [0, 2, 0], [2, 0, 2]],
    )


@pytest.fixture()
def in_memory_store() -> LedgerStore:
    """A fresh in-memory SQLite ledger store."""
    return LedgerStore(":memory:")


@pytest.fixture()
def default_config() -> TribunalSettings:
    """Default application settings."""
    return TribunalSettings()
