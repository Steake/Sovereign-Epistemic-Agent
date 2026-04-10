"""Utility helpers: unique ID generation."""

from __future__ import annotations

import uuid


def new_id(prefix: str = "") -> str:
    """Return a new UUID4 string, optionally with a prefix."""
    uid = str(uuid.uuid4())
    return f"{prefix}{uid}" if prefix else uid


def short_id(prefix: str = "") -> str:
    """Return an 8-character hex short ID."""
    uid = uuid.uuid4().hex[:8]
    return f"{prefix}{uid}" if prefix else uid
