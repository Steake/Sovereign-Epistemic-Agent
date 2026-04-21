"""Failure-memory layer — generic metacognitive failure-signature protocol.

This subpackage provides:

- **models** — ``FailureSignature`` (retrospective, labelled),
  ``FailureProbe`` (online, observable-only), ``FailureMatch``.
- **extractor** — post-evaluation signature extraction.
- **store** — SQLite-backed persistent storage.
- **query** — pre-adjudication lookup returning per-trace penalties.
"""

from __future__ import annotations
