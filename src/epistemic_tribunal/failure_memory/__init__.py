"""Failure-memory layer — generic metacognitive failure-signature protocol.

This subpackage provides:

- **models** — ``FailureSignature`` (retrospective, labelled),
  ``FailureProbe`` (online, observable-only), ``FailureMatch``,
  ``FailureConstraints`` (Strange Loop pre-generation guidance).
- **extractor** — post-evaluation signature extraction.
- **store** — SQLite-backed persistent storage.
- **query** — pre-adjudication lookup returning per-trace penalties.
- **constraint_builder** — Strange Loop v1: pre-generation constraint
  injection derived from prior failures.
"""

from __future__ import annotations
