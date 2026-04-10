# Roadmap

This repository houses the **Sovereign Epistemic Agent** project. The **Epistemic Tribunal** is the first implemented module inside that broader effort. The terms below name different layers of the same direction, not interchangeable labels for the current codebase.

## What exists now

### Sovereign Epistemic Agent

At present, this is the **project identity and research direction**, not a fully realised runtime architecture. The repository contains one concrete subsystem that embodies part of that doctrine.

### Epistemic Tribunal

This is the **implemented experimental adjudication stack**. It runs competing generator strategies, extracts invariants, critiques traces, measures disagreement, selects or abstains, and writes structured outcomes to a SQLite ledger.

### Failure ledger

The current ledger is **real and operational**. It stores traces, decisions, invariant violations, and failure records. Its present role is diagnostic, evaluative, and penalty-oriented.

## What is experimental but not yet implemented

### Strange Loop writable memory

This is the next stronger memory concept beyond the current ledger. Unlike the existing SQLite store, it would be queryable during reasoning and writable as part of live deliberation. It is not implemented in the current stack.

### Optional persistence and future extensions

The current project assumes persistent storage is useful, but stronger persistence models remain open:

- alternative ledger backends
- richer retrieval over past failures
- memory components that affect generation, critique, or resampling in real time
- optional deployment modes where persistence can be reduced, swapped, or disabled

These are extensions to the current tribunal environment, not present features.

## What is doctrinal guidance for future work

### Gödlø / operator-mind direction

This names a longer-range line of work around co-agency, operator participation, and reasoning systems that can sustain structured internal opposition without collapsing into a single brittle stance too early. It is a **conceptual direction**, not an implemented module in this repository.

## Relationship summary

- **Sovereign Epistemic Agent** is the umbrella project.
- **Epistemic Tribunal** is the first implemented experimental module.
- The **failure ledger** is the current memory bridge: persistent, structured, and post-hoc.
- **Strange Loop writable memory** is a future architecture for memory during reasoning.
- **Gödlø / operator-mind** is the broader research direction shaping what later modules may become.

The immediate purpose of the repository is therefore clear: use the Epistemic Tribunal as a disciplined test environment for adjudication, disagreement, and failure reuse while keeping a strict distinction between present implementation and future doctrine.
