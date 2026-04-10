# Sovereign Epistemic Agent — Roadmap

This document maps the conceptual architecture of the project against what currently exists, what is under active development, and what belongs to the longer-range research direction.

---

## Concept hierarchy

```
Sovereign Epistemic Agent          ← the broader project
└── Epistemic Tribunal             ← first concrete experimental module
    ├── Generator bank             ← heuristic scaffold (current)
    ├── Invariant extractor
    ├── Trace critic
    ├── Uncertainty analyzer
    ├── Tribunal aggregator
    └── Failure ledger             ← diagnostic memory (current)
                                      └── Strange Loop memory (future)
```

---

## What exists now

**Epistemic Tribunal** is a fully working adjudication pipeline. It is the first module of the Sovereign Epistemic Agent project and the primary experimental substrate.

| Component | Status |
|---|---|
| Generator bank (5 heuristic strategies) | Implemented |
| Invariant extractor (7 structural checks) | Implemented |
| TraceCritic (5 scoring dimensions) | Implemented |
| UncertaintyAnalyzer (entropy, margin, coalition mass, etc.) | Implemented |
| Tribunal aggregator + weighted scoring | Implemented |
| SELECT / RESAMPLE / ABSTAIN decision logic | Implemented |
| SQLite failure ledger (6-table schema) | Implemented |
| Post-hoc failure penalisation | Implemented |
| CLI (`tribunal run`, `benchmark`, `ledger`) | Implemented |
| Pluggable interfaces for all components | Implemented |
| Synthetic ARC-like benchmark tasks (5 tasks) | Implemented |

The current generator bank is a **heuristic scaffold** — greedy colour mapping, stochastic perturbation, rule enumeration, and Occam's-razor simplification. These strategies exist to populate the adjudication layer with meaningfully different candidates. They are not sovereign reasoning agents. They are the minimal environment in which tribunal logic can be exercised and validated.

---

## What is experimental but not yet implemented

These are concrete next steps with clear interfaces already defined:

**LLM-backed generator strategies**
The `BaseGenerator` interface is designed for substitution. Replacing one or more heuristic generators with real LLM calls (via any API) requires only implementing `generate(task) -> CandidateTrace`. The adjudication layer is unaffected.

**Token log-probability uncertainty**
The current `UncertaintyAnalyzer` derives all signals from inter-trace structural disagreement. A `TokenProbAnalyzer` implementing `BaseUncertaintyAnalyzer` would substitute real softmax distributions where available, sharpening entropy and margin estimates.

**Expanded failure taxonomy**
The ledger schema supports richer failure classification than is currently populated. Structured failure types — invariant-class failures, coalition-collapse failures, adversarial-generator wins — would improve the quality of `failure_similarity_penalty` matching.

**Expanded benchmark task set**
The five synthetic tasks exercise the pipeline end-to-end but do not stress-test the adjudication logic under genuine ambiguity. A larger and harder task set is a prerequisite for meaningful accuracy claims.

---

## What is conceptual doctrine guiding future work

These directions are not near-term implementation tasks. They are the research questions that give the project its purpose.

### Strange Loop writable memory

The current failure ledger is post-hoc: it records what went wrong after a decision is made, and those records influence future runs via the `failure_similarity_penalty`. This is useful but structurally limited — the memory is consulted as a scoring signal, not as an active participant in reasoning.

A Strange Loop memory would be **queryable during generation**. Before a generator strategy produces its candidate, it would be able to consult the system's prior failure record and adjust its reasoning accordingly. This is a different epistemic architecture: the memory acts as an internal corrective organ rather than an audit trail. Failures would shape the formation of hypotheses, not merely penalise their scores.

Implementing this requires solving non-trivial interface questions: what can a generator query, at what level of specificity, without collapsing the diversity of the candidate pool? That question is open.

### Co-agency and the operator-mind direction

The current generator bank produces candidates that differ only in strategy, not in epistemic stance. A co-agency architecture would involve reasoning agents with genuinely distinct internal states — different priors, different failure histories, different confidence calibrations. The tribunal would adjudicate not just between answers but between perspectives.

This direction requires re-framing what a "generator" is. In the heuristic scaffold, generators are stateless functions. In a co-agency model, each generator would be a stateful reasoning agent with its own accumulated experience. The tribunal layer would remain the adjudication authority, but what it adjudicates would be richer.

### Gödelian self-reference and undecidable cases

The tribunal's current decision logic terminates: it selects, resamples, or abstains. It does not have a mechanism for recognising when a task is structurally undecidable by the current generator pool — when the disagreement is not a symptom of insufficient evidence but of a genuine ambiguity in the task specification.

Handling this requires the system to reason about the limits of its own adjudication logic. This is the Gödelian direction: building in structured acknowledgment of cases the tribunal cannot resolve, and distinguishing those from cases where it simply lacks confidence.

---

## Relationship between concepts

| Concept | Relationship |
|---|---|
| Sovereign Epistemic Agent | The project. The goal is a reasoning system with genuine epistemic sovereignty — capable of distrusting brittle internal consensus, adjudicating competing stances, and learning from failure. |
| Epistemic Tribunal | The first experimental module. It implements the adjudication layer. It is not the finished sovereign agent; it is the substrate on which that agent will be built and tested. |
| Failure ledger | The current memory architecture. Diagnostic, post-hoc, structurally sound. A bridge toward Strange Loop memory, not its destination. |
| Strange Loop memory | The target memory architecture. Live, writable, queryable during reasoning. Not yet implemented. |
| Co-agency / operator-mind | A future framing in which generator strategies become stateful reasoning agents with distinct epistemic stances. Changes what the tribunal adjudicates, not how it adjudicates. |
| Gödelian self-reference | The outer limit of the project's ambition: a system that can reason about what it cannot resolve, and handle that honestly rather than by forced selection. |

---

The Epistemic Tribunal is the foundation. Everything above it depends on getting the adjudication layer right first.
