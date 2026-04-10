# Sovereign Epistemic Agent — Roadmap

This repository houses the **Sovereign Epistemic Agent** project. The **Epistemic Tribunal** is the first implemented module inside that broader effort. This roadmap keeps three layers separate: what exists now, what is a plausible next extension of the current stack, and what remains doctrinal guidance for longer-range research.

---

## Concept hierarchy

```text
Sovereign Epistemic Agent            ← umbrella project / research direction
└── Epistemic Tribunal               ← first implemented experimental module
    ├── Generator bank              ← heuristic scaffold (implemented)
    ├── Invariant extractor         ← implemented
    ├── Trace critic                ← implemented
    ├── Uncertainty analyzer        ← implemented
    ├── Tribunal aggregator         ← implemented
    └── Failure ledger              ← implemented diagnostic memory
        └── Strange Loop memory     ← future live memory during reasoning
```

The names in this hierarchy are not interchangeable. **Sovereign Epistemic Agent** names the project and direction. **Epistemic Tribunal** names the concrete adjudication stack currently implemented in this repository.

---

## What exists now

### Sovereign Epistemic Agent

At present this is the **umbrella identity and research direction**, not a completed runtime architecture. The repository currently implements one concrete module inside that broader project.

### Epistemic Tribunal

This is the **implemented experimental adjudication stack**. It runs competing generator strategies, extracts invariants, critiques traces, measures disagreement, selects or abstains, and records structured outcomes in SQLite.

| Component | Status | Role now |
|---|---|---|
| Generator bank | Implemented | Produces multiple candidate traces from heuristic strategies |
| Invariant extractor | Implemented | Infers structural constraints from training pairs |
| Trace critic | Implemented | Scores consistency, rule coherence, morphology, failure similarity, and invariant compliance |
| Uncertainty analyzer | Implemented | Measures disagreement, margin, coalition mass, and related signals |
| Tribunal aggregator | Implemented | Combines signals into SELECT / RESAMPLE / ABSTAIN decisions |
| Failure ledger | Implemented | Stores traces, decisions, violations, and failures for inspection and later penalisation |
| CLI and sample tasks | Implemented | Provides a runnable reference environment for experiments |

The current generator bank is a **heuristic scaffold**. It is there to populate the adjudication layer with meaningfully different candidate traces. It should not be mistaken for the finished sovereign agent.

### Failure ledger

The current ledger is **real and operational**, but its role is still limited. It is diagnostic, post-hoc, and penalty-oriented. It is best understood as a **bridge toward Strange Loop memory**, not as the destination.

---

## What is experimental but not yet implemented

These are extensions of the present tribunal stack, not features the repository already has.

| Extension | Status | Why it matters |
|---|---|---|
| Strange Loop memory | Not implemented | Would make memory queryable and writable during reasoning rather than only after the run |
| Model-backed generators | Not implemented | Would replace heuristic trace producers without changing the adjudication layer |
| Richer uncertainty inputs | Not implemented | Would allow token-probability or comparable signals instead of only structural disagreement proxies |
| Stronger failure retrieval and taxonomy | Not implemented | Would tighten how prior failures shape scoring, resampling, and later experiments |
| Expanded benchmark and evaluation regime | Not implemented | Would test the stack under harder and more genuinely ambiguous conditions |

A future **Strange Loop memory** would move memory from archive to participant. Instead of merely recording a failure and penalising similar traces later, the system would be able to consult prior failures during generation or revision. That distinction matters: the present ledger supports post-hoc learning signals, while Strange Loop memory would support live corrective influence.

---

## What is doctrinal guidance for future work

These ideas explain the direction of the project. They do **not** describe implemented runtime behaviour.

### Co-agency and operator-mind direction

The current generator bank differs by heuristic strategy, not by durable epistemic stance. A future co-agency or operator-mind architecture would involve candidate producers with more persistent differences: distinct priors, different failure histories, different calibration, or explicit operator participation in the reasoning loop. The tribunal would still adjudicate conflict, but the conflict would be richer than the present scaffold provides.

### Gödelian self-reference and undecidable-case handling

The current stack can select, resample, or abstain. It does not yet model the stronger question of whether some cases are undecidable for the current generator pool or under-specified by the task itself. The longer-range Gödelian direction is about handling those limits honestly: distinguishing low confidence from structural irresolvability, and building mechanisms that can acknowledge when adjudication itself has reached a boundary.

---

## Relationship summary

| Concept | Status | Relationship |
|---|---|---|
| Sovereign Epistemic Agent | Project identity / research direction | Umbrella under which later modules may be developed |
| Epistemic Tribunal | Implemented | First experimental adjudication module in the repository |
| Generator bank | Implemented | Heuristic scaffold that feeds the tribunal candidate traces |
| Failure ledger | Implemented | Current memory layer; structured, persistent, and post-hoc |
| Strange Loop memory | Future extension | Stronger memory architecture for live use during reasoning |
| Co-agency / operator-mind | Future doctrine | Possible future framing for richer candidate producers and operator participation |
| Gödelian limit-handling | Future doctrine | Possible future treatment of undecidable or structurally unresolved cases |

The immediate purpose of the repository is therefore narrow and concrete: use the **Epistemic Tribunal** as a disciplined test bed for adjudication, disagreement handling, and failure reuse without confusing that implemented stack with the full **Sovereign Epistemic Agent** end-state.
