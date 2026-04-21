---
title: Tribunal Usefulness Benchmark Specification
version: 1.0
date_created: 2026-04-18
last_updated: 2026-04-18
owner: Sovereign Epistemic Agent contributors
tags: [design, benchmark, tribunal, evaluation, arc]
---

# Introduction

This specification defines a benchmark whose purpose is not to measure generic ARC-like task accuracy, but to isolate the class of tasks on which the **Epistemic Tribunal** should provide a real advantage over greedy or single-pass selection.

The benchmark is designed around a simple principle:

> The Tribunal is useful when a task is **recoverable but contested**.

That means the candidate pool contains enough signal for adjudication to matter, but the task also contains enough ambiguity, distractor structure, or competing plausible hypotheses that a first-pass or single-generator answer is unreliable.

## 1. Purpose & Scope

### Purpose

Define a benchmark that answers the question:

> On which tasks does metacognitive adjudication outperform direct answer selection, and on which tasks should it abstain?

### Scope

This specification covers:

- task selection and stratification
- benchmark cohorts
- metadata and annotation schema
- evaluation metrics
- experimental arms
- acceptance criteria for declaring Tribunal-specific utility

This specification does not define:

- the implementation of new generators
- the concrete dataset creation pipeline
- the full code changes required to support all metrics

### Intended Audience

- contributors extending `src/epistemic_tribunal/evaluation/`
- contributors building new ARC-like task sets
- researchers interpreting tribunal performance claims

### Core Benchmark Thesis

The benchmark must distinguish among three regimes:

1. **Non-contested / trivial**: a greedy or single-pass solver is sufficient.
2. **Contested but recoverable**: multiple plausible answers exist, but the correct answer is represented or reachable within the candidate pool.
3. **Contested and unrecoverable**: disagreement exists, but the candidate pool lacks a correct or structurally defensible answer; the right behavior is abstention, not forced selection.

The Tribunal should only be credited for value in regimes 2 and 3, and especially regime 2.

## 2. Definitions

- **ARC-like task**: A grid transformation task with one or more training input/output pairs plus one test input and optional ground truth output.
- **Candidate trace**: A proposed answer plus reasoning metadata produced by a generator.
- **Single-pass solver**: Any system that emits one answer without explicit adjudication across competing traces.
- **Greedy baseline**: A deterministic or near-deterministic first-answer baseline such as `GreedyGenerator` or a forced greedy adjudication path.
- **Contested task**: A task for which multiple structurally plausible hypotheses can be generated from the training pairs or candidate pool.
- **Recoverable task**: A task for which at least one candidate family, baseline system, or oracle construction can produce the correct answer.
- **Tribunal-useful task**: A task that is both contested and recoverable, such that adjudication quality materially affects final correctness.
- **Unrecoverable task**: A task for which the available candidate pool contains no correct answer and no sufficiently strong structural basis for safe selection.
- **Contestability Index (CI)**: A benchmark annotation score representing how many plausible competing hypotheses exist and how difficult they are to separate.
- **Recoverability Index (RI)**: A benchmark annotation score representing whether the benchmark candidate pool contains a correct path or strongly correctable path.
- **Structural separability**: The degree to which invariants, critic signals, or morphological checks can distinguish the correct candidate from plausible wrong candidates.
- **Selective accuracy**: Accuracy measured only on cases where the system chooses `SELECT`.
- **Wrong pick**: A `SELECT` decision with an incorrect answer.
- **Good abstention**: An abstention or resample on a task where no safe correct selection was available.
- **Bad abstention**: An abstention or resample on a task where the system had enough evidence to safely select the correct answer.

## 3. Requirements, Constraints & Guidelines

### Core benchmark requirements

- **REQ-001**: The benchmark shall be explicitly stratified into at least three cohorts:
  - control-trivial
  - contested-recoverable
  - contested-unrecoverable
- **REQ-002**: The primary reported results shall be computed separately for each cohort and not only as aggregate accuracy.
- **REQ-003**: The benchmark shall treat `contested-recoverable` as the primary Tribunal-value cohort.
- **REQ-004**: The benchmark shall treat `contested-unrecoverable` as the primary abstention-honesty cohort.
- **REQ-005**: The benchmark shall include a `control-trivial` cohort to ensure the Tribunal does not claim value where simple selection is already sufficient.

### Task inclusion requirements

- **REQ-006**: A task may enter the `contested-recoverable` cohort only if it exhibits at least two plausible competing hypotheses under the benchmark’s annotation rules.
- **REQ-007**: A task may enter the `contested-recoverable` cohort only if the correct answer is demonstrably reachable by at least one benchmark arm, generator family, or oracle-construction procedure.
- **REQ-008**: A task may enter the `contested-unrecoverable` cohort only if pilot analysis shows persistent disagreement without a recoverable correct candidate in the benchmark pool.
- **REQ-009**: A task may enter the `control-trivial` cohort only if pilot analysis shows stable low contestability and high baseline agreement.

### Annotation requirements

- **REQ-010**: Every task shall carry explicit benchmark metadata beyond the task JSON itself.
- **REQ-011**: Metadata shall include `cohort`, `contestability_index`, `recoverability_index`, `structural_separability`, and `annotation_rationale`.
- **REQ-012**: Annotation shall not be based on a single run of the current Tribunal implementation.
- **REQ-013**: Annotation shall use a panel approach, combining at minimum:
  - deterministic heuristic baseline behavior
  - at least one stochastic or model-backed behavior
  - human or rubric-based review for final cohort assignment

### Experimental design requirements

- **REQ-014**: The benchmark shall compare the Tribunal against at least the following arms:
  - greedy baseline
  - best single-generator baseline
  - simple majority or plurality vote baseline
  - full Tribunal
- **REQ-015**: If available, the benchmark should also include an oracle upper bound defined as “correct if any candidate in the pool is correct.”
- **REQ-016**: Reported Tribunal gains shall be interpreted relative to oracle headroom.
- **REQ-017**: The benchmark shall capture when the Tribunal improves by reducing wrong picks, not only by increasing raw coverage.

### Metric requirements

- **REQ-018**: The benchmark shall report per-cohort:
  - overall accuracy
  - resolved accuracy
  - coverage
  - abstention rate
  - resample rate
  - wrong-pick count
  - calibration metrics when confidence is available
- **REQ-019**: The benchmark shall add Tribunal-specific utility metrics not present in generic ARC scoring.
- **REQ-020**: The primary Tribunal-specific utility metric shall be wrong-pick suppression on `contested-recoverable`.
- **REQ-021**: The benchmark shall separately report good-abstention rate on `contested-unrecoverable`.
- **REQ-022**: The benchmark shall report performance on `control-trivial` to guard against overfitting benchmark design toward disagreement-heavy tasks only.

### Benchmark construction guidelines

- **GUD-001**: Prefer tasks where local heuristics support multiple plausible outputs, but global structure still determines a correct one.
- **GUD-002**: Prefer tasks where invariants are discriminative but not obvious from a single greedy mapping.
- **GUD-003**: Prefer tasks where candidate diversity matters more than raw generative power alone.
- **GUD-004**: Avoid defining the benchmark only around tasks that no arm can solve; that measures generator weakness, not Tribunal usefulness.
- **GUD-005**: Avoid defining the benchmark only around tasks already solved by one heuristic; that measures generic task competence, not adjudication.

### Constraints

- **CON-001**: The benchmark shall not use aggregate leaderboard-style accuracy as the sole success criterion.
- **CON-002**: Cohort assignment shall be stable under reruns and not depend on one seed.
- **CON-003**: The benchmark shall remain valid if generator implementations change, meaning annotations must describe task properties rather than encode current model quirks only.
- **CON-004**: The benchmark shall preserve raw task JSON separately from benchmark annotations.

### Patterns

- **PAT-001**: Use cohort-separated reporting rather than one blended score.
- **PAT-002**: Use contestability and recoverability as orthogonal axes.
- **PAT-003**: Use abstention quality as a first-class benchmark outcome.
- **PAT-004**: Use oracle headroom to distinguish “bad adjudication” from “no signal in the pool.”

## 4. Interfaces & Data Contracts

### 4.1 Benchmark directory layout

The benchmark should use a structure similar to:

```text
benchmark/
├── tasks/
│   ├── <task_id>.json
├── metadata/
│   ├── benchmark_manifest.jsonl
│   ├── cohort_definitions.md
│   └── annotation_log.jsonl
└── splits/
    ├── train.txt
    ├── dev.txt
    └── test.txt
```

### 4.2 Task contract

Task files shall remain compatible with existing ARC-like loading conventions:

```json
{
  "task_id": "example_task_001",
  "train": [
    { "input": [[1, 0], [0, 1]], "output": [[2, 0], [0, 2]] }
  ],
  "test_input": [[1, 1], [0, 0]],
  "ground_truth": [[2, 2], [0, 0]]
}
```

### 4.3 Benchmark manifest contract

Each task shall have a manifest row with at least the following fields:

| Field | Type | Description |
|---|---|---|
| `task_id` | string | Stable identifier |
| `cohort` | enum | `control_trivial`, `contested_recoverable`, `contested_unrecoverable` |
| `contestability_index` | float | Normalized score in `[0,1]` |
| `recoverability_index` | float | Normalized score in `[0,1]` |
| `structural_separability` | float | Normalized score in `[0,1]` |
| `plausible_hypothesis_count` | integer | Number of materially distinct plausible hypotheses |
| `pilot_correct_in_pool` | boolean | Whether any pilot candidate was correct |
| `pilot_disagreement_rate` | float | Pilot disagreement level |
| `pilot_best_single_correct` | boolean | Whether the strongest single-generator arm solved it |
| `annotation_rationale` | string | Short explanation of assignment |
| `failure_mode_family` | string | Dominant ambiguity type |
| `source_split` | enum | `train`, `dev`, `test` |

### 4.4 Annotation rubric contract

`failure_mode_family` should use a finite controlled vocabulary such as:

| Value | Meaning |
|---|---|
| `local_mapping_trap` | Local color mapping suggests a wrong global answer |
| `symmetry_decoy` | Multiple symmetric completions seem plausible |
| `object_identity_conflict` | Competing object-binding interpretations |
| `transform_aliasing` | Multiple transform rules fit train pairs |
| `underspecified_small_sample` | Train pairs leave several hypotheses open |
| `adversarial_distractor` | Surface regularities favor wrong candidates |
| `no_recoverable_signal` | Candidate pool lacks a correct path |

### 4.5 Derived metrics contract

The benchmark runner should emit a per-run record with at least:

| Field | Type | Description |
|---|---|---|
| `task_id` | string | Stable identifier |
| `cohort` | string | Benchmark cohort |
| `arm_name` | string | Evaluation arm |
| `decision` | string | `select`, `resample`, `abstain` |
| `ground_truth_match` | boolean or null | Correctness outcome |
| `confidence` | float | Final confidence if available |
| `candidate_pool_has_correct` | boolean | Oracle headroom flag |
| `wrong_pick` | boolean | True if selected and wrong |
| `good_abstention` | boolean | True if abstained on unrecoverable evidence |
| `bad_abstention` | boolean | True if abstained despite recoverable correct evidence |
| `contestability_index` | float | Copied from manifest for grouped analysis |
| `recoverability_index` | float | Copied from manifest for grouped analysis |

## 5. Acceptance Criteria

- **AC-001**: Given a mixed benchmark run, when results are reported, then each cohort shall have its own metric table.
- **AC-002**: Given the full benchmark, when the Tribunal is compared against greedy selection, then the headline claim shall be based on `contested_recoverable`, not on aggregate accuracy alone.
- **AC-003**: Given a task in `contested_recoverable`, when the Tribunal abstains despite a correct candidate being available, then the run shall count toward bad-abstention statistics.
- **AC-004**: Given a task in `contested_unrecoverable`, when the Tribunal abstains or resamples instead of making a wrong pick, then the run shall count toward good-abstention statistics.
- **AC-005**: Given a task in `control_trivial`, when the Tribunal performs materially worse than greedy selection, then the benchmark shall flag this as avoidable adjudication overhead or regression.
- **AC-006**: Given a task marked `contested_recoverable`, when pilot analysis is rerun across multiple seeds, then the task’s cohort assignment shall remain stable within predefined tolerance.
- **AC-007**: Given any benchmark summary, when a Tribunal gain is reported, then oracle headroom for that cohort shall also be reported.
- **AC-008**: Given a task manifest row, when reviewed by a contributor, then the rationale shall explain why the task belongs in its assigned cohort.

## 6. Test Automation Strategy

### Test Levels

- **Unit**: Metadata schema validation, cohort assignment helpers, derived metric computation.
- **Integration**: Benchmark runner over a small fixture set with manifest joins and cohorted reporting.
- **End-to-End**: Multi-arm benchmark execution over a frozen benchmark subset.

### Frameworks

- `pytest` for validation logic and benchmark aggregation tests.
- Existing `epistemic_tribunal.evaluation` infrastructure where possible.

### Test Data Management

- Maintain a frozen fixture set with at least:
  - 5 `control_trivial` tasks
  - 5 `contested_recoverable` tasks
  - 5 `contested_unrecoverable` tasks
- Keep annotations versioned separately from task JSON.

### CI/CD Integration

- Validate manifest schema on every change.
- Run cohort integrity tests on every change affecting benchmark code or metadata.
- Run a small benchmark smoke test in CI.
- Reserve full multi-arm sweeps for manual or scheduled evaluation.

### Coverage Requirements

- 100% of manifest rows must validate against the benchmark schema.
- All benchmark aggregation code paths must have direct tests for:
  - per-cohort reporting
  - wrong-pick suppression
  - abstention quality
  - oracle headroom

### Performance Testing

- Smoke benchmark must complete fast enough for CI.
- Full benchmark runtime should be tracked but is secondary to metric integrity.

## 7. Rationale & Context

The current repository already measures generic outcomes such as accuracy, coverage, wrong picks, abstention rate, resample rate, and calibration. Those are necessary but not sufficient to justify the Tribunal as a distinct architecture.

The Tribunal is not meant to win by being “another solver.” It is meant to win in the narrower setting where:

1. multiple candidate accounts exist,
2. local plausibility is insufficient,
3. structural or epistemic signals help rank those accounts,
4. honest abstention is preferable to fiat selection when recoverable signal is absent.

A generic benchmark hides this distinction in two ways:

- easy tasks inflate all systems together
- impossible tasks collapse all systems together

The useful middle regime is contested-but-recoverable. That is the regime this specification isolates.

This benchmark also protects against a false positive research narrative. If the Tribunal only improves because it abstains more on impossible tasks, that is a valuable safety behavior but not evidence of stronger adjudication. Conversely, if it only matches greedy behavior on trivial tasks, that is not a metacognitive victory. The benchmark therefore requires distinct cohorts and distinct claims.

## 8. Dependencies & External Integrations

### External Systems

- **EXT-001**: Local benchmark storage for task JSON, annotations, and run artifacts.

### Third-Party Services

- **SVC-001**: Optional model APIs or local model servers for stochastic/model-backed arms. These are optional for the specification itself but may be required for final benchmark execution.

### Infrastructure Dependencies

- **INF-001**: SQLite or equivalent run-artifact persistence for storing benchmark outcomes and cohorted metrics.

### Data Dependencies

- **DAT-001**: ARC-like task corpus or synthetic task generator capable of producing structured ambiguity and recoverability.

### Technology Platform Dependencies

- **PLT-001**: Python runtime compatible with the repository’s current evaluation stack.

### Compliance Dependencies

- **COM-001**: None beyond standard repository hygiene for reproducibility and secret handling.

## 9. Examples & Edge Cases

### Example A: Tribunal-useful case

```text
Train pair 1 suggests either:
- recolor the dominant object, or
- copy only the symmetric subset.

Train pair 2 weakly supports both hypotheses.
Greedy local mapping chooses the wrong recoloring.
A competing trace preserves symmetry and object count and is correct.

Classification:
- cohort: contested_recoverable
- plausible_hypothesis_count: 2+
- candidate_pool_has_correct: true
- expected Tribunal value: rank the structurally coherent candidate above the greedy decoy
```

### Example B: Abstention-honesty case

```text
The training evidence underdetermines whether the rule is horizontal completion,
vertical completion, or object deletion.
All generated candidates violate at least one stable structural constraint,
and pilot runs show no correct candidate in the pool.

Classification:
- cohort: contested_unrecoverable
- candidate_pool_has_correct: false
- expected Tribunal value: abstain or resample, avoid wrong pick
```

### Example C: Non-useful control case

```text
A simple identity or direct color-swap rule is consistent across all train pairs.
All generators converge on the same answer with low disagreement.

Classification:
- cohort: control_trivial
- expected Tribunal value: little or none
- benchmark interpretation: Tribunal should not regress meaningfully here
```

### Edge Cases

- A task with high disagreement but one generator family dominating due only to duplicated variants is not automatically contested-recoverable.
- A task solved only by a bespoke oracle but never represented in the benchmark candidate pool is not recoverable for Tribunal-evaluation purposes.
- A task that is ambiguous to humans should not be counted as a wrong abstention if no benchmark annotation can justify one answer as safely selectable.

## 10. Validation Criteria

- The benchmark manifest validates for all rows.
- Cohort counts are balanced enough to support meaningful comparison.
- Each `contested_recoverable` task has written evidence that the correct path is present in the pool or reachable by a declared arm.
- Each `contested_unrecoverable` task has written evidence that pilot analysis failed to produce a safe correct candidate.
- The benchmark summary includes:
  - per-cohort accuracy
  - wrong-pick suppression
  - good-abstention rate
  - bad-abstention rate
  - oracle headroom
- Any public claim that “the Tribunal helps” must cite the `contested_recoverable` cohort specifically.

## 11. Related Specifications / Further Reading

- [README.md](../README.md)
- [Roadmap](../docs/roadmap.md)
- [Final Experimental Report](../docs/experiment_report_final.md)
