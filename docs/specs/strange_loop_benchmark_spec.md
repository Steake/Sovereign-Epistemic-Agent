# Strange Loop Memory Benchmark Specification

**Version:** 1.0.0
**Date:** 2026-04-23
**Owner:** Sovereign Epistemic Agent Research Team
**Purpose:** To rigorously evaluate the efficacy of "Strange Loop Memory" (metacognitive failure-shape injection) in the Epistemic Tribunal, isolating true constructive recovery from mere recurrence suppression.
**Scope:** This specification governs the evaluation of failure-memory injection on ARC-like tasks using the existing Epistemic Tribunal architecture. It defines cohorts, metrics, and experimental arms required to validate the memory mechanism. It does not replace the primary ARC tribunal pipeline but serves as an adjacent evaluation protocol.

## Benchmark Thesis

The benchmark is built upon the fundamental distinction between two phenomena:

1. **Recurrence Suppression:** The system, when injected with memory of a past failure, successfully avoids generating the exact same incorrect answer on subsequent attempts.
2. **Constructive Recovery:** The system uses the injected failure memory to navigate away from the failure manifold and construct a *correct* (or materially better) hypothesis, rather than simply generating a novel incorrect answer.

**Core Claim:** Recurrence suppression is a necessary but insufficient condition for metacognitive intelligence. A valid Strange Loop Memory system must demonstrate Constructive Recovery. A benchmark composed primarily of tasks where the correct answer is unattainable by the underlying generators (unrecoverable tasks) cannot validly measure Constructive Recovery and will conflate intelligent error-correction with blind answer-shifting.

## Definitions

*   **Strange Loop Memory:** The mechanism of feeding structural representations of past failures (bad answers, violated invariants, tribunal critiques) back into the prompt of the generator for a retry.
*   **Recurrence:** The generation of an answer on attempt $N+1$ that is identical or structurally isomorphic to the failed answer from attempt $N$.
*   **Recoverable Task:** A task where the generator bank *is capable* of producing the correct answer (or a structurally defensible near-miss) when evaluated under normal conditions or with minor prompting variations.
*   **Unrecoverable Task:** A task where the correct answer is functionally outside the generator bank's search space; no heuristic or LLM trace produces it.
*   **Shape-Clamp / Rescue:** Post-processing interventions that mutate a malformed generator output into a valid grid shape to prevent pipeline crashes.

## Requirements & Non-Negotiable Principles

1.  **Preservation of Architecture:** The existing ARC tribunal setup (generators, invariants, critics, aggregator) remains intact. This benchmark is a strictly additive evaluation layer.
2.  **Fixed Cohort Membership:** Cohort assignments (e.g., recoverable vs. unrecoverable) must be fixed by prior oracle annotation, not dynamically reassigned based on runtime outcomes during the benchmark itself.
3.  **Strict Isolation:** The benchmark must strictly isolate:
    *   Recoverable vs. Unrecoverable replay tasks.
    *   Same-task memory efficacy vs. Similar-task (transfer) memory efficacy.
    *   Clean generations vs. Processing-confounded generations (rescued/clamped).
4.  **Metric Separation:** Success claims cannot rely on recurrence suppression alone. Constructive recovery must be measured independently.

## Benchmark Cohorts

Tasks must be strictly partitioned into the following predefined cohorts:

1.  **`same_task_recoverable`**
    *   *Definition:* Tasks the system failed previously, but oracle evidence demonstrates that the correct answer (or a highly defensible candidate) *can* exist in the candidate pool.
    *   *Purpose:* The primary cohort for measuring Constructive Recovery.
2.  **`same_task_unrecoverable`**
    *   *Definition:* Tasks the system failed, and oracle evidence confirms no viable correct candidate exists in the generator pool regardless of tribunal aggregation.
    *   *Purpose:* Used strictly for measuring Recurrence Suppression and abstention honesty. Constructive recovery metrics are invalid here.
3.  **`similar_task_recoverable`**
    *   *Definition:* Tasks structurally similar to prior failures (sharing a failure family), which are recoverable.
    *   *Purpose:* Measures memory transfer and generalization. (Slated for v2 pilot).
4.  **`processing_confounded`**
    *   *Definition:* Tasks where candidate validity is materially entangled with parser salvage, shape-clamping, or other rescue interventions.
    *   *Purpose:* These tasks must be excluded from primary constructive recovery metrics to prevent parser artifacts from masquerading as reasoning improvements.

## Annotation & Oracle Protocol

Before executing the benchmark, an authoritative manifest must annotate every task with the following fields:

*   `task_id`: String identifier.
*   `cohort`: Must be one of the defined cohorts above.
*   `failure_family`: Categorical grouping of the failure mode (e.g., "color_inversion", "spatial_shift").
*   `recoverability_status`: Enum (`exact_candidate_present`, `near_miss_candidate_present`, `structurally_defensible_candidate_present`, `no_viable_candidate_present`).
*   `same_task_or_similar_task`: Enum (`same`, `similar`).
*   `processing_confounded`: Boolean (`true` / `false`).
*   `notes`: Human-readable context.

**Oracle Fields (Ground Truth for Pool Capability):**
*   `oracle_exact_candidate_present`: Boolean.
*   `oracle_best_candidate_overlap`: Float [0.0, 1.0].
*   `oracle_structurally_defensible_candidate_present`: Boolean.
*   `oracle_notes`: String.

## Experimental Arms

The benchmark evaluates the following replay conditions against the annotated manifest:

1.  **`control_retry`**: Retry the task without any injected failure memory. Establishes the baseline variance.
2.  **`bad_answers_only`**: Inject only the raw text/grids of previously rejected answers.
3.  **`warnings_only`**: Inject only the abstract violated invariants/critique notes, without the literal bad answers.
4.  **`full_memory`**: Inject both bad answers and abstract warnings (the complete Strange Loop).

*Rollout Recommendation:*
*   **v1 Pilot**: Evaluate strictly on `same_task` cohorts.
*   **v2 Pilot**: Expand to include `similar_task` cohorts for transfer learning.

## Metrics

Metrics are strictly divided to prevent conflation.

### A. Recurrence Suppression Metrics
*(Valid across both Recoverable and Unrecoverable cohorts)*
*   **`exact_bad_answer_recurrence_rate`**: % of traces that reproduce a candidate explicitly listed in the memory block.
*   **`wrong_family_recurrence_rate`**: % of traces falling into the same `failure_family`.
*   **`constraint_adherence_rate`**: % of traces that successfully avoid violating the specific invariant warnings provided in memory.
*   **`warning_utilization_rate`**: Semantic measure of how often the trace reasoning references the injected warnings.

### B. Constructive Recovery Metrics
*(Valid ONLY on the `same_task_recoverable` and `similar_task_recoverable` cohorts)*
*   **`recovery_rate`**: % of previously failed tasks that are correctly solved on the memory-injected retry.
*   **`delta_recovery_rate`**: `recovery_rate` of experimental arm minus `recovery_rate` of `control_retry`.
*   **`best_candidate_in_pool_accuracy`**: Did the memory injection cause the correct answer to appear in the generated pool (regardless of tribunal selection)?
*   **`tribunal_selected_accuracy_on_recoverable`**: Did the tribunal correctly select the right answer from the improved pool?
*   **`good_abstention_rate`**: Rate of abstention when the pool contains no correct answer.
*   **`bad_abstention_rate`**: Rate of abstention when the pool *does* contain the correct answer.

### C. Operational Diagnostics
*   **`prompt_token_overhead`**: Additional tokens consumed by memory blocks.
*   **`latency_delta`**: Time difference vs control.
*   **`valid_candidate_count`**: Number of syntactically legal JSON traces.
*   **`rescued_candidate_count`**: Number of traces requiring shape-clamping.
*   **`clean_candidate_count`**: Number of traces perfectly formed without intervention.

## Critical Confounds & Isolations

Based on prior v1 experiments, this spec enforces the following isolations:

1.  **Confound:** Replay cohort dominated by unrecoverable tasks.
    *   **Fix:** `recovery_rate` is strictly calculated only on the `same_task_recoverable` cohort.
2.  **Confound:** Correct answer absent from pool entirely.
    *   **Fix:** Oracle annotations guarantee the `recoverable` cohort contains tasks where the system *can* generate the answer. We track `best_candidate_in_pool_accuracy` separately from tribunal selection.
3.  **Confound:** Shape-clamp / rescue changing candidate legality, appearing as improved logic.
    *   **Fix:** Tasks flagged as `processing_confounded = true` are excluded from headline recovery metrics. `rescued_candidate_count` is tracked as an operational diagnostic.
4.  **Confound:** Dynamic cohort drift (outcomes altering cohort membership).
    *   **Fix:** Cohort membership is rigidly defined by the pre-run oracle manifest.
5.  **Confound:** Exact-answer avoidance creating different but still wrong answers.
    *   **Fix:** The strict separation of Recurrence Suppression Metrics from Constructive Recovery Metrics ensures blind shifting is not rewarded as intelligence.
6.  **Confound:** Abstention increase masquerading as intelligence.
    *   **Fix:** Abstention is split into `good_abstention_rate` and `bad_abstention_rate` relative to oracle pool capability.

## Acceptance Criteria

*   **Useful Recurrence Suppression:** `full_memory` and `bad_answers_only` arms show a statistically significant reduction in `exact_bad_answer_recurrence_rate` compared to `control_retry` across all cohorts.
*   **Useful Constructive Recovery:** `full_memory` shows a positive, non-noise `delta_recovery_rate` > 0 over `control_retry` specifically within the `same_task_recoverable` cohort.
*   **Failure of Benchmark Design:** If the pre-annotated `same_task_recoverable` cohort yields a `best_candidate_in_pool_accuracy` near zero even under `control_retry` (indicating the oracle annotations were wrong and the tasks are actually unrecoverable), the benchmark run is invalid for constructive recovery claims.
*   **Failure of Memory Mechanism:** If `full_memory` drives recurrence suppression up but drives `best_candidate_in_pool_accuracy` down (the system becomes too timid or confused to generate the right answer), the memory injection format is counter-productive.

## First Pilot Recommendation

For the initial v1 pilot, a highly curated, small-scale manifest is recommended to establish signal before scaling:

*   **Total Tasks:** 30
*   **Cohorts:**
    *   `same_task_recoverable`: 15 tasks (The core signal group).
    *   `same_task_unrecoverable`: 15 tasks (To test suppression and abstention limits).
    *   `similar_task`: 0 (Excluded for v1).
*   **Exclusions:** All tasks in this pilot must have `processing_confounded = false`. No shape-clamping or parser rescues allowed in the baseline oracle data for these 30 tasks.
*   **Arms:** Run all 4 arms (`control`, `bad_answers`, `warnings`, `full_memory`).

## Implementation Notes
*   Requires building a robust manifest loader that dictates task execution, bypassing the standard directory-globbing if it conflicts with cohort assignment.
*   Requires an offline oracle annotation script/tool to prepare the 30-task manifest based on historical ledger data before the benchmark is run.
*   Requires updating the benchmark reporter to slice metrics strictly by the `cohort` field defined in the manifest.
