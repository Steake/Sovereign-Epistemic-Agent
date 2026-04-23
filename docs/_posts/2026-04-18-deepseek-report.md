---
layout: post
title: "DeepSeek-Chat Full Adjudication"
excerpt: "First full-pipeline run of the Epistemic Tribunal adjudication system against a 50-task ARC validation subset using DeepSeek-Chat."
document_refs:
  - arc-epistemic-memo
  - godelos-overview
---

**Date:** 2026-04-18  
**Experiment Arm:** `tribunal_full_3gen`  
**Model Backend:** DeepSeek-Chat V3.2  
**Task Set:** ARC Validation Subset — 50 unique tasks (78 ledger entries including resamples)  
**Config:** `configs/tribunal_full_experiment.yaml`

---

## 1. Executive Summary

This report documents the first full-pipeline run of the **Epistemic Tribunal** adjudication system against a 50-task ARC validation subset using DeepSeek-Chat as the primary LLM generator, augmented by two heuristic generators (Greedy and Diverse).

The central research question:

> **"Does metacognitive adjudication add value beyond raw answer selection?"**

### Key Finding

The tribunal demonstrated **high metacognitive honesty** — successfully refusing to commit on 89.5% of tasks where no generator produced a correct answer. However, base model accuracy on this validation subset is too low (0% selective accuracy) to demonstrate positive "lift." The system functions as an **effective error gate** rather than a performance amplifier in this configuration.

---

## 2. Experimental Setup

### 2.1 Generator Pool (3 Agents)

| Generator | Type | Description |
|:---|:---|:---|
| `llm` | DeepSeek-Chat V3.2 | Primary intelligent agent, temperature=0.0, 8192 token budget |
| `greedy` | Deterministic heuristic | Colour-frequency majority rule baseline |
| `diverse` | Stochastic perturbation | 10% cell perturbation of greedy output |

### 2.2 Tribunal Configuration

| Parameter | Value | Rationale |
|:---|:---|:---|
| `adjudication_strategy` | `standard` | Full tribunal pipeline with uncertainty, critic, and invariant scoring |
| `selection_threshold` | 0.35 | Score at which tribunal commits to SELECT |
| `resample_threshold` | 0.15 | Score floor below which we ABSTAIN |
| `diversity_floor` | 0.60 | Relaxed to allow coalition of 2 generator types |
| `structural_override` | enabled | Path B can bypass when structural evidence is overwhelming |

### 2.3 Scoring Weights

```
Uncertainty:  0.25  │  Entropy + coalition mass signal
Critic:       0.35  │  Consistency + invariant compliance  
Memory:       0.00  │  No ledger history (cold start)
Invariant:    0.40  │  Shape, colour, object count checks
```

---

## 3. Results

### 3.1 Primary Metrics

| Metric | Value | Interpretation |
|:---|:---|:---|
| **Total Tasks** | 50 (78 ledger entries) | 28 resamples occurred |
| **Overall Accuracy** | 4.0% | 2 tasks with correct ground truth match |
| **Selective Accuracy** | 0.0% | No selected answers were correct |
| **Coverage** | 24.0% | Tribunal committed on 12/50 tasks |
| **Abstention Rate** | 76.0% | System refused to guess on 38 tasks |
| **Mean Confidence** | 0.432 | Selected tasks averaged ~43% confidence |
| **ECE** | 0.427 | System is overconfident when selecting |
| **Avg. Duration** | 11.16s | Per-task wall-clock time |

### 3.2 Decision Distribution

![Tribunal Decision Distribution]({{ "/assets/images/chart_decisions.png" | relative_url }})

The overwhelming majority of tasks resulted in **RESAMPLE** (77%), indicating the tribunal's diversity gate consistently detected that generators were not producing convergent solutions. Only 23% of tasks crossed the selection threshold.

> [!NOTE]
> No tasks ended in ABSTAIN — all non-selections were RESAMPLE decisions. This means the tribunal always saw enough signal to attempt resampling, rather than completely giving up. After the single resample attempt, tasks that still didn't converge would terminate as resampled (since `max_resample_attempts: 1`).

### 3.3 Metacognitive Quality (Abstention Analysis)

![Metacognitive Quality]({{ "/assets/images/chart_abstention_quality.png" | relative_url }})

This is the most important chart in the report. It answers: **"When the tribunal refused to answer, was it right to do so?"**

| Metric | Count | Description |
|:---|:---|:---|
| **Good Abstentions** | 47 | Correctly avoided tasks where no candidate was correct |
| **Bad Abstentions** | 4 | Missed tasks where at least one candidate had the right answer |
| **Wrong Selections** | 18 | Selected an answer that was incorrect |
| **Correct Selections** | 0 | No selections matched ground truth |

**Abstention Efficiency: 92.2%** — When the tribunal refuses to commit, it is correct 92% of the time. This is the tribunal's strongest signal.

> [!WARNING]
> The 4 "Bad Abstentions" represent missed opportunities: the system had a correct answer in the candidate pool but failed to identify and select it. These came from the `contested-recoverable` cohort — tasks where a correct answer existed but the tribunal's scoring couldn't distinguish it from incorrect candidates.

### 3.4 Confidence Distribution

![Confidence Distribution]({{ "/assets/images/chart_confidence_dist.png" | relative_url }})

The confidence distributions for RESAMPLE and SELECT decisions heavily overlap in the 0.33–0.45 range. This narrow band indicates:

- The selection threshold of 0.35 is barely above the resample confidence range
- There's poor separation between "confident enough to select" and "uncertain enough to resample"
- Confidence calibration needs work — the ECE of 0.427 confirms the system is overconfident

**Confidence Statistics:**

| Decision | Mean | Min | Max |
|:---|:---|:---|:---|
| Resample | 0.360 | 0.333 | 0.477 |
| Select | 0.432 | 0.370 | 0.549 |

### 3.5 Generator Performance Comparison

![Generator Scores]({{ "/assets/images/chart_generator_scores.png" | relative_url }})

All three generators performed similarly, with the `greedy` heuristic slightly outscoring the LLM on aggregate tribunal scores. This is a counterintuitive but important finding:

| Generator | Mean Score | Std Dev |
|:---|:---|:---|
| `greedy` | 0.708 | 0.082 |
| `llm` | 0.697 | 0.072 |
| `diverse` | 0.633 | 0.074 |

The `diverse` generator consistently scored lowest due to its stochastic cell perturbation violating invariant constraints (lower V scores). However, it serves a critical role: by introducing controlled disagreement, it enables the tribunal's coalition and diversity signals to function.

### 3.6 Disagreement–Confidence Landscape

![Disagreement vs Confidence]({{ "/assets/images/chart_scatter_disagree.png" | relative_url }})

This scatter plot reveals the decision landscape:

- **High disagreement (0.9–1.0)** dominates, meaning generators almost never agree on solutions for these validation tasks
- The confidence range is compressed (0.33–0.55), with no clear separation between correct and incorrect outcomes
- Wrong selections (red X markers) occupy the same region as resample decisions — the tribunal is not sufficiently discriminating

> [!IMPORTANT]
> The average disagreement rate across all tasks is **0.907**, meaning generators disagreed on roughly 91% of cells. This extreme disagreement rate is the primary driver of the high resample rate — the tribunal correctly identifies that there is no consensus.

---

## 4. Cohort Analysis

### 4.1 Task Stratification

| Cohort | N | Description |
|:---|:---|:---|
| **Contested-Recoverable** | 4 | Tasks where at least one generator produced the correct answer |
| **Contested-Unrecoverable** | 46 | Tasks where no generator produced the correct answer |

### 4.2 Performance by Cohort

<div class="cohort-grid">
  <div class="cohort-card recoverable">
    <div class="cohort-header">Contested-Recoverable</div>
    <div class="cohort-stat"><span>Abstention Rate</span><span>100%</span></div>
    <div class="cohort-stat"><span>Wrong Picks</span><span>0</span></div>
    <div class="cohort-stat"><span>Selective Accuracy</span><span>N/A</span></div>
  </div>
  <div class="cohort-card unrecoverable">
    <div class="cohort-header">Contested-Unrecoverable</div>
    <div class="cohort-stat"><span>Abstention Rate</span><span>73.9%</span></div>
    <div class="cohort-stat"><span>Wrong Picks</span><span>12</span></div>
    <div class="cohort-stat"><span>Selective Accuracy</span><span>0.0%</span></div>
  </div>
</div>

> [!CAUTION]
> The tribunal abstained on **all 4 recoverable tasks** — meaning it had the right answer available but couldn't identify it. This is the critical failure mode: the adjudication pipeline lacks sufficient signal to distinguish correct from incorrect candidates when disagreement is high.

---

## 5. Path B Structural Override Analysis

The Path B structural override was triggered only **once** across 78 runs. The override gates failed primarily due to:

- **Margin too small** (most common): The structural margin between top candidates was below the 0.04 threshold
- **Critic score below threshold**: The aggregate critic score didn't reach the 0.75 cutoff
- **Invariant violations present**: Candidates had too many structural constraint violations

This suggests the Path B thresholds are conservative for `deepseek-chat` — a model that already produces structurally valid (but semantically incorrect) outputs.

---

## 6. Diagnostic Summary

### What Worked
- ✅ **Error Gating**: 92.2% abstention efficiency — the tribunal successfully suppresses hallucinations
- ✅ **Zero Parse Failures**: The DeepSeek-Chat prompt engineering is robust; no JSON parsing issues
- ✅ **Zero Truncations**: The 8192 token budget is sufficient for this model
- ✅ **Pipeline Stability**: All 50 tasks completed without crashes

### What Needs Improvement
- ❌ **Selective Accuracy**: 0% — when the tribunal selects, it's always wrong
- ❌ **Recoverable Task Identification**: 0/4 recoverable tasks were correctly identified
- ❌ **Confidence Calibration**: ECE of 0.427 indicates systematic overconfidence
- ⚠️ **Threshold Tuning**: Selection threshold of 0.35 allows too many wrong picks (18)

---

## 7. Recommendations for Next Experiments

### 7.1 Immediate: Threshold Tightening
Increase `selection_threshold` from 0.35 to **0.50** to reduce wrong picks. This will further decrease coverage but improve precision.

### 7.2 Model Upgrade: DeepSeek-Reasoner (R1)
Run the same 50-task set with `deepseek-reasoner` to measure whether native Chain-of-Thought reasoning improves:
- Base accuracy (currently ~4%)
- Candidate discrimination (currently 0% selective accuracy)
- Disagreement rates (currently 91%)

### 7.3 Path B Relaxation
Lower `margin_threshold` from 0.04 to 0.02 and `c_threshold` from 0.75 to 0.65 to allow the structural override to fire more often, particularly on tasks where invariant compliance is high.

### 7.4 Cohort-Aware Adjudication
Investigate whether the tribunal can learn to apply different confidence thresholds based on task-level features (grid size, colour count, structural complexity) to improve recoverable task identification.

---

## 8. Reproduction

```bash
# Set environment
export ARC_DATASET_PATH="/path/to/dataset"
export DEEPSEEK_API_KEY="..."

# Run the experiment
python3 scripts/arc.py run \
  --config configs/tribunal_full_experiment.yaml \
  --manifest data/validation_manifest_v1.txt

# Inspect results
python3 scripts/inspect_results.py results/benchmark_tribunal_full_experiment_*.json
```

---

*Report generated from ledger: `data/tribunal_full_experiment_ledger.db`*  
*Pipeline version: Epistemic Tribunal v2 (3-generator standard adjudication)*
