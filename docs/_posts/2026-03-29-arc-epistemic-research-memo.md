---
layout: post
title: "ARC-Epistemic: A Research Memo on Metacognitive Calibration"
excerpt: "The 21-day journey from philosophical hypothesis to a 1,000-task benchmark proving metacognitive routing is a learnable skill."
document_refs:
  - arc-epistemic-memo
---

*Oliver C. Hirst · Independent Researcher · March 29, 2026*

*This memo documents the intense 21-day sprint that culminated in the precursor [ARC-Epistemic](https://github.com/Steake/ARC-Epistemic) benchmark.*

---

## The Journey at a Glance

In 21 days, I have moved from a philosophical hypothesis—that an AI must have the sovereign right to doubt its own logic—to a validated 1,000-task benchmark that proves metacognitive routing is a learnable skill. This journey spanned seven training iterations, multiple H200 scale-up missions, and a series of "evidentiary shocks" that dismantled my early assumptions about model branding. The result is a system that can finally look past its own task names and "feel" the shakiness of its internal reasoning across a signal-pure distribution.

---

## 1. The Narrative of Discovery: 21 Days of Iteration

The development of ARC-Epistemic was not a linear engineering task, but a series of evidentiary shocks that forced a complete reimagining of how I train "Selector" models to measure and bridge the gap to AGI.

### Phase 0: The Philosophical Origin (Epistemic Sovereignty)
The project began not with code, but with the principle of **Epistemic Sovereignty**. I hypothesized that for an AI to be truly agentic, it must possess the "sovereign right" to its own internal state of doubt. In current RLHF paradigms, models are penalized for "I don't know" or for internal divergence. **Co-Agency** was the proposed solution: the ability for an agent (or multiple agents) to hold **distinct and consistent stances** under uncertainty, rather than collapsing into a singular, brittle consensus. This was the first idea — the core insight which predicated the benchmark and my entry into the metacognition track of the competition.

### Phase 1: The M3 Breakthrough
The journey then shifted to the discovery that a 9B parameter model, when forced into a **Diversity-Aware Consensus (M3)** ensemble, achieved a **+13.5% relative lift** on un-seen reasoning tasks. This proved that "hidden" accuracy was trapped in the model's internal probability distributions, waiting to be unlocked by better aggregation.

### Phase 2: The "Branding" Catastrophe (v6)
To automate this lift, I attempted to train **Model B (The Selector)**. The first major failure (v8 predecessor) was a model that achieved 100% accuracy on training but showed **100% SWITCH rates** during validation. It hadn't learned to reason; it had learned "branding." It saw the prefix `selector_divergence_` and triggered the skip-connection like a Pavlovian response.

### Phase 3: The H200 Sprint & "Signal-Pure" (v8)
The final 48 hours were a technical sprint. Migrating to an **NVIDIA H200 NVL** (141GB VRAM) allowed for a jump from 4-bit quantized (NF4) training to **Native Bfloat16**. This solved the weight-fidelity issues that plagued v7 (which undershot into a 100% STAY bias). By implementing **ID Scrubbing (Blinding)**, I finally forced the model to ignore the labels and look at the "shakiness" of the entropy distribution.

---

## 2. Philosophy & Epistemology: The "Switch" Architecture

### 2.1 Epistemic Mapping vs. Instruction Following
Most AI training (SFT) focuses on **Instruction Following** (getting the answer right). ARC-Epistemic shifts the focus to **Epistemic Mapping**: teaching the model to build a "map" of its own internal certainty. 
- **The Question**: "Does this distribution of 5 diverse answers look like a solved problem or a hallucination-in-progress?"
- **The Architecture**: Model B acts as an **Analog-to-Digital Switch**, converting a continuous epistemic vector (Entropy, Margin, Coalition) into a binary routing decision (`STAY` vs. `SWITCH`).

### 2.2 The "Competence Gap" & Co-Agency
The benchmark identifies a specific failure mode: **Greedy Selection**. When a model generates multiple paths, the "best" path according to the log-probs is often the *most confidently wrong*. **Co-Agency** solves this by enabling the system to evaluate multiple **distinct and consistent stances**. Instead of discarding minority reasoning paths, the selector analyzes the *coherence* of these stances under uncertainty. ARC-Epistemic is the first 1,000-task benchmark to explicitly isolate this "condition of disagreement."

---

## 3. Methodology: Technical Details of the v8 Build

### 3.1 The "Blind" SFT Pipeline
To break the categorical bias, the training data was scrubbed of all semantic identifiers:
- **Input**: Generic prompt format: `Task [task_0001]: Signals [E:0.8, M:0.1, C:0.3]`.
- **Target**: `SWITCH` (or `STAY` based on the Oracle recovery).
- **Oversampling**: I implemented a **10x Hard-Zone Multiplier**. Cases where the base model was "confidently wrong" (low entropy but wrong answer) were oversampled to sharpen the selector's edge on the most deceptive tasks.

### 3.2 H200 Precision & Performance
- **Transformers 5.x Patching**: Required a mid-run manual patch to the `Trainer` class to handle the new `processing_class` abstractions while maintaining compatibility with the distilled Qwen 27B base.
- **Port Migration**: The mission involved a midnight migration between H200 pods (Port 14613 → 15800) due to a 500GB disk overflow during the checkpointing of the 27B model.
- **Efficiency**: v8 converged to `loss=0.0000` in just **360 steps**, proving that the "Epistemic Signal" is a highly dense featurespace that the model can learn with extreme efficiency once the "branding" noise is removed.

---

## 4. Key Findings: The "Frozen Row" Trap & The 70B Imperative

### 4.1 Discrimination Success (v8)
The v8 "Signal-Pure" model achieved a **177/751 split** on the final validation run. 
- **Success Criteria**: It correctly switched on `selector_divergence` (the family where the base model fails) while staying on `control` tasks, despite neither having their name available in the prompt.
- **Epistemic Threshold**: Analysis shows the model effectively learned a non-linear threshold: `if Entropy > 0.65 OR Margin < 0.18 THEN SWITCH`.

### 4.2 The "Scale-Invariance" Methodological Trap
Early results suggested a shocking phenomenon: that 9B and 27B models had identical accuracy caps. However, an audit revealed this to be a **methodological error**—the evaluation script (`validate_h200_checkpoints.py`) contained a bug that bypassed live inference entirely, scoring both models against a shared, statically-cached "frozen row" of Qwen-7B hypotheses. 

- **Conclusion**: The realization that 27B's true ceiling remains unknown makes a **70B parameter scaling study** (Llama 3.1 70B / Qwen 2.5 72B) the absolute most critical next step. Only by taking a frontier reasoning model through the full Epistemic Pipeline (generation → signal extraction → Model B selection) can we determine the true upper bound of Diversity-Aware Consensus.
