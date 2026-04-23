---
layout: post
title: "Future Directions"
excerpt: "Strategic engineering path for the Epistemic Tribunal following the 8-cycle EQBSL tuning campaign."
document_refs:
  - eqbsl
  - plenum
  - gm7
---

This document outlines the strategic engineering path for the Epistemic Tribunal following the 8-cycle EQBSL (Evidence Quantified Belief Subjective Logic) tuning campaign. 

## 1. The Evidence Ceiling (Pool & Oracle Limitations)

The primary finding of the EQBSL implementation campaign is that the tribunal's adjudication logic (both baseline `weighted_sum` and `EQBSL`) is now highly mature. However, the system is fundamentally bottlenecked by the quality of its external evidence sources rather than its fusion policy.

* **Verification Hallucinations:** The current Trace Critic frequently hallucinates with high confidence on hard boundary cases. It has been observed confidently contradicting the ground truth in GSM8K and falsely supporting structural distractors in ARC.
* **The Threshold Trap:** Because the remaining failures are predominantly caused by these false oracle signals or a lack of valid candidates in the pool, further tuning of selector thresholds yields diminishing returns. Selector tuning at this stage merely shifts errors rather than resolving the underlying structural ambiguity.

## 2. Upgrading the Trace Critic (The Oracle)

The single most critical engineering investment for the next phase is migrating the Trace Critic to a natively stronger reasoning model (e.g., DeepSeek Reasoner, OpenAI o1, etc.).

* **Calibration is Key:** EQBSL directly amplifies explicit source opinions. If the critic hallucinates a contradiction with 1.0 confidence, EQBSL will predictably fail. The Trace Critic needs better intrinsic epistemic calibration—knowing when it is uncertain.
* **Focus Shift:** Development focus should shift from "how to fuse opinions" to "how to extract more accurate, less brittle opinions" from the trace critic.

## 3. Expanding the Candidate Pool

The generator bank acts as the foundation of the tribunal. If the correct answer is not present, or is present but lacks a reliable rationale, the tribunal cannot select it.

* **Stronger Generators:** Implement explicit model-backed generators with varied reasoning priors (e.g., explicit structured chain-of-thought, code-backed execution, diverse sampling temperatures) to reduce homogenous failure modes.
* **Pool Diversity:** Ensure the true answer is generated with diverse, reliable rationales before it ever reaches the selector layer.

## 4. EQBSL Deployment Strategy

* **Domain-Scoped Usage:** EQBSL should remain experimental and domain-scoped for now. While it provides vastly superior epistemic transparency (especially in tracking structural coalitions, base-rates, and belief/disbelief tension), its current reliance on noisy external verification makes it brittle cross-domain when compared to the robust `weighted_sum` baseline.
* **Hybrid Policies:** Future iterations may explore dynamic switching—falling back to `weighted_sum` when EQBSL's uncertainty metrics detect that the critic's signal is chaotic or unsupported by the pool geometry.

## 5. Transitioning to Live Memory (Strange Loop)

As outlined in the core roadmap, the current failure ledger is diagnostic and post-hoc. The next architectural frontier for failure handling is moving toward a true **Strange Loop memory**.

* Instead of strictly penalizing traces post-hoc via the trace critic or tribunal memory aggregation, prior structural failures should directly inject constraints into the generator bank *during* candidate production. This ensures the system explicitly attempts to avoid known failure classes before the adjudication phase even begins.
