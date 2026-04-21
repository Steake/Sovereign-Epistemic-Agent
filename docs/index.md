---
layout: default
title: Home
---

# Sovereign Epistemic Agent

**Research blog for the Epistemic Tribunal project.**

The Epistemic Tribunal is a metacognitive adjudication stack for reasoning tasks. It stages a governed contest between competing internal accounts of a problem, scores those accounts against structural constraints and prior failure patterns, and decides whether any candidate deserves selection.

The central object is not "the answer." It is the governed conflict between candidate hypotheses.

---

## Experiment Reports

<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
      <br>
      <small>{{ post.date | date: "%B %d, %Y" }}</small>
      {% if post.excerpt %}
        <p>{{ post.excerpt | strip_html | truncatewords: 40 }}</p>
      {% endif %}
    </li>
  {% endfor %}
</ul>

---

## Architecture

```
Task
 → Generator Bank (competing strategies: LLM, LLM-CoT, Greedy, Diverse)
 → Invariant Extractor (structural constraints from training pairs)
 → Trace Critic (consistency, rule coherence, morphology, failure memory)
 → Uncertainty Analyzer (entropy, margin, coalition mass, disagreement)
 → Tribunal Aggregator (weighted_sum or EQBSL fusion → SELECT / RESAMPLE / ABSTAIN)
 → Failure Ledger (SQLite persistence for post-hoc analysis and future penalisation)
```

## Links

- [Source Code](https://github.com/Steake/Sovereign-Epistemic-Agent)
- [Roadmap](https://github.com/Steake/Sovereign-Epistemic-Agent/blob/main/docs/roadmap.md)
