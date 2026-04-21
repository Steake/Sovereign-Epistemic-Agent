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

```mermaid
flowchart TD
    Task[ARC Task Input] --> GB[Generator Bank]
    
    subgraph Generators
        GB --> LLM[LLM / LLM-CoT]
        GB --> GR[Greedy Heuristic]
        GB --> DIV[Diverse Perturbation]
    end
    
    LLM --> IE[Invariant Extractor]
    GR --> IE
    DIV --> IE
    
    IE --> TC[Trace Critic]
    TC --> UA[Uncertainty Analyzer]
    UA --> TA[Tribunal Aggregator]
    
    TA -->|weighted_sum or EQBSL| DEC{Decision}
    
    DEC -->|SELECT| OUT[Final Output]
    DEC -->|RESAMPLE| RET[Retry/Abstain]
    
    TA --> FL[(Failure Ledger SQLite)]
    FL -.->|Future Penalisation| TC
```

## Links

- [Source Code](https://github.com/Steake/Sovereign-Epistemic-Agent)
- [Roadmap](https://github.com/Steake/Sovereign-Epistemic-Agent/blob/main/docs/roadmap.md)
