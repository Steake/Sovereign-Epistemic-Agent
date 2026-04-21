---
layout: default
title: Sovereign Epistemic
---

<section class="hero">
  <h1>Reasoned<br>Dialectic</h1>
  <p>The Epistemic Tribunal is a metacognitive adjudication stack for reasoning tasks. It stages a governed contest between competing internal accounts of a problem, refusing to treat the first plausible answer as sovereign.</p>
</section>

<div class="tabs-container">
  <div class="tabs-header">
    <button class="tab-btn active" data-target="tab-reports">Research Reports</button>
    <button class="tab-btn" data-target="tab-architecture">System Architecture</button>
    <button class="tab-btn" data-target="tab-about">About</button>
  </div>

  <div id="tab-reports" class="tab-content active">
    <div class="post-grid">
      {% for post in site.posts %}
        <a href="{{ post.url | relative_url }}" class="post-card">
          <span class="post-card-date">{{ post.date | date: "%B %d, %Y" }}</span>
          <h3 class="post-card-title">{{ post.title }}</h3>
          {% if post.excerpt %}
            <p class="post-card-excerpt">{{ post.excerpt | strip_html | truncatewords: 30 }}</p>
          {% endif %}
        </a>
      {% endfor %}
    </div>
  </div>

  <div id="tab-architecture" class="tab-content">
    <p>The system evaluates multiple accounts against structural constraints, computing uncertainty signals before triggering a decision node.</p>
    
    <div class="mermaid" markdown="0">
{% raw %}
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
{% endraw %}
    </div>
    
    <div style="margin-top: 1rem; text-align: center; color: var(--text-muted); font-size: 0.9rem;">
      <em>Interactive diagram: Pan and zoom logic handled by Mermaid.js default interaction bindings.</em>
    </div>
  </div>

  <div id="tab-about" class="tab-content">
    <h2>The Philosophy</h2>
    <p>It is a vanity to assume that simply exposing a Large Language Model to a problem of logic is sufficient to produce reasoning. The central object here is not "the answer." It is the governed conflict between candidate hypotheses.</p>
    
    <div style="margin-top: 2rem;">
      <a href="https://github.com/Steake/Sovereign-Epistemic-Agent" class="icon-btn" style="display: inline-flex; width: auto; padding: 0.8rem 1.5rem; background: var(--card-hover); border: 1px solid var(--border-glass); border-radius: 8px;">
        View Source Code →
      </a>
    </div>
  </div>
</div>
