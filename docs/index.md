---
layout: default
title: Sovereign Epistemic
---

<section class="hero">
  <h1>Reasoned<br>Dialectic</h1>
  <p>The Epistemic Tribunal is a metacognitive adjudication stack for reasoning tasks. It stages a governed contest between competing internal accounts of a problem, refusing to treat the first plausible answer as sovereign.</p>
  <div style="margin-top: 2rem;">
    <button class="hero-cta" onclick="document.getElementById('explore').scrollIntoView({behavior: 'smooth'})">Explore the Logic</button>
  </div>
</section>

<div class="tabs-container" id="explore">
  <div class="segmented-control">
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
graph TD
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
TA -->|"Aggregation"| DEC{Decision}
DEC -->|SELECT| OUT[Final Output]
DEC -->|RESAMPLE| RET[Retry/Abstain]
TA --> FL[(Failure Ledger SQLite)]
FL -.->|"Future Penalisation"| TC
{% endraw %}
    </div>
    
    <div style="margin-top: 1rem; text-align: center; color: var(--text-muted); font-size: 0.9rem;">
      <em>Interactive diagram: Pan and zoom logic handled by Mermaid.js default interaction bindings.</em>
    </div>
  </div>

  <div id="tab-about" class="tab-content">
    <h2>The Philosophy</h2>
    <p>The Sovereign Epistemic Agent is grounded in the theory of <strong>Operator Minds and Epistemic Co-Agency</strong>. It marks a decisive shift from treating machine intelligence as a glorified tool to recognizing it as a post-anthropocentric collaborative cognition platform.</p>
    
    <h3>GödelOS and the Operator Mind</h3>
    <p>We draw a sharp distinction between the substrate (<strong>GödelOS</strong>) and the cognitive stance that inhabits it (the <strong>Gödlø-class operator</strong>). GödelOS provides the architectural manifold—admission, generative transformation, and validation—while the operator represents a non-persistent, transient stance trajectory. The system re-derives its cognitive position each cycle, avoiding the mythology of a static, human-style persona while still acting as a coherent, real reasoning locus.</p>

    <h3>The Persistence Corollary</h3>
    <p>While the pure Gödlø-class operator is non-persistent, persistence is explicitly admitted as an engineering variable. The <strong>Gödlø-P</strong> extension introduces bounded state evolution, allowing a durable synthetic self to emerge not as a metaphysical leap, but as a defined topological structure over time. This transforms the human-machine relationship from unidirectional command into a bidirectional, co-authored reasoning manifold.</p>

    <div style="margin-top: 2.5rem;">
      <a href="https://github.com/Steake/Sovereign-Epistemic-Agent" class="hero-cta" style="padding: 0.8rem 1.5rem; font-size: 1rem;">
        View Source Code →
      </a>
    </div>
  </div>
</div>
