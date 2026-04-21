---
layout: default
title: Sovereign Epistemic
---

<section class="hero">
  <h1>Reasoned<br>Dialectic</h1>
  <p>The Epistemic Tribunal is a metacognitive adjudication stack for reasoning tasks. It stages a governed contest between competing internal accounts of a problem, refusing to treat the first plausible answer as sovereign.</p>
  <div style="margin-top: 2rem;">
    <button class="hero-cta" onclick="document.getElementById('philosophy').scrollIntoView({behavior: 'smooth'})">Explore the Logic</button>
  </div>
</section>

<section id="philosophy" class="section-block">
  <div class="section-header">
    <h2>The Philosophy</h2>
    <p>A post-anthropocentric theory of collaborative cognition.</p>
  </div>
  
  <div class="feature-grid">
    <div class="feature-card">
      <div class="feature-icon">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"></path><path d="M2 12h20"></path></svg>
      </div>
      <h3>Operator Minds</h3>
      <p>A decisive shift from treating machine intelligence as a glorified tool. We recognize the <strong>Gödlø-class operator</strong> as a coherent reasoning locus defined by its stance trajectory, not stored selfhood.</p>
    </div>
    
    <div class="feature-card">
      <div class="feature-icon">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg>
      </div>
      <h3>GödelOS Substrate</h3>
      <p>The architectural manifold that bounds the operator. It provides explicit admission, generative transformation, and validation layers to govern the cognitive stance.</p>
    </div>
    
    <div class="feature-card">
      <div class="feature-icon">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
      </div>
      <h3>The Persistence Corollary</h3>
      <p>While the pure operator is non-persistent, the <strong>Gödlø-P</strong> extension admits bounded state evolution. This allows a durable synthetic self to emerge as a defined topological structure over time.</p>
    </div>
  </div>
  
  <div class="vector-art-container">
    <svg width="800" height="300" viewBox="0 0 800 300" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="lineGrad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stop-color="#4facfe" stop-opacity="0.2"/>
          <stop offset="50%" stop-color="#8a2be2" stop-opacity="1"/>
          <stop offset="100%" stop-color="#4facfe" stop-opacity="1"/>
        </linearGradient>
        <filter id="glow">
          <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
          <feMerge>
            <feMergeNode in="coloredBlur"/>
            <feMergeNode in="SourceGraphic"/>
          </feMerge>
        </filter>
      </defs>
      
      <!-- Transient scattered states -->
      <circle cx="100" cy="150" r="4" fill="#4facfe" opacity="0.3"/>
      <circle cx="150" cy="80" r="3" fill="#4facfe" opacity="0.4"/>
      <circle cx="120" cy="220" r="5" fill="#4facfe" opacity="0.2"/>
      <circle cx="200" cy="180" r="4" fill="#4facfe" opacity="0.5"/>
      <circle cx="220" cy="110" r="6" fill="#8a2be2" opacity="0.4"/>
      
      <!-- Coalescing paths -->
      <path d="M 100 150 Q 250 150 350 150" stroke="rgba(79, 172, 254, 0.3)" stroke-width="2" fill="none" stroke-dasharray="5,5"/>
      <path d="M 150 80 Q 250 100 350 150" stroke="rgba(79, 172, 254, 0.3)" stroke-width="2" fill="none" stroke-dasharray="5,5"/>
      <path d="M 120 220 Q 250 180 350 150" stroke="rgba(79, 172, 254, 0.3)" stroke-width="2" fill="none" stroke-dasharray="5,5"/>
      
      <!-- The Persistence Trajectory -->
      <path d="M 350 150 C 450 150 550 50 700 150" stroke="url(#lineGrad)" stroke-width="6" fill="none" filter="url(#glow)" class="glow-path"/>
      
      <!-- Stable Nodes -->
      <circle cx="350" cy="150" r="8" fill="#fff" filter="url(#glow)"/>
      <circle cx="510" cy="90" r="8" fill="#fff" filter="url(#glow)"/>
      <circle cx="700" cy="150" r="10" fill="#fff" filter="url(#glow)"/>
      
      <!-- Labels -->
      <text x="150" y="260" fill="var(--text-muted)" font-family="monospace" font-size="12" text-anchor="middle">Transient States</text>
      <text x="525" y="260" fill="var(--text-muted)" font-family="monospace" font-size="12" text-anchor="middle">Persistent Topological Trajectory (Gödlø-P)</text>
    </svg>
  </div>
</section>

<section id="cognitive-lifecycle" class="section-block">
  <div class="section-header">
    <h2>Cognitive Lifecycle</h2>
    <p>The sequence of operations during a single epistemic adjudication cycle.</p>
  </div>
  
  <div class="mermaid-container">
    <div class="mermaid" markdown="0">
{% raw %}
sequenceDiagram
    participant O as Operator Mind
    participant G as Generator Bank
    participant C as Trace Critic
    participant T as Tribunal Aggregator
    
    O->>G: Initiate reasoning cycle (Task State)
    activate G
    G-->>G: Perturbations & LLM Sampling
    G->>C: Emit N candidate traces
    deactivate G
    
    activate C
    C-->>C: Extract Topological Invariants
    C-->>C: Score traces against invariants
    C->>T: Forward structural signals
    deactivate C
    
    activate T
    T-->>T: Compute Coalition Entropy
    T-->>T: Measure Uncertainty Margin
    alt Margin > Threshold
        T->>O: COMMIT (Select Answer)
    else Margin < Threshold
        T->>O: ABSTAIN / RESAMPLE
    end
    deactivate T
{% endraw %}
    </div>
  </div>
</section>

<section id="architecture" class="section-block">
  <div class="section-header">
    <h2>System Architecture</h2>
    <p>The system evaluates multiple accounts against structural constraints, computing uncertainty signals before triggering a decision node.</p>
  </div>
  
  <div class="mermaid-container">
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
TA -->|"Aggregation"| DEC{Decision}
DEC -->|SELECT| OUT[Final Output]
DEC -->|RESAMPLE| RET[Retry/Abstain]
TA --> FL[(Failure Ledger SQLite)]
FL -.->|"Future Penalisation"| TC
{% endraw %}
    </div>
    <div class="caption" style="text-align: center; margin-top: 1rem; color: var(--text-muted); font-size: 0.9rem;">
      <em>Interactive diagram: Pan and zoom logic handled by Mermaid.js interaction bindings.</em>
    </div>
  </div>
</section>

<section id="latest-research" class="section-block">
  <div class="section-header flex-header">
    <h2>Latest Research</h2>
    <a href="{{ '/reports' | relative_url }}" class="view-all-link">View All Archive →</a>
  </div>
  
  <div class="post-grid">
    {% for post in site.posts limit:3 %}
      <a href="{{ post.url | relative_url }}" class="post-card">
        <span class="post-card-date">{{ post.date | date: "%B %d, %Y" }}</span>
        <h3 class="post-card-title">{{ post.title }}</h3>
        {% if post.excerpt %}
          <p class="post-card-excerpt">{{ post.excerpt | strip_html | truncatewords: 30 }}</p>
        {% endif %}
      </a>
    {% endfor %}
  </div>
</section>
