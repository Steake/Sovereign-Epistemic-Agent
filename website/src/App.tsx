import './App.css'

function App() {
  return (
    <div className="app">
      <nav className="navbar">
        <div className="logo-text">
          Sovereign <span className="logo-accent">Epistemic Agent</span>
        </div>
        <div className="nav-links">
          <a href="#architecture">Architecture</a>
          <a href="#features">Features</a>
          <a href="https://github.com/Steake/Sovereign-Epistemic-Agent" target="_blank" rel="noopener noreferrer">GitHub</a>
        </div>
      </nav>

      <header className="hero">
        <div className="container">
          <h1>Epistemic Tribunal</h1>
          <p>
            A metacognitive adjudication stack for Sovereign AI. 
            Moving beyond greedy single-pass solvers to disciplined internal deliberation.
          </p>
          <div className="cta-group">
            <button className="btn-primary" onClick={() => document.getElementById('architecture')?.scrollIntoView({ behavior: 'smooth' })}>
              Explore Architecture
            </button>
            <a href="https://github.com/Steake/Sovereign-Epistemic-Agent" target="_blank" rel="noopener noreferrer">
              <button className="btn-secondary">View on GitHub</button>
            </a>
          </div>
        </div>
      </header>

      <section id="concept" className="container">
        <div className="section-title">
          <h2>The Sovereign Approach</h2>
          <p>Why we need a Tribunal for AI reasoning.</p>
        </div>
        <div className="grid grid-2">
          <div>
            <h3>The Problem: Greedy Solvers</h3>
            <p>
              Traditional AI solvers often treat the first plausible answer as sovereign. 
              They lack a mechanism for self-distrust, leading to brittle consensus and a failure to learn from past mistakes.
            </p>
          </div>
          <div>
            <h3>The Solution: Epistemic Tribunal</h3>
            <p>
              The Tribunal stages a contest between competing internal accounts. 
              It scores candidates against structural constraints and prior failure patterns, 
              ensuring that only the most robust hypotheses are selected.
            </p>
          </div>
        </div>
      </section>

      <section id="architecture" className="container">
        <div className="section-title">
          <h2>Architecture Overview</h2>
          <p>A modular stack for metacognitive adjudication.</p>
        </div>
        
        <div className="architecture-diagram">
          {/* Simple SVG representation of the mermaid diagram */}
          <svg viewBox="0 0 800 400" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect x="350" y="20" width="100" height="40" rx="8" fill="#1e293b" stroke="#38bdf8" />
            <text x="400" y="45" fill="#f8fafc" textAnchor="middle" fontSize="12">Task JSON</text>
            
            <path d="M400 60 V100" stroke="#64748b" markerEnd="url(#arrowhead)" />
            
            <rect x="250" y="100" width="120" height="50" rx="8" fill="#1e293b" stroke="#38bdf8" />
            <text x="310" y="125" fill="#f8fafc" textAnchor="middle" fontSize="12">Generator Bank</text>
            <text x="310" y="140" fill="#94a3b8" textAnchor="middle" fontSize="10">Multiple Hypotheses</text>
            
            <rect x="430" y="100" width="120" height="50" rx="8" fill="#1e293b" stroke="#38bdf8" />
            <text x="490" y="125" fill="#f8fafc" textAnchor="middle" fontSize="12">Invariant Extractor</text>
            <text x="490" y="140" fill="#94a3b8" textAnchor="middle" fontSize="10">Structural Constraints</text>
            
            <path d="M310 150 V200" stroke="#64748b" />
            <path d="M490 150 V200" stroke="#64748b" />
            <path d="M310 200 H490" stroke="#64748b" />
            
            <rect x="340" y="210" width="120" height="50" rx="8" fill="#1e293b" stroke="#38bdf8" />
            <text x="400" y="235" fill="#f8fafc" textAnchor="middle" fontSize="12">Trace Critic</text>
            <text x="400" y="250" fill="#94a3b8" textAnchor="middle" fontSize="10">Internal Consistency</text>
            
            <rect x="580" y="210" width="120" height="50" rx="8" fill="#1e293b" stroke="#38bdf8" />
            <text x="640" y="235" fill="#f8fafc" textAnchor="middle" fontSize="12">Uncertainty Analyzer</text>
            <text x="640" y="250" fill="#94a3b8" textAnchor="middle" fontSize="10">Entropy & Margin</text>
            
            <path d="M400 260 V310" stroke="#64748b" />
            <path d="M640 260 V310 H460" stroke="#64748b" />
            
            <rect x="340" y="310" width="120" height="50" rx="8" fill="#0369a1" stroke="#38bdf8" />
            <text x="400" y="335" fill="#f8fafc" textAnchor="middle" fontSize="12" fontWeight="bold">Tribunal Aggregator</text>
            <text x="400" y="350" fill="#e2e8f0" textAnchor="middle" fontSize="10">Weighted Decision</text>
            
            <path d="M460 335 H550" stroke="#64748b" strokeDasharray="4" />
            <rect x="550" y="315" width="100" height="40" rx="8" fill="#1e293b" stroke="#fbbf24" />
            <text x="600" y="340" fill="#fbbf24" textAnchor="middle" fontSize="12">Failure Ledger</text>
            
            <defs>
              <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="0" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#64748b" />
              </marker>
            </defs>
          </svg>
        </div>

        <div className="grid grid-3">
          <div className="card">
            <h3>1. Generator Bank</h3>
            <p>Produces multiple candidate reasoning traces using competing strategies like Greedy, Diverse, and Rule-First.</p>
          </div>
          <div className="card">
            <h3>2. Invariant Extractor</h3>
            <p>Infers lightweight structural constraints (e.g., color preservation, symmetry) from training pairs.</p>
          </div>
          <div className="card">
            <h3>3. Trace Critic</h3>
            <p>Scores every candidate for internal consistency, rule coherence, and morphology.</p>
          </div>
          <div className="card">
            <h3>4. Uncertainty Analyzer</h3>
            <p>Computes signals like entropy and coalition mass to gauge the system's overall confidence.</p>
          </div>
          <div className="card">
            <h3>5. Tribunal Aggregator</h3>
            <p>Combines all signals to elect a winner, request a resample, or abstain entirely.</p>
          </div>
          <div className="card">
            <h3>6. Failure Ledger</h3>
            <p>Persists structured failure records to penalize similar traces in future deliberations.</p>
          </div>
        </div>
      </section>

      <section id="comparison" className="container">
        <div className="section-title">
          <h2>The Difference</h2>
          <p>Why the Tribunal is a step forward.</p>
        </div>
        <div className="architecture-diagram">
          <table className="comparison-table">
            <thead>
              <tr>
                <th>Aspect</th>
                <th>Greedy / Single-Pass</th>
                <th>Epistemic Tribunal</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Candidate Generation</td>
                <td>One answer</td>
                <td>Multiple competing strategies</td>
              </tr>
              <tr>
                <td>Invariant Awareness</td>
                <td>None</td>
                <td>Extracted and enforced</td>
              </tr>
              <tr>
                <td>Self-Critique</td>
                <td>None</td>
                <td>Continuous scoring of all traces</td>
              </tr>
              <tr>
                <td>Uncertainty</td>
                <td>Ignored</td>
                <td>Core decision signal</td>
              </tr>
              <tr>
                <td>Failure Memory</td>
                <td>None</td>
                <td>Persistent and reusable</td>
              </tr>
              <tr>
                <td>Abstention</td>
                <td>Never</td>
                <td>Valid, confidence-based output</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section id="features" className="container">
        <div className="section-title">
          <h2>Key Features</h2>
          <p>Built for robustness and research.</p>
        </div>
        <div className="grid grid-3">
          <div className="card">
            <h3>Epistemic Sovereignty</h3>
            <p>The system does not treat the first answer as sovereign; it governs the conflict between candidate hypotheses.</p>
          </div>
          <div className="card">
            <h3>Domain Agnostic</h3>
            <p>Currently tested on ARC-like tasks, but the adjudication pattern is applicable to any reasoning domain.</p>
          </div>
          <div className="card">
            <h3>Failure-Feedback Loop</h3>
            <p>The system learns not just whether it was wrong, but <em>how</em> it was wrong, using that data to improve.</p>
          </div>
        </div>
      </section>

      <footer className="footer">
        <p>&copy; 2026 Sovereign Epistemic Agent Initiative. All rights reserved.</p>
        <p>Built for the future of robust AI reasoning.</p>
      </footer>
    </div>
  )
}

export default App
