---
layout: default
title: Research Library
permalink: /library/
---

<div class="library-hero">
  <p class="library-eyebrow">PDF Corpus</p>
  <h1>Research Library</h1>
  <p>Primary papers, companion manuscripts, and direct Zenodo records underlying the Sovereign Epistemic site. Every entry exposes an in-site PDF, and Zenodo-published items also link out to their record and DOI.</p>
</div>

<div class="library-stat-grid">
  <div class="library-stat-card">
    <span class="library-stat-value">17</span>
    <span class="library-stat-label">Curated PDFs</span>
  </div>
  <div class="library-stat-card">
    <span class="library-stat-value">13</span>
    <span class="library-stat-label">Zenodo Records</span>
  </div>
  <div class="library-stat-card">
    <span class="library-stat-value">4</span>
    <span class="library-stat-label">Research Tracks</span>
  </div>
</div>

<section class="section-block">
  <div class="section-header section-header-left">
    <h2>Tribunal Lineage</h2>
    <p>Documents that connect the current Epistemic Tribunal implementation back to the ARC-Epistemic precursor benchmark and its metacognitive routing thesis.</p>
  </div>
  {% assign tribunal_docs = site.data.research_library | where: "section", "tribunal" | sort: "order" %}
  <div class="document-grid">
    {% for doc in tribunal_docs %}
      {% include document-card.html doc=doc %}
    {% endfor %}
  </div>
</section>

<section class="section-block">
  <div class="section-header section-header-left">
    <h2>Operator-Mind Foundations</h2>
    <p>The GödelOS and Gödlø-class corpus that informs the site's operator-mind framing, persistence corollary, and post-anthropocentric vocabulary.</p>
  </div>
  {% assign foundation_docs = site.data.research_library | where: "section", "foundations" | sort: "order" %}
  <div class="document-grid">
    {% for doc in foundation_docs %}
      {% include document-card.html doc=doc %}
    {% endfor %}
  </div>
</section>

<section class="section-block">
  <div class="section-header section-header-left">
    <h2>Trust And Collective Systems</h2>
    <p>Papers that extend the same epistemic posture into trust computation, collective coordination, cryptographic attestations, and anti-cartel consensus.</p>
  </div>
  {% assign systems_docs = site.data.research_library | where: "section", "systems" | sort: "order" %}
  <div class="document-grid">
    {% for doc in systems_docs %}
      {% include document-card.html doc=doc %}
    {% endfor %}
  </div>
</section>

<section class="section-block">
  <div class="section-header section-header-left">
    <h2>Companion Manuscripts</h2>
    <p>Additional local PDFs folded into the site for direct download when a line of research is present in the corpus but not yet linked to a Zenodo record in the local metadata.</p>
  </div>
  {% assign companion_docs = site.data.research_library | where: "section", "companions" | sort: "order" %}
  <div class="document-grid">
    {% for doc in companion_docs %}
      {% include document-card.html doc=doc %}
    {% endfor %}
  </div>
</section>

<section class="section-block">
  <div class="section-header section-header-left">
    <h2>Zenodo Landing</h2>
    <p>For a broader external record index beyond the in-site corpus, the canonical Zenodo search landing page remains available alongside the PDF library above.</p>
  </div>
  <div class="document-actions">
    <a class="document-btn primary" href="https://zenodo.org/search?q=hirst+oliver+shadowgraph+labs" target="_blank" rel="noopener">Open Zenodo Search</a>
    <a class="document-btn" href="{{ '/reports' | relative_url }}">Read Site Reports</a>
  </div>
</section>
