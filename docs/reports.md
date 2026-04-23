---
layout: default
title: Research Archive
---

<div class="reports-header">
  <h1>Research Archive</h1>
  <p>Chronicles of the Sovereign Epistemic Agent, experimental protocols, and the persistence corollary.</p>
</div>

{% assign latest_post = site.posts.first %}
{% if latest_post %}
<section class="featured-research">
  <div class="featured-label">Featured Report</div>
  <a href="{{ latest_post.url | relative_url }}" class="featured-post-card">
    <div class="featured-content">
      <span class="post-card-date">{{ latest_post.date | date: "%B %d, %Y" }}</span>
      <h2 class="featured-title">{{ latest_post.title }}</h2>
      <p class="featured-excerpt">{{ latest_post.excerpt | strip_html | truncatewords: 45 }}</p>
      <span class="read-more">Read the Full Report →</span>
    </div>
  </a>
</section>
{% endif %}

<section class="reports-reference-band">
  <div class="section-header flex-header">
    <div>
      <h2>Primary Documents</h2>
      <p>Posts on this site now link into a shared PDF corpus with direct Zenodo and DOI paths.</p>
    </div>
    <a href="{{ '/library' | relative_url }}" class="view-all-link">Browse Library →</a>
  </div>

  {% include document-list.html ids="arc-epistemic-memo, gm7, eqbsl" compact=true %}
</section>

<section class="research-archive">
  <h2>Previous Reports</h2>
  <div class="archive-list">
    {% for post in site.posts offset:1 %}
      <a href="{{ post.url | relative_url }}" class="archive-item">
        <div class="archive-date">{{ post.date | date: "%b %d, %Y" }}</div>
        <div class="archive-info">
          <h3>{{ post.title }}</h3>
          <p>{{ post.excerpt | strip_html | truncatewords: 25 }}</p>
        </div>
      </a>
    {% endfor %}
  </div>
</section>
