---
layout: default
title: Sovereign Epistemic Agent — Research Blog
---

# Sovereign Epistemic Agent

Research dispatches from the **Epistemic Tribunal** project — a metacognitive adjudication stack that stages governed conflict between competing hypotheses rather than trusting the first plausible answer.

---

## Posts

<ul>
  {% for post in site.posts %}
    <li>
      <strong>{{ post.date | date: "%Y-%m-%d" }}</strong> —
      <a href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a>
      <br><em>{{ post.excerpt | strip_html | truncatewords: 30 }}</em>
    </li>
  {% endfor %}
</ul>

---

## About the Project

The **Epistemic Tribunal** does not treat the first plausible answer as sovereign. It stages a contest between competing internal accounts of a task, scores those accounts against structural constraints and prior failure patterns, and then decides whether any candidate deserves selection.

The central object is not "the answer." It is the **governed conflict between candidate hypotheses**.

- **[Source Code](https://github.com/Steake/Sovereign-Epistemic-Agent)**
- **[Architecture & README](https://github.com/Steake/Sovereign-Epistemic-Agent#architecture-overview)**
- **[Roadmap](https://github.com/Steake/Sovereign-Epistemic-Agent/blob/main/docs/roadmap.md)**
