---
title: Home
hide:
  - navigation
  - toc
---

<style>
.hero-section {
  text-align: center;
  padding: 4rem 2rem 3rem 2rem;
  background: linear-gradient(135deg, rgba(0, 150, 136, 0.1) 0%, rgba(0, 121, 107, 0.1) 100%);
  border-radius: 1rem;
  margin-bottom: 3rem;
}

.hero-section h1 {
  font-size: 3.5rem;
  font-weight: 700;
  margin-bottom: 1rem;
  background: linear-gradient(135deg, #009688 0%, #00796B 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.hero-section .tagline {
  font-size: 1.5rem;
  color: var(--md-default-fg-color--light);
  margin-bottom: 2rem;
  font-weight: 300;
}

.hero-buttons {
  display: flex;
  gap: 1rem;
  justify-content: center;
  flex-wrap: wrap;
  margin-top: 2rem;
}

.feature-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 2rem;
  margin: 3rem 0;
}

.feature-card {
  padding: 2rem;
  border-radius: 0.75rem;
  background: var(--md-code-bg-color);
  border: 1px solid var(--md-default-fg-color--lightest);
  transition: all 0.3s ease;
  text-align: center;
}

.feature-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
  border-color: var(--md-primary-fg-color);
}

.feature-icon {
  font-size: 3rem;
  margin-bottom: 1rem;
  display: block;
}

.feature-card h3 {
  margin-top: 0;
  margin-bottom: 0.5rem;
  font-size: 1.25rem;
}

.feature-card p {
  color: var(--md-default-fg-color--light);
  margin: 0;
  font-size: 0.95rem;
  line-height: 1.6;
}

.stats-banner {
  display: flex;
  justify-content: space-around;
  padding: 2rem;
  background: var(--md-code-bg-color);
  border-radius: 0.75rem;
  margin: 3rem 0;
  text-align: center;
  flex-wrap: wrap;
  gap: 2rem;
}

.stat-item {
  flex: 1;
  min-width: 150px;
}

.stat-number {
  font-size: 2.5rem;
  font-weight: 700;
  color: var(--md-primary-fg-color);
  display: block;
}

.stat-label {
  color: var(--md-default-fg-color--light);
  font-size: 0.9rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.architecture-section {
  margin: 4rem 0;
  padding: 2rem;
  background: var(--md-code-bg-color);
  border-radius: 0.75rem;
}

.quick-links {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1.5rem;
  margin: 3rem 0;
}

.quick-link-card {
  padding: 1.5rem;
  border-left: 4px solid var(--md-primary-fg-color);
  background: var(--md-code-bg-color);
  border-radius: 0.5rem;
  transition: all 0.2s ease;
  text-decoration: none;
  display: block;
}

.quick-link-card:hover {
  background: var(--md-default-fg-color--lightest);
  transform: translateX(4px);
}

.quick-link-card h3 {
  margin: 0 0 0.5rem 0;
  font-size: 1.1rem;
  color: var(--md-primary-fg-color);
}

.quick-link-card p {
  margin: 0;
  color: var(--md-default-fg-color--light);
  font-size: 0.9rem;
}

@media screen and (max-width: 768px) {
  .hero-section h1 {
    font-size: 2.5rem;
  }

  .hero-section .tagline {
    font-size: 1.2rem;
  }

  .hero-buttons {
    flex-direction: column;
    align-items: stretch;
  }

  .feature-grid {
    grid-template-columns: 1fr;
  }

  .stats-banner {
    flex-direction: column;
  }
}
</style>

<div class="hero-section">

<h1>flixOpt</h1>

<p class="tagline">Energy and Material Flow Optimization Framework</p>

<p>Model, optimize, and analyze complex energy systems with a powerful Python framework designed for flexibility and performance.</p>

<p class="hero-buttons">
  <a href="getting-started/" class="md-button md-button--primary">🚀 Get Started</a>
  <a href="examples/" class="md-button">💡 View Examples</a>
  <a href="https://github.com/flixOpt/flixopt" class="md-button">⭐ GitHub</a>
</p>

</div>

---

## :material-star-four-points: Key Features

<div class="feature-grid" markdown="1">

<div class="feature-card" markdown="1">

:material-lightning-bolt:{ .feature-icon }

### High Performance

Efficient optimization algorithms powered by industry-standard solvers for fast computation of complex energy systems

</div>

<div class="feature-card" markdown="1">

:material-puzzle:{ .feature-icon }

### Modular Design

Flexible component-based architecture allowing you to build systems from flows, buses, storage, and converters

</div>

<div class="feature-card" markdown="1">

:material-chart-line:{ .feature-icon }

### Advanced Modeling

Support for piecewise linearization, on/off parameters, investment decisions, and duration tracking

</div>

<div class="feature-card" markdown="1">

:material-code-tags:{ .feature-icon }

### Pythonic API

Clean, intuitive interface with comprehensive type hints and excellent documentation

</div>

<div class="feature-card" markdown="1">

:material-file-document-multiple:{ .feature-icon }

### Well Documented

Extensive guides, mathematical notation, examples, and API reference to get you productive quickly

</div>

<div class="feature-card" markdown="1">

:material-open-source-initiative:{ .feature-icon }

### Open Source

MIT licensed and community-driven development with contributions welcome

</div>

</div>

---

{%
   include-markdown "../README.md"
   start="## Installation"
   end="## License"
%}

---

## 🏗️ Framework Architecture

<div class="architecture-section" markdown="1">

<figure markdown>
  ![FlixOpt Conceptual Usage](./images/architecture_flixOpt.png)
  <figcaption>Conceptual Usage and IO operations of FlixOpt</figcaption>
</figure>

**FlixOpt** provides a complete workflow for energy system optimization:

- **:material-file-code: Define** your system using Python components
- **:material-cog: Optimize** with powerful solvers (HiGHS, Gurobi, CPLEX)
- **:material-chart-box: Analyze** results with built-in visualization tools
- **:material-export: Export** to various formats for further analysis

</div>

---

## :material-map-marker-path: Quick Navigation

<div class="quick-links">
  <a href="getting-started/" class="quick-link-card">
    <h3>🚀 Getting Started</h3>
    <p>New to FlixOpt? Start here with installation and your first model</p>
  </a>

  <a href="examples/" class="quick-link-card">
    <h3>💡 Examples Gallery</h3>
    <p>Explore real-world examples from simple to complex systems</p>
  </a>

  <a href="api-reference/" class="quick-link-card">
    <h3>📚 API Reference</h3>
    <p>Detailed documentation of all classes, methods, and parameters</p>
  </a>

  <a href="user-guide/recipes/" class="quick-link-card">
    <h3>📖 Recipes</h3>
    <p>Common patterns and best practices for modeling energy systems</p>
  </a>

  <a href="user-guide/mathematical-notation/" class="quick-link-card">
    <h3>∫ Mathematical Notation</h3>
    <p>Understand the mathematical formulations behind the framework</p>
  </a>

  <a href="roadmap/" class="quick-link-card">
    <h3>🛣️ Roadmap</h3>
    <p>See what's coming next and contribute to the future of FlixOpt</p>
  </a>
</div>

---

## :material-account-group: Community & Support

<div class="feature-grid" markdown="1">

<div class="feature-card" markdown="1">

:fontawesome-brands-github:{ .feature-icon }

### GitHub

Report issues, request features, and contribute to the codebase

[Visit Repository →](https://github.com/flixOpt/flixopt){target="_blank"}

</div>

<div class="feature-card" markdown="1">

:material-forum:{ .feature-icon }

### Discussions

Ask questions and share your projects with the community

[Join Discussion →](https://github.com/flixOpt/flixopt/discussions){target="_blank"}

</div>

<div class="feature-card" markdown="1">

:material-book-open-page-variant:{ .feature-icon }

### Contributing

Help improve FlixOpt by contributing code, docs, or examples

[Learn How →](contribute/){target="_blank"}

</div>

</div>

---

## :material-file-document-edit: Recent Updates

!!! tip "What's New in v3.0.0"
    Major improvements and breaking changes. Check the [Migration Guide](user-guide/migration-guide-v3.md) for upgrading from v2.x.

📋 See the full [Release Notes](changelog/) for detailed version history.

---

<div style="text-align: center; margin: 3rem 0; padding: 2rem; background: var(--md-code-bg-color); border-radius: 0.75rem;">

<h3>Ready to optimize your energy system?</h3>

<p>
  <a href="getting-started/" class="md-button md-button--primary md-button--lg">▶️ Start Building</a>
</p>

</div>
