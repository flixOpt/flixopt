---
title: Home
hide:
  - navigation
  - toc
---

<div class="hero-section">

<h1>flixOpt</h1>

<p class="backronym"><span class="letter">F</span>lexible · <span class="letter">L</span>ow-entry · <span class="letter">I</span>nvestment · <span class="letter">X</span>-sector · <span class="letter">OPT</span>imization</p>

<p class="backronym-desc"><em>Model more than costs</em> · <em>Easy to prototype</em> · <em>Based on dispatch</em> · <em>Sector coupling</em> · <em>Mathematical optimization</em></p>

<p>Model, optimize, and analyze complex energy systems with a powerful Python framework designed for flexibility and performance.</p>

<p class="hero-buttons">
  <a href="getting-started/" class="md-button md-button--primary">🚀 Get Started</a>
  <a href="examples/" class="md-button">💡 View Examples</a>
  <a href="https://github.com/flixOpt/flixopt" class="md-button" target="_blank" rel="noopener noreferrer">⭐ GitHub</a>
</p>

</div>

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

## :material-account-group: Community & Support

<div class="feature-grid" markdown="1">

<div class="feature-card" markdown="1">

:fontawesome-brands-github:{ .feature-icon }

### GitHub

Report issues, request features, and contribute to the codebase

[Visit Repository →](https://github.com/flixOpt/flixopt){target="_blank" rel="noopener noreferrer"}

</div>

<div class="feature-card" markdown="1">

:material-forum:{ .feature-icon }

### Discussions

Ask questions and share your projects with the community

[Join Discussion →](https://github.com/flixOpt/flixopt/discussions){target="_blank" rel="noopener noreferrer"}

</div>

<div class="feature-card" markdown="1">

:material-book-open-page-variant:{ .feature-icon }

### Contributing

Help improve FlixOpt by contributing code, docs, or examples

[Learn How →](contribute/){target="_blank" rel="noopener noreferrer"}

</div>

</div>


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

---

{%
   include-markdown "../README.md"
   start="## 🛠️ Installation"
   end="## 📄 License"
%}
