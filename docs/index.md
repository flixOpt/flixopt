---
title: Home
---

<div class="hero-section">

<h1>flixOpt</h1>

<p class="backronym"><span class="letter">F</span>lexible · <span class="letter">L</span>ow-entry · <span class="letter">I</span>nvestment · <span class="letter">X</span>-sector · <span class="letter">OPT</span>imization</p>

<p class="backronym-desc"><em>Model more than costs</em> · <em>Easy to prototype</em> · <em>Based on dispatch</em> · <em>Sector coupling</em> · <em>Mathematical optimization</em></p>

<p>Model, optimize, and analyze complex energy systems with a powerful Python framework designed for flexibility and performance.</p>

<p class="hero-buttons">
  <a href="home/installation/" class="md-button md-button--primary">🚀 Get Started</a>
  <a href="notebooks/" class="md-button">💡 View Examples</a>
  <a href="https://github.com/flixOpt/flixopt" class="md-button" target="_blank" rel="noopener noreferrer">⭐ GitHub</a>
</p>

</div>

## :material-map-marker-path: Quick Navigation

<div class="grid cards" markdown>

- :rocket: **[Getting Started](home/installation/)**

    ---

    New to FlixOpt? Start here with installation and your first model

- :bulb: **[Examples Gallery](notebooks/)**

    ---

    Explore real-world examples from simple to complex systems

- :books: **[API Reference](api-reference/)**

    ---

    Detailed documentation of all classes, methods and parameters

- :book: **[Recipes](user-guide/recipes/)**

    ---

    Common patterns and best practices for modeling energy systems

- :material-math-integral: **[Mathematical Notation](user-guide/mathematical-notation/)**

    ---

    Understand the mathematical formulations behind the framework

- :material-road: **[Roadmap](roadmap/)**

    ---

    See what's coming next and contribute to the future of FlixOpt

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

<div class="grid cards" markdown>

- :fontawesome-brands-github: **GitHub**

    ---

    Report issues, request features, and contribute to the codebase

    [Visit Repository →](https://github.com/flixOpt/flixopt){target="_blank" rel="noopener noreferrer"}

- :material-forum: **Discussions**

    ---

    Ask questions and share your projects with the community

    [Join Discussion →](https://github.com/flixOpt/flixopt/discussions){target="_blank" rel="noopener noreferrer"}

- :material-book-open-page-variant: **Contributing**

    ---

    Help improve FlixOpt by contributing code, docs, or examples

    [Learn How →](contribute/){target="_blank" rel="noopener noreferrer"}

</div>


## :material-file-document-edit: Recent Updates

!!! tip "What's New in v3.0.0"
    Major improvements and breaking changes. Check the [Migration Guide](user-guide/migration-guide-v3.md) for upgrading from v2.x.

📋 See the full [Release Notes](changelog/) for detailed version history.

---

<div style="text-align: center; margin: 3rem 0; padding: 2rem; background: var(--md-code-bg-color); border-radius: 0.75rem;">

<h3>Ready to optimize your energy system?</h3>

<p>
  <a href="home/installation/" class="md-button md-button--primary md-button--lg">▶️ Start Building</a>
</p>

</div>

---

{%
   include-markdown "../README.md"
   start="## 🛠️ Installation"
   end="## 📄 License"
%}
