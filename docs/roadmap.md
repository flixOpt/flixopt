# Roadmap and Vision

## 🎯 Our Vision

**FlixOpt aims to be the most accessible, flexible, and universal Python framework for energy and material flow optimization.**

We believe optimization modeling should be **approachable for beginners** yet **powerful for experts**, minimizing context switching between **short-term dispatch** and **long-term investment** planning.

---

## 🚀 Short-term (Next 6 months)

- **Recipe collection** - Community-driven library of common modeling patterns, data manipulation techniques, and optimization strategies
- **Examples of stochastic and multi-period modeling** - The new v3.0 features currently lack comprehensive showcases
- **Advanced result analysis** - Automated reporting and enhanced visualization options
- **Interactive tutorials** - Browser-based, reactive tutorials for learning FlixOpt without local installation using [Marimo](https://marimo.io/)

## 🔮 Medium-term (6-12 months)

- **Modeling to Generate Alternatives (MGA)** - Built-in support for exploring near-optimal solution spaces to produce more robust, diverse solutions under uncertainty. See [PyPSA](https://docs.pypsa.org/latest/user-guide/optimization/modelling-to-generate-alternatives/) and [Calliope](https://calliope.readthedocs.io/en/latest/examples/modes/) for reference implementations
- **Advanced stochastic optimization** - Build sophisticated new `Optimization` classes to perform different stochastic optimization approaches, like PyPSA's [two-stage stochastic programming and risk preferences with Conditional Value-at-Risk (CVaR)](https://docs.pypsa.org/latest/user-guide/optimization/stochastic/)
- **Enhanced component library** - More pre-built, domain-specific components (sector coupling, hydrogen systems, thermal networks, demand-side management)

## 🌟 Long-term (12+ months)

- **Showcase universal applicability** - FlixOpt already handles any flow-based system (supply chains, water networks, production planning, chemical processes) - we need more examples and domain-specific component libraries to demonstrate this
- **Community ecosystem** - Rich library of user-contributed components, examples, and domain-specific extensions

---

## 🤝 How to Help

- **Code**: Implement features, fix bugs, add tests
- **Docs**: Write tutorials, improve examples, create case studies
- **Components**: Contribute domain-specific components
- **Feedback**: [Report issues](https://github.com/flixOpt/flixopt/issues), [join discussions](https://github.com/flixOpt/flixopt/discussions)

See our [contribution guide](contribute.md) to get started.

---

## 📅 Release Philosophy

FlixOpt follows [semantic versioning](https://semver.org/):
- **Major** (v3→v4): Breaking changes, major features
- **Minor** (v3.0→v3.1): New features, backward compatible
- **Patch** (v3.0.0→v3.0.1): Bug fixes only

Target: Patch releases as needed, minor releases every 2-3 months.
