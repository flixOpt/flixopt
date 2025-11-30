# FlixOpt: Progressive Flow System Optimization

<p align="center">
  <b>F</b>lexible &nbsp;•&nbsp; <b>L</b>ow-entry &nbsp;•&nbsp; <b>I</b>nvestment &nbsp;•&nbsp; <b>X</b>-sector &nbsp;•&nbsp; <b>OPT</b>imization
</p>

<p align="center">
  <i>Model more than costs</i> · <i>Easy to prototype</i> · <i>Based on dispatch</i> · <i>Sector coupling</i> · <i>Mathematical optimization</i>
</p>

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://flixopt.github.io/flixopt/latest/)
[![Build Status](https://github.com/flixOpt/flixopt/actions/workflows/python-app.yaml/badge.svg)](https://github.com/flixOpt/flixopt/actions/workflows/python-app.yaml)
[![PyPI version](https://img.shields.io/pypi/v/flixopt)](https://pypi.org/project/flixopt/)
[![PyPI status](https://img.shields.io/pypi/status/flixopt.svg)](https://pypi.org/project/flixopt/)
[![Python Versions](https://img.shields.io/pypi/pyversions/flixopt.svg)](https://pypi.org/project/flixopt/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI downloads](https://img.shields.io/pypi/dm/flixopt)](https://pypi.org/project/flixopt/)
[![GitHub last commit](https://img.shields.io/github/last-commit/flixOpt/flixopt)](https://github.com/flixOpt/flixopt/commits/main)
[![GitHub issues](https://img.shields.io/github/issues/flixOpt/flixopt)](https://github.com/flixOpt/flixopt/issues)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Powered by linopy](https://img.shields.io/badge/powered%20by-linopy-blue)](https://github.com/PyPSA/linopy/)
[![Powered by xarray](https://img.shields.io/badge/powered%20by-xarray-blue)](https://xarray.dev/)
[![DOI](https://zenodo.org/badge/540378857.svg)](https://doi.org/10.5281/zenodo.17448623)
[![DOI](https://img.shields.io/badge/DOI-10.18086%2Feurosun.2022.04.07-blue)](https://doi.org/10.18086/eurosun.2022.04.07)
[![GitHub stars](https://img.shields.io/github/stars/flixOpt/flixopt?style=social)](https://github.com/flixOpt/flixopt/stargazers)

---

**FlixOpt is a Python framework for progressive flow system optimization** - from district heating networks to industrial production lines, from renewable energy portfolios to supply chain logistics.

Build simple models quickly, then incrementally add investment decision, multi-period planning, stochastic scenarios, and custom constraints without refactoring.

---

## 🚀 Quick Start

```bash
pip install flixopt
```

That's it! FlixOpt comes with the [HiGHS](https://highs.dev/) solver included. You're ready to optimize.

**The basic workflow:**

```python
import flixopt as fx

# 1. Define your system structure
flow_system = fx.FlowSystem(timesteps)
flow_system.add_elements(buses, components, effects)

# 2. Create and solve
optimization = fx.Optimization("MyModel", flow_system)
optimization.solve(fx.solvers.HighsSolver())

# 3. Analyze results
optimization.results.solution
```

**Get started with real examples:**
- 📚 [Full Documentation](https://flixopt.github.io/flixopt/latest/)
- 💡 [Examples Gallery](https://flixopt.github.io/flixopt/latest/examples/) - Complete working examples from simple to complex
- 🔧 [API Reference](https://flixopt.github.io/flixopt/latest/api-reference/)

---

## 🌟 Why FlixOpt?

### Progressive Enhancement - Your Model Grows With You

**Start simple:**
```python
# Basic single-period model
flow_system = fx.FlowSystem(timesteps)
boiler = fx.Boiler("Boiler", eta=0.9, ...)
```

**Add complexity incrementally:**
- **Investment decisions** → Add `InvestParameters` to components
- **Multi-period planning** → Add `periods` dimension to FlowSystem
- **Uncertainty modeling** → Add `scenarios` dimension with probabilities
- **Custom constraints** → Extend with native linopy syntax

**No refactoring required.** Your component definitions stay the same - periods, scenarios, and features are added as dimensions and parameters.

→ [Learn more about multi-period and stochastic modeling](https://flixopt.github.io/flixopt/latest/user-guide/mathematical-notation/dimensions/)

### For Everyone

- **Beginners:** High-level components that "just work"
- **Experts:** Full access to modify models with linopy
- **Researchers:** Quick prototyping with customization options
- **Engineers:** Reliable, tested components without black boxes
- **Students:** Clear, Pythonic interfaces for learning optimization

### Key Features

**Multi-criteria optimization:** Model costs, emissions, resource use - any custom metric. Optimize single objectives or use weighted combinations and ε-constraints.
→ [Effects documentation](https://flixopt.github.io/flixopt/latest/user-guide/mathematical-notation/effects-and-dimensions/)

**Performance at any scale:** Choose optimization modes without changing your model - Optimization, SegmentedOptimization, or ClusteredOptimization (using [TSAM](https://github.com/FZJ-IEK3-VSA/tsam)).
→ [Optimization modes](https://flixopt.github.io/flixopt/latest/api-reference/optimization/)

**Built for reproducibility:** Self-contained NetCDF result files with complete model information. Load results months later - everything is preserved.
→ [Results documentation](https://flixopt.github.io/flixopt/latest/api-reference/results/)

**Flexible data operations:** Transform FlowSystems with xarray-style operations (`sel()`, `resample()`) for multi-stage optimization.

---

## 🎯 What is FlixOpt?

### A General-Purpose Flow Optimization Framework

FlixOpt models **any system involving flows and conversions:**

- **Energy systems:** District heating/cooling, microgrids, renewable portfolios, sector coupling
- **Material flows:** Supply chains, production lines, chemical processes
- **Integrated systems:** Water-energy nexus, industrial symbiosis

While energy systems are our primary focus, the same foundation applies universally. This enables coupling different system types within integrated models.

### Modern Foundations

Built on [linopy](https://github.com/PyPSA/linopy/) and [xarray](https://github.com/pydata/xarray), FlixOpt delivers **performance** and **transparency**. Full access to variables, constraints, and model structure. Extend anything with native linopy syntax.

### Our Position

We bridge the gap between high-level strategic models (like [FINE](https://github.com/FZJ-IEK3-VSA/FINE)) and low-level dispatch tools - similar to [PyPSA](https://docs.pypsa.org/latest/). FlixOpt is the sweet spot for detailed operational planning and long-term investment analysis in the **same framework**.

### Academic Roots

Originally developed at [TU Dresden](https://github.com/gewv-tu-dresden) for the SMARTBIOGRID project (funded by the German Federal Ministry for Economic Affairs and Energy, FKZ: 03KB159B). FlixOpt evolved from the MATLAB-based flixOptMat framework while incorporating best practices from [oemof/solph](https://github.com/oemof/oemof-solph).

---

## 🛣️ Roadmap

**FlixOpt aims to be the most accessible, flexible, and universal Python framework for energy and material flow optimization.** We believe optimization modeling should be approachable for beginners yet powerful for experts, minimizing context switching between different planning horizons.

**Current focus:**
- Enhanced component library (sector coupling, hydrogen, thermal networks)
- Examples showcasing multi-period and stochastic modeling
- Advanced result analysis and visualization

**Future vision:**
- Modeling to generate alternatives (MGA) for robust decision-making
- Advanced stochastic optimization (two-stage, CVaR)
- Community ecosystem of user-contributed components

→ [Full roadmap and vision](https://flixopt.github.io/flixopt/latest/roadmap/)

---

## 🛠️ Installation

### Basic Installation

```bash
pip install flixopt
```

Includes the [HiGHS](https://highs.dev/) solver - you're ready to optimize immediately.

### Full Installation

For additional features (interactive network visualization, time series aggregation):

```bash
pip install "flixopt[full]"
```

### Solver Support

FlixOpt supports many solvers via linopy: **HiGHS** (included), **Gurobi**, **CPLEX**, **CBC**, **GLPK**, and more.

→ [Installation guide](https://flixopt.github.io/flixopt/latest/getting-started/)

---

## 🤝 Contributing

FlixOpt thrives on community input. Whether you're fixing bugs, adding components, improving docs, or sharing use cases - **we welcome your contributions.**

→ [Contribution guide](https://flixopt.github.io/flixopt/latest/contribute/)

---

## 📖 Citation

If FlixOpt supports your research or project, please cite:

- **Main Citation:** [DOI:10.18086/eurosun.2022.04.07](https://doi.org/10.18086/eurosun.2022.04.07)
- **Short Overview:** [DOI:10.13140/RG.2.2.14948.24969](https://doi.org/10.13140/RG.2.2.14948.24969)

To pinpoint which version you used in your work, please reference one of these doi's here:
- [![DOI](https://zenodo.org/badge/540378857.svg)](https://doi.org/10.5281/zenodo.17448623)

---

## 📄 License

MIT License - See [LICENSE](https://github.com/flixopt/flixopt/blob/main/LICENSE) for details.
