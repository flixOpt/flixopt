# FlixOpt: Energy and Material Flow Optimization Framework

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://flixopt.github.io/flixopt/latest/)
[![Build Status](https://github.com/flixOpt/flixopt/actions/workflows/python-app.yaml/badge.svg)](https://github.com/flixOpt/flixopt/actions/workflows/python-app.yaml)
[![PyPI version](https://img.shields.io/pypi/v/flixopt)](https://pypi.org/project/flixopt/)
[![PyPI status](https://img.shields.io/pypi/status/flixopt.svg)](https://pypi.org/project/flixopt/)
[![Python Versions](https://img.shields.io/pypi/pyversions/flixopt.svg)](https://pypi.org/project/flixopt/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI downloads](https://img.shields.io/pypi/dm/flixopt)](https://pypi.org/project/flixopt/)
[![GitHub last commit](https://img.shields.io/github/last-commit/flixOpt/flixopt)](https://github.com/flixOpt/flixopt/commits/main)
[![GitHub issues](https://img.shields.io/github/issues/flixOpt/flixopt)](https://github.com/flixOpt/flixopt/issues)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/flixOpt/flixopt/main.svg)](https://results.pre-commit.ci/latest/github/flixOpt/flixopt/main)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Powered by linopy](https://img.shields.io/badge/powered%20by-linopy-blue)](https://github.com/PyPSA/linopy/)
[![Powered by xarray](https://img.shields.io/badge/powered%20by-xarray-blue)](https://xarray.dev/)
[![DOI](https://img.shields.io/badge/DOI-10.18086%2Feurosun.2022.04.07-blue)](https://doi.org/10.18086/eurosun.2022.04.07)
[![GitHub stars](https://img.shields.io/github/stars/flixOpt/flixopt?style=social)](https://github.com/flixOpt/flixopt/stargazers)

---

## 🎯 Vision

**FlixOpt aims to be the most accessible, flexible and universal Python framework for energy and material flow optimization.**

We believe that optimization modeling should be **approachable for beginners** yet **powerful for experts**. Further we belive in **minimizing context switching**.

Flixopt aims to be the tool of choice for both short term planning with high grade of detail, and for long term investment planning. We belive that using the same tool and context for both holds a lot of value.

### Where We're Going

**Short-term goals:**
- **Enhanced component library**: More pre-built, domain-specific components (sector coupling, hydrogen systems, thermal networks, demand-side management)
- **Advanced result analysis**: Automated reporting and even more visualization options
- **Examples of stochastic and Multi-Period Modeling**: THe new feature is currently lacking showcases.
- **Interactive tutorials**: Browser-based, reactive tutorials for learning FlixOpt without local installation

**Medium-term vision:**
- **Modeling to generate alternatives (MGA)**: Built-in support for exploring near-optimal solution spaces to produce more robust, diverse solutions under uncertainty. See [PyPSA](https://docs.pypsa.org/latest/user-guide/optimization/modelling-to-generate-alternatives/) and [calliope](https://calliope.readthedocs.io/en/latest/examples/modes/)
- **Stochastic Optimization**: Build sophisticated new `Calculation` classes to perform differnet Stochastic optimization. Like PyPSA's new [**Two-Stage Stochastic-Programming** or **Risk Preferences with Conditional Value-at-Risk (CVaR)**](https://docs.pypsa.org/latest/user-guide/optimization/stochastic/)

**Long-term vision:**
- **Showcase universal applicability**: FlixOpt already handles any flow-based system (supply chains, water networks, production planning, chemical processes) - we need more examples and domain-specific component libraries to demonstrate this
- **Community ecosystem**: Rich library of user-contributed components, examples, and domain-specific extensions

### Why FlixOpt Exists

FlixOpt is a **general-purpose framework for modeling any system involving flows and conversions** - energy, materials, fluids, goods, or data. While energy systems are the primary use case, the same mathematical foundation applies to supply chains, water networks, production lines, and more. THis also enables the coupling of such systems with an energy system model.

We bridge the gap between high-level strategic models (like [FINE](https://github.com/FZJ-IEK3-VSA/FINE)) for long-term planning and low-level tools for short term operation/dispatch.
This approach is similar to the mature [PyPSA](https://docs.pypsa.org/latest/) project. FlixOpt is the **sweet spot** for:

- **Researchers** who need to prototype quickly but may require deep customization later
- **Engineers** who want reliable, tested components without black-box abstractions
- **Students** learning optimization who benefit from clear, Pythonic interfaces
- **Practitioners** who need to move from model to production-ready results
- **Domain experts** from any field where things flow, transform, and need optimizing

Built on modern foundations ([linopy](https://github.com/PyPSA/linopy/) and [xarray](https://github.com/pydata/xarray)), FlixOpt delivers both **performance** and **transparency**. You can inspect everything, extend anything, and trust that your model does exactly what you designed.

Originally developed at [TU Dresden](https://github.com/gewv-tu-dresden) for the SMARTBIOGRID project (funded by the German Federal Ministry for Economic Affairs and Energy, FKZ: 03KB159B), FlixOpt has evolved from the Matlab-based flixOptMat framework while incorporating the best ideas from [oemof/solph](https://github.com/oemof/oemof-solph).

---

## 🌟 What Makes FlixOpt Different

### Start Simple, Scale Complex
Define a working model in minutes with high-level components, then drill down to fine-grained control when needed. No rewriting, no framework switching.

```python
import flixopt as fx

# Simple start
boiler = fx.Boiler("Boiler", eta=0.9, ...)

# Advanced control when needed - extend with native linopy
boiler.model.add_constraints(custom_constraint, name="my_constraint")
```

### Multi-Criteria Optimization Done Right
Model costs, emissions, resource use, and any custom metric simultaneously as **Effects**. Effects use intuitive `share_from_*` syntax showing clear contribution sources. Optimize any single Effect, use weighted combinations, or apply ε-constraints:

```python
# Simple start
costs = fx.Effect('costs', '€', 'Total costs', objective=True)
co2 = fx.Effect('CO2', 'kg', 'Emissions')

# Later: Add effect relationships without changing component definitions
costs = fx.Effect('costs', '€', 'Total costs', objective=True,
                  share_from_temporal={'CO2': 180},  # 180 €/tCO2 carbon pricing
                  share_from_periodic={'land': 100})  # 100 €/m² land cost
co2 = fx.Effect('CO2', 'kg', 'Emissions', maximum_periodic=50000)  # Add constraint
```

### Performance at Any Scale
Choose the right calculation mode for your problem:
- **Full** - Maximum accuracy for smaller problems
- **Segmented** - Rolling horizon for large time series
- **Aggregated** - Typical periods using [TSAM](https://github.com/FZJ-IEK3-VSA/tsam) for massive models

Switch between modes without changing your model definition.

### Built for Reproducibility
Every result file is self-contained with complete model information. Full NetCDF/JSON serialization support with round-trip fidelity. Load results months later and know exactly what you optimized - complete with the original FlowSystem. Export to NetCDF, share with colleagues, archive for compliance.

### Flexible Data Manipulation
Transform your FlowSystem on the fly for different analyses:
```python
# Subset to specific time ranges
system_q2 = flow_system.sel(time=slice("2025-04", "2025-06"))

# Extract specific scenarios for comparison
system_high = flow_system.sel(scenario="high_demand")

# Resample to different temporal resolutions for multi-stage optimization
system_hourly = flow_system.resample(time="h")
system_daily = flow_system.resample(time="D")
```

### User Friendly
Flixopt is object oriented and well documented. We try to make the project and the resulting code as readable as possible.
If you have any issues with naming or definitions of parameters feel free to propose a rename.

---

## 🚀 Quick Start

```bash
pip install flixopt
```

That's it. FlixOpt comes with the [HiGHS](https://highs.dev/) solver included - you're ready to optimize.
Many more solvers are supported (gurobi, cplex, cbc, glpk, ...)

For additional features (interactive network visualization, time series aggregation):
```bash
pip install "flixopt[full]"
```

**Next steps:**
- 📚 [Full Documentation](https://flixopt.github.io/flixopt/latest/)
- 💡 [Examples](https://flixopt.github.io/flixopt/latest/examples/)
- 🔧 [API Reference](https://flixopt.github.io/flixopt/latest/api-reference/)

---

## 🤝 Contributing

FlixOpt thrives on community input. Whether you're fixing bugs, adding components, improving docs, or sharing use cases - we welcome your contributions.

See our [contribution guide](https://flixopt.github.io/flixopt/latest/contribute/) to get started.

---

## 📖 Citation

If FlixOpt supports your research or project, please cite:

- **Main Citation:** [DOI:10.18086/eurosun.2022.04.07](https://doi.org/10.18086/eurosun.2022.04.07)
- **Short Overview:** [DOI:10.13140/RG.2.2.14948.24969](https://doi.org/10.13140/RG.2.2.14948.24969)

---

## 📄 License

MIT License - See [LICENSE](https://github.com/flixopt/flixopt/blob/main/LICENSE) for details.
