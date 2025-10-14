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

**FlixOpt is a Python framework for optimizing energy and material flow systems** - from district heating networks to industrial production lines, from renewable energy portfolios to supply chain logistics.

**Start simple, scale complex:** Build a working optimization model in minutes, then progressively add detail - multi-period investments, stochastic scenarios, custom constraints - without rewriting your code.

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
calculation = fx.FullCalculation("MyModel", flow_system)
calculation.solve()

# 3. Analyze results
calculation.results.solution
```

**Get started with real examples:**
- 📚 [Full Documentation](https://flixopt.github.io/flixopt/latest/)
- 💡 [Examples Gallery](https://flixopt.github.io/flixopt/latest/examples/) - Complete working examples from simple to complex
- 🔧 [API Reference](https://flixopt.github.io/flixopt/latest/api-reference/)

---

## 🌟 Why FlixOpt?

### For Everyone: Start Simple, Scale Complex

**Beginners** get working models in minutes with high-level components:
```python
boiler = fx.Boiler("Boiler", eta=0.9, ...)  # Just works
```

**Experts** can drill down to fine-grained control when needed:
```python
boiler.submodel.add_constraints(custom_constraint, name="my_constraint")
```

**No framework switching.** Your initial model structure stays intact as you add complexity.

### Grow Your Model Incrementally

Start simple, then progressively enhance without refactoring:

```python
# Step 1: Simple single-period model
flow_system = fx.FlowSystem(timesteps=timesteps)
boiler = fx.Boiler("Boiler", eta=0.9, ...)
flow_system.add_component(boiler)

# Step 2: Add investment decisions
boiler.Q_th.invest_parameters = fx.InvestParameters(...)

# Step 3: Multi-period planning? Just add periods!
periods = pd.Index([2025, 2030, 2035])
flow_system_mp = fx.FlowSystem(timesteps=timesteps, periods=periods)
flow_system_mp.add_component(boiler)  # Same component!

# Step 4: Model uncertainty? Add scenarios!
scenarios = pd.Index(['high_demand', 'base_case', 'low_demand'])
flow_system_stoch = fx.FlowSystem(
    timesteps=timesteps,
    periods=periods,
    scenarios=scenarios,
    weights=[0.3, 0.5, 0.2]
)
```

**The key:** Your component definitions stay the same. Periods and scenarios are dimensions of the FlowSystem, not structural changes to your model.

### Multi-Criteria Optimization

Model multiple metrics simultaneously - costs, emissions, resource use, any custom metric:

```python
# Start simple
costs = fx.Effect('costs', '€', 'Total costs', is_objective=True)
co2 = fx.Effect('CO2', 'kg', 'Emissions')

# Add sophisticated relationships later
costs = fx.Effect('costs', '€', 'Total costs', is_objective=True,
                  share_from_temporal={'CO2': 180},  # Carbon pricing: 180 €/tCO2
                  share_from_periodic={'land': 100})  # Land cost: 100 €/m²
co2 = fx.Effect('CO2', 'kg', 'Emissions',
                maximum_periodic=50000)  # Emission constraint
```

This enables to model and evaluate cost structures in a system directly from the optimization solution (Costs from Investments, Costs from Operation, Revenues, Funding, Net Costs)

### Performance at Any Scale

Choose the calculation mode that fits your problem - **without changing your model definition:**

- **Full** - Maximum accuracy for detailed models (hours to days)
- **Segmented** - Rolling horizon for long time series (months to years)
- **Aggregated** - Typical periods using [TSAM](https://github.com/FZJ-IEK3-VSA/tsam) for massive models (decades)

```python
# Same model, different calculation modes
fx.FullCalculation("MyModel", flow_system)
fx.SegmentedCalculation("MyModel", flow_system, segment_length=168)
fx.AggregatedCalculation("MyModel", flow_system, n_clusters=12)
```

### Built for Reproducibility

Every result file is **self-contained** with complete model information:

- Full NetCDF/JSON serialization with round-trip fidelity
- Load results months later and know exactly what you optimized
- Original FlowSystem included - no manual reconstruction needed
- Export to NetCDF, share with colleagues, archive for compliance

```python
# Save
calculation.results.to_file('my_results.nc')

# Load later - everything is there
results = fx.results.CalculationResults.from_file('my_results.nc')
original_system = results.flow_system  # Automatically restored!
```

### Flexible Data Manipulation

Transform your FlowSystem on the fly using familiar xarray-style operations:

```python
# Subset to specific time ranges
system_q2 = flow_system.sel(time=slice("2025-04", "2025-06"))

# Extract specific scenarios for comparison
system_high = flow_system.sel(scenario="high_demand")

# Resample for multi-stage optimization
system_daily = flow_system.resample(time="D")
```

### User-Friendly Design

- **Object-oriented** and Pythonic - feels natural if you know Python
- **Comprehensively documented** - every parameter explained
- **Readable code** - we prioritize clarity in both framework and user code
- **Open to feedback** - unclear naming? We welcome your suggestions!

---

## 🎯 What is FlixOpt?

### A General-Purpose Flow Optimization Framework

FlixOpt models **any system involving flows and conversions:**

- **Energy systems:** District heating/cooling, microgrids, renewable portfolios, sector coupling
- **Material flows:** Supply chains, production lines, chemical processes, recycling networks
- **Integrated systems:** Water-energy nexus, industrial symbiosis, smart cities

While energy systems are our primary focus, the same mathematical foundation applies universally. This enables coupling different system types within integrated models.

### The Sweet Spot

We bridge the gap between high-level strategic models (like [FINE](https://github.com/FZJ-IEK3-VSA/FINE)) for long-term planning and low-level tools for short-term operation and dispatch. This approach is similar to the mature [PyPSA](https://docs.pypsa.org/latest/) project.

**FlixOpt is ideal for:**

- **Researchers** who need quick prototyping but may require deep customization later
- **Engineers** who want reliable, tested components without black-box abstractions
- **Students** learning optimization who benefit from clear, Pythonic interfaces
- **Practitioners** who need to move from model to production-ready results
- **Domain experts** from any field where things flow, transform, and need optimizing

### Modern Foundations

Built on [linopy](https://github.com/PyPSA/linopy/) and [xarray](https://github.com/pydata/xarray), FlixOpt delivers both **performance** and **transparency**:

- **Inspect everything** - full access to variables, constraints, and model structure
- **Extend anything** - add custom constraints using native linopy syntax
- **Trust your model** - you control exactly what gets optimized

### Academic Roots

Originally developed at [TU Dresden](https://github.com/gewv-tu-dresden) for the SMARTBIOGRID project (funded by the German Federal Ministry for Economic Affairs and Energy, FKZ: 03KB159B). FlixOpt evolved from the MATLAB-based flixOptMat framework while incorporating best practices from [oemof/solph](https://github.com/oemof/oemof-solph).

---

## 🛣️ Roadmap

### Our Vision

**FlixOpt aims to be the most accessible, flexible, and universal Python framework for energy and material flow optimization.**

We believe optimization modeling should be **approachable for beginners** yet **powerful for experts**. We also believe in **minimizing context switching** - use the same tool for short-term operational planning and long-term investment analysis.

### Short-term Goals

- **Enhanced component library:** More pre-built, domain-specific components (sector coupling, hydrogen systems, thermal networks, demand-side management)
- **Advanced result analysis:** Automated reporting and enhanced visualization options
- **Examples of stochastic and multi-period modeling:** The new features currently lack comprehensive showcases
- **Interactive tutorials:** Browser-based, reactive tutorials for learning FlixOpt without local installation

### Medium-term Vision

- **Modeling to generate alternatives (MGA):** Built-in support for exploring near-optimal solution spaces to produce more robust, diverse solutions under uncertainty. See [PyPSA](https://docs.pypsa.org/latest/user-guide/optimization/modelling-to-generate-alternatives/) and [Calliope](https://calliope.readthedocs.io/en/latest/examples/modes/) for reference implementations
- **Advanced stochastic optimization:** Build sophisticated new `Calculation` classes to perform different stochastic optimization approaches, like PyPSA's [two-stage stochastic programming and risk preferences with Conditional Value-at-Risk (CVaR)](https://docs.pypsa.org/latest/user-guide/optimization/stochastic/)

### Long-term Vision

- **Showcase universal applicability:** FlixOpt already handles any flow-based system (supply chains, water networks, production planning, chemical processes) - we need more examples and domain-specific component libraries to demonstrate this
- **Community ecosystem:** Rich library of user-contributed components, examples, and domain-specific extensions

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

FlixOpt supports many solvers via linopy:
- **HiGHS** (included, open-source, recommended for most users)
- **Gurobi** (commercial, fast for large problems)
- **CPLEX** (commercial)
- **CBC, GLPK** (open-source alternatives)

---

## 🤝 Contributing

FlixOpt thrives on community input. Whether you're fixing bugs, adding components, improving docs, or sharing use cases - **we welcome your contributions.**

See our [contribution guide](https://flixopt.github.io/flixopt/latest/contribute/) to get started.

---

## 📖 Citation

If FlixOpt supports your research or project, please cite:

- **Main Citation:** [DOI:10.18086/eurosun.2022.04.07](https://doi.org/10.18086/eurosun.2022.04.07)
- **Short Overview:** [DOI:10.13140/RG.2.2.14948.24969](https://doi.org/10.13140/RG.2.2.14948.24969)

---

## 📄 License

MIT License - See [LICENSE](https://github.com/flixopt/flixopt/blob/main/LICENSE) for details.
