# flixOpt: Energy and Material Flow Optimization Framework

**flixOpt** is a Python-based optimization framework designed to tackle energy and material flow problems using mixed-integer linear programming (MILP). Combining flexibility and efficiency, it provides a powerful platform for both dispatch and investment optimization challenges.

## 🚀 Introduction

flixOpt was developed by [TU Dresden](https://github.com/gewv-tu-dresden) as part of the SMARTBIOGRID project, funded by the German Federal Ministry for Economic Affairs and Energy. Building on the Matlab-based flixOptMat framework, flixOpt also incorporates concepts from [oemof/solph](https://github.com/oemof/oemof-solph).

Although flixOpt is in its early stages, it is fully functional and ready for experimentation. Feedback and collaboration are highly encouraged to help shape its future.

## 🌟 Key Features

- **High-level Interface** with low-level control
    - User-friendly interface for defining energy systems
    - Fine-grained control for advanced configurations
    - Pre-defined components like CHP, Heat Pump, Cooling Tower, etc.

- **Investment Optimization**
    - Combined dispatch and investment optimization
    - Size and discrete investment decisions
    - Integration with On/Off variables and constraints

- **Multiple Effects**
    - Couple effects (e.g., specific CO2 costs)
    - Set constraints (e.g., max CO2 emissions)
    - Easily switch optimization targets (e.g., costs vs CO2)

- **Calculation Modes**
    - **Full Mode** - Exact solutions with high computational requirements
    - **Segmented Mode** - Speed up complex systems with variable time overlap
    - **Aggregated Mode** - Typical periods for large-scale simulations

## 🛠️ Getting Started

See the [Getting Started Guide](getting-started.md) to start using flixOpt.

See the [Examples](examples/) section for detailed examples.

## ⚙️ How It Works

See our [Concepts & Math](concepts-and-math/index.md) to understand the core concepts of flixOpt.

## 🛠️ Compatible Solvers

flixOpt works with various solvers:

- HiGHS (installed by default)
- CBC
- GLPK
- Gurobi
- CPLEX

## 📝 Citation

If you use flixOpt in your research or project, please cite:

- **Main Citation:** [DOI:10.18086/eurosun.2022.04.07](https://doi.org/10.18086/eurosun.2022.04.07)
- **Short Overview:** [DOI:10.13140/RG.2.2.14948.24969](https://doi.org/10.13140/RG.2.2.14948.24969)
