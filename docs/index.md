# flixOpt: Energy and Material Flow Optimization Framework

**flixOpt** is a Python-based optimization framework designed to tackle energy and material flow problems using mixed-integer linear programming (MILP). Combining flexibility and efficiency, it provides a powerful platform for both dispatch and investment optimization challenges.

## 🚀 Introduction

flixOpt was developed by [TU Dresden](https://github.com/gewv-tu-dresden) as part of the SMARTBIOGRID project, funded by the German Federal Ministry for Economic Affairs and Energy. Building on the Matlab-based flixOptMat framework, flixOpt also incorporates concepts from [oemof/solph](https://github.com/oemof/oemof-solph).

Although flixOpt is in its early stages, it is fully functional and ready for experimentation. Feedback and collaboration are highly encouraged to help shape its future.

## 🌟 Key Features

- **High-level Interface** with low-level control
    - User-friendly interface for defining flow systems
    - Pre-defined components like CHP, Heat Pump, Cooling Tower, etc.
    - Fine-grained control for advanced configurations

- **Investment Optimization**
    - Combined dispatch and investment optimization
    - Size optimization and discrete investment decisions
    - Combined with On/Off variables and constraints

- **Effects, not only Costs --> Multi-criteria Optimization**
    - flixopt abstracts costs as so called 'Effects'. This allows to model costs, CO2-emissions, primary-energy-demand or area-demand at the same time.
    - Effects can interact with each other(e.g., specific CO2 costs)
    - Any of these `Effects` can be used as the optimization objective.
    - A **Weigted Sum**of Effects can be used as the optimization objective.
    - Every Effect can be constrained ($\epsilon$-constraint method).

- **Calculation Modes**
    - **Full** - Solve the model with highest accuracy and computational requirements.
    - **Segmented** - Speed up solving by using a rolling horizon. 
    - **Aggregated** - Speed up solving by identifying typical periods using [TSAM](https://github.com/FZJ-IEK3-VSA/tsam). Suitable for large models.

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
