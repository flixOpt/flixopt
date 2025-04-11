# FlixOpt: Energy and Material Flow Optimization Framework

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://flixopt.github.io/flixopt/latest/)
[![Build Status](https://github.com/flixOpt/flixopt/actions/workflows/python-app.yaml/badge.svg)](https://github.com/flixOpt/flixopt/actions/workflows/python-app.yaml)
[![PyPI version](https://img.shields.io/pypi/v/flixopt)](https://pypi.org/project/flixopt/)
[![Python Versions](https://img.shields.io/pypi/pyversions/flixopt.svg)](https://pypi.org/project/flixopt/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🚀 Purpose

**flixopt** is a Python-based optimization framework designed to tackle energy and material flow problems using mixed-integer linear programming (MILP).

**flixopt** bridges the gap between high-level energy systems models like [FINE](https://github.com/FZJ-IEK3-VSA/FINE) used for design and (multi-period) investment decisions and low-level dispatch optimization tools used for operation decisions.

**flixopt** leverages the fast and efficient [linopy](https://github.com/PyPSA/linopy/) for the mathematical modeling and [xarray](https://github.com/pydata/xarray) for data handling.

**flixopt** provides a user-friendly interface with options for advanced users.

It was originally developed by [TU Dresden](https://github.com/gewv-tu-dresden) as part of the SMARTBIOGRID project, funded by the German Federal Ministry for Economic Affairs and Energy (FKZ: 03KB159B). Building on the Matlab-based flixOptMat framework (developed in the FAKS project), FlixOpt also incorporates concepts from [oemof/solph](https://github.com/oemof/oemof-solph). 

---

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
    - A **Weigted Sum** of Effects can be used as the optimization objective.
    - Every Effect can be constrained ($\epsilon$-constraint method).

- **Calculation Modes**
    - **Full** - Solve the model with highest accuracy and computational requirements.
    - **Segmented** - Speed up solving by using a rolling horizon. 
    - **Aggregated** - Speed up solving by identifying typical periods using [TSAM](https://github.com/FZJ-IEK3-VSA/tsam). Suitable for large models.

---

## 📦 Installation

Install FlixOpt via pip.
`pip install flixopt`
With [HiGHS](https://github.com/ERGO-Code/HiGHS?tab=readme-ov-file) included out of the box, flixopt is ready to use..

We recommend installing FlixOpt with all dependencies, which enables additional features like interactive network visualizations ([pyvis](https://github.com/WestHealth/pyvis)) and time series aggregation ([tsam](https://github.com/FZJ-IEK3-VSA/tsam)).
`pip install "flixopt[full]"`

---

## 📚 Documentation

The documentation is available at [https://flixopt.github.io/flixopt/latest/](https://flixopt.github.io/flixopt/latest/)

---

## 🛠️ Solver Integration

By default, FlixOpt uses the open-source solver [HiGHS](https://highs.dev/) which is installed by default. However, it is compatible with additional solvers such as:  

- [Gurobi](https://www.gurobi.com/)  
- [CBC](https://github.com/coin-or/Cbc)  
- [GLPK](https://www.gnu.org/software/glpk/)
- [CPLEX](https://www.ibm.com/analytics/cplex-optimizer)

For detailed licensing and installation instructions, refer to the respective solver documentation.  

---

## 📖 Citation

If you use FlixOpt in your research or project, please cite the following:  

- **Main Citation:** [DOI:10.18086/eurosun.2022.04.07](https://doi.org/10.18086/eurosun.2022.04.07)  
- **Short Overview:** [DOI:10.13140/RG.2.2.14948.24969](https://doi.org/10.13140/RG.2.2.14948.24969)  
