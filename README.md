# flixOpt: Energy and Material Flow Optimization Framework

[![📚 Documentation](https://img.shields.io/badge/📚_docs-online-brightgreen.svg)](https://flixopt.github.io/flixopt/)
[![CI](https://github.com/flixOpt/flixopt/actions/workflows/python-app.yaml/badge.svg)](https://github.com/flixOpt/flixopt/actions/workflows/python-app.yaml)
[![PyPI version](https://badge.fury.io/py/flixopt.svg)](https://badge.fury.io/py/flixopt)
[![Python Versions](https://img.shields.io/pypi/pyversions/flixopt.svg)](https://pypi.org/project/flixopt/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🚀 Purpose

**flixopt** is a Python-based optimization framework designed to tackle energy and material flow problems using mixed-integer linear programming (MILP).

**flixopt** bridges the gap between high-level energy systems models like [FINE](https://github.com/FZJ-IEK3-VSA/FINE) used for design and (multi-period) investment decisions and low-level dispatch optimization tools used for operation decisions.

**flixopt** leverages the fast and efficient [linopy](https://github.com/PyPSA/linopy/) for the mathematical modeling and [xarray](https://github.com/pydata/xarray) for data handling.

**flixopt** provides a user-friendly interface with options for advanced users.

It was originally developed by [TU Dresden](https://github.com/gewv-tu-dresden) as part of the SMARTBIOGRID project, funded by the German Federal Ministry for Economic Affairs and Energy (FKZ: 03KB159B). Building on the Matlab-based flixOptMat framework (developed in the FAKS project), flixOpt also incorporates concepts from [oemof/solph](https://github.com/oemof/oemof-solph). 

---

## 📦 Installation

Install flixOpt via pip.
`pip install flixopt`
With [HiGHS](https://github.com/ERGO-Code/HiGHS?tab=readme-ov-file) included out of the box, flixopt is ready to use..

We recommend installing flixOpt with all dependencies, which enables additional features like interactive network visualizations ([pyvis](https://github.com/WestHealth/pyvis)) and time series aggregation ([tsam](https://github.com/FZJ-IEK3-VSA/tsam)).
`pip install "flixopt[full]"`

---

## 📚 Documentation

The documentation is available at [https://flixopt.github.io/flixopt/](https://flixopt.github.io/flixopt/)

---

## 🛠️ Solver Integration

By default, flixOpt uses the open-source solver [HiGHS](https://highs.dev/) which is installed by default. However, it is compatible with additional solvers such as:  

- [Gurobi](https://www.gurobi.com/)  
- [CBC](https://github.com/coin-or/Cbc)  
- [GLPK](https://www.gnu.org/software/glpk/)
- [CPLEX](https://www.ibm.com/analytics/cplex-optimizer)

For detailed licensing and installation instructions, refer to the respective solver documentation.  

---

## 📖 Citation

If you use flixOpt in your research or project, please cite the following:  

- **Main Citation:** [DOI:10.18086/eurosun.2022.04.07](https://doi.org/10.18086/eurosun.2022.04.07)  
- **Short Overview:** [DOI:10.13140/RG.2.2.14948.24969](https://doi.org/10.13140/RG.2.2.14948.24969)  
