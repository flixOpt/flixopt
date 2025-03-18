# flixOpt: Energy and Material Flow Optimization Framework

**flixOpt** is a Python-based optimization framework designed to tackle energy and material flow problems using mixed-integer linear programming (MILP). Combining flexibility and efficiency, it provides a powerful platform for both dispatch and investment optimization challenges.

---

## 🚀 Introduction

flixOpt was developed by [TU Dresden](https://github.com/gewv-tu-dresden) as part of the SMARTBIOGRID project, funded by the German Federal Ministry for Economic Affairs and Energy (FKZ: 03KB159B). Building on the Matlab-based flixOptMat framework (developed in the FAKS project), flixOpt also incorporates concepts from [oemof/solph](https://github.com/oemof/oemof-solph). 

Although flixOpt is in its early stages, it is fully functional and ready for experimentation. It is used for investment and operation decisions by energy providing companys as well as research institutions. Feedback and collaboration are highly encouraged to help shape its future.

---

## 📦 Installation

Install flixOpt directly into your environment using pip. Thanks to [HiGHS](https://github.com/ERGO-Code/HiGHS?tab=readme-ov-file), flixOpt can be used without further setup.
`pip install git+https://github.com/flixOpt/flixOpt.git`

We recommend installing flixOpt with all dependencies, which enables interactive network visualizations by [pyvis](https://github.com/WestHealth/pyvis) and time series aggregation by [tsam](https://github.com/FZJ-IEK3-VSA/tsam).
`pip install "flixOpt[full] @ git+https://github.com/flixOpt/flixOpt.git"`

---

## 🌟 Key Features and Concepts

### 💡 High-level Interface...
  - flixOpt aims to provide a user-friendly interface for defining and solving energy systems, without sacrificing fine-grained control where necessary.
  - This is achieved through a high-level interface with many optional or default parameters.
  - The most important concepts are:
    - **FlowSystem**: Represents the System that is modeled.
    - **Flow**: A Flow represents a stream of matter or energy. In an Energy-System, it could be electricity [kW]
    - **Bus**: A Bus represents a balancing node in the Energy-System, typically connecting a demand to a supply.
    - **Component**: A Component is a physical entity that consumes or produces matter or energy. It can also transform matter or energy into other kinds of matter or energy.
    - **Effect**: Flows and Components can have Effects, related to their usage (or size). Common effects are *costs*, *CO2-emissions*, *primary-energy-demand* or *area-demand*. One Effect is used as the optimization target. The others can be constrained.
  - To simplify the modeling process, high-level **Components** (CHP, Boiler, Heat Pump, Cooling Tower, Storage, etc.) are availlable.

### 🎛️ ...with low-level control
- **Segmented Linear Correlations**  
  - Accurate modeling for efficiencies, investment effects, and sizes.
- **On/Off Variables**
  - Modeling On/Off-Variables and their constraints.
    - On-Hours/Off-Hours
    - Consecutive On-Hours/ Off-Hours
    - Switch On/Off

### 💰 Investment Optimization
- flixOpt combines dispatch optimization with investment optimization in one model.
- Size and/or discrete investment decisions can be modeled
- Investment decisions can be combined with Modeling On/Off-Variables and their constraints

### Further Features
- **Multiple Effects**
  - Couple effects (e.g., specific CO2 costs) and set constraints (e.g., max CO2 emissions).
  - Easily switch between optimization targets (e.g., minimize CO2 or costs).
  - This allows to solve questions like "How much does it cost to reduce CO2 emissions by 20%?"

- **Advanced Time Handling**
  - Non-equidistant timesteps supported.  
  - Energy prices or effects in general can always be defined per hour (or per MWh...)

  - A variety of predefined constraints for operational and investment optimization can be applied.
  - Many of these are optional and only applied when necessary, keeping the amount o variables and equations low.

---

## 🖥️ Usage Example
![Usage Example](https://github.com/user-attachments/assets/fa0e12fa-2853-4f51-a9e2-804abbefe20c)

**Plotting examples**:
![flixOpt plotting](/pics/flixOpt_plotting.jpg)

## ⚙️ Calculation Modes

flixOpt offers three calculation modes, tailored to different performance and accuracy needs:

- **Full Mode**  
  - Provides exact solutions with high computational requirements.  
  - Recommended for detailed analyses and investment decision problems.

- **Segmented Mode**  
  - Solving a Model segmentwise, this mode can speed up the solving process for complex systems, while being fairly accurate.
  - Utilizes variable time overlap to improve accuracy.
  - Not suitable for large storage systems or investment decisions.

- **Aggregated Mode**  
  - Automatically generates typical periods using [TSAM](https://github.com/FZJ-IEK3-VSA/tsam).  
  - Balances speed and accuracy, making it ideal for large-scale simulations.


## 🏗️ Architecture

- **Minimal coupling to Pyomo**
  - Included independent module is used to organize variables and equations, independently of a specific modeling language.
  - While currently only working with [Pyomo](http://www.pyomo.org/), flixOpt is designed to work with different modeling languages with minor modifications ([cvxpy](https://www.cvxpy.org)).

- **File-based Post-Processing Unit**
  - Results are saved to .json and .yaml files for easy access and analysis anytime.
  - Internal plotting functions utilizing matplotlib, plotly and pandas simplify results visualization and reporting.

![Architecture Diagram](/pics/architecture_flixOpt.png)

---

## 🛠️ Solver Integration

By default, flixOpt uses the open-source solver [HiGHS](https://highs.dev/) which is installed by default. However, it is compatible with additional solvers such as:  

- [CBC](https://github.com/coin-or/Cbc)  
- [GLPK](https://www.gnu.org/software/glpk/)  
- [Gurobi](https://www.gurobi.com/)  
- [CPLEX](https://www.ibm.com/analytics/cplex-optimizer)

Executables can be found for example [here for CBC](https://portal.ampl.com/dl/open/cbc/) and [here for GLPK](https://sourceforge.net/projects/winglpk/) (Windows: You have to put solver-executables to the PATH-variable)

For detailed licensing and installation instructions, refer to the respective solver documentation.  

---

## 📖 Citation

If you use flixOpt in your research or project, please cite the following:  

- **Main Citation:** [DOI:10.18086/eurosun.2022.04.07](https://doi.org/10.18086/eurosun.2022.04.07)  
- **Short Overview:** [DOI:10.13140/RG.2.2.14948.24969](https://doi.org/10.13140/RG.2.2.14948.24969)  

---

## 🔧 Development and Testing

Run the tests using:  

```bash
python -m unittest discover -s tests
