# Features

flixOpt is a comprehensive framework for modeling and optimizing energy and material flow systems. It provides a powerful set of features for both operational and investment optimization.

## Core Capabilities

### :material-cog: Optimization Types

- **Operational Optimization** - Dispatch optimization with given capacities
- **Investment Optimization** - Capacity expansion planning with binary or continuous sizing
- **Multi-Period Planning** - Sequential investment decisions over multiple periods
- **Two-Stage Optimization** - Separate investment and operational decisions

### :material-chart-line: Modeling Features

#### Components

- **Flow** - Energy or material transfer with variable flow rates
- **Bus** - Nodal balance point for connecting multiple flows
- **Storage** - Energy storage with charge/discharge dynamics and efficiency losses
- **LinearConverter** - Linear conversion relationships between flows

#### Advanced Features

- **Investment Parameters** - Binary or continuous capacity decisions with sizing constraints
- **On/Off Parameters** - Discrete operational states with minimum run/idle times
- **Piecewise Linearization** - Non-linear relationships approximated with piecewise segments
- **Effects** - System-wide tracking of costs, emissions, and other impacts

### :material-math-integral: Mathematical Formulation

- **Mixed-Integer Linear Programming (MILP)** - Exact optimization with commercial and open-source solvers
- **Time Series Support** - Native handling of time-indexed variables and parameters
- **Flexible Constraints** - User-defined custom constraints and objectives
- **Penalty Variables** - Soft constraint violations for infeasibility analysis

### :material-puzzle: Solvers

flixOpt supports multiple optimization solvers:

- **HiGHS** - Open-source, high-performance LP/MIP solver (default)
- **Gurobi** - Commercial solver with academic licenses available
- **CPLEX** - IBM's commercial optimization solver
- **GLPK** - GNU Linear Programming Kit

### :material-file-export: Data Handling

- **Flexible Input** - Python dictionaries, pandas DataFrames, or direct parameters
- **Time Series** - Native support for time-indexed data with automatic alignment
- **Results Export** - Comprehensive results in structured formats
- **Visualization** - Built-in plotting capabilities for results analysis

## Use Cases

### Energy Systems

- Power system dispatch and expansion planning
- Combined heat and power (CHP) optimization
- Renewable energy integration
- Battery storage optimization
- District heating networks

### Industrial Applications

- Process optimization with material flows
- Multi-commodity networks
- Supply chain optimization
- Resource allocation

## Performance

- **Scalable** - Handles systems from simple examples to large-scale energy systems
- **Efficient** - Optimized model building and constraint generation
- **Fast** - Modern solvers with warm-start capabilities
- **Flexible** - Modular design allows for easy extension and customization
