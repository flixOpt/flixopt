# flixOpt Concepts & Mathematical Description

flixOpt is built around a set of core concepts that work together to represent and optimize energy and material flow systems. This page provides a high-level overview of these concepts and how they interact.

## Mathematical Notation & Naming Conventions

flixOpt uses the following naming conventions:

- All optimization variables are denoted by italic letters (e.g., $x$, $y$, $z$)
- All parameters and constants are denoted by non italic small letters (e.g., $\text{a}$, $\text{b}$, $\text{c}$)
- All Sets are denoted by greek capital letters (e.g., $\mathcal{F}$, $\mathcal{E}$)
- All units of a set are denoted by greek small letters (e.g., $\mathcal{f}$, $\mathcal{e}$)
- The letter $i$ is used to denote an index (e.g., $i=1,\dots,\text n$)
- All time steps are denoted by the letter $\text{t}$ (e.g., $\text{t}_0$, $\text{t}_1$, $\text{t}_i$)

## Core Concepts

### FlowSystem

The [`FlowSystem`][flixOpt.flow_system.FlowSystem] is the central organizing unit in flixOpt. 
Every flixOpt model starts with creating a FlowSystem. It:

- Defines the timesteps for the optimization
- Contains and connects [components](#components), [buses](#buses), and [flows](#flows)
- Manages the [effects](#effects) (objectives and constraints)

### Timesteps
Time steps are defined as a sequence of discrete time steps $\text{t}_i \in \mathcal{T} \quad \text{for} \quad i \in \{1, 2, \dots, \text{n}\}$ (left-aligned in its timespan).
From this sequence, the corresponding time intervals $\Delta \text{t}_i \in \Delta \mathcal{T}$ are derived as 

$$\Delta \text{t}_i = \text{t}_{i+1} - \text{t}_i \quad \text{for} \quad i \in \{1, 2, \dots, \text{n}-1\}$$

The final time interval $\Delta \text{t}_\text n$ defaults to $\Delta \text{t}_\text n = \Delta \text{t}_{\text n-1}$, but is of course customizable.
Non-equidistant time steps are also supported.

### Buses

[`Bus`][flixOpt.elements.Bus] objects represent nodes or connection points in a FlowSystem. They:

- Balance incoming and outgoing flows
- Can represent physical networks like heat, electricity, or gas 
- Handle infeasible balances gently by allowing the balance to be closed in return for a big Penalty (optional)

#### Mathematical Notation

The balance equation for a bus is:

$$ \label{eq:bus_balance}
  \sum_{f_\text{in} \in \mathcal{F}_\text{in}} p_{f_\text{in}}(\text{t}_i) =
  \sum_{f_\text{out} \in \mathcal{F}_\text{out}} p_{f_\text{out}}(\text{t}_i)
$$

Optionally, a Bus can have a `excess_penalty_per_flow_hour` parameter, which allows to penalize the balance for missing or excess flow-rates.
This is usefull as it handles a possible ifeasiblity gently.

This changes the balance to

$$ \label{eq:bus_balance-excess}
  \sum_{f_\text{in} \in \mathcal{F}_\text{in}} p_{f_ \text{in}}(\text{t}_i) + \phi_\text{in}(\text{t}_i) =
  \sum_{f_\text{out} \in \mathcal{F}_\text{out}} p_{f_\text{out}}(\text{t}_i) + \phi_\text{out}(\text{t}_i)
$$

The penalty term is defined as

$$ \label{eq:bus_penalty}
  s_{b \rightarrow \Phi}(\text{t}_i) =
      \text a_{b \rightarrow \Phi}(\text{t}_i) \cdot \Delta \text{t}_i
      \cdot [ \phi_\text{in}(\text{t}_i) + \phi_\text{out}(\text{t}_i) ]
$$

With:

- $\mathcal{F}_\text{in}$ and $\mathcal{F}_\text{out}$ being the set of all incoming and outgoing flows
- $p_{f_\text{in}}(\text{t}_i)$ and $p_{f_\text{out}}(\text{t}_i)$ being the flow-rate at time $\text{t}_i$ for flow $f_\text{in}$ and $f_\text{out}$, respectively
- $\phi_\text{in}(\text{t}_i)$ and $\phi_\text{out}(\text{t}_i)$ being the missing or excess flow-rate at time $\text{t}_i$, respectively
- $\text{t}_i$ being the time step
- $s_{b \rightarrow \Phi}(\text{t}_i)$ being the penalty term
- $\text a_{b \rightarrow \Phi}(\text{t}_i)$ being the penalty coefficient (`excess_penalty_per_flow_hour`)


### Flows

[`Flow`][flixOpt.elements.Flow] objects represent the movement of energy or material between components and buses. They:

- Have a size (fixed or part of an investment decision)
- Can have fixed profiles (for demands or renewable generation)
- Can have constraints (min/max, total flow hours, etc.)
- Can have [Effects](#effects) associated by their use (operation, investment, on/off, ...)

### Components

[`Component`][flixOpt.elements.Component] objects usually represent physical entities in your system that interact with [`Flows`][flixOpt.elements.Flow]. They include:

- [`LinearConverters`][flixOpt.components.LinearConverter] - Converts input flows to output flows with (piecewise) linear relationships
- [`Storages`][flixOpt.components.Storage] - Stores energy or material over time
- [`Sources`][flixOpt.components.Source] / [`Sinks`][flixOpt.components.Sink] / [`SourceAndSinks`][flixOpt.components.SourceAndSink] - Produce or consume flows. They are usually used to model external demands or supplies.
- [`Transmissions`][flixOpt.components.Transmission] - Moves flows between locations with possible losses
- Specialized [`LinearConverters`][flixOpt.components.LinearConverter] like [`Boilers`][flixOpt.linear_converters.Boiler], [`HeatPumps`][flixOpt.linear_converters.HeatPump], [`CHPs`][flixOpt.linear_converters.CHP], etc. These simplify the usage of the `LinearConverter` class and can also be used as blueprint on how to define custom classes or parameterize existing ones.

### Effects

[`Effect`][flixOpt.effects.Effect] objects represent impacts or metrics related to your system, such as:

- Costs (investment, operation)
- Emissions (CO₂, NOx, etc.)
- Resource consumption

These can be freely defined and crosslink to each other (`CO₂` ──[specific CO₂-costs]─→ `Costs`).
One effect is designated as the **optimization objective** (typically Costs), while others can have constraints.
This effect can incorporate several other effects, which woul result in a weighted objective from multiple effects.

### Calculation Modes

flixOpt offers different calculation approaches:

- [`FullCalculation`][flixOpt.calculation.FullCalculation] - Solves the entire problem at once
- [`SegmentedCalculation`][flixOpt.calculation.SegmentedCalculation] - Solves the problem in segments (with optioinal overlap), improving performance for large problems
- [`AggregatedCalculation`][flixOpt.calculation.AggregatedCalculation] - Uses typical periods to reduce computational requirements

## How These Concepts Work Together

1. You create a `FlowSystem` with a specified time series
2. You add elements to the FLowSystem:
    - `Bus` objects as connection points
    - `Component` objects like Boilers, Storages, etc.. They include `Flow` which define the connection to a Bus.
    - `Effect` objects to represent costs, emissions, etc.
3.You choose a calculation mode and solver
4.flixOpt converts your model into a mathematical optimization problem
5.The solver finds the optimal solution
6.You analyze the results with built-in or external tools

## Advanced Usage
flixOpt uses [linopy](https://github.com/PyPSA/linopy) to model the mathematical optimization problem.
Any model created with flixOpt can be extended or modified using the great [linopy API](https://linopy.readthedocs.io/en/latest/api.html).
This allows to adjust your model to very specific requirements without loosing the convenience of flixOpt.


## Architechture (outdated)
![Architecture](../images/architecture_flixOpt.png)


<!--## Next Steps-->
<!---->
<!--Now that you understand the basic concepts, learn more about each one:-->
<!---->
<!--- [FlowSystem](api/flow_system.md) - Time series and system organization-->
<!--- [Components](api/components.md) - Available component types and how to use them-->
<!--- [Effects](apieffects.md) - Costs, emissions, and other impacts-->
<!--- [Calculation Modes](api/calculation.md) - Different approaches to solving your model-->
