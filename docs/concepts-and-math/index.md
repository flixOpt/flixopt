# flixOpt Concepts & Mathematical Description

flixOpt is built around a set of core concepts that work together to represent and optimize energy and material flow systems. This page provides a high-level overview of these concepts and how they interact.

## Core Concepts

### FlowSystem

The [`FlowSystem`][flixOpt.flow_system.FlowSystem] is the central organizing unit in flixOpt. It:

- Defines the time series for the simulation
- Contains all components, buses, and flows
- Manages the effects (objectives and constraints)

Every flixOpt model starts with creating a FlowSystem.

### Timesteps
Time steps are defined as a sequence of discrete time steps $\text{t}_i \in \mathcal{T} \text{for} \quad i \in \{1, 2, \dots, \text{n}\}$ (left-aligned in its timespan).
From this sequence, the corresponding time intervals $\Delta \text{t}_i \in \Delta \mathcal{T}$ are derived as 

$$\Delta \text{t}_i = \text{t}_{i+1} - \text{t}_i \quad \text{for} \quad i \in \{1, 2, \dots, \text{n}-1\}$$

Non-equidistant time steps are supported. 
The final time interval $\Delta \text{t}_\text n$ defaults to $\Delta \text{t}_\text n = \Delta \text{t}_{\text n-1}$, but is of course customizable.


### Buses

[`Bus`][flixOpt.elements.Bus] objects represent nodes or connection points in your system. They:

- Balance incoming and outgoing flows
- Can represent physical networks like heat, electricity, or gas 
- Handle infeasable balances gently by allowing the balance to be closed in return for a big Penalty (optional)

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


## Mathematical Notation & Naming Conventions

flixOpt uses the following naming conventions:

- All optimization variables are denoted by italic letters (e.g., $x$, $y$, $z$)
- All parameters and constants are denoted by non italic small letters (e.g., $\text{a}$, $\text{b}$, $\text{c}$)
- The letter $i$ is used to denote an index (e.g., $i=1,\dots,\text n$)
- All time steps are denoted by the letter $\text{t}$ (e.g., $\text{t}_0$, $\text{t}_1$, $\text{t}_i$)


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
