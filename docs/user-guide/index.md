# flixOpt Concepts

flixOpt is built around a set of core concepts that work together to represent and optimize energy and material flow systems. This page provides a high-level overview of these concepts and how they interact.

## Core Concepts

### FlowSystem

The [`FlowSystem`][flixOpt.flow_system.FlowSystem] is the central organizing unit in flixOpt. 
Every flixOpt model starts with creating a FlowSystem. It:

- Defines the timesteps for the optimization
- Contains and connects [components](#components), [buses](#buses), and [flows](#flows)
- Manages the [effects](#effects) (objectives and constraints)

### Flows

[`Flow`][flixOpt.elements.Flow] objects represent the movement of energy or material between a [Bus](#buses) and a [Component](#components) in a predefined direction.

- Have a `flow_rate`, which is the main optimization variable of a Flow
- Have a `size` which defines how much energy or material can be moved (fixed or part of an investment decision)
- Have constraints to limit the flow-rate (min/max, total flow hours, on/off etc.)
- Can have fixed profiles (for demands or renewable generation)
- Can have [Effects](#effects) associated by their use (operation, investment, on/off, ...)

### Buses

[`Bus`][flixOpt.elements.Bus] objects represent nodes or connection points in a FlowSystem. They:

- Balance incoming and outgoing flows
- Can represent physical networks like heat, electricity, or gas 
- Handle infeasible balances gently by allowing the balance to be closed in return for a big Penalty (optional)

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
- Area demand

These can be freely defined and crosslink to each other (`CO₂` ──[specific CO₂-costs]─→ `Costs`).
One effect is designated as the **optimization objective** (typically Costs), while others can be constrained.
This approach allows for a multi-criteria optimization using both...
 - ... the **Weigted Sum**Method, by Optimizing a theoretical Effect which other Effects crosslink to.
 - ... the ($\epsilon$-constraint method) by constraining effects.

### Calculation

A [`FlowSystem`][flixOpt.flow_system.FlowSystem] can be converted to a Model and optimized by creating a [`Calculation`][flixOpt.calculation.Calculation] from it.

flixOpt offers different calculation modes:

- [`FullCalculation`][flixOpt.calculation.FullCalculation] - Solves the entire problem at once
- [`SegmentedCalculation`][flixOpt.calculation.SegmentedCalculation] - Solves the problem in segments (with optioinal overlap), improving performance for large problems
- [`AggregatedCalculation`][flixOpt.calculation.AggregatedCalculation] - Uses typical periods to reduce computational requirements

### Results

The results of a calculation are stored in a [`CalculationResults`][flixOpt.results.CalculationResults] object.
This object contains the solutions of the optimization as well as all information about the [`Calculation`][flixOpt.calculation.Calculation] and the [`FlowSystem`][flixOpt.flow_system.FlowSystem] it was created from.
The solutions is stored as an `xarray.Dataset`, but can be accessed through their assotiated Component, Bus or Effect.

This [`CalculationResults`][flixOpt.results.CalculationResults] object can be saved to file and reloaded from file, allowing you to analyze the results anytime after the solve.

## How These Concepts Work Together

The process of working with flixOpt can be divided into 3 steps:

1. Create a [`FlowSystem`][flixOpt.flow_system.FlowSystem], containing all the elements and data of your system
     -  Define the time series of your system
     -  Add [`Components`][flixOpt.components] like [`Boilers`][flixOpt.linear_converters.Boiler], [`HeatPumps`][flixOpt.linear_converters.HeatPump], [`CHPs`][flixOpt.linear_converters.CHP], etc.
     -  Add [`Buses`][flixOpt.elements.Bus] as connection points in your system
     -  Add [`Effects`][flixOpt.effects.Effect] to represent costs, emissions, etc.
     - *This [`FlowSystem`][flixOpt.flow_system.FlowSystem] can also be loaded from a netCDF file*
2. Translate the model to a mathematical optimization problem
     - Create a [`Calculation`][flixOpt.calculation.Calculation] from your FlowSystem and choose a Solver
     - ...The Calculation is translated internaly to a mathematical optimization problem...
     - ...and solved by the chosen solver.
3. Analyze the results
     - The results are stored in a [`CalculationResults`][flixOpt.results.CalculationResults] object
     - This object can be saved to file and reloaded from file, retaining all information about the calculation
     - As it contains the used [`FlowSystem`][flixOpt.flow_system.FlowSystem], it can be used to start a new calculation

<figure markdown>
  ![flixOpt Conceptual Usage](../images/architecture_flixOpt.png)
  <figcaption>Conceptual Usage and IO operations of flixOpt</figcaption>
</figure>

## Advanced Usage
As flixopt is build on [linopy](https://github.com/PyPSA/linopy), any model created with flixOpt can be extended or modified using the great [linopy API](https://linopy.readthedocs.io/en/latest/api.html).
This allows to adjust your model to very specific requirements without loosing the convenience of flixOpt.

<!--## Next Steps-->
<!---->
<!--Now that you understand the basic concepts, learn more about each one:-->
<!---->
<!--- [FlowSystem](api/flow_system.md) - Time series and system organization-->
<!--- [Components](api/components.md) - Available component types and how to use them-->
<!--- [Effects](apieffects.md) - Costs, emissions, and other impacts-->
<!--- [Calculation Modes](api/calculation.md) - Different approaches to solving your model-->
