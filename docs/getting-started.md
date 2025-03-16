# Getting Started with flixOpt

This guide will help you install flixOpt, understand its basic concepts, and run your first optimization model.

## Installation

### Basic Installation

Install flixOpt directly into your environment using pip:

```bash
pip install git+https://github.com/flixOpt/flixOpt.git
```

This provides the core functionality with the HiGHS solver included.

### Full Installation

For all features including interactive network visualizations and time series aggregation:

```bash
pip install "flixOpt[full] @ git+https://github.com/flixOpt/flixOpt.git"
```

### Development Installation

For development purposes, clone the repository and install in editable mode:

```bash
git clone https://github.com/flixOpt/flixOpt.git
cd flixOpt
pip install -e ".[full]"
```

## Basic Workflow

Working with flixOpt follows a general pattern:

1. **Create a FlowSystem** with a time series
2. **Define Buses** as connection points in your system
3. **Create Flows** to represent energy/material streams
4. **Add Components** like converters, storage, sources/sinks
5. **Define Effects** (costs, emissions, etc.)
6. **Run Calculations** to optimize your system
7. **Analyze Results** using built-in visualization tools

## Simple Example

Here's a minimal example of a simple system with a heat demand and a boiler:

```python
import flixOpt as fo
import numpy as np

# Create time steps - hourly for one day
time_series = fo.create_datetime_array('2023-01-01', steps=24, freq='1h')
system = fo.FlowSystem(time_series)

# Create buses as connection points
heat_bus = fo.Bus("Heat")
fuel_bus = fo.Bus("Fuel")

# Create a demand profile (sine wave + base load)
heat_demand_profile = 100 * np.sin(np.linspace(0, 2*np.pi, 24))**2 + 50

# Create flows connecting to buses
heat_demand = fo.Flow(
    label="heat_demand", 
    bus=heat_bus,
    fixed_relative_profile=heat_demand_profile  # Fixed demand profile
)

fuel_supply = fo.Flow(
    label="fuel_supply",
    bus=fuel_bus
)

heat_output = fo.Flow(
    label="heat_output",
    bus=heat_bus
)

# Create a boiler component
boiler = fo.linear_converters.Boiler(
    label="Boiler",
    eta=0.9,  # 90% efficiency
    Q_fu=fuel_supply,
    Q_th=heat_output
)

# Create a sink for the heat demand
heat_sink = fo.Sink(
    label="Heat Demand",
    sink=heat_demand
)

# Add effects (costs)
fuel_cost = fo.Effect(
    label="costs",
    unit="€",
    description="Operational costs",
    is_objective=True  # This effect will be minimized
)

# Add elements to the system
system.add_effects(fuel_cost)
system.add_components(boiler, heat_sink)

# Run optimization
calculation = fo.FullCalculation("Simple_Example", system)
solver = fo.HighsSolver()  # Using the default solver

# Optimize the system
calculation.do_modeling()
calculation.solve(solver, save_results=True)

# Print results summary
print(f"Objective value: {calculation.system_model.result_of_objective}")
```

## Visualization

flixOpt includes tools to visualize your results. Here's a simple example to plot flow rates:

```python
import flixOpt.results as results

# Load results from a previous calculation
result = results.CalculationResults("Simple_Example", "results")

# Plot heat flows
result.plot_operation("Heat", mode="area", show=True)
```

## Next Steps

Now that you've installed flixOpt and understand the basic workflow, you can:

- Learn about the [core concepts](concepts/overview.md)
- Explore more complex [examples](examples/basic.md)
- Check the [API reference](api/flow-system.md) for detailed documentation

For more in-depth guidance, continue to the [Concepts](concepts/overview.md) section.
