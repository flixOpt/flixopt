# Quick Start

Get up and running with flixOpt in 5 minutes! This guide walks you through creating and solving your first energy system optimization.

## Installation

First, install flixOpt:

```bash
pip install "flixopt[full]"
```

## Your First Model

Let's create a simple energy system with a generator, demand, and battery storage.

### 1. Import flixOpt

```python
import flixopt as fx
import numpy as np
import pandas as pd
```

### 2. Define your time horizon

```python
# 24h period with hourly timesteps
timesteps = pd.date_range('2024-01-01', periods=24, freq='h')
```

### 2. Set Up the Flow System

```python
# Create the flow system
flow_system = fx.FlowSystem(timesteps)

# Define an effect to minimize (costs)
costs = fx.Effect('costs', 'EUR', 'Minimize total system costs', is_objective=True)
flow_system.add_elements(costs)
```

### 4. Add Components

```python
# Electricity bus
electricity_bus = fx.Bus('electricity')

# Solar generator with time-varying output
solar_profile = np.array([0, 0, 0, 0, 0, 0, 0.2, 0.5, 0.8, 1.0,
                          1.0, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1, 0,
                          0, 0, 0, 0, 0, 0])

solar = fx.Port(
    'solar',
    imports=[fx.Flow(
        bus='electricity',
        size=100,  # 100 kW capacity
        relative_maximum=solar_profile
    )
])

# Demand
demand_profile = np.array([30, 25, 20, 20, 25, 35, 50, 70, 80, 75,
                           70, 65, 60, 65, 70, 80, 90, 95, 85, 70,
                           60, 50, 40, 35])

demand = fx.Port('demand', exports=[
    fx.Flow(bus='electricity',
            size=1,
            fixed_relative_profile=demand_profile)
])

# Battery storage
battery = fx.Storage(
    'battery',
    charging=fx.Flow(bus='electricity', size=50),
    discharging=fx.Flow(bus='electricity', size=50),
    capacity_in_flow_hours=100,  # 100 kWh capacity
    initial_charge_state=50,  # Start at 50%
    eta_charge=0.95,
    eta_discharge=0.95,
)

# Add all components to system
flow_system.add_elements(solar, demand, battery, electricity_bus)
```

### 5. Visualize and Run Optimization

```python
# Optional: visualize your system structure
flow_system.topology.plot(path='system.html')

# Run optimization
flow_system.optimize(fx.solvers.HighsSolver())
```

### 6. Access and Visualize Results

```python
# Access raw solution data
print(flow_system.solution)

# Use statistics for aggregated data
print(flow_system.statistics.flow_hours)

# Access component-specific results
print(flow_system.components['battery'].solution)

# Visualize results
flow_system.statistics.plot.balance('electricity')
flow_system.statistics.plot.storage('battery')
```

### 7. Save Results (Optional)

```python
# Save the flow system (includes inputs and solution)
flow_system.to_netcdf('results/solar_battery.nc')

# Load it back later
loaded_fs = fx.FlowSystem.from_netcdf('results/solar_battery.nc')
```

## What's Next?

Now that you've created your first model, you can:

- **Learn the concepts** - Read the [Core Concepts](../user-guide/core-concepts.md) guide
- **Explore examples** - Check out more [Examples](../notebooks/index.md)
- **Deep dive** - Study the [Mathematical Formulation](../user-guide/mathematical-notation/index.md)
- **Build complex models** - Use [Recipes](../user-guide/recipes/index.md) for common patterns

## Common Workflow

Most flixOpt projects follow this pattern:

1. **Define time series** - Set up the temporal resolution
2. **Create flow system** - Initialize with time series and effects
3. **Add buses** - Define connection points
4. **Add components** - Create generators, storage, converters, loads
5. **Verify structure** - Use `flow_system.topology.plot()` to visualize
6. **Run optimization** - Call `flow_system.optimize(solver)`
7. **Analyze results** - Via `flow_system.statistics` and `.solution`
8. **Visualize** - Use `flow_system.statistics.plot.*` methods

## Tips

- Start simple and add complexity incrementally
- Use meaningful names for components and flows
- Check solver status before analyzing results
- Enable logging during development for debugging
- Visualize results to verify model behavior
