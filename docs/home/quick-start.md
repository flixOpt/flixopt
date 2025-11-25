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
system.add_effects(costs)
```

### 4. Add Components

```python
# Electricity bus
electricity_bus = fx.Bus('electricity', 'kW')

# Solar generator with time-varying output
solar_profile = np.array([0, 0, 0, 0, 0, 0, 0.2, 0.5, 0.8, 1.0,
                          1.0, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1, 0,
                          0, 0, 0, 0, 0, 0])

solar = fx.Source(
    'solar',
    outputs=[fx.Flow(
        'power',
        bus=electricity_bus,
        size=100,  # 100 kW capacity
        relative_maximum=solar_profile
    )
])

# Demand
demand_profile = np.array([30, 25, 20, 20, 25, 35, 50, 70, 80, 75,
                           70, 65, 60, 65, 70, 80, 90, 95, 85, 70,
                           60, 50, 40, 35])

demand = fx.Sink('demand', inputs=[
    fx.Flow('consumption',
            bus=electricity_bus,
            size=1,
            fixed_relative_value=demand_profile)
])

# Battery storage
battery = fx.Storage(
    'battery',
    charging=fx.Flow('charge', bus=electricity_bus, size=50),
    discharging=fx.Flow('discharge', bus=electricity_bus, size=50),
    capacity_in_flow_hours=100,  # 100 kWh capacity
    initial_charge_state=50,  # Start at 50%
    eta_charge=0.95,
    eta_discharge=0.95,
    effects_per_flow_hour={costs: 0.01}  # Small battery wear cost
)

# Add all components to system
flow_system.add_components(solar, demand, battery)
```

### 5. Run Optimization

```python
# Create and run optimization
calc = fx.Optimization('solar_battery_optimization', system).solve()
```

### 6. Save Results

```python
# This includes the modeled FlowSystem. SO you can restore both results and inputs
calc.results.to_file()
```

## What's Next?

Now that you've created your first model, you can:

- **Learn the concepts** - Read the [Core Concepts](../user-guide/core-concepts.md) guide
- **Explore examples** - Check out more [Examples](../examples/index.md)
- **Deep dive** - Study the [Mathematical Formulation](../user-guide/mathematical-notation/index.md)
- **Build complex models** - Use [Recipes](../user-guide/recipes/index.md) for common patterns

## Common Workflow

Most flixOpt projects follow this pattern:

1. **Define time series** - Set up the temporal resolution
2. **Create flow system** - Initialize with time series and effects
3. **Add buses** - Define connection points
4. **Add components** - Create generators, storage, converters, loads
5. **Configure flows** - Set capacity, bounds, and cost parameters
6. **Run calculation** - Solve the optimization
7. **Save Results** - For later analysis. Or only extract needed data

## Tips

- Start simple and add complexity incrementally
- Use meaningful names for components and flows
- Check solver status before analyzing results
- Enable logging during development for debugging
- Visualize results to verify model behavior
