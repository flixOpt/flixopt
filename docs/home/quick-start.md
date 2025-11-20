# Quick Start

Get up and running with flixOpt in 5 minutes! This guide walks you through creating and solving your first energy system optimization.

## Installation

First, install flixOpt:

```bash
pip install flixopt
```

## Your First Model

Let's create a simple energy system with a generator, demand, and battery storage.

### 1. Import flixOpt

```python
import flixopt as fx
import numpy as np
```

### 2. Create a Time Series

```python
# 24-hour period with hourly timesteps
timesteps = 24
time_series = fx.TimeSeriesData(
    'hours',
    0,
    timesteps,
    1
)
```

### 3. Set Up the Flow System

```python
# Create the flow system
system = fx.FlowSystem(time_series)

# Define an effect to minimize (costs)
costs = fx.Effect('costs', 'EUR', 'Minimize total system costs')
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

solar = fx.Component('solar', inputs=[], outputs=[
    fx.Flow('power',
            bus=electricity_bus,
            size=100,  # 100 kW capacity
            relative_maximum=solar_profile,
            effects_per_flow_hour={costs: 0})  # Free solar
])

# Demand
demand_profile = np.array([30, 25, 20, 20, 25, 35, 50, 70, 80, 75,
                           70, 65, 60, 65, 70, 80, 90, 95, 85, 70,
                           60, 50, 40, 35])

demand = fx.Component('demand', inputs=[
    fx.Flow('consumption',
            bus=electricity_bus,
            size=1,
            fixed_relative_value=demand_profile)
], outputs=[])

# Battery storage
battery = fx.Storage(
    'battery',
    charging=fx.Flow('charge', bus=electricity_bus, size=50),
    discharging=fx.Flow('discharge', bus=electricity_bus, size=50),
    capacity_in_flow_hours=100,  # 100 kWh capacity
    initial_charge_state=0.5,  # Start at 50%
    eta_charge=0.95,
    eta_discharge=0.95,
    effects_per_flow_hour={costs: 0.01}  # Small battery wear cost
)

# Add all components to system
system.add_components(solar, demand, battery)
```

### 5. Run Optimization

```python
# Create calculation
calc = fx.FullCalculation('solar_battery_optimization', system)

# Solve
calc.solve()
```

### 6. View Results

```python
# Print some key results
print(f"Total costs: {calc.results.objective_value:.2f} EUR")

# Access time series results
solar_output = calc.results.get_component('solar').get_flow('power').flow_rate
battery_soc = calc.results.get_component('battery').charge_state

# Plot results (requires matplotlib)
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

ax1.plot(solar_output, label='Solar Output')
ax1.plot(demand_profile, label='Demand')
ax1.set_ylabel('Power (kW)')
ax1.legend()
ax1.grid(True)

ax2.plot(battery_soc)
ax2.set_ylabel('Battery SOC (kWh)')
ax2.set_xlabel('Hour')
ax2.grid(True)

plt.tight_layout()
plt.show()
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
7. **Analyze results** - Extract and visualize outcomes

## Tips

- Start simple and add complexity incrementally
- Use meaningful names for components and flows
- Check solver status before analyzing results
- Enable logging during development for debugging
- Visualize results to verify model behavior
