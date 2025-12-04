# Analyzing Results

After running an optimization, flixOpt provides powerful tools to access, analyze, and visualize your results.

## Accessing Solution Data

### Raw Solution

The `solution` property contains all optimization variables as an xarray Dataset:

```python
# Run optimization
flow_system.optimize(fx.solvers.HighsSolver())

# Access the full solution dataset
solution = flow_system.solution
print(solution)

# Access specific variables
print(solution['Boiler(Q_th)|flow_rate'])
print(solution['Battery|charge_state'])
```

### Element-Specific Solutions

Access solution data for individual elements:

```python
# Component solutions
boiler = flow_system.components['Boiler']
print(boiler.solution)  # All variables for this component

# Flow solutions
flow = flow_system.flows['Boiler(Q_th)']
print(flow.solution)

# Bus solutions (if imbalance is allowed)
bus = flow_system.buses['Heat']
print(bus.solution)
```

## Statistics Accessor

The `statistics` accessor provides pre-computed aggregations for common analysis tasks:

```python
# Access via the statistics property
stats = flow_system.statistics
```

### Available Data Properties

| Property | Description |
|----------|-------------|
| `flow_rates` | All flow rate variables as xarray Dataset |
| `flow_hours` | Flow hours (flow_rate × hours_per_timestep) |
| `sizes` | All size variables (fixed and optimized) |
| `charge_states` | Storage charge state variables |
| `temporal_effects` | Temporal effects per contributor per timestep |
| `periodic_effects` | Periodic (investment) effects per contributor |
| `total_effects` | Total effects (temporal + periodic) per contributor |
| `effect_share_factors` | Conversion factors between effects |

### Examples

```python
# Get all flow rates
flow_rates = flow_system.statistics.flow_rates
print(flow_rates)

# Get flow hours (energy)
flow_hours = flow_system.statistics.flow_hours
total_heat = flow_hours['Boiler(Q_th)'].sum()

# Get sizes (capacities)
sizes = flow_system.statistics.sizes
print(f"Boiler size: {sizes['Boiler(Q_th)'].values}")

# Get storage charge states
charge_states = flow_system.statistics.charge_states

# Get effect breakdown by contributor
temporal = flow_system.statistics.temporal_effects
print(temporal['costs'])  # Costs per contributor per timestep

# Group by component
temporal['costs'].groupby('component').sum()
```

### Effect Analysis

Analyze how effects (costs, emissions, etc.) are distributed:

```python
# Access effects via the new properties
stats = flow_system.statistics

# Temporal effects per timestep (costs, CO2, etc. per contributor)
stats.temporal_effects['costs']  # DataArray with dims [time, contributor]
stats.temporal_effects['costs'].sum('contributor')  # Total per timestep

# Periodic effects (investment costs, etc.)
stats.periodic_effects['costs']  # DataArray with dim [contributor]

# Total effects (temporal + periodic combined)
stats.total_effects['costs'].sum('contributor')  # Grand total

# Group by component or component type
stats.total_effects['costs'].groupby('component').sum()
stats.total_effects['costs'].groupby('component_type').sum()
```

## Plotting Results

The `statistics.plot` accessor provides visualization methods:

```python
# Balance plots
flow_system.statistics.plot.balance('HeatBus')
flow_system.statistics.plot.balance('Boiler')

# Heatmaps
flow_system.statistics.plot.heatmap('Boiler(Q_th)|flow_rate')

# Duration curves
flow_system.statistics.plot.duration_curve('Boiler(Q_th)')

# Sankey diagrams
flow_system.statistics.plot.sankey()
```

See [Plotting Results](../results-plotting.md) for comprehensive plotting documentation.

## Network Visualization

The `topology` accessor lets you visualize and inspect your system structure:

### Static HTML Visualization

Generate an interactive network diagram using PyVis:

```python
# Default: saves to 'flow_system.html' and opens in browser
flow_system.topology.plot()

# Custom options
flow_system.topology.plot(
    path='output/my_network.html',
    controls=['nodes', 'layout', 'physics'],
    show=True
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | str, Path, or False | `'flow_system.html'` | Where to save the HTML file |
| `controls` | bool or list | `True` | UI controls to show |
| `show` | bool | `None` | Whether to open in browser |

### Interactive App

Launch a Dash/Cytoscape application for exploring the network:

```python
# Start the visualization server
flow_system.topology.start_app()

# ... interact with the visualization in your browser ...

# Stop when done
flow_system.topology.stop_app()
```

!!! note "Optional Dependencies"
    The interactive app requires additional packages:
    ```bash
    pip install flixopt[network_viz]
    ```

### Network Structure Info

Get node and edge information programmatically:

```python
nodes, edges = flow_system.topology.infos()

# nodes: dict mapping labels to properties
# {'Boiler': {'label': 'Boiler', 'class': 'Component', 'infos': '...'}, ...}

# edges: dict mapping flow labels to properties
# {'Boiler(Q_th)': {'label': 'Q_th', 'start': 'Boiler', 'end': 'Heat', ...}, ...}

print(f"Components and buses: {list(nodes.keys())}")
print(f"Flows: {list(edges.keys())}")
```

## Saving and Loading

Save the FlowSystem (including solution) for later analysis:

```python
# Save to NetCDF (recommended for large datasets)
flow_system.to_netcdf('results/my_system.nc')

# Load later
loaded_fs = fx.FlowSystem.from_netcdf('results/my_system.nc')
print(loaded_fs.solution)

# Save to JSON (human-readable, smaller datasets)
flow_system.to_json('results/my_system.json')
loaded_fs = fx.FlowSystem.from_json('results/my_system.json')
```

## Working with xarray

All result data uses [xarray](https://docs.xarray.dev/), giving you powerful data manipulation:

```python
solution = flow_system.solution

# Select specific times
summer = solution.sel(time=slice('2024-06-01', '2024-08-31'))

# Aggregate over dimensions
daily_avg = solution.resample(time='D').mean()

# Convert to pandas
df = solution['Boiler(Q_th)|flow_rate'].to_dataframe()

# Export to various formats
solution.to_netcdf('full_solution.nc')
df.to_csv('boiler_flow.csv')
```

## Complete Example

```python
import flixopt as fx
import pandas as pd

# Build and optimize
timesteps = pd.date_range('2024-01-01', periods=168, freq='h')
flow_system = fx.FlowSystem(timesteps)
# ... add elements ...
flow_system.optimize(fx.solvers.HighsSolver())

# Visualize network structure
flow_system.topology.plot(path='system_network.html')

# Analyze results
print("=== Flow Statistics ===")
print(flow_system.statistics.flow_hours)

print("\n=== Effect Breakdown ===")
print(flow_system.statistics.total_effects)

# Create plots
flow_system.statistics.plot.balance('HeatBus')
flow_system.statistics.plot.heatmap('Boiler(Q_th)|flow_rate')

# Save for later
flow_system.to_netcdf('results/optimized_system.nc')
```

## Next Steps

- [Plotting Results](../results-plotting.md) - Detailed plotting documentation
- [Examples](../../examples/index.md) - Working code examples
