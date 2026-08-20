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

!!! tip "Contributors"
    Contributors are automatically detected from the optimization solution and include:

    - **Flows**: Individual flows with `effects_per_flow_hour`
    - **Components**: Components with `effects_per_active_hour` or similar direct effects

    Each contributor has associated metadata (`component` and `component_type` coordinates) for flexible groupby operations.

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

# Effects breakdown
flow_system.statistics.plot.effects()  # Total costs by component
flow_system.statistics.plot.effects(effect='costs', by='contributor')  # By individual flows
flow_system.statistics.plot.effects(aspect='temporal', by='time')  # Over time
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

## Comparing Multiple Systems

Use the [`Comparison`][flixopt.comparison.Comparison] class to analyze and visualize multiple FlowSystems side-by-side. This is useful for:

- Comparing different design alternatives (with/without CHP, different storage sizes)
- Analyzing optimization method trade-offs (full vs. two-stage, different aggregation levels)
- Sensitivity analysis (different scenarios, parameter variations)

### Basic Usage

```python
import flixopt as fx

# Optimize two system variants
fs_baseline = create_system()
fs_baseline.name = 'Baseline'
fs_baseline.optimize(solver)

fs_with_storage = create_system_with_storage()
fs_with_storage.name = 'With Storage'
fs_with_storage.optimize(solver)

# Create comparison
comp = fx.Comparison([fs_baseline, fs_with_storage])

# Side-by-side balance plots (auto-faceted by 'case' dimension)
comp.statistics.plot.balance('Heat')

# Access combined data with 'case' dimension
comp.statistics.flow_rates  # xr.Dataset with dims: (time, case)
comp.solution  # Combined solution dataset
```

### Requirements

All FlowSystems must have **matching core dimensions** (`time`, `period`, `scenario`). Auxiliary dimensions like `cluster_boundary` are ignored. If core dimensions differ, use `.transform.sel()` to align them first:

```python
# Systems with different scenarios
fs_both = flow_system  # Has 'Mild Winter' and 'Harsh Winter' scenarios
fs_mild = flow_system.transform.sel(scenario='Mild Winter')  # Single scenario

# Cannot compare directly - scenario dimension mismatch!
# fx.Comparison([fs_both, fs_mild])  # Raises ValueError

# Instead, select matching dimensions
fs_both_mild = fs_both.transform.sel(scenario='Mild Winter')
comp = fx.Comparison([fs_both_mild, fs_mild])  # Works!

# Auxiliary dimensions are OK (e.g., expanded clustered solutions)
fs_expanded = fs_clustered.transform.expand()  # Has cluster_boundary dim
comp = fx.Comparison([fs_full, fs_expanded])  # Works! cluster_boundary is ignored
```

!!! note "Component Differences"
    Systems can have different components. The Comparison aligns data where possible,
    and variables unique to specific systems will be `NaN` for others. This is useful
    for comparing scenarios like "with vs. without storage" where one system has
    Storage components and the other doesn't.

### Available Properties

The `Comparison.statistics` accessor mirrors all `StatisticsAccessor` properties, returning combined datasets with an added `'case'` dimension:

| Property | Description |
|----------|-------------|
| `flow_rates` | All flow rate variables |
| `flow_hours` | Flow hours (energy) |
| `sizes` | Component sizes |
| `storage_sizes` | Storage capacities |
| `charge_states` | Storage charge states |
| `temporal_effects` | Effects per timestep |
| `periodic_effects` | Investment effects |
| `total_effects` | Combined effects |

### Available Plot Methods

All standard plot methods work on the comparison, with the `'case'` dimension automatically used for faceting:

```python
comp = fx.Comparison([fs_baseline, fs_modified])

# Balance plots - faceted by case
comp.statistics.plot.balance('Heat')
comp.statistics.plot.balance('Electricity', mode='area')

# Flow plots
comp.statistics.plot.flows(component='CHP')

# Effect breakdowns
comp.statistics.plot.effects()

# Heatmaps
comp.statistics.plot.heatmap('Boiler(Q_th)')

# Duration curves
comp.statistics.plot.duration_curve('CHP(Q_th)')

# Storage plots
comp.statistics.plot.storage('Battery')
```

### Computing Differences

Use the `diff()` method to compute differences relative to a reference case:

```python
# Differences relative to first case (default)
differences = comp.diff()

# Differences relative to specific case
differences = comp.diff(reference='Baseline')
differences = comp.diff(reference=0)  # By index

# Analyze differences
print(differences['costs'])  # Cost difference per case
```

### Naming Systems

System names come from `FlowSystem.name` by default. Override with the `names` parameter:

```python
# Using FlowSystem.name (default)
fs1.name = 'Scenario A'
fs2.name = 'Scenario B'
comp = fx.Comparison([fs1, fs2])

# Or override explicitly
comp = fx.Comparison([fs1, fs2], names=['Base Case', 'Alternative'])
```

### Example: Comparing Optimization Methods

```python
# Full optimization
fs_full = flow_system.copy()
fs_full.name = 'Full Optimization'
fs_full.optimize(solver)

# Two-stage optimization
fs_sizing = flow_system.transform.resample('4h')
fs_sizing.optimize(solver)
fs_dispatch = flow_system.transform.fix_sizes(fs_sizing.statistics.sizes)
fs_dispatch.name = 'Two-Stage'
fs_dispatch.optimize(solver)

# Compare results
comp = fx.Comparison([fs_full, fs_dispatch])
comp.statistics.plot.balance('Heat')

# Check cost difference
diff = comp.diff()
print(f"Cost difference: {diff['costs'].sel(case='Two-Stage').item():.0f} €")
```

## Next Steps

- [Plotting Results](../results-plotting.md) - Detailed plotting documentation
- [Examples](../../notebooks/index.md) - Working code examples
