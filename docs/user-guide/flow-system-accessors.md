# FlowSystem Accessors

The `FlowSystem` class provides several accessor properties that give you convenient access to different aspects of your model. These accessors group related functionality and follow a consistent pattern.

## Overview

| Accessor | Purpose | When to Use |
|----------|---------|-------------|
| [`optimize`](#optimize) | Run optimization | After building your model |
| [`transform`](#transform) | Create transformed versions | Before optimization (e.g., clustering) |
| [`statistics`](#statistics) | Analyze optimization results | After optimization |
| [`topology`](#topology) | Inspect and visualize network structure | Anytime |

## optimize

The `optimize` accessor provides methods to run the optimization.

### Basic Usage

```python
import flixopt as fx

# Simple one-liner: build + solve
flow_system.optimize(fx.solvers.HighsSolver())

# Access results
print(flow_system.solution)
print(flow_system.components['Boiler'].solution)
```

### How It Works

Calling `flow_system.optimize(solver)` is equivalent to:

```python
flow_system.build_model()
flow_system.solve(solver)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `solver` | Solver | *required* | The solver to use (e.g., `HighsSolver`, `GurobiSolver`) |
| `normalize_weights` | bool | `True` | Normalize scenario/period weights to sum to 1 |

### Returns

Returns the `FlowSystem` itself for method chaining:

```python
solution = flow_system.optimize(solver).solution
```

---

## transform

The `transform` accessor provides methods to create transformed versions of your FlowSystem.

### Clustering

Create a time-aggregated version of your FlowSystem for faster optimization:

```python
# Define clustering parameters
params = fx.ClusteringParameters(
    hours_per_period=24,     # Hours per typical period
    nr_of_periods=8,         # Number of typical periods
)

# Create clustered FlowSystem
clustered_fs = flow_system.transform.cluster(params)

# Optimize the clustered version
clustered_fs.optimize(fx.solvers.HighsSolver())
```

### Available Methods

| Method | Description |
|--------|-------------|
| `cluster(parameters, components_to_clusterize=None)` | Create a time-clustered FlowSystem |

### Clustering Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `hours_per_period` | int | Duration of each typical period in hours |
| `nr_of_periods` | int | Number of typical periods to create |
| `fix_storage_flows` | bool | Whether to fix storage flows during clustering |
| `aggregate_data_and_fix_non_binary_vars` | bool | Whether to aggregate data |

---

## statistics

The `statistics` accessor provides aggregated data and plotting methods for optimization results.

!!! note
    The FlowSystem must have a solution (from `optimize()` or `solve()`) before using statistics.

### Data Properties

Access pre-computed aggregations as xarray Datasets:

```python
flow_system.optimize(solver)

# Get aggregated data
flow_system.statistics.flow_rates      # All flow rates
flow_system.statistics.flow_hours      # Flow hours (energy)
flow_system.statistics.sizes           # All flow sizes/capacities
flow_system.statistics.charge_states   # Storage charge states
flow_system.statistics.effects_per_component  # Effect breakdown by component
```

### Available Data Properties

| Property | Returns | Description |
|----------|---------|-------------|
| `flow_rates` | `xr.Dataset` | All flow rate variables |
| `flow_hours` | `xr.Dataset` | Flow hours (flow_rate × hours_per_timestep) |
| `sizes` | `xr.Dataset` | All size variables (fixed and optimized) |
| `charge_states` | `xr.Dataset` | Storage charge state variables |
| `effects_per_component` | `xr.Dataset` | Effect totals broken down by component |
| `effect_share_factors` | `dict` | Cross-effect conversion factors |

### Plotting

Access plotting methods through the nested `plot` accessor:

```python
# Balance plots
flow_system.statistics.plot.balance('HeatBus')
flow_system.statistics.plot.balance('Boiler', mode='area')

# Heatmaps
flow_system.statistics.plot.heatmap('Boiler(Q_th)|flow_rate')

# Line and bar charts
flow_system.statistics.plot.line('Battery|charge_state')
flow_system.statistics.plot.bar('costs', by='component')
```

### Available Plot Methods

| Method | Description |
|--------|-------------|
| `balance(node, ...)` | Energy/material balance at a bus or component |
| `heatmap(variables, ...)` | Time series heatmap with automatic reshaping |
| `line(variables, ...)` | Line chart of variables over time |
| `bar(variables, ...)` | Bar chart for comparison |

See [Plotting Results](results-plotting.md) for detailed documentation of all plot methods.

### Effect Analysis

Analyze effect contributions:

```python
# Get effect shares for a specific element
shares = flow_system.statistics.get_effect_shares(
    element='Boiler',
    effect='costs',
    mode='temporal',
    include_flows=True
)
```

---

## topology

The `topology` accessor provides methods to inspect and visualize the network structure.

### Inspecting Structure

Get node and edge information:

```python
# Get topology as dictionaries
nodes, edges = flow_system.topology.infos()

# nodes: {'Boiler': {'label': 'Boiler', 'class': 'Component', 'infos': '...'}, ...}
# edges: {'Boiler(Q_th)': {'label': 'Q_th', 'start': 'Boiler', 'end': 'Heat', 'infos': '...'}, ...}

print(f"Components and buses: {list(nodes.keys())}")
print(f"Flows: {list(edges.keys())}")
```

### Static Visualization

Generate an interactive HTML network diagram using PyVis:

```python
# Default: save to 'flow_system.html' and open in browser
flow_system.topology.plot()

# Custom path and options
flow_system.topology.plot(
    path='output/network.html',
    controls=['nodes', 'layout', 'physics'],
    show=True
)

# Create but don't save
network = flow_system.topology.plot(path=False, show=False)
```

### Plot Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | str, Path, or False | `'flow_system.html'` | Where to save the HTML file |
| `controls` | bool or list | `True` | UI controls to show |
| `show` | bool | `None` | Whether to open in browser |

Available controls: `'nodes'`, `'edges'`, `'layout'`, `'interaction'`, `'manipulation'`, `'physics'`, `'selection'`, `'renderer'`

### Interactive Visualization

Launch an interactive Dash/Cytoscape application for exploring the network:

```python
# Start the app (opens in browser)
flow_system.topology.start_app()

# ... interact with the visualization ...

# Stop the server when done
flow_system.topology.stop_app()
```

!!! note "Optional Dependencies"
    The interactive app requires additional dependencies:
    ```bash
    pip install flixopt[network_viz]
    # or
    pip install dash dash-cytoscape dash-daq networkx werkzeug
    ```

### Available Methods

| Method | Description |
|--------|-------------|
| `infos()` | Get node/edge dictionaries for the network |
| `plot(...)` | Generate static HTML visualization (PyVis) |
| `start_app()` | Start interactive visualization server (Dash) |
| `stop_app()` | Stop the visualization server |

---

## Accessor Pattern

All accessors follow a consistent pattern:

```python
# Access via property
accessor = flow_system.accessor_name

# Call methods on the accessor
result = accessor.method(...)

# Or chain directly
flow_system.accessor_name.method(...)
```

### Caching

- `statistics` is cached and invalidated when the solution changes
- `topology`, `optimize`, and `transform` create new instances each time

### Method Chaining

Many methods return the FlowSystem for chaining:

```python
# Chain optimization and access
solution = flow_system.optimize(solver).solution

# Chain transform and optimize
clustered_fs = flow_system.transform.cluster(params)
clustered_fs.optimize(solver)
```

## Complete Example

```python
import flixopt as fx
import pandas as pd

# Create FlowSystem
timesteps = pd.date_range('2024-01-01', periods=168, freq='h')
flow_system = fx.FlowSystem(timesteps)

# Add elements...
flow_system.add_elements(heat_bus, gas_bus, boiler, heat_pump)

# 1. Inspect topology before optimization
flow_system.topology.plot(path='system_structure.html')

# 2. Optionally transform for faster solving
clustered = flow_system.transform.cluster(
    fx.ClusteringParameters(hours_per_period=24, nr_of_periods=7)
)

# 3. Optimize
clustered.optimize(fx.solvers.HighsSolver())

# 4. Analyze results
print(clustered.statistics.flow_rates)
clustered.statistics.plot.balance('HeatBus')
clustered.statistics.plot.heatmap('Boiler(Q_th)|flow_rate')
```
