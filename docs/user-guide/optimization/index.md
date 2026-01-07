# Running Optimizations

This section covers how to run optimizations in flixOpt, including different optimization modes and solver configuration.

## Verifying Your Model

Before running an optimization, it's helpful to visualize your system structure:

```python
# Generate an interactive network diagram
flow_system.topology.plot(path='my_system.html')

# Or get structure info programmatically
nodes, edges = flow_system.topology.infos()
print(f"Components: {[n for n, d in nodes.items() if d['class'] == 'Component']}")
print(f"Buses: {[n for n, d in nodes.items() if d['class'] == 'Bus']}")
print(f"Flows: {list(edges.keys())}")
```

## Standard Optimization

The recommended way to run an optimization is directly on the `FlowSystem`:

```python
import flixopt as fx

# Simple one-liner
flow_system.optimize(fx.solvers.HighsSolver())

# Access results directly
print(flow_system.solution['Boiler(Q_th)|flow_rate'])
print(flow_system.components['Boiler'].solution)
```

For more control over the optimization process, you can split model building and solving:

```python
# Build the model first
flow_system.build_model()

# Optionally inspect or modify the model
print(flow_system.model.constraints)

# Then solve
flow_system.solve(fx.solvers.HighsSolver())
```

**Best for:**

- Small to medium problems
- When you need the globally optimal solution
- Problems without time-coupling simplifications

## Clustered Optimization

For large problems, use time series clustering to reduce computational complexity:

```python
# Cluster to 12 typical days
fs_clustered = flow_system.transform.cluster(
    n_clusters=12,
    cluster_duration='1D',
    time_series_for_high_peaks=['HeatDemand(Q)|fixed_relative_profile'],
)

# Optimize the clustered system
fs_clustered.optimize(fx.solvers.HighsSolver())

# Expand back to full resolution
fs_expanded = fs_clustered.transform.expand()
```

**Best for:**

- Investment planning problems
- Year-long optimizations
- When computational speed is critical

**Trade-offs:**

- Much faster solve times
- Approximates the full problem
- Best when patterns repeat (e.g., typical days)

See the **[Clustering Guide](clustering.md)** for details on storage modes, peak selection, and multi-dimensional support.

## Choosing an Optimization Mode

| Mode | Problem Size | Solve Time | Solution Quality |
|------|-------------|------------|------------------|
| Standard | Small-Medium | Slow | Optimal |
| Clustered | Very Large | Fast | Approximate |

## Transform Accessor

The `transform` accessor provides methods to create modified copies of your FlowSystem. All transform methods return a **new FlowSystem without a solution** — you must re-optimize the transformed system.

### Selecting Subsets

Select a subset of your data by label or index:

```python
# Select by label (like xarray.sel)
fs_january = flow_system.transform.sel(time=slice('2024-01-01', '2024-01-31'))
fs_scenario = flow_system.transform.sel(scenario='base')

# Select by integer index (like xarray.isel)
fs_first_week = flow_system.transform.isel(time=slice(0, 168))
fs_first_scenario = flow_system.transform.isel(scenario=0)

# Re-optimize the subset
fs_january.optimize(fx.solvers.HighsSolver())
```

### Resampling Time Series

Change the temporal resolution of your FlowSystem:

```python
# Resample to 4-hour intervals
fs_4h = flow_system.transform.resample(time='4h', method='mean')

# Resample to daily
fs_daily = flow_system.transform.resample(time='1D', method='mean')

# Re-optimize with new resolution
fs_4h.optimize(fx.solvers.HighsSolver())
```

**Available resampling methods:** `'mean'`, `'sum'`, `'max'`, `'min'`, `'first'`, `'last'`

### Clustering

See the **[Clustering Guide](clustering.md)** for comprehensive documentation.

### Use Cases

| Method | Use Case |
|--------|----------|
| `sel()` / `isel()` | Analyze specific time periods, scenarios, or periods |
| `resample()` | Reduce problem size, test at lower resolution |
| `cluster()` | Investment planning with typical periods |

## Custom Constraints

flixOpt is built on [linopy](https://github.com/PyPSA/linopy), allowing you to add custom constraints beyond what's available through the standard API.

### Adding Custom Constraints

To add custom constraints, build the model first, then access the underlying linopy model:

```python
# Build the model (without solving)
flow_system.build_model()

# Access the linopy model
model = flow_system.model

# Access variables from the solution namespace
# Variables are named: "ElementLabel|variable_name"
boiler_flow = model.variables['Boiler(Q_th)|flow_rate']
chp_flow = model.variables['CHP(Q_th)|flow_rate']

# Add a custom constraint: Boiler must produce at least as much as CHP
model.add_constraints(
    boiler_flow >= chp_flow,
    name='boiler_min_chp'
)

# Solve with the custom constraint
flow_system.solve(fx.solvers.HighsSolver())
```

### Common Use Cases

**Minimum runtime constraint:**
```python
# Require component to run at least 100 hours total
on_var = model.variables['CHP|on']  # Binary on/off variable
hours = flow_system.hours_per_timestep
model.add_constraints(
    (on_var * hours).sum() >= 100,
    name='chp_min_runtime'
)
```

**Linking flows across components:**
```python
# Heat pump and boiler combined must meet minimum base load
hp_flow = model.variables['HeatPump(Q_th)|flow_rate']
boiler_flow = model.variables['Boiler(Q_th)|flow_rate']
model.add_constraints(
    hp_flow + boiler_flow >= 50,  # At least 50 kW combined
    name='min_heat_supply'
)
```

**Seasonal constraints:**
```python
import pandas as pd

# Different constraints for summer vs winter
summer_mask = flow_system.timesteps.month.isin([6, 7, 8])
winter_mask = flow_system.timesteps.month.isin([12, 1, 2])

flow_var = model.variables['Boiler(Q_th)|flow_rate']

# Lower capacity in summer
model.add_constraints(
    flow_var.sel(time=flow_system.timesteps[summer_mask]) <= 100,
    name='summer_limit'
)
```

### Inspecting the Model

Before adding constraints, inspect available variables and existing constraints:

```python
flow_system.build_model()
model = flow_system.model

# List all variables
print(model.variables)

# List all constraints
print(model.constraints)

# Get details about a specific variable
print(model.variables['Boiler(Q_th)|flow_rate'])
```

### Variable Naming Convention

Variables follow this naming pattern:

| Element Type | Pattern | Example |
|--------------|---------|---------|
| Flow rate | `Component(FlowLabel)\|flow_rate` | `Boiler(Q_th)\|flow_rate` |
| Flow size | `Component(FlowLabel)\|size` | `Boiler(Q_th)\|size` |
| On/off status | `Component\|on` | `CHP\|on` |
| Charge state | `Storage\|charge_state` | `Battery\|charge_state` |
| Effect totals | `effect_name\|total` | `costs\|total` |

## Solver Configuration

### Available Solvers

| Solver | Type | Speed | License |
|--------|------|-------|---------|
| **HiGHS** | Open-source | Fast | Free |
| **Gurobi** | Commercial | Fastest | Academic/Commercial |
| **CPLEX** | Commercial | Fastest | Academic/Commercial |
| **GLPK** | Open-source | Slower | Free |

**Recommendation:** Start with HiGHS (included by default). Use Gurobi/CPLEX for large models or when speed matters.

### Solver Options

```python
# Basic usage with defaults
flow_system.optimize(fx.solvers.HighsSolver())

# With custom options
flow_system.optimize(
    fx.solvers.GurobiSolver(
        time_limit_seconds=3600,
        mip_gap=0.01,
        extra_options={
            'Threads': 4,
            'Presolve': 2
        }
    )
)
```

Common solver parameters:

- `time_limit_seconds` - Maximum solve time
- `mip_gap` - Acceptable optimality gap (0.01 = 1%)
- `log_to_console` - Show solver output

## Performance Tips

### Model Size Reduction

- Use longer timesteps where acceptable
- Use `flow_system.transform.cluster()` for long horizons
- Remove unnecessary components
- Simplify constraint formulations

### Solver Tuning

- Enable presolve and cuts
- Adjust optimality tolerances for faster (approximate) solutions
- Use parallel threads when available

### Problem Formulation

- Avoid unnecessary binary variables
- Use continuous investment sizes when possible
- Tighten variable bounds
- Remove redundant constraints

## Debugging

### Infeasibility

If your model has no feasible solution:

1. **Enable excess penalties on buses** to allow balance violations:
   ```python
   # Allow imbalance with high penalty cost (default is 1e5)
   heat_bus = fx.Bus('Heat', excess_penalty_per_flow_hour=1e5)

   # Or disable penalty to enforce strict balance
   electricity_bus = fx.Bus('Electricity', excess_penalty_per_flow_hour=None)
   ```
   When `excess_penalty_per_flow_hour` is set, the optimization can violate bus balance constraints by paying a penalty, helping identify which constraints cause infeasibility.

2. **Use Gurobi for infeasibility analysis** - When using GurobiSolver and the model is infeasible, flixOpt automatically extracts and logs the Irreducible Inconsistent Subsystem (IIS):
   ```python
   # Gurobi provides detailed infeasibility analysis
   flow_system.optimize(fx.solvers.GurobiSolver())
   # If infeasible, check the model documentation file for IIS details
   ```
   The infeasible constraints are saved to the model documentation file in the results folder.

3. Check balance constraints - can supply meet demand?
4. Verify capacity limits are consistent
5. Review storage state requirements
6. Simplify model to isolate the issue

See [Troubleshooting](../troubleshooting.md) for more details.

### Unexpected Results

If solutions don't match expectations:

1. Verify input data (units, scales)
2. Enable logging: `fx.CONFIG.exploring()`
3. Visualize intermediate results
4. Start with a simpler model
5. Check constraint formulations

## Next Steps

- See [Examples](../../notebooks/index.md) for working code
- Learn about [Mathematical Notation](../mathematical-notation/index.md)
- Explore [Recipes](../recipes/index.md) for common patterns
