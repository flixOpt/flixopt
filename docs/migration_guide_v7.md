# Migration Guide: flixopt v7

This guide covers migrating to flixopt v7, which introduces a new batched/vectorized architecture bringing significant performance improvements and powerful new analytical capabilities.

## TL;DR

**For most users: No changes required!** The public API (`FlowSystem`, `add_elements`, `solve`) is unchanged.

**If you access `model.variables` or `model.constraints` directly:**
```python
# v6 and earlier
rate = model.variables['Boiler(Q_th)|rate']

# v7
rate = model.variables['flow|rate'].sel(flow='Boiler(Q_th)')
```

**What you get:**
- 7-28x faster model building
- 4-13x faster LP file writing
- Native xarray operations for analysis

---

## What's New in v7

### Architecture Overview

**v6 and earlier:** One model instance per element
- `FlowModel` per Flow, `StorageModel` per Storage
- Result: Hundreds of separate variables (`Flow1|rate`, `Flow2|rate`, ...)

**v7:** One model instance per element *type*
- `FlowsModel` for ALL flows, `StoragesModel` for ALL storages
- Result: Few batched variables with element dimension (`flow|rate` with coords)

```
OLD: 859 variables, 997 constraints (720h, 50 converters)
NEW:  21 variables,  30 constraints (same system)
```

### Variable Naming Convention

| Old Pattern | New Pattern |
|-------------|-------------|
| `Boiler(Q_th)|rate` | `flow|rate` with coord `flow='Boiler(Q_th)'` |
| `Boiler(Q_th)|status` | `flow|status` with coord `flow='Boiler(Q_th)'` |
| `Boiler(Q_th)|size` | `flow|size` with coord `flow='Boiler(Q_th)'` |
| `HeatStorage|charge_state` | `storage|charge_state` with coord `storage='HeatStorage'` |

---

## Breaking Changes

### 1. Variable Access Pattern

```python
# v6 - Direct name lookup
rate = model.variables['Boiler(Q_th)|rate']

# v7 - Batched variable + selection
rate = model.variables['flow|rate'].sel(flow='Boiler(Q_th)')
```

### 2. Constraint Access Pattern

```python
# v6
constraint = model.constraints['Boiler(Q_th)|hours_min']

# v7
constraint = model.constraints['flow|hours_min'].sel(flow='Boiler(Q_th)')
```

### 3. Iterating Over Elements

```python
# v6
for flow_label in flow_labels:
    var = model.variables[f'{flow_label}|rate']
    # do something

# v7 - Vectorized (preferred)
rates = model.variables['flow|rate']  # All at once
# Then use xarray operations

# v7 - If you need to iterate
for flow_label in model.variables['flow|rate'].coords['flow'].values:
    var = model.variables['flow|rate'].sel(flow=flow_label)
```

---

## Migration Examples

### Example 1: Access a Specific Flow's Rate

```python
# v6
boiler_rate = model.variables['Boiler(Q_th)|rate']

# v7
boiler_rate = model.variables['flow|rate'].sel(flow='Boiler(Q_th)')

# With drop=True to remove the flow dimension
boiler_rate = model.variables['flow|rate'].sel(flow='Boiler(Q_th)', drop=True)
```

### Example 2: Get Multiple Flows

```python
# v6
rates = {
    'boiler': model.variables['Boiler(Q_th)|rate'],
    'chp': model.variables['CHP(Q_th)|rate'],
}

# v7 - Single selection
rates = model.variables['flow|rate'].sel(flow=['Boiler(Q_th)', 'CHP(Q_th)'])
# Returns DataArray with shape (2, time, ...)
```

### Example 3: Calculate Total Energy

```python
# v6
total = 0
for flow_label in heat_flow_labels:
    rate = model.variables[f'{flow_label}|rate']
    total += rate.sum()

# v7 - Vectorized
heat_rates = model.variables['flow|rate'].sel(flow=heat_flow_labels)
total = heat_rates.sum()  # Single operation
```

### Example 4: Check Which Flows Have Status Variables

```python
# v6
status_flows = []
for flow_label in all_flows:
    if f'{flow_label}|status' in model.variables:
        status_flows.append(flow_label)

# v7 - Check coordinate
status_var = model.variables.get('flow|status')
if status_var is not None:
    status_flows = list(status_var.coords['flow'].values)
```

### Example 5: Access After Solving

```python
# Build and solve
flow_system.build_model()
flow_system.solve(solver='highs')

# v6 - Individual variable values
boiler_rate_values = model.solution['Boiler(Q_th)|rate']

# v7 - Batched solution access
solution = model.solution
all_rates = solution['flow|rate']  # (flow, time, ...)
boiler_rates = solution['flow|rate'].sel(flow='Boiler(Q_th)')
```

---

## New Capabilities

The batched architecture enables powerful new features beyond just speed.

### 1. Vectorized Selection and Filtering

```python
# Select multiple elements at once
selected = model.variables['flow|rate'].sel(
    flow=['Boiler(Q_th)', 'CHP(Q_th)', 'HeatPump(Q_th)']
)

# Pattern-based selection
all_flows = model.variables['flow|rate'].coords['flow'].values
heat_flows = [f for f in all_flows if 'Q_th' in f]
heat_rates = model.variables['flow|rate'].sel(flow=heat_flows)
```

### 2. Vectorized Aggregations

```python
rates = model.variables['flow|rate']

# Sum across all flows
total_rate = rates.sum('flow')  # (time, ...)

# Mean per flow
avg_per_flow = rates.mean('time')  # (flow, ...)

# Max across both
max_rate = rates.max()  # scalar
```

### 3. Time Series Operations

```python
solution = model.solution
rates = solution['flow|rate']

# Resample to daily
daily_avg = rates.resample(time='1D').mean()

# Select time range
jan_rates = rates.sel(time='2024-01')

# Rolling average
rolling = rates.rolling(time=24).mean()
```

### 4. Boolean Masking

```python
rates = solution['flow|rate']

# Find when flows are active
is_active = rates > 0.01
hours_active = is_active.astype(int).sum('time')

# Filter to only active periods
active_rates = rates.where(is_active)
```

### 5. Cross-Element Analysis

```python
# Compare all flows at once
rates = solution['flow|rate']
sizes = solution['flow|size']

# Capacity factor for all flows
capacity_factor = rates / sizes  # Broadcasts automatically

# Correlation between flows
import xarray as xr
correlation = xr.corr(
    rates.sel(flow='Boiler(Q_th)'),
    rates.sel(flow='CHP(Q_th)'),
    dim='time'
)
```

### 6. GroupBy Operations

```python
# If you have metadata about flows
flow_types = xr.DataArray(
    ['boiler', 'chp', 'chp', 'heatpump'],
    dims='flow',
    coords={'flow': flow_labels}
)

# Group and aggregate
rates_by_type = rates.groupby(flow_types).sum('flow')
```

### 7. Easy Export to Pandas

```python
# Convert to DataFrame for further analysis
rates_df = solution['flow|rate'].to_dataframe()

# Pivot for wide format
rates_wide = rates_df.unstack('flow')
```

### 8. Dimensional Broadcasting

```python
# Parameters automatically broadcast
rates = solution['flow|rate']       # (flow, time, period, scenario)
sizes = solution['flow|size']       # (flow, period, scenario)

# Division broadcasts time dimension automatically
utilization = rates / sizes         # (flow, time, period, scenario)

# Aggregate keeping some dimensions
energy_per_period = rates.sum('time')  # (flow, period, scenario)
```

---

## Performance Comparison

### Build Time

| System | Old | New | Speedup |
|--------|-----|-----|---------|
| Small (168h, 17 components) | 1,095ms | 158ms | **6.9x** |
| Medium (720h, 30 components) | 5,278ms | 388ms | **13.6x** |
| Large (720h, 65 components) | 13,364ms | 478ms | **28.0x** |
| XL (2000h, 355 components) | 59,684ms | 5,978ms | **10.0x** |

### LP File Write

| System | Old | New | Speedup |
|--------|-----|-----|---------|
| Small | 600ms | 47ms | **12.9x** |
| Medium | 2,613ms | 230ms | **11.3x** |
| Large | 4,552ms | 449ms | **10.1x** |
| XL | 37,374ms | 8,684ms | **4.3x** |

### Model Size

| System | Old Vars | New Vars | Old Cons | New Cons |
|--------|----------|----------|----------|----------|
| Medium | 370 | 21 | 428 | 30 |
| Large | 859 | 21 | 997 | 30 |
| XL | 4,917 | 21 | 5,715 | 30 |

---

## Common Patterns

### Pattern 1: Find All Elements of a Type

```python
# Get all flow labels
flow_labels = list(model.variables['flow|rate'].coords['flow'].values)

# Get all storage labels
storage_labels = list(model.variables['storage|charge_state'].coords['storage'].values)

# Get all buses
bus_labels = list(model.constraints['bus|balance'].coords['bus'].values)
```

### Pattern 2: Check If Variable Exists for Element

```python
def has_status(model, flow_label):
    """Check if a flow has status variables."""
    status_var = model.variables.get('flow|status')
    if status_var is None:
        return False
    return flow_label in status_var.coords['flow'].values

def has_investment(model, flow_label):
    """Check if a flow has investment variables."""
    size_var = model.variables.get('flow|size')
    if size_var is None:
        return False
    return flow_label in size_var.coords['flow'].values
```

### Pattern 3: Extract Results to Dictionary

```python
def get_flow_results(solution, variable_name='rate'):
    """Extract flow results as a dictionary."""
    var = solution[f'flow|{variable_name}']
    return {
        flow: var.sel(flow=flow).values
        for flow in var.coords['flow'].values
    }

# Usage
rates_dict = get_flow_results(model.solution, 'rate')
sizes_dict = get_flow_results(model.solution, 'size')
```

### Pattern 4: Aggregate by Component Type

```python
def sum_by_component_type(solution, component_types: dict[str, list[str]]):
    """
    Sum flow rates by component type.

    Args:
        solution: Model solution dataset
        component_types: Dict mapping type name to list of flow labels
                        e.g., {'boilers': ['Boiler1(Q_th)', 'Boiler2(Q_th)'], ...}
    """
    rates = solution['flow|rate']
    results = {}
    for type_name, flow_labels in component_types.items():
        type_rates = rates.sel(flow=flow_labels)
        results[type_name] = type_rates.sum('flow')
    return results
```

---

## Troubleshooting

### KeyError: Variable Not Found

```python
# v6 code that fails
rate = model.variables['Boiler(Q_th)|rate']  # KeyError!

# FIX: Use new pattern
rate = model.variables['flow|rate'].sel(flow='Boiler(Q_th)')
```

### Element Not in Coordinates

```python
# If a flow doesn't have status, it won't be in status variable coords
try:
    status = model.variables['flow|status'].sel(flow='SimpleFlow(Q)')
except KeyError:
    # This flow doesn't have status tracking
    status = None
```

### Dimension Mismatch in Operations

```python
# If you're combining variables with different dimensions, use xarray alignment
rates = model.variables['flow|rate']    # (flow, time, ...)
sizes = model.variables['flow|size']    # (flow, period, scenario)

# This works - xarray broadcasts automatically
utilization = rates / sizes

# But be careful with manual operations
# Make sure dimensions align or use .reindex()
```

---

## Summary

flixopt v7 brings:

1. **Performance**: 7-28x faster model building, 4-13x faster LP writing
2. **Cleaner Structure**: 21 variables instead of 859 for large models
3. **Powerful Analysis**: Native xarray operations for vectorized analysis
4. **Better Scalability**: Performance scales with number of element *types*, not elements

The main migration effort is updating variable/constraint access patterns from direct name lookup to batched variable + `.sel()` selection.

For most users who only use the high-level API (`FlowSystem`, `add_elements`, `solve`), **no code changes are required** - the public API remains unchanged. The migration is only needed if you directly access `model.variables` or `model.constraints`.
