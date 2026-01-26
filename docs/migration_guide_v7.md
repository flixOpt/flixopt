# Migration Guide: flixopt v7

## What's New

### Performance

| System | v6 | v7 | Speedup |
|--------|-----|-----|---------|
| Medium (720h, 30 components) | 5,278ms | 388ms | **13.6x** |
| Large (720h, 65 components) | 13,364ms | 478ms | **28.0x** |
| XL (2000h, 355 components) | 59,684ms | 5,978ms | **10.0x** |

LP file writing is also 4-13x faster.

### Fewer Variables, Same Model

v7 uses batched variables with element coordinates instead of individual variables per element:

```
v6: 859 variables, 997 constraints (720h, 50 converters)
v7:  21 variables,  30 constraints (same model!)
```

| v6 | v7 |
|----|-----|
| `Boiler(Q_th)\|rate` | `flow\|rate` with coord `flow='Boiler(Q_th)'` |
| `Boiler(Q_th)\|size` | `flow\|size` with coord `flow='Boiler(Q_th)'` |
| `HeatStorage\|charge_state` | `storage\|charge_state` with coord `storage='HeatStorage'` |

### Native xarray Access

After solving, results are xarray DataArrays with full analytical capabilities:

```python
solution = model.solution
rates = solution['flow|rate']  # (flow, time, ...)

# Select elements
rates.sel(flow='Boiler(Q_th)')
rates.sel(flow=['Boiler(Q_th)', 'CHP(Q_th)'])

# Aggregations
rates.sum('flow')
rates.mean('time')

# Time series operations
rates.resample(time='1D').mean()
rates.groupby('time.hour').mean()

# Export
rates.to_dataframe()
```

---

## Breaking Changes

### Solution Variable Names

The main breaking change is how variables are named in `model.solution`:

```python
solution = model.solution

# v6 style - NO LONGER EXISTS
solution['Boiler(Q_th)|rate']      # KeyError!
solution['Boiler(Q_th)|size']      # KeyError!

# v7 style - Use batched name + .sel()
solution['flow|rate'].sel(flow='Boiler(Q_th)')
solution['flow|size'].sel(flow='Boiler(Q_th)')
```

#### Variable Name Mapping

| v6 Name | v7 Name |
|---------|---------|
| `{flow}\|rate` | `flow\|rate` with `.sel(flow='{flow}')` |
| `{flow}\|size` | `flow\|size` with `.sel(flow='{flow}')` |
| `{flow}\|status` | `flow\|status` with `.sel(flow='{flow}')` |
| `{storage}\|charge_state` | `storage\|charge_state` with `.sel(storage='{storage}')` |
| `{storage}\|size` | `storage\|size` with `.sel(storage='{storage}')` |

#### Migration Pattern

```python
# v6
def get_flow_rate(solution, flow_name):
    return solution[f'{flow_name}|rate']

# v7
def get_flow_rate(solution, flow_name):
    return solution['flow|rate'].sel(flow=flow_name)
```

### Iterating Over Results

```python
# v6 - iterate over individual variable names
for flow_name in flow_names:
    rate = solution[f'{flow_name}|rate']
    process(rate)

# v7 - use xarray iteration or vectorized operations
rates = solution['flow|rate']

# Option 1: Vectorized (preferred)
total = rates.sum('flow')

# Option 2: Iterate if needed
for flow_name in rates.coords['flow'].values:
    rate = rates.sel(flow=flow_name)
    process(rate)
```

### Getting All Flow/Storage Names

```python
# v7 - get element names from coordinates
flow_names = list(solution['flow|rate'].coords['flow'].values)
storage_names = list(solution['storage|charge_state'].coords['storage'].values)
```

---

## Quick Reference

### Available Batched Variables

| Variable | Dimensions |
|----------|------------|
| `flow\|rate` | (flow, time, period?, scenario?) |
| `flow\|size` | (flow, period?, scenario?) |
| `flow\|status` | (flow, time, ...) |
| `storage\|charge_state` | (storage, time, ...) |
| `storage\|size` | (storage, period?, scenario?) |
| `bus\|balance` | (bus, time, ...) |

### Common Operations

```python
solution = model.solution

# Get all rates
rates = solution['flow|rate']

# Select one element
boiler = rates.sel(flow='Boiler(Q_th)')

# Select multiple
selected = rates.sel(flow=['Boiler(Q_th)', 'CHP(Q_th)'])

# Filter by pattern
heat_flows = [f for f in rates.coords['flow'].values if 'Q_th' in f]
heat_rates = rates.sel(flow=heat_flows)

# Aggregate
total_by_time = rates.sum('flow')
total_by_flow = rates.sum('time')

# Time operations
daily = rates.resample(time='1D').mean()
hourly_pattern = rates.groupby('time.hour').mean()
```
