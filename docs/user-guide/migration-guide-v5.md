# Migration Guide: v4.x → v5.0.0

!!! tip "Quick Start"
    ```bash
    pip install --upgrade flixopt
    ```
    The new API is simpler and more intuitive. Review this guide to update your code.

---

## Overview

v5.0.0 introduces a streamlined API for optimization and results access. The key changes are:

| Aspect | Old API (v4.x) | New API (v5.0.0) |
|--------|----------------|------------------|
| **Optimization** | `fx.Optimization` class | `FlowSystem.optimize()` method |
| **Results access** | `element.submodel.variable.solution` | `flow_system.solution['variable_name']` |
| **Results storage** | `Results` class | `xarray.Dataset` on `flow_system.solution` |

---

## 💥 Breaking Changes in v5.0.0

### Optimization API

The `Optimization` class is removed. Use `FlowSystem.optimize()` directly.

=== "v4.x (Old)"
    ```python
    import flixopt as fx

    # Create flow system
    flow_system = fx.FlowSystem(timesteps)
    flow_system.add_elements(...)

    # Create Optimization object
    optimization = fx.Optimization('my_model', flow_system)
    optimization.do_modeling()
    optimization.solve(fx.solvers.HighsSolver())

    # Access results via Optimization object
    results = optimization.results
    costs = results.model['costs'].solution.item()
    ```

=== "v5.0.0 (New)"
    ```python
    import flixopt as fx

    # Create flow system
    flow_system = fx.FlowSystem(timesteps)
    flow_system.add_elements(...)

    # Optimize directly on FlowSystem
    flow_system.optimize(fx.solvers.HighsSolver())

    # Access results via flow_system.solution
    costs = flow_system.solution['costs'].item()
    ```

!!! note "Two-step alternative"
    If you need access to the model before solving:
    ```python
    flow_system.build_model()  # Creates flow_system.model
    flow_system.solve(fx.solvers.HighsSolver())
    ```

---

### Results Access

Results are now accessed via `flow_system.solution`, which is an `xarray.Dataset`.

#### Effect Values

=== "v4.x (Old)"
    ```python
    # Via element reference
    costs = flow_system.effects['costs']
    total_costs = costs.submodel.total.solution.item()

    # Or via results object
    total_costs = optimization.results.model['costs'].solution.item()
    ```

=== "v5.0.0 (New)"
    ```python
    # Direct access via solution Dataset
    total_costs = flow_system.solution['costs'].item()

    # Temporal and periodic components
    temporal_costs = flow_system.solution['costs(temporal)'].values
    periodic_costs = flow_system.solution['costs(periodic)'].values
    per_timestep = flow_system.solution['costs(temporal)|per_timestep'].values
    ```

#### Flow Rates

=== "v4.x (Old)"
    ```python
    boiler = flow_system.components['Boiler']
    flow_rate = boiler.thermal_flow.submodel.flow_rate.solution.values
    ```

=== "v5.0.0 (New)"
    ```python
    flow_rate = flow_system.solution['Boiler(Q_th)|flow_rate'].values
    ```

#### Investment Variables

=== "v4.x (Old)"
    ```python
    boiler = flow_system.components['Boiler']
    size = boiler.thermal_flow.submodel.investment.size.solution.item()
    invested = boiler.thermal_flow.submodel.investment.invested.solution.item()
    ```

=== "v5.0.0 (New)"
    ```python
    size = flow_system.solution['Boiler(Q_th)|size'].item()
    invested = flow_system.solution['Boiler(Q_th)|invested'].item()
    ```

#### Status Variables

=== "v4.x (Old)"
    ```python
    boiler = flow_system.components['Boiler']
    status = boiler.thermal_flow.submodel.status.status.solution.values
    startup = boiler.thermal_flow.submodel.status.startup.solution.values
    shutdown = boiler.thermal_flow.submodel.status.shutdown.solution.values
    ```

=== "v5.0.0 (New)"
    ```python
    status = flow_system.solution['Boiler(Q_th)|status'].values
    startup = flow_system.solution['Boiler(Q_th)|startup'].values
    shutdown = flow_system.solution['Boiler(Q_th)|shutdown'].values
    ```

#### Storage Variables

=== "v4.x (Old)"
    ```python
    storage = flow_system.components['Speicher']
    charge_state = storage.submodel.charge_state.solution.values
    netto_discharge = storage.submodel.netto_discharge.solution.values
    ```

=== "v5.0.0 (New)"
    ```python
    charge_state = flow_system.solution['Speicher|charge_state'].values
    netto_discharge = flow_system.solution['Speicher|netto_discharge'].values
    final_charge = flow_system.solution['Speicher|charge_state|final'].item()
    ```

---

## Variable Naming Convention

The new API uses a consistent naming pattern:

```text
ComponentLabel(FlowLabel)|variable_name
```

### Pattern Reference

| Variable Type | Pattern | Example |
|--------------|---------|---------|
| **Flow rate** | `Component(Flow)\|flow_rate` | `Boiler(Q_th)\|flow_rate` |
| **Size** | `Component(Flow)\|size` | `Boiler(Q_th)\|size` |
| **Invested** | `Component(Flow)\|invested` | `Boiler(Q_th)\|invested` |
| **Status** | `Component(Flow)\|status` | `Boiler(Q_th)\|status` |
| **Startup** | `Component(Flow)\|startup` | `Boiler(Q_th)\|startup` |
| **Shutdown** | `Component(Flow)\|shutdown` | `Boiler(Q_th)\|shutdown` |
| **Inactive** | `Component(Flow)\|inactive` | `Boiler(Q_th)\|inactive` |
| **Active hours** | `Component(Flow)\|active_hours` | `Boiler(Q_th)\|active_hours` |
| **Total flow** | `Component(Flow)\|total_flow_hours` | `Boiler(Q_th)\|total_flow_hours` |
| **Storage charge** | `Storage\|charge_state` | `Speicher\|charge_state` |
| **Storage final** | `Storage\|charge_state\|final` | `Speicher\|charge_state\|final` |
| **Netto discharge** | `Storage\|netto_discharge` | `Speicher\|netto_discharge` |

### Effects Pattern

| Variable Type | Pattern | Example |
|--------------|---------|---------|
| **Total** | `effect_label` | `costs` |
| **Temporal** | `effect_label(temporal)` | `costs(temporal)` |
| **Periodic** | `effect_label(periodic)` | `costs(periodic)` |
| **Per timestep** | `effect_label(temporal)\|per_timestep` | `costs(temporal)\|per_timestep` |
| **Contribution** | `Component(Flow)->effect(temporal)` | `Gastarif(Q_Gas)->costs(temporal)` |

---

## Discovering Variable Names

Use these methods to find available variable names:

```python
# List all variables in the solution
print(list(flow_system.solution.data_vars))

# Filter for specific patterns
costs_vars = [v for v in flow_system.solution.data_vars if 'costs' in v]
boiler_vars = [v for v in flow_system.solution.data_vars if 'Boiler' in v]
```

---

## Results I/O

### Saving Results

=== "v4.x (Old)"
    ```python
    optimization.results.to_file(folder='results', name='my_model')
    ```

=== "v5.0.0 (New)"
    ```python
    # Save entire FlowSystem with solution
    flow_system.to_netcdf('results/my_model.nc4')

    # Or save just the solution Dataset
    flow_system.solution.to_netcdf('results/solution.nc4')
    ```

### Loading Results

=== "v4.x (Old)"
    ```python
    results = fx.results.Results.from_file('results', 'my_model')
    ```

=== "v5.0.0 (New)"
    ```python
    import xarray as xr

    # Load FlowSystem with solution
    flow_system = fx.FlowSystem.from_netcdf('results/my_model.nc4')

    # Or load just the solution
    solution = xr.open_dataset('results/solution.nc4')
    ```

### Migrating Old Result Files

If you have result files saved with the old API (v4.x), you can migrate them to the new format using `FlowSystem.from_old_results()`. This method:

- Loads the old multi-file format (`*--flow_system.nc4`, `*--solution.nc4`)
- Renames deprecated parameters in the FlowSystem structure (e.g., `on_off_parameters` → `status_parameters`)
- Attaches the solution data to the FlowSystem

```python
# Load old results
flow_system = fx.FlowSystem.from_old_results('results_folder', 'my_model')

# Access basic solution data (flow rates, sizes, charge states, etc.)
flow_system.solution['Boiler(Q_th)|flow_rate'].plot()

# Save in new single-file format
flow_system.to_netcdf('results/my_model_migrated.nc4')
```

!!! warning "Limitations"
    This is a best-effort migration for accessing old results:

    - **Solution variable names are NOT renamed** - only basic variables work
      (flow rates, sizes, charge states, effect totals)
    - Advanced variable access may require using the original variable names
    - Summary metadata (solver info, timing) is not loaded

    For full compatibility, re-run optimizations with the new API.

---

## Working with xarray Dataset

The `flow_system.solution` is an `xarray.Dataset`, giving you powerful data manipulation:

```python
# Access a single variable
costs = flow_system.solution['costs']

# Get values as numpy array
values = flow_system.solution['Boiler(Q_th)|flow_rate'].values

# Get scalar value
total = flow_system.solution['costs'].item()

# Sum over time dimension
total_flow = flow_system.solution['Boiler(Q_th)|flow_rate'].sum(dim='time')

# Select by time
subset = flow_system.solution.sel(time=slice('2020-01-01', '2020-01-02'))

# Convert to DataFrame
df = flow_system.solution.to_dataframe()
```

---

## Segmented & Clustered Optimization

The new API also applies to advanced optimization modes:

=== "v4.x (Old)"
    ```python
    calc = fx.SegmentedOptimization('model', flow_system,
                                     timesteps_per_segment=96)
    calc.do_modeling_and_solve(solver)
    results = calc.results
    ```

=== "v5.0.0 (New)"
    ```python
    # Use transform accessor for segmented optimization
    flow_system.transform.segment(timesteps_per_segment=96)
    flow_system.optimize(solver)
    # Results in flow_system.solution
    ```

---

## Statistics Accessor

The new `statistics` accessor provides convenient aggregated data:

```python
stats = flow_system.statistics

# Flow data (clean labels, no |flow_rate suffix)
stats.flow_rates['Boiler(Q_th)']  # Not 'Boiler(Q_th)|flow_rate'
stats.flow_hours['Boiler(Q_th)']
stats.sizes['Boiler(Q_th)']
stats.charge_states['Battery']

# Effect breakdown by contributor (replaces effects_per_component)
stats.temporal_effects['costs']  # Per timestep, per contributor
stats.periodic_effects['costs']  # Investment costs per contributor
stats.total_effects['costs']     # Total per contributor

# Group by component or component type
stats.total_effects['costs'].groupby('component').sum()
stats.total_effects['costs'].groupby('component_type').sum()
```

---

## 🔧 Quick Reference

### Common Conversions

| Old Pattern | New Pattern |
|-------------|-------------|
| `optimization.results.model['costs'].solution.item()` | `flow_system.solution['costs'].item()` |
| `comp.flow.submodel.flow_rate.solution.values` | `flow_system.solution['Comp(Flow)\|flow_rate'].values` |
| `comp.flow.submodel.investment.size.solution.item()` | `flow_system.solution['Comp(Flow)\|size'].item()` |
| `comp.flow.submodel.status.status.solution.values` | `flow_system.solution['Comp(Flow)\|status'].values` |
| `storage.submodel.charge_state.solution.values` | `flow_system.solution['Storage\|charge_state'].values` |
| `effects['CO2'].submodel.total.solution.item()` | `flow_system.solution['CO2'].item()` |

---

## ✅ Migration Checklist

| Task | Description |
|------|-------------|
| **Replace Optimization class** | Use `flow_system.optimize(solver)` instead |
| **Update results access** | Use `flow_system.solution['var_name']` pattern |
| **Update I/O code** | Use `to_netcdf()` / `from_netcdf()` |
| **Migrate old result files** | Use `FlowSystem.from_old_results(folder, name)` |
| **Update transform methods** | Use `flow_system.transform.sel/isel/resample()` instead |
| **Test thoroughly** | Verify results match v4.x outputs |
| **Remove deprecated imports** | Remove `fx.Optimization`, `fx.Results` |

---

## Transform Methods Moved to Accessor

The `sel()`, `isel()`, and `resample()` methods have been moved from `FlowSystem` to the `TransformAccessor`:

=== "Old (deprecated)"
    ```python
    # These still work but emit deprecation warnings
    fs_subset = flow_system.sel(time=slice('2023-01-01', '2023-06-30'))
    fs_indexed = flow_system.isel(time=slice(0, 24))
    fs_resampled = flow_system.resample(time='4h', method='mean')
    ```

=== "New (recommended)"
    ```python
    # Use the transform accessor
    fs_subset = flow_system.transform.sel(time=slice('2023-01-01', '2023-06-30'))
    fs_indexed = flow_system.transform.isel(time=slice(0, 24))
    fs_resampled = flow_system.transform.resample(time='4h', method='mean')
    ```

!!! info "Solution is dropped"
    All transform methods return a **new FlowSystem without a solution**. You must re-optimize the transformed system:
    ```python
    fs_subset = flow_system.transform.sel(time=slice('2023-01-01', '2023-01-31'))
    fs_subset.optimize(solver)  # Re-optimize the subset
    ```

---

## Deprecation Timeline

| Version | Status |
|---------|--------|
| v4.x | `Optimization` and `Results` classes available |
| v5.0.0 | `Optimization` and `Results` deprecated, new API available |

!!! warning "Update your code"
    The `Optimization` and `Results` classes are deprecated and will be removed in a future version.
    The `flow_system.sel()`, `flow_system.isel()`, and `flow_system.resample()` methods are deprecated
    in favor of `flow_system.transform.sel/isel/resample()`.
    Update your code to the new API to avoid breaking changes when upgrading.

---

:material-book: [Docs](https://flixopt.github.io/flixopt/) • :material-github: [Issues](https://github.com/flixOpt/flixopt/issues)

!!! success "Welcome to the new flixopt API! 🎉"
