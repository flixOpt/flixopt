# Migration Guide: v2.x → v3.0.0

Quick guide for migrating flixopt from v2.x to v3.0.0.

!!! tip "Quick Start"
    ```bash
    pip install --upgrade flixopt
    ```
    Review breaking changes below, update deprecated parameters, and test thoroughly.

---

## Breaking Changes

**Effect Sharing System** - ⚠️ No deprecation warning

Effects now "pull" shares instead of "pushing" them:

=== "v2.x (Old)"

    ```python
    # Effects "pushed" shares to other effects
    CO2 = fx.Effect('CO2', 'kg', 'CO2 emissions',
        specific_share_to_other_effects_operation={'costs': 0.2})

    land = fx.Effect('land', 'm²', 'Land usage',
        specific_share_to_other_effects_invest={'costs': 100})

    costs = fx.Effect('costs', '€', 'Total costs')
    ```

=== "v3.0.0 (New)"

    ```python
    # Effects "pull" shares from other effects (clearer direction)
    CO2 = fx.Effect('CO2', 'kg', 'CO2 emissions')

    land = fx.Effect('land', 'm²', 'Land usage')

    costs = fx.Effect('costs', '€', 'Total costs',
        share_from_temporal={'CO2': 0.2},      # From temporal (operation) effects
        share_from_periodic={'land': 100})     # From periodic (investment) effects
    ```

**Migration:** Move share definitions to receiving effect and rename:
- `specific_share_to_other_effects_operation` → `share_from_temporal`
- `specific_share_to_other_effects_invest` → `share_from_periodic`

---

**Variable Renaming in Results**

Multiple variables renamed following terminology changes.

| Category         | Old (v2.x)                         | New (v3.0.0)   |
|------------------|------------------------------------|----------------|
| Investment       | `is_invested`                      | `invested`     |
| Switch tracking  | `switch_on`                        | `switch\|on`   |
| Switch tracking  | `switch_off`                       | `switch\|off`  |
| Switch tracking  | `switch_on_nr`                     | `switch\|count` |
| Effect submodels | `Effect(invest)\|total`            | `Effect(periodic)` |
| Effect submodels | `Effect(operation)\|total`         | `Effect(temporal)` |
| Effect submodels | `Effect(operation)\|total_per_timestep` | `Effect(temporal)\|per_timestep` |
| Effect submodels | `Effect\|total`           | `Effect` |

=== "v2.x (Old)"

    ```python
    # Investment decision
    results.solution['component|is_invested']

    # Switch tracking
    results.solution['component|switch_on']
    results.solution['component|switch_off']
    results.solution['component|switch_on_nr']

    # Effect variables
    results.solution['costs(invest)|total']
    results.solution['costs(operation)|total']
    results.solution['costs(operation)|total_per_timestep']
    results.solution['costs|total']
    ```

=== "v3.0.0 (New)"

    ```python
    # Investment decision
    results.solution['component|invested']

    # Switch tracking
    results.solution['component|switch|on']
    results.solution['component|switch|off']
    results.solution['component|switch|count']

    # Effect variables (with new terminology)
    results.solution['costs(periodic)']
    results.solution['costs(temporal)']
    results.solution['costs(temporal)|per_timestep']
    results.solution['costs']
    ```

---

**Calculation API** - `do_modeling()` now returns `Calculation` object for method chaining

=== "v2.x (Old)"

    ```python
    calculation = fx.FullCalculation('my_calc', flow_system)
    linopy_model = calculation.do_modeling()  # Returned linopy.Model

    # Access model directly from return value
    print(linopy_model)
    ```

=== "v3.0.0 (New)"

    ```python
    calculation = fx.FullCalculation('my_calc', flow_system)
    calculation.do_modeling()  # Returns Calculation object
    linopy_model = calculation.model  # Access model via property

    # This enables chaining operations
    fx.FullCalculation('my_calc', flow_system).do_modeling().solve()
    ```

---

**Storage Charge State** - Arrays no longer have extra timestep

=== "v2.x (Old)"

    ```python
    # Array with extra timestep
    storage = fx.Storage(
        'storage',
        relative_minimum_charge_state=np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # 5 values for 4 timesteps
    )
    ```

=== "v3.0.0 (New)"

    ```python
    # Array matches timesteps
    storage = fx.Storage(
        'storage',
        relative_minimum_charge_state=np.array([0.2, 0.2, 0.2, 0.2]),  # 4 values for 4 timesteps
        relative_minimum_final_charge_state=0.3  # Specify the final value directly
    )
    ```

---

**Plotting** - `mode` parameter renamed to `style`

=== "v2.x (Old)"

    ```python
    results.plot_heatmap('component|variable', mode='line')
    ```

=== "v3.0.0 (New)"

    ```python
    results.plot_heatmap('component|variable', style='line')
    ```

---

## Deprecated Parameters

!!! info "Still Work"
    These parameters still work but will be removed in a future version. Deprecation warnings will guide you.

**InvestParameters:**

| Old Parameter (v2.x) | New Parameter (v3.0.0) |
|---------------------|----------------------|
| `fix_effects` | `effects_of_investment` |
| `specific_effects` | `effects_of_investment_per_size` |
| `divest_effects` | `effects_of_retirement` |
| `piecewise_effects` | `piecewise_effects_of_investment` |

=== "v2.x (Deprecated)"

    ```python
    fx.InvestParameters(
        fix_effects=1000,
        specific_effects={'costs': 10},
        divest_effects=100,
        piecewise_effects=my_piecewise,
    )
    ```

=== "v3.0.0 (Recommended)"

    ```python
    fx.InvestParameters(
        effects_of_investment=1000,
        effects_of_investment_per_size={'costs': 10},
        effects_of_retirement=100,
        piecewise_effects_of_investment=my_piecewise,
    )
    ```

**Effect:**

| Old Parameter (v2.x) | New Parameter (v3.0.0) |
|---------------------|----------------------|
| `minimum_investment` | `minimum_periodic` |
| `maximum_investment` | `maximum_periodic` |
| `minimum_operation` | `minimum_temporal` |
| `maximum_operation` | `maximum_temporal` |
| `minimum_operation_per_hour` | `minimum_per_hour` |
| `maximum_operation_per_hour` | `maximum_per_hour` |

=== "v2.x (Deprecated)"

    ```python
    fx.Effect(
        'my_effect', 'unit', 'description',
        minimum_investment=10,
        maximum_investment=100,
        minimum_operation=5,
        maximum_operation=50,
        minimum_operation_per_hour=1,
        maximum_operation_per_hour=10,
    )
    ```

=== "v3.0.0 (Recommended)"

    ```python
    fx.Effect(
        'my_effect', 'unit', 'description',
        minimum_periodic=10,
        maximum_periodic=100,
        minimum_temporal=5,
        maximum_temporal=50,
        minimum_per_hour=1,
        maximum_per_hour=10,
    )
    ```

**Component Parameters:**

=== "v2.x (Deprecated)"

    ```python
    fx.Source('my_source', source=flow)

    fx.Sink('my_sink', sink=flow)

    fx.SourceAndSink(
        'my_source_sink',
        source=flow1,
        sink=flow2,
        prevent_simultaneous_sink_and_source=True
    )
    ```

=== "v3.0.0 (Recommended)"

    ```python
    fx.Source('my_source', outputs=flow)

    fx.Sink('my_sink', inputs=flow)

    fx.SourceAndSink(
        'my_source_sink',
        outputs=flow1,
        inputs=flow2,
        prevent_simultaneous_flow_rates=True
    )
    ```

**TimeSeriesData:**

=== "v2.x (Deprecated)"

    ```python
    fx.TimeSeriesData(
        agg_group='group1',
        agg_weight=2.0
    )
    ```

=== "v3.0.0 (Recommended)"

    ```python
    fx.TimeSeriesData(
        aggregation_group='group1',
        aggregation_weight=2.0
    )
    ```

**Calculation:**

=== "v2.x (Deprecated)"

    ```python
    calculation = fx.FullCalculation(
        'calc',
        flow_system,
        active_timesteps=[0, 1, 2]
    )
    ```

=== "v3.0.0 (Recommended)"

    ```python
    # Use FlowSystem selection methods
    flow_system_subset = flow_system.sel(time=slice('2020-01-01', '2020-01-03'))
    calculation = fx.FullCalculation('calc', flow_system_subset)

    # Or with isel for index-based selection
    flow_system_subset = flow_system.isel(time=slice(0, 3))
    calculation = fx.FullCalculation('calc', flow_system_subset)
    ```

---

## New Features

**Multi-Period Investments** - Model transformation pathways with distinct decisions per period:

```python
import pandas as pd

# Define multiple investment periods
periods = pd.Index(['2020', '2030'])
flow_system = fx.FlowSystem(time=timesteps, periods=periods)

# Components can now invest differently in each period
solar = fx.Source(
    'solar',
    outputs=[fx.Flow(
        'P_el',
        bus='electricity',
        size=fx.InvestParameters(
            minimum_size=0,
            maximum_size=1000,
            effects_of_investment_per_size={'costs': 100}
        )
    )]
)
```

**Scenario-Based Stochastic Optimization** - Model uncertainty with weighted scenarios:

```python
# Define scenarios with probabilities
scenarios = pd.Index(['low_demand', 'base', 'high_demand'], name='scenario')
scenario_weights = [0.2, 0.6, 0.2]  # Probabilities

flow_system = fx.FlowSystem(
    time=timesteps,
    scenarios=scenarios,
    scenario_weights=scenario_weights
)

# Define scenario-dependent data
demand = xr.DataArray(
    data=[[70, 80, 90],    # low_demand scenario
          [90, 100, 110],   # base scenario
          [110, 120, 130]], # high_demand scenario
    dims=['scenario', 'time'],
    coords={'scenario': scenarios, 'time': timesteps}
)

```

**Enhanced I/O** - Save, load, and manipulate FlowSystems:

```python
# Save and load FlowSystem
flow_system.to_netcdf('my_system.nc')
flow_system_loaded = fx.FlowSystem.from_netcdf('my_system.nc')

# Manipulate FlowSystem
fs_subset = flow_system.sel(time=slice('2020-01', '2020-06'))
fs_resampled = flow_system.resample(time='D')  # Resample to daily
fs_copy = flow_system.copy()

# Access FlowSystem from results (lazily loaded)
results = calculation.results
original_fs = results.flow_system  # No manual restoration needed
```

**Effects Per Component** - Analyze component impacts including indirect effects:

```python
# Get dataset showing contribution of each component to all effects
effects_ds = calculation.results.effects_per_component()

print(effects_ds['costs'])  # Total costs by component
print(effects_ds['CO2'])    # CO2 emissions by component (including indirect)
```

**Balanced Storage** - Force equal charging/discharging capacities:

```python
storage = fx.Storage(
    'storage',
    charging=fx.Flow('charge', bus='electricity', size=fx.InvestParameters(effects_per_size=100, minimum_size=5)),
    discharging=fx.Flow('discharge', bus='electricity', size=fx.InvestParameters(),
    balanced=True,  # Ensures charge_size == discharge_size
    capacity_in_flow_hours=100
)
```

**Final Charge State Control** - Set bounds on storage end state:

```python
storage = fx.Storage(
    'storage',
    charging=fx.Flow('charge', bus='electricity', size=100),
    discharging=fx.Flow('discharge', bus='electricity', size=100),
    capacity_in_flow_hours=10,
    relative_minimum_final_charge_state=0.5,  # End at least 50% charged
    relative_maximum_final_charge_state=0.8   # End at most 80% charged
)
```

---

## Configuration

**Logging (v2.2.0+)** - Console and file logging now disabled by default:

```python
import flixopt as fx

# Enable console logging
fx.CONFIG.Logging.console = True
fx.CONFIG.Logging.level = 'INFO'
fx.CONFIG.apply()

# Enable file logging
fx.CONFIG.Logging.file = 'flixopt.log'
fx.CONFIG.apply()

# Deprecated: change_logging_level() - will be removed in future
# fx.change_logging_level('INFO')  # ❌ Old way
```

---

## Testing

**Check for Deprecation Warnings:**

```python
import warnings
warnings.filterwarnings('default', category=DeprecationWarning)

# Run your flixopt code
# Review any DeprecationWarning messages
```

**Validate Results:**

```python
# Save v2.x results before upgrading
calculation.results.to_file('results_v2.nc')

# After upgrading, compare
results_v3 = calculation.results
results_v2 = fx.CalculationResults.from_file('results_v2.nc')

# Check key variables match (within numerical tolerance)
import numpy as np
v2_costs = results_v2['effect_values'].sel(effect='costs')
v3_costs = results_v3['effect_values'].sel(effect='costs')
np.testing.assert_allclose(v2_costs, v3_costs, rtol=1e-5)
```

---

## Common Issues

**"Effect share parameters not working"**
→ Move shares to receiving effect using `share_from_temporal`/`share_from_periodic`

**"Storage charge state has wrong dimensions"**
→ Remove extra timestep; use `relative_minimum_final_charge_state`

**"Bus assignment error"**
→ Use string labels instead of Bus objects:

```python
# Old
my_bus = fx.Bus('electricity')
flow = fx.Flow('P_el', bus=my_bus)  # ❌

# New
my_bus = fx.Bus('electricity')
flow = fx.Flow('P_el', bus='electricity')  # ✅
```

**"KeyError when accessing results"**
→ Update variable names:
  - `is_invested` → `invested`
  - `switch_on` → `switch|on`, `switch_off` → `switch|off`, `switch_on_nr` → `switch|count`
  - `Effect(invest)|total` → `Effect(periodic)`
  - `Effect(operation)|total` → `Effect(temporal)`
  - `Effect(operation)|total_per_timestep` → `Effect(temporal)|per_timestep`
  - `Effect|total` → `Effect`

**"AttributeError: SystemModel"**
→ Rename `SystemModel` → `FlowSystemModel`

**"No logging output"**
→ Enable explicitly: `fx.CONFIG.Logging.console = True; fx.CONFIG.apply()`

---

## Checklist

- [ ] Update flixopt: `pip install --upgrade flixopt`
- [ ] Update effect sharing syntax (no deprecation warning!)
- [ ] Update `Calculation.do_modeling()` usage
- [ ] Fix storage charge state array dimensions
- [ ] Rename `mode` → `style` in plotting calls
- [ ] Update deprecated parameter names (optional, but recommended)
- [ ] Enable logging explicitly if needed
- [ ] Test your code thoroughly
- [ ] Explore new features (periods, scenarios, enhanced I/O)

---

**Resources:**
[Documentation](https://flixopt.github.io/flixopt/) •
[GitHub Issues](https://github.com/flixOpt/flixopt/issues) •
[Full Changelog](https://flixopt.github.io/flixopt/latest/changelog/99984-v3.0.0/)

**Welcome to flixopt v3.0.0!** 🎉
