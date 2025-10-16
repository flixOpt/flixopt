# Migration Guide: Upgrading to v3.0.0

This guide helps you migrate your flixopt code from v2.x to v3.0.0. Version 3.0.0 introduces powerful new features like multi-period investments and scenario-based stochastic optimization, along with a redesigned effect sharing system.

!!! tip "Quick Start"
    1. **Update your installation:**
       ```bash
       pip install --upgrade flixopt
       ```
    2. **Review breaking changes** in the sections below
    3. **Update deprecated parameters** to their new names
    4. **Test your code** with the new version

---

## Breaking Changes

### 1. Effect Sharing System Redesign

!!! warning "Breaking Change - No Deprecation"
    The effect sharing syntax has been inverted and simplified. This change was made WITHOUT deprecation warnings due to the fundamental restructuring.

**What changed:** Effects now "pull" shares from other effects instead of "pushing" them.

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

!!! success "Migration Steps"
    1. Find all uses of `specific_share_to_other_effects_operation` and `specific_share_to_other_effects_invest`
    2. Move the share definition to the **receiving** effect
    3. Rename parameters:
         - `specific_share_to_other_effects_operation` → `share_from_temporal`
         - `specific_share_to_other_effects_invest` → `share_from_periodic`

---

### 2. Variable Renaming in Results

!!! warning "Breaking Change"
    Multiple variables have been renamed in the optimization results. Update all result access accordingly.

**Investment Variables:**

=== "v2.x (Old)"

    ```python
    # Investment decision variable
    results.solution['element|is_invested']
    ```

=== "v3.0.0 (New)"

    ```python
    # Investment decision variable
    results.solution['element|invested']
    ```

**Switch Tracking Variables (OnOffModel):**

=== "v2.x (Old)"

    ```python
    # Switch state tracking
    results.solution['component|switch_on']
    results.solution['component|switch_off']
    results.solution['component|switch_on_nr']
    ```

=== "v3.0.0 (New)"

    ```python
    # Switch state tracking with pipe delimiter
    results.solution['component|switch|on']
    results.solution['component|switch|off']
    results.solution['component|switch|count']
    ```

**Quick Reference Table:**

| Old Variable Name (v2.x) | New Variable Name (v3.0.0) | Component Type |
|--------------------------|---------------------------|----------------|
| `is_invested` | `invested` | Investment |
| `switch_on` | `switch|on` | OnOff |
| `switch_off` | `switch|off` | OnOff |
| `switch_on_nr` | `switch|count` | OnOff |

---

### 3. FlowSystem Independence

!!! warning "Breaking Change"
    FlowSystems can no longer be shared across multiple Calculations.

**What changed:** Each `Calculation` now automatically creates its own copy of the FlowSystem, making calculations fully independent.

**Impact:**
- Mutations to one calculation's FlowSystem won't affect others
- Each `Subcalculation` in `SegmentedCalculation` has its own distinct FlowSystem
- Memory usage may increase slightly due to copying

=== "v2.x (Old)"

    ```python
    # FlowSystem was shared
    flow_system = fx.FlowSystem(time=timesteps)
    calc1 = fx.FullCalculation('calc1', flow_system)
    calc2 = fx.FullCalculation('calc2', flow_system)

    # Both calculations shared the same FlowSystem object
    # Changes in one affected the other
    ```

=== "v3.0.0 (New)"

    ```python
    # Each calculation gets its own copy
    flow_system = fx.FlowSystem(time=timesteps)
    calc1 = fx.FullCalculation('calc1', flow_system)  # Gets a copy
    calc2 = fx.FullCalculation('calc2', flow_system)  # Gets another copy

    # Calculations are now independent
    # Changes to calc1's FlowSystem won't affect calc2
    ```

!!! tip "Migration"
    If you relied on shared FlowSystem behavior (which you most likely did not), you should copy the flow_system before passing it to another calculation.

---

### 4. Bus and Effect Object Assignment

!!! warning "Breaking Change"
    Direct assignment of Bus and Effect objects is no longer supported. Use string labels instead.

**Bus Assignment:**

=== "v2.x (Old)"

    ```python
    my_bus = fx.Bus('electricity')
    flow = fx.Flow('P_el', bus=my_bus)  # ❌ Object assignment
    ```

=== "v3.0.0 (New)"

    ```python
    my_bus = fx.Bus('electricity')
    flow = fx.Flow('P_el', bus='electricity')  # ✅ String label
    ```

**Effect Shares Assignment:**

=== "v2.x (Old)"

    ```python
    CO2 = fx.Effect('CO2', 'kg', 'CO2 emissions')
    costs = fx.Effect('costs', '€', 'Total costs',
        share_from_temporal={CO2: 0.2})  # ❌ Effect object
    ```

=== "v3.0.0 (New)"

    ```python
    CO2 = fx.Effect('CO2', 'kg', 'CO2 emissions')
    costs = fx.Effect('costs', '€', 'Total costs',
        share_from_temporal={'CO2': 0.2})  # ✅ String label
    ```

---

### 5. Calculation API Change

!!! info "Method Chaining Support"
    `Calculation.do_modeling()` now returns the Calculation object to enable method chaining.

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

!!! tip "Migration"
    If you used the return value of `do_modeling()`, update to access `.model` property instead.

---

### 6. Storage Charge State Bounds

!!! warning "Array Dimensions Changed"
    `relative_minimum_charge_state` and `relative_maximum_charge_state` no longer have an extra timestep.

**Impact:** If you provided arrays with `len(timesteps) + 1` elements, reduce to `len(timesteps)`.

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

!!! note "Final State Control"
    Use the new `relative_minimum_final_charge_state` and `relative_maximum_final_charge_state` parameters to explicitly control the final charge state.

---

### 7. Plotting Parameter Rename

=== "v2.x (Old)"

    ```python
    results.plot_heatmap('component|variable', mode='line')
    ```

=== "v3.0.0 (New)"

    ```python
    results.plot_heatmap('component|variable', style='line')
    ```

---

### 8. Logging Configuration

!!! warning "Breaking Change (from v2.2.0)"
    Console and file logging are now **disabled by default**.

**What changed:** In v2.2.0 (before v3.0.0), the default logging behavior was changed to be opt-in rather than opt-out.

**Impact:** If you're upgrading from v2.1.x or earlier to v3.0.0, you may notice that logging output is no longer displayed unless explicitly enabled.

=== "v2.1.x and earlier"

    ```python
    import flixopt as fx

    # Logging was enabled by default
    calculation = fx.FullCalculation('calc', flow_system)
    calculation.solve()  # Logs were shown automatically
    ```

=== "v2.2.0+ and v3.0.0"

    ```python
    import flixopt as fx

    # Enable console logging explicitly
    fx.CONFIG.Logging.console = True
    fx.CONFIG.Logging.level = 'INFO'
    fx.CONFIG.Logging.file = 'flixopt.log' # Optional: Enable file logging
    fx.CONFIG.apply()

    calculation = fx.FullCalculation('calc', flow_system)
    calculation.solve()  # Now logs are shown
    ```

!!! tip "Migration"
    Add logging configuration at the start of your scripts if you want to see log output:
    ```python
    import flixopt as fx
    fx.CONFIG.Logging.console = True
    fx.CONFIG.apply()
    ```

---

## Deprecated Parameters (Still Supported)

!!! info "Gradual Migration"
    These parameters still work but will be removed in a future version. Update them at your convenience - deprecation warnings will guide you.

### InvestParameters

**Parameter Changes:**

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

### Effect

**Parameter Changes:**

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

### Component Parameters

=== "v2.x (Deprecated)"

    ```python
    flow1 = fx.Source('Buy', bus='Gas)

    flow2 = fx.Flow('Sell', bus='Gas)

    fx.SourceAndSink(
        'my_source_sink',
        source=flow1,
        sink=flow2,
        prevent_simultaneous_sink_and_source=True
    )
    ```

=== "v3.0.0 (Recommended)"

    ```python
    flow1 = fx.Source('Buy', bus='Gas)

    flow2 = fx.Flow('Sell', bus='Gas)

    fx.SourceAndSink(
        'my_source_sink',
        outputs=[flow1],
        inputs=[flow2],
        prevent_simultaneous_flow_rates=True
    )
    ```

### TimeSeriesData

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

### Calculation

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

## New Features in v3.0.0

### 1. Multi-Period Investments

Model transformation pathways with distinct investment decisions in each period:

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

### 2. Scenario-Based Stochastic Optimization

Model uncertainty with weighted scenarios:

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

**Control variable independence:**
```python
# By default: investment sizes are shared across scenarios, flow rates vary
# To make sizes scenario-independent:
flow_system = fx.FlowSystem(
    time=timesteps,
    scenarios=scenarios,
    scenario_independent_sizes=True  # Each scenario gets its own capacity
)
```

### 3. Enhanced I/O and Data Handling

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

### 4. Effects Per Component

Analyze the impact of each component, including indirect effects through effect shares:

```python
# Get dataset showing contribution of each component to all effects
effects_ds = calculation.results.effects_per_component()

print(effects_ds['costs'])  # Total costs by component
print(effects_ds['CO2'])    # CO2 emissions by component (including indirect)
```

### 5. Balanced Storage

Force charging and discharging capacities to be equal:

```python
storage = fx.Storage(
    'storage',
    charging=fx.Flow('charge', bus='electricity', size=fx.InvestParameters(effects_per_size=100, minimum_size=5)),
    discharging=fx.Flow('discharge', bus='electricity', size=fx.InvestParameters(),
    balanced=True,  # Ensures charge_size == discharge_size
    capacity_in_flow_hours=100
)
```

### 6. Final Charge State Control

Set bounds on the storage state at the end of the optimization:

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

## Configuration Changes

### Logging (v2.2.0+)

**Breaking change:** Console and file logging are now disabled by default.

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

## Testing Your Migration

### 1. Check for Deprecation Warnings

Run your code and watch for deprecation warnings:

```python
import warnings
warnings.filterwarnings('default', category=DeprecationWarning)

# Run your flixopt code
# Review any DeprecationWarning messages
```

### 2. Validate Results

Compare results from v2.x and v3.0.0 to ensure consistency:

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

## Common Migration Issues

### Issue: "Effect share parameters not working"

**Solution:** Effect sharing was completely redesigned. Move share definitions to the **receiving** effect using `share_from_temporal` and `share_from_periodic`.

### Issue: "Storage charge state has wrong dimensions"

**Solution:** Remove the extra timestep from charge state bound arrays.

### Issue: "Import error with Bus assignment"

**Solution:** Pass bus labels (strings) instead of Bus objects to `Flow.bus`.

```python
# Old
my_bus = fx.Bus('electricity')
flow = fx.Flow('P_el', bus=my_bus)  # ❌

# New
my_bus = fx.Bus('electricity')
flow = fx.Flow('P_el', bus='electricity')  # ✅
```

### Issue: "AttributeError: module 'flixopt' has no attribute 'SystemModel'"

**Solution:** Rename `SystemModel` → `FlowSystemModel`

### Issue: "KeyError when accessing results variables"

**Solution:** Variable names have changed. Update your result access:

```python
# Old variable names
results.solution['component|is_invested']   # ❌
results.solution['component|switch_on']     # ❌
results.solution['component|switch_off']    # ❌
results.solution['component|switch_on_nr']  # ❌

# New variable names
results.solution['component|invested']      # ✅
results.solution['component|switch|on']     # ✅
results.solution['component|switch|off']    # ✅
results.solution['component|switch|count']  # ✅
```

### Issue: "FlowSystem changes affecting multiple calculations"

**Solution:** FlowSystems are now automatically copied for each Calculation. If you need to access the FlowSystem from a calculation, use `calculation.flow_system` instead of keeping a reference to the original.

### Issue: "No logging output"

**Solution:** Logging is disabled by default in v2.2.0+. Enable it explicitly:

```python
import flixopt as fx
fx.CONFIG.Logging.console = True
fx.CONFIG.apply()
```

---

## Getting Help

- **Documentation:** [https://flixopt.github.io/flixopt/](https://flixopt.github.io/flixopt/)
- **GitHub Issues:** [https://github.com/flixOpt/flixopt/issues](https://github.com/flixOpt/flixopt/issues)
- **Changelog:** [Full v3.0.0 release notes](https://flixopt.github.io/flixopt/latest/changelog/99984-v3.0.0/)

---

## Summary Checklist

**Critical (Breaking Changes):**

- [ ] Update flixopt: `pip install --upgrade flixopt`
- [ ] Update effect sharing syntax (no deprecation warning!) - move shares to receiving effect
- [ ] Update all variable names in result access (`is_invested` → `invested`, `switch_on` → `switch|on`, etc.)
- [ ] Replace Bus/Effect object assignments with string labels
- [ ] Remove any code that relies on shared FlowSystem objects across Calculations
- [ ] Update `Calculation.do_modeling()` usage if accessing return value
- [ ] Fix storage charge state array dimensions (remove extra timestep)

**Important:**

- [ ] Rename `mode` → `style` in plotting calls
- [ ] Enable logging explicitly if needed (`fx.CONFIG.Logging.console = True; fx.CONFIG.apply()`)
- [ ] Update deprecated parameter names (optional, but recommended)

**Testing:**

- [ ] Test your code thoroughly
- [ ] Check for deprecation warnings
- [ ] Validate results match v2.x output (if upgrading)

**Optional:**

- [ ] Explore new features (periods, scenarios, enhanced I/O, balanced storage, final charge state control)

**Welcome to flixopt v3.0.0!** 🎉
