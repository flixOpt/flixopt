# Migration Guide: v2.x → v3.0.0

Quick guide for migrating flixopt from v2.x to v3.0.0.

!!! tip "Quick Start"
    ```bash
    pip install --upgrade flixopt
    ```
    Review breaking changes, update deprecated parameters, and test thoroughly.

---

## 💥 Breaking Changes

### Effect System Redesign

Effect domains renamed and sharing system inverted (no deprecation warnings).

| Old Term (v2.x) | New Term (v3.0.0) | Meaning                                                                 |
|-----------------|-------------------|-------------------------------------------------------------------------|
| `operation` | `temporal` | Time-varying effects (e.g., operational costs, occurring over time)      |
| `invest` / `investment` | `periodic` | Investment-related effects (e.g., fixed costs per period, annuity, ...) |

**Sharing system:** Effects now "pull" shares instead of "pushing" them.

=== "v2.x"
    ```python
    CO2 = fx.Effect('CO2', 'kg', 'CO2 emissions',
        specific_share_to_other_effects_operation={'costs': 0.2})
    land = fx.Effect('land', 'm²', 'Land usage',
        specific_share_to_other_effects_invest={'costs': 100})
    costs = fx.Effect('costs', '€', 'Total costs')
    ```

=== "v3.0.0"
    ```python
    CO2 = fx.Effect('CO2', 'kg', 'CO2 emissions')
    land = fx.Effect('land', 'm²', 'Land usage')
    costs = fx.Effect('costs', '€', 'Total costs',
        share_from_temporal={'CO2': 0.2},      # Pulls from temporal
        share_from_periodic={'land': 100})     # Pulls from periodic
    ```

!!! warning "No Deprecation Warning"
    This change was made WITHOUT deprecation warnings due to fundamental restructuring.

**Migration:**

1. Move share definitions to the receiving effect
2. Update parameter names:
   - `specific_share_to_other_effects_operation` → `share_from_temporal`
   - `specific_share_to_other_effects_invest` → `share_from_periodic`
3. Update terminology throughout your code:
   - Replace "operation" with "temporal" in effect-related contexts
   - Replace "invest/investment" with "periodic" in effect-related contexts

---

### Variable Names in Results

| Category | Old (v2.x) | New (v3.0.0) |
|----------|------------|--------------|
| Investment | `is_invested` | `invested` |
| Switch tracking | `switch_on` | `switch\|on` |
| Switch tracking | `switch_off` | `switch\|off` |
| Switch tracking | `switch_on_nr` | `switch\|count` |
| Effect submodels | `Effect(invest)\|total` | `Effect(periodic)` |
| Effect submodels | `Effect(operation)\|total` | `Effect(temporal)` |
| Effect submodels | `Effect(operation)\|total_per_timestep` | `Effect(temporal)\|per_timestep` |
| Effect submodels | `Effect\|total` | `Effect` |

=== "v2.x"
    ```python
    # Investment
    results.solution['component|is_invested']

    # Switch tracking
    results.solution['component|switch_on']
    results.solution['component|switch_off']
    results.solution['component|switch_on_nr']

    # Effects
    results.solution['costs(invest)|total']
    results.solution['costs(operation)|total']
    results.solution['costs(operation)|total_per_timestep']
    results.solution['costs|total']
    ```

=== "v3.0.0"
    ```python
    # Investment
    results.solution['component|invested']

    # Switch tracking
    results.solution['component|switch|on']
    results.solution['component|switch|off']
    results.solution['component|switch|count']

    # Effects
    results.solution['costs(periodic)']
    results.solution['costs(temporal)']
    results.solution['costs(temporal)|per_timestep']
    results.solution['costs']
    ```

---

### Bus and Effect Assignment

Use string labels instead of object references.

=== "v2.x"
    ```python
    # Bus assignment
    my_bus = fx.Bus('electricity')
    flow = fx.Flow('P_el', bus=my_bus)  # ❌ Object

    # Effect shares
    CO2 = fx.Effect('CO2', 'kg', 'CO2 emissions')
    costs = fx.Effect('costs', '€', 'Total costs',
        share_from_temporal={CO2: 0.2})  # ❌ Object
    ```

=== "v3.0.0"
    ```python
    # Bus assignment
    my_bus = fx.Bus('electricity')
    flow = fx.Flow('P_el', bus='electricity')  # ✅ String

    # Effect shares
    CO2 = fx.Effect('CO2', 'kg', 'CO2 emissions')
    costs = fx.Effect('costs', '€', 'Total costs',
        share_from_temporal={'CO2': 0.2})  # ✅ String
    ```

---

### FlowSystem Independence

Each Calculation now receives its own FlowSystem copy.

=== "v2.x"
    ```python
    # FlowSystem was shared
    flow_system = fx.FlowSystem(time=timesteps)
    calc1 = fx.FullCalculation('calc1', flow_system)  # Shared reference
    calc2 = fx.FullCalculation('calc2', flow_system)  # Same reference
    # Changes in calc1's FlowSystem would affect calc2
    ```

=== "v3.0.0"
    ```python
    # Each calculation gets a copy
    flow_system = fx.FlowSystem(time=timesteps)
    calc1 = fx.FullCalculation('calc1', flow_system)
    calc2 = fx.FullCalculation('calc2', flow_system)  # Gets copy
    # Calculations are now independent
    ```

!!! info "Impact"
    - Mutations to one calculation's FlowSystem won't affect others
    - Each `Subcalculation` in `SegmentedCalculation` has its own distinct FlowSystem
    - Memory usage may increase slightly due to copying

!!! tip "Migration"
    If you relied on shared FlowSystem behavior (which you most likely did not or by accident), you should copy the flow_system before passing it to another calculation.

---

### Calculation API

`do_modeling()` now returns Calculation object for method chaining.

=== "v2.x"
    ```python
    calculation = fx.FullCalculation('my_calc', flow_system)
    linopy_model = calculation.do_modeling()  # Returned linopy.Model

    # Access model directly from return value
    print(linopy_model)
    ```

=== "v3.0.0"
    ```python
    calculation = fx.FullCalculation('my_calc', flow_system)
    calculation.do_modeling()  # Returns Calculation object
    linopy_model = calculation.model  # Access model via property

    # Enables method chaining
    fx.FullCalculation('my_calc', flow_system).do_modeling().solve()
    ```

!!! tip "Migration"
    If you used the return value of `do_modeling()`, update to access `.model` property instead.

---

### Storage Charge State

Arrays now match timestep count (no extra element).

=== "v2.x"
    ```python
    # Array with extra timestep
    storage = fx.Storage(
        'storage',
        relative_minimum_charge_state=np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # 5 for 4 timesteps
    )
    ```

=== "v3.0.0"
    ```python
    # Array matches timesteps
    storage = fx.Storage(
        'storage',
        relative_minimum_charge_state=np.array([0.2, 0.2, 0.2, 0.2]),  # 4 values for 4 timesteps
        relative_minimum_final_charge_state=0.3  # Specify the final value directly if needed
    )
    ```

!!! note "Final State Control"
    You only need to specify `relative_minimum_final_charge_state` if it differs from the last value of `relative_minimum_charge_state`.

!!! info "Impact"
    If you provided arrays with `len(timesteps) + 1` elements, reduce to `len(timesteps)`.

---

### Other Breaking Changes

=== "Plotting"
    ```python
    # v2.x
    results.plot_heatmap('component|variable', mode='line')

    # v3.0.0
    results.plot_heatmap('component|variable', style='line')
    ```

=== "Class Names"
    ```python
    # v2.x
    from flixopt import SystemModel, Model

    # v3.0.0
    from flixopt import FlowSystemModel, Submodel
    ```

=== "Logging"
    ```python
    # v2.x - enabled by default
    # (no explicit configuration needed)

    # v3.0.0 - disabled by default
    import flixopt as fx
    fx.CONFIG.Logging.console = True
    fx.CONFIG.apply()
    ```

!!! warning "Breaking Change (from v2.2.0)"
    If you're upgrading from v2.1.x or earlier to v3.0.0, logging output is no longer displayed unless explicitly enabled. See [Configuration](#configuration) for details.

---

## 🗑️ Deprecated Parameters

!!! info "Still Supported"
    These parameters still work but will be removed in a future version. Deprecation warnings guide migration.

### InvestParameters

| Old (v2.x) | New (v3.0.0) |
|------------|--------------|
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

---

### Effect

| Old (v2.x) | New (v3.0.0) |
|------------|--------------|
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

---

### Component Parameters

| Old (v2.x) | New (v3.0.0) |
|------------|--------------|
| `source` (parameter) | `outputs` |
| `sink` (parameter) | `inputs` |
| `prevent_simultaneous_sink_and_source` | `prevent_simultaneous_flow_rates` |

=== "v2.x (Deprecated)"
    ```python
    fx.Source('my_source', source=flow)
    fx.Sink('my_sink', sink=flow)
    fx.SourceAndSink('my_source_sink', source=flow1, sink=flow2,
                     prevent_simultaneous_sink_and_source=True)
    ```

=== "v3.0.0 (Recommended)"
    ```python
    fx.Source('my_source', outputs=flow)
    fx.Sink('my_sink', inputs=flow)
    fx.SourceAndSink('my_source_sink', outputs=[flow1], inputs=[flow2],
                     prevent_simultaneous_flow_rates=True)
    ```

---

### TimeSeriesData

| Old (v2.x) | New (v3.0.0) |
|------------|--------------|
| `agg_group` | `aggregation_group` |
| `agg_weight` | `aggregation_weight` |

=== "v2.x (Deprecated)"
    ```python
    fx.TimeSeriesData(agg_group='group1', agg_weight=2.0)
    ```

=== "v3.0.0 (Recommended)"
    ```python
    fx.TimeSeriesData(aggregation_group='group1', aggregation_weight=2.0)
    ```

---

### Calculation

=== "v2.x (Deprecated)"
    ```python
    calculation = fx.FullCalculation('calc', flow_system, active_timesteps=[0, 1, 2])
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

## ✨ New Features

### Multi-Period Investments

Model transformation pathways with distinct decisions per period.

```python
import pandas as pd

# Define multiple investment periods
periods = pd.Index(['2020', '2030'])
flow_system = fx.FlowSystem(time=timesteps, periods=periods)

# Components can invest differently in each period
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

---

### Scenario-Based Stochastic Optimization

Model uncertainty with weighted scenarios.

```python
# Define scenarios with probabilities
scenarios = pd.Index(['low_demand', 'base', 'high_demand'], name='scenario')
scenario_weights = [0.2, 0.6, 0.2]

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

---

### Enhanced I/O

Save, load, and manipulate FlowSystems.

```python
# Save and load
flow_system.to_netcdf('system.nc')
fs = fx.FlowSystem.from_netcdf('system.nc')

# Manipulate
fs_subset = flow_system.sel(time=slice('2020-01', '2020-06'))
fs_resampled = flow_system.resample(time='D')  # Resample to daily
fs_copy = flow_system.copy()

# Access from results
original_fs = results.flow_system  # Lazily loaded
```

---

### Effects Per Component

Analyze component impacts including indirect effects.

```python
# Get dataset showing contribution of each component to all effects
effects_ds = results.effects_per_component

print(effects_ds['total'].sel(effect='costs'))  # Total costs by component
print(effects_ds['temporal'].sel(effect='CO2'))    # Temporal CO2 emissions by component (including indirect)
```

---

### Balanced Storage

Force equal charging/discharging sizes.

```python
storage = fx.Storage(
    'storage',
    charging=fx.Flow('charge', bus='electricity',
                     size=fx.InvestParameters(effects_of_investment_per_size=100, minimum_size=5)),
    discharging=fx.Flow('discharge', bus='electricity', size=fx.InvestParameters()),
    balanced=True,  # Ensures charge_size == discharge_size
    capacity_in_flow_hours=100
)
```

---

### Final Charge State Control

Set bounds on storage end state.

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

## ⚙️ Configuration

### Logging

Console and file logging now disabled by default (changed in v2.2.0).

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
    fx.CONFIG.apply()

    # Enable file logging (optional)
    fx.CONFIG.Logging.file = 'flixopt.log'
    fx.CONFIG.apply()

    calculation = fx.FullCalculation('calc', flow_system)
    calculation.solve()  # Now logs are shown
    ```

!!! tip "Quick Enable"
    Add at the start of your scripts:
    ```python
    import flixopt as fx
    fx.CONFIG.Logging.console = True
    fx.CONFIG.apply()
    ```

---

## 🧪 Testing

### Check Deprecation Warnings

```python
import warnings
warnings.filterwarnings('default', category=DeprecationWarning)

# Run your flixopt code
# Review any DeprecationWarning messages
```

---

### Validate Results

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

## 🔧 Common Issues

!!! failure "Effect shares not working"
    **Solution:** Effect sharing was completely redesigned. See [Effect System Redesign](#effect-system-redesign).

!!! failure "Storage dimensions wrong"
    **Solution:** Remove extra timestep from charge state arrays. See [Storage Charge State](#storage-charge-state).

!!! failure "Bus assignment error"
    **Solution:** Use string labels instead of Bus objects. See [Bus and Effect Assignment](#bus-and-effect-assignment).

!!! failure "KeyError when accessing results"
    **Solution:** Variable names have changed. See [Variable Names in Results](#variable-names-in-results) for the complete mapping.

!!! failure "AttributeError: Element x has no attribute `model`"
    **Solution:** Rename `.model` → `.submodel`. See [Other Breaking Changes](#other-breaking-changes).

!!! failure "No logging output"
    **Solution:** Logging is disabled by default. See [Configuration](#configuration) to enable it.

---

## ✅ Checklist

**Critical (Breaking Changes):**

- [ ] Update flixopt: `pip install --upgrade flixopt`
- [ ] Update [effect sharing syntax](#effect-system-redesign) (no deprecation warning!)
- [ ] Update [variable names in results](#variable-names-in-results)
- [ ] Update [Bus/Effect assignments](#bus-and-effect-assignment) to use string labels
- [ ] Update [Calculation API](#calculation-api) usage if accessing return value
- [ ] Fix [storage charge state](#storage-charge-state) array dimensions
- [ ] Rename plotting `mode` → `style`
- [ ] Update class names: `SystemModel` → `FlowSystemModel`, `Model` → `Submodel`

**Important:**

- [ ] Enable [logging](#configuration) explicitly if needed
- [ ] Update [deprecated parameters](#deprecated-parameters) (optional, but recommended)

**Testing:**

- [ ] Test your code thoroughly
- [ ] Check for [deprecation warnings](#check-deprecation-warnings)
- [ ] [Validate results](#validate-results) match v2.x output (if upgrading)

**Optional:**

- [ ] Explore [new features](#new-features)

---

## 📚 Getting Help

:material-book: [Documentation](https://flixopt.github.io/flixopt/)
:material-github: [GitHub Issues](https://github.com/flixOpt/flixopt/issues)
:material-text-box: [Full Changelog](https://flixopt.github.io/flixopt/latest/changelog/99984-v3.0.0/)

---

!!! success "Welcome to flixopt v3.0.0! 🎉"
