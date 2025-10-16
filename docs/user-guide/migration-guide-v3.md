# Migration Guide: v2.x → v3.0.0

!!! tip "Quick Start"
    ```bash
    pip install --upgrade flixopt
    ```
    Review [breaking changes](#breaking-changes), update [deprecated parameters](#deprecated-parameters), test thoroughly.

---

## 💥 Breaking Changes

### Effect System Redesign

Terminology changed: `operation` → `temporal`, `invest/investment` → `periodic`. Sharing inverted: effects now "pull" shares.

=== "v2.x"
    ```python
    CO2 = fx.Effect('CO2', 'kg', 'CO2',
        specific_share_to_other_effects_operation={'costs': 0.2})
    costs = fx.Effect('costs', '€', 'Total')
    ```

=== "v3.0.0"
    ```python
    CO2 = fx.Effect('CO2', 'kg', 'CO2')
    costs = fx.Effect('costs', '€', 'Total',
        share_from_temporal={'CO2': 0.2})
    ```

!!! warning "No deprecation warning"
    - Move shares to receiving effect
    - `specific_share_to_other_effects_operation` → `share_from_temporal`
    - `specific_share_to_other_effects_invest` → `share_from_periodic`

---

### Variable Names

| Old | New | Old | New |
|-----|-----|-----|-----|
| `is_invested` | `invested` | `switch_on` | `switch\|on` |
| `switch_off` | `switch\|off` | `switch_on_nr` | `switch\|count` |
| `Effect(invest)\|total` | `Effect(periodic)` | `Effect(operation)\|total` | `Effect(temporal)` |
| `Effect(operation)\|total_per_timestep` | `Effect(temporal)\|per_timestep` | `Effect\|total` | `Effect` |

---

### String Labels

Use strings instead of objects for Bus/Effect references.

=== "v2.x"
    ```python
    flow = fx.Flow('P_el', bus=my_bus)  # ❌ Object
    costs = fx.Effect('costs', '€', share_from_temporal={CO2: 0.2})  # ❌
    ```

=== "v3.0.0"
    ```python
    flow = fx.Flow('P_el', bus='electricity')  # ✅ String
    costs = fx.Effect('costs', '€', share_from_temporal={'CO2': 0.2})  # ✅
    ```

---

### FlowSystem & Calculation

- **FlowSystem**: Each `Calculation` gets its own copy (independent)
- **do_modeling()**: Returns `Calculation` (access model via `.model` property)
- **Storage**: Arrays match timestep count (no extra element)
  - Use `relative_minimum_final_charge_state` for final state control

---

### Other Changes

| Category | Old | New |
|----------|-----|-----|
| Plotting | `mode='line'` | `style='line'` |
| Classes | `SystemModel`, `Model` | `FlowSystemModel`, `Submodel` |
| Logging | Enabled by default | Disabled (enable: `fx.CONFIG.Logging.console = True; fx.CONFIG.apply()`) |

---

## 🗑️ Deprecated Parameters

??? abstract "InvestParameters"

    `fix_effects` → `effects_of_investment` • `specific_effects` → `effects_of_investment_per_size` • `divest_effects` → `effects_of_retirement` • `piecewise_effects` → `piecewise_effects_of_investment`

??? abstract "Effect"

    `minimum_investment` → `minimum_periodic` • `maximum_investment` → `maximum_periodic` • `minimum_operation` → `minimum_temporal` • `maximum_operation` → `maximum_temporal` • `minimum_operation_per_hour` → `minimum_per_hour` • `maximum_operation_per_hour` → `maximum_per_hour`

??? abstract "Components"

    `source` → `outputs` • `sink` → `inputs` • `prevent_simultaneous_sink_and_source` → `prevent_simultaneous_flow_rates`

??? abstract "TimeSeriesData & Calculation"

    - `agg_group` → `aggregation_group`
    - `agg_weight` → `aggregation_weight`
    - `active_timesteps` → Use `flow_system.sel()` or `flow_system.isel()`

---

## ✨ New Features

??? success "Multi-Period Investments"

    ```python
    periods = pd.Index(['2020', '2030'])
    flow_system = fx.FlowSystem(time=timesteps, periods=periods)
    ```

??? success "Scenario-Based Optimization"

    ```python
    scenarios = pd.Index(['low', 'base', 'high'], name='scenario')
    flow_system = fx.FlowSystem(time=timesteps, scenarios=scenarios,
        scenario_weights=[0.2, 0.6, 0.2], scenario_independent_sizes=True)
    ```

??? success "Enhanced I/O"

    `flow_system.to_netcdf()` • `fx.FlowSystem.from_netcdf()` • `flow_system.sel()` • `flow_system.resample()` • `results.flow_system`

??? success "Effects Per Component"

    ```python
    effects_ds = results.effects_per_component
    print(effects_ds['total'].sel(effect='costs'))
    ```

??? success "Storage Features"

    **Balanced**: `balanced=True` ensures charge_size == discharge_size
    **Final State**: `relative_minimum_final_charge_state=0.5`, `relative_maximum_final_charge_state=0.8`

---

## 🔧 Common Issues

| Issue | Solution |
|-------|----------|
| Effect shares not working | See [Effect System Redesign](#effect-system-redesign) |
| Storage dimensions wrong | See [FlowSystem & Calculation](#flowsystem-calculation) |
| Bus assignment error | See [String Labels](#string-labels) |
| KeyError in results | See [Variable Names](#variable-names) |
| `AttributeError: model` | Rename `.model` → `.submodel` |
| No logging | See [Other Changes](#other-changes) |

---

## ✅ Checklist

- [ ] `pip install --upgrade flixopt`
- [ ] Update [effect sharing](#effect-system-redesign), [variable names](#variable-names), [string labels](#string-labels)
- [ ] Fix [storage arrays](#flowsystem-calculation), [Calculation API](#flowsystem-calculation)
- [ ] Rename `mode` → `style`, update [class names](#other-changes)
- [ ] Enable [logging](#other-changes) if needed
- [ ] Update [deprecated parameters](#deprecated-parameters)
- [ ] Test & validate results

---

:material-book: [Docs](https://flixopt.github.io/flixopt/) • :material-github: [Issues](https://github.com/flixOpt/flixopt/issues) • :material-text-box: [Changelog](https://flixopt.github.io/flixopt/latest/changelog/99984-v3.0.0/)

!!! success "Welcome to flixopt v3.0.0! 🎉"
