# Migration Guide: v2.x → v3.0.0

!!! tip "Quick Start"
    ```bash
    pip install --upgrade flixopt
    ```
    Review [breaking changes](#breaking-changes), update [deprecated parameters](#deprecated-parameters), test thoroughly.

---

## 💥 Breaking Changes

### Effect System Redesign

Terminology changed and sharing system inverted: effects now "pull" shares.

| Concept | Old (v2.x) | New (v3.0.0) |
|---------|------------|--------------|
| Time-varying effects | `operation` | `temporal` |
| Investment effects | `invest` / `investment` | `periodic` |
| Share to other effects (operation) | `specific_share_to_other_effects_operation` | `share_from_temporal` |
| Share to other effects (invest) | `specific_share_to_other_effects_invest` | `share_from_periodic` |

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
        share_from_temporal={'CO2': 0.2})  # Pull from CO2
    ```

!!! warning "No deprecation warning"
    Move shares to receiving effect and update parameter names throughout your code.

---

### Variable Names

| Category                        | Old (v2.x) | New (v3.0.0) |
|---------------------------------|------------|--------------|
| Investment                      | `is_invested` | `invested` |
| Switching                       | `switch_on` | `switch|on` |
| Switching                       | `switch_off` | `switch|off` |
| Switching                       | `switch_on_nr` | `switch|count` |
| Effects                         | `Effect(invest)|total` | `Effect(periodic)` |
| Effects        | `Effect(operation)|total` | `Effect(temporal)` |
| Effects | `Effect(operation)|total_per_timestep` | `Effect(temporal)|per_timestep` |
| Effects                    | `Effect|total` | `Effect` |

---

### String Labels

| What | Old (v2.x) | New (v3.0.0) |
|------|------------|--------------|
| Bus assignment | `bus=my_bus` (object) | `bus='electricity'` (string) |
| Effect shares | `{CO2: 0.2}` (object key) | `{'CO2': 0.2}` (string key) |

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

| Change | Description |
|--------|-------------|
| **FlowSystem copying** | Each `Calculation` gets its own copy (independent) |
| **do_modeling() return** | Returns `Calculation` object (access model via `.model` property) |
| **Storage arrays** | Arrays match timestep count (no extra element) |
| **Final charge state** | Use `relative_minimum_final_charge_state` / `relative_maximum_final_charge_state` |

---

### Other Changes

| Category               | Old (v2.x) | New (v3.0.0+) |
|------------------------|------------|---------------|
| System model class     | `SystemModel` | `FlowSystemModel` |
| Element submodel       | `Model` | `Submodel` |
| Logging default        | Enabled | Disabled (silent) |
| Enable console logging | (default) | `fx.CONFIG.Logging.enable_console('INFO')` or `fx.CONFIG.exploring()` |

---

## 🗑️ Deprecated Parameters

??? abstract "InvestParameters"

    | Old (v2.x) | New (v3.0.0) |
    |------------|--------------|
    | `fix_effects` | `effects_of_investment` |
    | `specific_effects` | `effects_of_investment_per_size` |
    | `divest_effects` | `effects_of_retirement` |
    | `piecewise_effects` | `piecewise_effects_of_investment` |

??? abstract "Effect"

    | Old (v2.x) | New (v3.0.0) |
    |------------|--------------|
    | `minimum_investment` | `minimum_periodic` |
    | `maximum_investment` | `maximum_periodic` |
    | `minimum_operation` | `minimum_temporal` |
    | `maximum_operation` | `maximum_temporal` |
    | `minimum_operation_per_hour` | `minimum_per_hour` |
    | `maximum_operation_per_hour` | `maximum_per_hour` |

??? abstract "Components"

    | Old (v2.x) | New (v3.0.0) |
    |------------|--------------|
    | `source` (parameter) | `outputs` |
    | `sink` (parameter) | `inputs` |
    | `prevent_simultaneous_sink_and_source` | `prevent_simultaneous_flow_rates` |

??? abstract "TimeSeriesData"

    | Old (v2.x) | New (v3.0.0) |
    |------------|--------------|
    | `agg_group` | `aggregation_group` |
    | `agg_weight` | `aggregation_weight` |

??? abstract "Calculation"

    | Old (v2.x) | New (v3.0.0) |
    |------------|--------------|
    | `active_timesteps=[0, 1, 2]` | Use `flow_system.sel()` or `flow_system.isel()` |

---

## ✨ New Features

??? success "Multi-Period Investments"

    ```python
    periods = pd.Index(['2020', '2030'])
    flow_system = fx.FlowSystem(time=timesteps, periods=periods)
    ```

??? success "Scenario-Based Optimization"

    | Parameter | Description | Example |
    |-----------|-------------|---------|
    | `scenarios` | Scenario index | `pd.Index(['low', 'base', 'high'], name='scenario')` |
    | `scenario_weights` | Probabilities | `[0.2, 0.6, 0.2]` |
    | `scenario_independent_sizes` | Separate capacities per scenario | `True` / `False` (default) |

    ```python
    flow_system = fx.FlowSystem(
        time=timesteps,
        scenarios=scenarios,
        scenario_weights=[0.2, 0.6, 0.2],
        scenario_independent_sizes=True
    )
    ```

??? success "Enhanced I/O"

    | Method | Description |
    |--------|-------------|
    | `flow_system.to_netcdf('file.nc')` | Save FlowSystem |
    | `fx.FlowSystem.from_netcdf('file.nc')` | Load FlowSystem |
    | `flow_system.sel(time=slice(...))` | Select by label |
    | `flow_system.isel(time=slice(...))` | Select by index |
    | `flow_system.resample(time='D')` | Resample timeseries |
    | `flow_system.copy()` | Deep copy |
    | `results.flow_system` | Access from results |

??? success "Effects Per Component"

    ```python
    effects_ds = results.effects_per_component

    # Access effect contributions by component
    print(effects_ds['total'].sel(effect='costs'))      # Total effects
    print(effects_ds['temporal'].sel(effect='CO2'))     # Temporal effects
    print(effects_ds['periodic'].sel(effect='costs'))   # Periodic effects
    ```

??? success "Storage Features"

    | Feature | Parameter | Description |
    |---------|-----------|-------------|
    | **Balanced storage** | `balanced=True` | Ensures charge_size == discharge_size |
    | **Final state min** | `relative_minimum_final_charge_state=0.5` | End at least 50% charged |
    | **Final state max** | `relative_maximum_final_charge_state=0.8` | End at most 80% charged |

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

| Category | Tasks |
|----------|-------|
| **Install** | • `pip install --upgrade flixopt` |
| **Breaking changes** | • Update [effect sharing](#effect-system-redesign)<br>• Update [variable names](#variable-names)<br>• Update [string labels](#string-labels)<br>• Fix [storage arrays](#flowsystem-calculation)<br>• Update [Calculation API](#flowsystem-calculation)<br>• Update [class names](#other-changes) |
| **Configuration** | • Enable [logging](#other-changes) if needed |
| **Deprecated** | • Update [deprecated parameters](#deprecated-parameters) (recommended) |
| **Testing** | • Test thoroughly<br>• Validate results match v2.x |

---

:material-book: [Docs](https://flixopt.github.io/flixopt/) • :material-github: [Issues](https://github.com/flixOpt/flixopt/issues) • :material-text-box: [Changelog](https://flixopt.github.io/flixopt/latest/changelog/99984-v3.0.0/)

!!! success "Welcome to flixopt v3.0.0! 🎉"
