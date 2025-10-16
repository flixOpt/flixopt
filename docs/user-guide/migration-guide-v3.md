# Migration Guide: Upgrading to v3.0.0

Quick guide for migrating flixopt from v2.x to v3.0.0.

## Quick Start

```bash
pip install --upgrade flixopt
```

Review breaking changes, update deprecated parameters, and test thoroughly.

---

## Breaking Changes

### 1. Effect System Redesign

⚠️ **Multiple effect-related changes** - terminology and sharing system redesigned.

**Terminology Changes:**

Effect domains have been renamed for clarity:

| Old Term (v2.x) | New Term (v3.0.0) | Meaning                                                                 |
|-----------------|-------------------|-------------------------------------------------------------------------|
| `operation` | `temporal` | Time-varying effects (e.g., operational costs, occuring over time)      |
| `invest` / `investment` | `periodic` | Investment-related effects (e.g., fixed costs per period, annuity, ...) |

**Effect Sharing System (⚠️ No deprecation warning):**

Effects now "pull" shares from other effects instead of "pushing" them.

**v2.x:**
```python
CO2 = fx.Effect('CO2', 'kg', 'CO2 emissions',
    specific_share_to_other_effects_operation={'costs': 0.2})  # operation → temporal
land = fx.Effect('land', 'm²', 'Land usage',
    specific_share_to_other_effects_invest={'costs': 100})     # invest → periodic
costs = fx.Effect('costs', '€', 'Total costs')
```

**v3.0.0:**
```python
CO2 = fx.Effect('CO2', 'kg', 'CO2 emissions')
land = fx.Effect('land', 'm²', 'Land usage')
costs = fx.Effect('costs', '€', 'Total costs',
    share_from_temporal={'CO2': 0.2},      # Pulls from temporal effects
    share_from_periodic={'land': 100})     # Pulls from periodic effects
```

**Migration:**
1. Move share definitions to the receiving effect
2. Update parameter names:
   - `specific_share_to_other_effects_operation` → `share_from_temporal`
   - `specific_share_to_other_effects_invest` → `share_from_periodic`
3. Update terminology throughout your code:
   - Replace "operation" with "temporal" in effect-related contexts
   - Replace "invest/investment" with "periodic" in effect-related contexts

---

### 2. Variable Renaming in Results

| Old (v2.x) | New (v3.0.0) |
|------------|--------------|
| `is_invested` | `invested` |
| `switch_on` | `switch\|on` |
| `switch_off` | `switch\|off` |
| `switch_on_nr` | `switch\|count` |

```python
# Old: results.solution['component|is_invested']
# New: results.solution['component|invested']
```

---

### 3. Bus and Effect Assignment - Use String Labels

Pass string labels instead of objects:

**Bus Assignment:**
```python
# Old: flow = fx.Flow('P_el', bus=my_bus_object)
# New: flow = fx.Flow('P_el', bus='electricity')
```

**Effect Shares:**
```python
# Old: costs = fx.Effect('costs', '€', share_from_temporal={CO2_object: 0.2})
# New: costs = fx.Effect('costs', '€', share_from_temporal={'CO2': 0.2})
```

---

### 4. Storage Charge State Bounds

Array length now matches timesteps (no extra element):

```python
storage = fx.Storage(
    'storage',
    relative_minimum_charge_state=np.array([0.2, 0.2, 0.2, 0.2]),  # Matches timesteps
    relative_minimum_final_charge_state=0.3  # New: control final state explicitly
)
```

---

### 5. FlowSystem Independence

Each `Calculation` now gets its own FlowSystem copy - calculations are fully independent.

```python
# v2.x: FlowSystem was shared across calculations
flow_system = fx.FlowSystem(time=timesteps)
calc1 = fx.FullCalculation('calc1', flow_system)  # Shared reference
calc2 = fx.FullCalculation('calc2', flow_system)  # Same reference

# v3.0.0: Each calculation gets a copy
flow_system = fx.FlowSystem(time=timesteps)
calc1 = fx.FullCalculation('calc1', flow_system)
calc2 = fx.FullCalculation('calc2', flow_system)  # Gets separate copy
```

---

### 6. Other Breaking Changes

- **`do_modeling()` return value:** Now returns `Calculation` object (access model via `.model` property)
- **Plotting:** `mode` parameter renamed to `style`
- **Class names:** `SystemModel` → `FlowSystemModel`, `Model` → `Submodel`
- **Logging:** Disabled by default (enable with `fx.CONFIG.Logging.console = True; fx.CONFIG.apply()`)

---

## Deprecated Parameters (Still Work)

### InvestParameters

| Old | New |
|-----|-----|
| `fix_effects` | `effects_of_investment` |
| `specific_effects` | `effects_of_investment_per_size` |
| `divest_effects` | `effects_of_retirement` |
| `piecewise_effects` | `piecewise_effects_of_investment` |

### Effect

| Old | New |
|-----|-----|
| `minimum_investment` | `minimum_periodic` |
| `maximum_investment` | `maximum_periodic` |
| `minimum_operation` | `minimum_temporal` |
| `maximum_operation` | `maximum_temporal` |
| `minimum_operation_per_hour` | `minimum_per_hour` |
| `maximum_operation_per_hour` | `maximum_per_hour` |

### SourceAndSink

| Old | New |
|-----|-----|
| `source` | `outputs` |
| `sink` | `inputs` |
| `prevent_simultaneous_sink_and_source` | `prevent_simultaneous_flow_rates` |

### TimeSeriesData

| Old | New |
|-----|-----|
| `agg_group` | `aggregation_group` |
| `agg_weight` | `aggregation_weight` |

### Calculation

Replace `active_timesteps` with FlowSystem selection:
```python
# Old: calculation = fx.FullCalculation('calc', flow_system, active_timesteps=[0, 1, 2])
# New:
fs_subset = flow_system.isel(time=slice(0, 3))
calculation = fx.FullCalculation('calc', fs_subset)
```

---

## New Features

### Multi-Period Investments

```python
periods = pd.Index(['2020', '2030'])
flow_system = fx.FlowSystem(time=timesteps, periods=periods)

solar = fx.Source('solar', outputs=[fx.Flow('P_el', bus='electricity',
    size=fx.InvestParameters(minimum_size=0, maximum_size=1000))])
```

### Scenario-Based Stochastic Optimization

```python
scenarios = pd.Index(['low', 'base', 'high'], name='scenario')
flow_system = fx.FlowSystem(time=timesteps, scenarios=scenarios,
    scenario_weights=[0.2, 0.6, 0.2],
    scenario_independent_sizes=True)  # Optional: scenario-specific capacities
```

### Enhanced I/O

```python
flow_system.to_netcdf('system.nc')
fs = fx.FlowSystem.from_netcdf('system.nc')
fs_subset = flow_system.sel(time=slice('2020-01', '2020-06'))
fs_resampled = flow_system.resample(time='D')
```

### Effects Per Component

```python
effects_ds = calculation.results.effects_per_component()
print(effects_ds['costs'])  # Costs by component
```

### Balanced Storage

```python
storage = fx.Storage('storage',
    charging=fx.Flow('charge', bus='electricity', size=fx.InvestParameters(...)),
    discharging=fx.Flow('discharge', bus='electricity', size=fx.InvestParameters(...)),
    balanced=True,  # Charge size == discharge size
    capacity_in_flow_hours=100)
```

---

## Common Issues

**"Effect share parameters not working"**
→ Move shares to receiving effect using `share_from_temporal`/`share_from_periodic`

**"Storage charge state has wrong dimensions"**
→ Remove extra timestep; use `relative_minimum_final_charge_state`

**"KeyError when accessing results"**
→ Update variable names (`is_invested` → `invested`, etc.)

**"No logging output"**
→ Enable explicitly: `fx.CONFIG.Logging.console = True; fx.CONFIG.apply()`

---

## Migration Checklist

**Critical:**
- [ ] Update flixopt: `pip install --upgrade flixopt`
- [ ] Update effect sharing syntax
- [ ] Update result variable names
- [ ] Replace object assignments with string labels
- [ ] Fix storage charge state arrays
- [ ] Update `do_modeling()` usage if needed
- [ ] Rename plotting `mode` → `style`
- [ ] Enable logging if needed

**Recommended:**
- [ ] Update deprecated parameter names
- [ ] Test thoroughly and validate results

**Optional:**
- [ ] Explore new features (periods, scenarios, balanced storage)

**Welcome to flixopt v3.0.0!** 🎉
---

## Resources

- **Docs:** https://flixopt.github.io/flixopt/latest/
- **Issues:** https://github.com/flixOpt/flixopt/issues
- **Changelog:** https://flixopt.github.io/flixopt/latest/changelog/99984-v3.0.0/
