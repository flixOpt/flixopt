# Batched Modeling Architecture

This document describes the architecture for batched (vectorized) modeling in flixopt, covering data organization, variable management, and constraint creation.

## Overview

The batched modeling architecture separates concerns into three layers:

```text
┌─────────────────────────────────────────────────────────────────┐
│                        User-Facing Layer                         │
│   Flow, Component, Storage, LinearConverter, Effect, Bus         │
│   (Individual elements with parameters)                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Data Layer                               │
│   FlowsData, StatusData, InvestmentData, StoragesData,           │
│   EffectsData, BusesData, ComponentsData, ConvertersData,        │
│   TransmissionsData                                              │
│   (Batched parameter access as xr.DataArray + validation)        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Model Layer                               │
│   FlowsModel, BusesModel, StoragesModel,                        │
│   InterclusterStoragesModel, ComponentsModel, ConvertersModel,   │
│   TransmissionsModel, EffectsModel                               │
│   (Variables, constraints, optimization logic)                   │
└─────────────────────────────────────────────────────────────────┘
```

## Design Decisions

### 1. Separation of Data and Model

**Problem:** Previously, individual element classes (Flow, Storage) contained both data and modeling logic, leading to:
- Repeated iteration over elements to build batched arrays
- Mixed concerns between parameter storage and optimization
- Difficulty in testing data preparation separately from constraint creation

**Solution:** Introduce dedicated `*Data` classes that:
- Batch parameters from individual elements into `xr.DataArray`
- Provide categorizations (e.g., `with_status`, `with_investment`)
- Cache computed properties for efficiency

```python
# Before: Repeated iteration in model code
for flow in flows:
    if flow.status_parameters is not None:
        # build arrays...

# After: Single property access
flow_ids_with_status = flows_data.with_status  # Cached list[str]
status_bounds = flows_data.uptime_bounds       # Cached xr.DataArray
```

### 2. Delegation Pattern for Nested Parameters

**Problem:** Parameters like `StatusParameters` and `InvestParameters` are nested within elements, requiring deep access patterns.

**Solution:** Create dedicated data classes that batch these nested parameters:

```python
class FlowsData:
    @cached_property
    def _status_data(self) -> StatusData | None:
        """Delegates to StatusData for status-related batching."""
        if not self.with_status:
            return None
        return StatusData(
            params=self.status_params,
            dim_name='flow',
            effect_ids=list(self._fs.effects.keys()),
            ...
        )

    # Properties delegate to _status_data
    @property
    def uptime_bounds(self) -> tuple[xr.DataArray, xr.DataArray] | None:
        return self._status_data.uptime_bounds if self._status_data else None
```

### 3. Effect Properties as DataArrays

**Problem:** Effect contributions (costs, emissions) were collected per-element, requiring complex aggregation.

**Solution:** Build effect factor arrays with `(element, effect, ...)` dimensions:

```python
# InvestmentData builds batched effect arrays
@cached_property
def effects_per_size(self) -> xr.DataArray | None:
    """(element, effect) - effects per unit size."""
    return self._build_effects('effects_of_investment_per_size')

# EffectsModel uses them directly
share = size_var * type_model.effects_per_size.fillna(0)
```

### 4. Builders for Shared Constraint Logic

**Problem:** Some constraint-creation logic (duration tracking, investment bounds, piecewise linearization) is shared across multiple `*Model` classes and shouldn't be duplicated.

**Solution:** Static `*Builder` classes in `features.py` contain reusable model-building algorithms. Unlike `*Data` classes (which batch **parameters**), Builders create **variables and constraints**:

| Builder Class | Used By | Purpose |
|---------------|---------|---------|
| `StatusBuilder` | `FlowsModel`, `ComponentsModel` | Duration tracking (uptime/downtime), startup/shutdown variables |
| `InvestmentBuilder` | `FlowsModel`, `StoragesModel` | Optional size bounds, linked periods, effect share creation |
| `MaskHelpers` | `BusesModel`, `ComponentsModel` | Mask matrices for batching element→flow relationships |

## Architecture Details

### Data Layer

All `*Data` classes live in `batched.py` and are accessed through the `BatchedAccessor` on `FlowSystem`:

```python
batched = flow_system.batched
batched.flows        # FlowsData
batched.storages     # StoragesData
batched.buses        # BusesData
batched.effects      # EffectsData
batched.components   # ComponentsData
batched.converters   # ConvertersData
batched.transmissions  # TransmissionsData
```

Instances are lazily created and cached. The same `*Data` instances used for validation are reused during model building.

#### FlowsData

Primary batched data container for flows (`dim_name='flow'`).

```python
class FlowsData:
    # Element access
    def __getitem__(self, label: str) -> Flow
    def get(self, label: str) -> Flow | None

    # Categorizations (list[str])
    with_status: list[str]           # Flows with status_parameters
    with_investment: list[str]       # Flows with invest_parameters
    with_effects: list[str]          # Flows with effects_per_flow_hour
    without_size: list[str]          # Flows without explicit size
    with_status_only: list[str]      # Status but no investment
    with_investment_only: list[str]  # Investment but no status
    with_status_and_investment: list[str]

    # Boolean masks (xr.DataArray, dim='flow')
    has_status: xr.DataArray
    has_investment: xr.DataArray
    has_size: xr.DataArray

    # Nested data (delegation)
    _status_data: StatusData | None
    _investment_data: InvestmentData | None

    # Parameter dicts (for nested data classes)
    invest_params: dict[str, InvestParameters]
    status_params: dict[str, StatusParameters]
```

#### StatusData

Batches `StatusParameters` for a group of elements. Reused by both `FlowsData` and `ComponentsData`.

```python
class StatusData:
    # Categorizations
    with_uptime_tracking: list[str]
    with_downtime_tracking: list[str]
    with_startup_limit: list[str]
    with_effects_per_active_hour: list[str]
    with_effects_per_startup: list[str]

    # Bounds (xr.DataArray with element dimension)
    min_uptime: xr.DataArray | None
    max_uptime: xr.DataArray | None
    min_downtime: xr.DataArray | None
    max_downtime: xr.DataArray | None

    # Previous durations
    previous_uptime: xr.DataArray | None
    previous_downtime: xr.DataArray | None

    # Effects
    effects_per_active_hour: xr.DataArray | None  # (element, effect)
    effects_per_startup: xr.DataArray | None      # (element, effect)
```

#### InvestmentData

Batches `InvestParameters` for a group of elements. Reused by `FlowsData` and `StoragesData`.

```python
class InvestmentData:
    # Categorizations
    with_optional: list[str]      # Non-mandatory investments
    with_mandatory: list[str]     # Mandatory investments

    # Size bounds
    size_minimum: xr.DataArray    # (element,)
    size_maximum: xr.DataArray    # (element,)

    # Effects (xr.DataArray with (element, effect) dims)
    effects_per_size: xr.DataArray | None
    effects_of_investment: xr.DataArray | None
    effects_of_retirement: xr.DataArray | None
```

#### StoragesData

Batched data container for storages (`dim_name='storage'` or `'intercluster_storage'`).

```python
class StoragesData:
    # Categorizations
    with_investment: list[str]
    with_optional_investment: list[str]
    with_mandatory_investment: list[str]
    with_balanced: list[str]

    # Storage parameters (xr.DataArray, dim='storage')
    eta_charge: xr.DataArray
    eta_discharge: xr.DataArray
    relative_loss_per_hour: xr.DataArray
    capacity_lower: xr.DataArray
    capacity_upper: xr.DataArray
    charge_state_lower_bounds: xr.DataArray
    charge_state_upper_bounds: xr.DataArray

    # Flow references
    charging_flow_ids: list[str]
    discharging_flow_ids: list[str]

    # Investment (delegation)
    investment_data: InvestmentData | None
```

#### EffectsData

Batched data container for effects (`dim_name='effect'`).

```python
class EffectsData:
    # Properties
    effect_ids: list[str]
    objective_effect_id: str
    penalty_effect_id: str

    # Bounds (xr.DataArray, dim='effect')
    minimum_periodic: xr.DataArray
    maximum_periodic: xr.DataArray
    minimum_temporal: xr.DataArray
    maximum_temporal: xr.DataArray
    minimum_total: xr.DataArray
    maximum_total: xr.DataArray
```

#### BusesData

Batched data container for buses (`dim_name='bus'`).

```python
class BusesData:
    element_ids: list[str]
    with_imbalance: list[str]       # Buses that allow imbalance
    imbalance_elements: list[Bus]   # Bus objects with imbalance settings
```

#### ComponentsData

Batched data container for generic components (`dim_name='component'`). Handles component-level status (not conversion or storage).

```python
class ComponentsData:
    element_ids: list[str]
    all_components: list[Component]
```

#### ConvertersData

Batched data container for linear converters (`dim_name='converter'`).

```python
class ConvertersData:
    element_ids: list[str]
    with_factors: list[LinearConverter]    # Standard linear conversion
    with_piecewise: list[LinearConverter]  # Piecewise conversion
```

#### TransmissionsData

Batched data container for transmissions (`dim_name='transmission'`).

```python
class TransmissionsData:
    element_ids: list[str]
    bidirectional: list[Transmission]  # Two-way transmissions
    balanced: list[Transmission]       # Balanced flow sizes
```

### Model Layer

All `*Model` classes extend `TypeModel` (from `structure.py`), which provides:
- Batched variable creation via `add_variables()`
- Batched constraint creation via `add_constraints()`
- Subscript access: `model['flow|rate']` returns the linopy variable
- Element slicing: `model.get_variable('flow|rate', 'Boiler(gas_in)')` returns a single element's variable

#### FlowsModel (`elements.py`)

Type-level model for ALL flows. Creates batched variables and constraints.

```python
class FlowsModel(TypeModel):
    data: FlowsData

    # Variables (linopy.Variable with 'flow' dimension)
    rate: linopy.Variable      # (flow, time, ...)
    status: linopy.Variable    # (flow, time, ...) — binary, masked to with_status
    size: linopy.Variable      # (flow, period, scenario) — masked to with_investment
    invested: linopy.Variable  # (flow, period, scenario) — binary, masked to optional

    # Status variables (masked to flows with status)
    startup: linopy.Variable
    shutdown: linopy.Variable
    active_hours: linopy.Variable
    startup_count: linopy.Variable
```

#### BusesModel (`elements.py`)

Type-level model for ALL buses. Creates balance constraints and imbalance variables.

```python
class BusesModel(TypeModel):
    data: BusesData

    # Variables (only for buses with imbalance)
    virtual_supply: linopy.Variable | None  # (bus, time, ...)
    virtual_demand: linopy.Variable | None  # (bus, time, ...)
```

#### StoragesModel (`components.py`)

Type-level model for ALL basic storages.

```python
class StoragesModel(TypeModel):
    data: StoragesData

    # Variables
    charge: linopy.Variable        # (storage, time+1, ...) — extra timestep
    netto: linopy.Variable         # (storage, time, ...)
    size: linopy.Variable | None   # (storage, period, scenario)
    invested: linopy.Variable | None  # (storage, period, scenario)
```

#### InterclusterStoragesModel (`components.py`)

Type-level model for intercluster storages (used in clustering/multi-period).

```python
class InterclusterStoragesModel(TypeModel):
    data: StoragesData  # dim_name='intercluster_storage'

    # Variables
    charge_state: linopy.Variable         # (intercluster_storage, time+1, ...)
    netto_discharge: linopy.Variable      # (intercluster_storage, time, ...)
    soc_boundary: linopy.Variable         # (cluster_boundary, intercluster_storage, ...)
    size: linopy.Variable | None
    invested: linopy.Variable | None
```

#### ComponentsModel (`elements.py`)

Handles component-level STATUS (not conversion). Links component status to flow statuses.

```python
class ComponentsModel(TypeModel):
    data: ComponentsData

    # Status variables (masked to components with status_parameters)
    status: linopy.Variable | None   # (component, time, ...)
    startup: linopy.Variable | None
    shutdown: linopy.Variable | None
    active_hours: linopy.Variable | None
    startup_count: linopy.Variable | None
```

#### ConvertersModel (`elements.py`)

Handles CONVERSION constraints for LinearConverter.

```python
class ConvertersModel(TypeModel):
    data: ConvertersData

    # Linear conversion: sum(flow_rate * coefficient * sign) == 0
    # Piecewise conversion: inside_piece, lambda0, lambda1 variables
```

#### TransmissionsModel (`elements.py`)

Handles transmission constraints (efficiency, balance, bidirectional logic).

```python
class TransmissionsModel(TypeModel):
    data: TransmissionsData
```

#### EffectsModel (`effects.py`)

Manages effect variables, contributions, and share aggregation.

```python
class EffectsModel:
    data: EffectsData

    # Variables (dim='effect')
    periodic: linopy.Variable          # (effect, period, scenario)
    temporal: linopy.Variable          # (effect, period, scenario)
    per_timestep: linopy.Variable      # (effect, time, ...)
    total: linopy.Variable             # (effect, period, scenario)
    total_over_periods: linopy.Variable  # (effect,)

    # Push-based contribution API
    def add_temporal_contribution(expr, ...)
    def add_periodic_contribution(expr, ...)
    def finalize_shares()  # Called after all models register contributions
```

## The Build Pipeline

The actual build sequence lives in `FlowSystemModel.build_model()` (`structure.py`). Before building, `connect_and_transform()` runs automatically to prepare the data.

### Pre-Build: `connect_and_transform()`

```text
1. _connect_network()           — wire flows to buses
2. _register_missing_carriers() — auto-register carriers from CONFIG
3. _assign_element_colors()     — assign default colors
4. _prepare_effects()           — create penalty effect if needed
5. element.transform_data()     — convert user inputs to xr.DataArray
6. _validate_system_integrity() — check cross-element references
7. _run_validation()            — run all *Data.validate() methods
```

### Build: `build_model()`

Each step creates a `*Model` instance which immediately creates its variables and constraints:

```text
1.  EffectsModel         — effect variables (periodic, temporal, total, ...)
2.  FlowsModel           — flow rate, status, size, investment constraints
3.  BusesModel            — bus balance constraints, imbalance variables
4.  StoragesModel         — charge state, energy balance, investment
5.  InterclusterStoragesModel — SOC boundary linking for clustering
6.  ComponentsModel       — component-level status features
7.  ConvertersModel       — linear/piecewise conversion constraints
8.  TransmissionsModel    — transmission efficiency/balance constraints
9.  Finalize:
      - _add_scenario_equality_constraints()
      - _populate_element_variable_names()
      - effects.finalize_shares()  ← collects all contributions
```

**Why this order matters:**

- `EffectsModel` is built first because other models register effect contributions into it via `add_temporal_contribution()` / `add_periodic_contribution()`.
- `FlowsModel` is built before `BusesModel`, `StoragesModel`, and `ComponentsModel` because they reference flow variables (e.g., bus balance sums flow rates; storages reference charging/discharging flows).
- `finalize_shares()` runs last to collect all effect contributions that were pushed during model building.

## Validation

Validation runs during `connect_and_transform()`, **after** element data is transformed to `xr.DataArray` but **before** model building.

### Validation Flow

```python
def _run_validation(self) -> None:
    batched = self.batched
    batched.buses.validate()                  # Bus config + DataArray checks
    batched.effects.validate()                # Effect config + share structure
    batched.flows.validate()                  # Flow config + DataArray checks
    batched.storages.validate()               # Storage config + capacity bounds
    batched.intercluster_storages.validate()  # Intercluster storage checks
    batched.converters.validate()             # Converter config
    batched.transmissions.validate()          # Transmission config + balanced sizes
    batched.components.validate()             # Generic component config
```

Each `*Data.validate()` method performs two categories of checks:

1. **Config validation** — calls `element.validate_config()` on each element (simple attribute checks)
2. **DataArray validation** — post-transformation checks on batched arrays (bounds consistency, capacity ranges, etc.)

Buses are validated first to catch structural issues (e.g., "Bus with no flows") before `FlowsData` tries to build arrays from an empty set.

The same cached `*Data` instances created during validation are reused during `build_model()`, so validation has zero redundant computation.

## Variable Storage

Variables are stored in each `*Model`'s `_variables` dict, keyed by their `type|name` string (e.g., `'flow|rate'`). `TypeModel` provides subscript access and optional element slicing:

```python
flows_model['flow|rate']                              # full batched variable
flows_model.get_variable('flow|rate', 'Boiler(gas)')  # single-element slice
'flow|status' in flows_model                          # membership test
```

For the complete list of variable names and dimensions, see [Variable Names](../variable_names.md).

## Data Flow

### Flow Rate Bounds Example

```text
Flow.relative_minimum (user input)
    │
    ▼
FlowsData._build_relative_bounds() [batched.py]
    │ Stacks into (flow, time, ...) DataArray
    ▼
FlowsData.relative_lower_bounds [cached property]
    │
    ▼
FlowsModel.rate [elements.py]
    │ Uses bounds in add_variables()
    ▼
linopy.Variable with proper bounds
```

### Investment Effects Example

```text
InvestParameters.effects_of_investment_per_size (user input)
    │
    ▼
InvestmentData._build_effects() [batched.py]
    │ Builds (element, effect) DataArray
    ▼
InvestmentData.effects_per_size [cached property]
    │
    ▼
FlowsModel.effects_per_size [elements.py]
    │ Delegates to data._investment_data
    ▼
EffectsModel._create_periodic_shares() [effects.py]
    │ Creates: share = size * effects_per_size
    ▼
effect|periodic constraint
```

## Performance Considerations

### xarray Access Patterns

Use `ds.variables[name]` for bulk metadata access (70-80x faster than `ds[name]`):

```python
# Fast: Access Variable objects directly
dims = {name: ds.variables[name].dims for name in ds.data_vars}

# Slow: Creates new DataArray each iteration
dims = {name: arr.dims for name, arr in ds.data_vars.items()}
```

### Cached Properties

All `*Data` classes use `@cached_property` for computed values:

```python
@cached_property
def uptime_bounds(self) -> tuple[xr.DataArray, xr.DataArray] | None:
    """Computed once, cached for subsequent access."""
    ...
```

### Single-Pass Building

Combine related computations to avoid repeated iteration:

```python
@cached_property
def uptime_bounds(self) -> tuple[xr.DataArray, xr.DataArray] | None:
    """Build both min and max in single pass."""
    ids = self.with_uptime_tracking
    if not ids:
        return None

    # Single iteration builds both arrays
    mins, maxs = [], []
    for eid in ids:
        p = self._params[eid]
        mins.append(p.minimum_uptime or 0)
        maxs.append(p.maximum_uptime or np.inf)

    min_arr = xr.DataArray(mins, dims=[self._dim], coords={self._dim: ids})
    max_arr = xr.DataArray(maxs, dims=[self._dim], coords={self._dim: ids})
    return min_arr, max_arr
```

## Summary

The batched modeling architecture provides:

1. **Clear separation**: Data preparation vs. optimization logic
2. **Efficient batching**: Single-pass array building with caching
3. **Consistent patterns**: All `*Model` classes follow similar structure
4. **Extensibility**: New element types can follow established patterns
5. **Testability**: Data classes can be tested independently

Key classes and their responsibilities:

| Class | Layer | File | Responsibility |
|-------|-------|------|----------------|
| `FlowsData` | Data | `batched.py` | Batch flow parameters, categorizations |
| `StatusData` | Data | `batched.py` | Batch status parameters (shared) |
| `InvestmentData` | Data | `batched.py` | Batch investment parameters (shared) |
| `StoragesData` | Data | `batched.py` | Batch storage parameters |
| `EffectsData` | Data | `batched.py` | Batch effect definitions and bounds |
| `BusesData` | Data | `batched.py` | Bus categorizations and imbalance info |
| `ComponentsData` | Data | `batched.py` | Generic component categorizations |
| `ConvertersData` | Data | `batched.py` | Converter categorizations |
| `TransmissionsData` | Data | `batched.py` | Transmission categorizations |
| `FlowsModel` | Model | `elements.py` | Flow variables and constraints |
| `BusesModel` | Model | `elements.py` | Bus balance constraints |
| `StoragesModel` | Model | `components.py` | Storage variables and constraints |
| `InterclusterStoragesModel` | Model | `components.py` | Intercluster storage linking |
| `ComponentsModel` | Model | `elements.py` | Component status features |
| `ConvertersModel` | Model | `elements.py` | Conversion constraints |
| `TransmissionsModel` | Model | `elements.py` | Transmission constraints |
| `EffectsModel` | Model | `effects.py` | Effect aggregation and shares |

### Design Principles

1. **Data classes batch, Model classes optimize**: Clear responsibility split
2. **Delegation for nested parameters**: StatusData/InvestmentData reusable across element types
3. **Cached properties**: Compute once, access many times
4. **Push-based effect collection**: Models push contributions to EffectsModel during build
5. **xarray for everything**: Consistent labeled array interface
