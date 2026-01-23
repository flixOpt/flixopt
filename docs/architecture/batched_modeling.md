# Batched Modeling Architecture

This document describes the architecture for batched (vectorized) modeling in flixopt, covering data organization, variable management, and constraint creation.

## Overview

The batched modeling architecture separates concerns into three layers:

```
┌─────────────────────────────────────────────────────────────────┐
│                        User-Facing Layer                         │
│   Flow, Component, Storage, LinearConverter, Effect, Bus         │
│   (Individual elements with parameters)                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Data Layer                               │
│   FlowsData, StatusData, InvestmentData                          │
│   (Batched parameter access as xr.DataArray)                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Model Layer                               │
│   FlowsModel, StoragesModel, ComponentsModel, ConvertersModel    │
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

### 4. Helpers for Complex Math

**Problem:** Some operations (duration tracking, piecewise linearization) involve complex math that shouldn't be duplicated.

**Solution:** Static helper classes contain reusable algorithms:

| Helper Class | Purpose |
|--------------|---------|
| `StatusHelpers` | Duration tracking (uptime/downtime), status feature creation |
| `InvestmentHelpers` | Optional size bounds, linked periods, effect stacking |
| `PiecewiseHelpers` | Segment variables, lambda interpolation, coupling constraints |
| `MaskHelpers` | Bounds masking, status-size interactions |

## Architecture Details

### Data Layer

#### FlowsData (`batched.py`)

Primary batched data container for flows. Accessed via `flow_system.batched.flows`.

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

    # Nested data (delegation)
    _status_data: StatusData | None
    _investment_data: InvestmentData | None

    # Batched parameters (xr.DataArray)
    absolute_lower_bounds: xr.DataArray  # (flow, time, ...)
    absolute_upper_bounds: xr.DataArray  # (flow, time, ...)
    effects_per_flow_hour: xr.DataArray  # (flow, effect, ...)
```

#### StatusData (`batched.py`)

Batches `StatusParameters` for a group of elements.

```python
class StatusData:
    # Categorizations
    with_uptime_tracking: list[str]
    with_downtime_tracking: list[str]
    with_startup_limit: list[str]

    # Bounds (xr.DataArray with element dimension)
    uptime_bounds: tuple[xr.DataArray, xr.DataArray] | None   # (min, max)
    downtime_bounds: tuple[xr.DataArray, xr.DataArray] | None

    # Previous durations (computed from previous_states)
    previous_uptime: xr.DataArray | None
    previous_downtime: xr.DataArray | None

    # Effects
    effects_per_active_hour: xr.DataArray | None  # (element, effect)
    effects_per_startup: xr.DataArray | None      # (element, effect)
```

#### InvestmentData (`batched.py`)

Batches `InvestParameters` for a group of elements.

```python
class InvestmentData:
    # Categorizations
    with_optional: list[str]      # Non-mandatory investments
    with_mandatory: list[str]     # Mandatory investments
    with_piecewise_effects: list[str]

    # Size bounds
    size_minimum: xr.DataArray    # (element,)
    size_maximum: xr.DataArray    # (element,)
    optional_size_minimum: xr.DataArray | None
    optional_size_maximum: xr.DataArray | None

    # Effects (xr.DataArray with (element, effect) dims)
    effects_per_size: xr.DataArray | None
    effects_of_investment: xr.DataArray | None
    effects_of_retirement: xr.DataArray | None

    # Constant effects (list for direct addition)
    effects_of_investment_mandatory: list[tuple[str, dict]]
    effects_of_retirement_constant: list[tuple[str, dict]]
```

### Model Layer

#### FlowsModel (`elements.py`)

Type-level model for ALL flows. Creates batched variables and constraints.

```python
class FlowsModel(TypeModel):
    # Data access
    @property
    def data(self) -> FlowsData

    # Variables (linopy.Variable with 'flow' dimension)
    rate: linopy.Variable      # (flow, time, ...)
    status: linopy.Variable    # (flow, time, ...) - binary
    size: linopy.Variable      # (flow, period, scenario)
    invested: linopy.Variable  # (flow, period, scenario) - binary

    # Status variables
    startup: linopy.Variable
    shutdown: linopy.Variable
    uptime: linopy.Variable
    downtime: linopy.Variable
    active_hours: linopy.Variable

    # Effect properties (delegating to data._investment_data)
    effects_per_size: xr.DataArray | None
    effects_of_investment: xr.DataArray | None
    effects_of_retirement: xr.DataArray | None
```

#### StoragesModel (`components.py`)

Type-level model for ALL storages.

```python
class StoragesModel(TypeModel):
    # Data access
    invest_params: dict[str, InvestParameters]
    _investment_data: InvestmentData | None

    # Variables
    charge_state: linopy.Variable   # (storage, time, ...)
    netto_discharge: linopy.Variable
    size: linopy.Variable           # (storage, period, scenario)
    invested: linopy.Variable       # (storage, period, scenario)

    # Effect properties (same interface as FlowsModel)
    effects_per_size: xr.DataArray | None
    effects_of_investment: xr.DataArray | None
    # ...
```

#### ComponentsModel (`elements.py`)

Handles component STATUS (not conversion). Links component status to flow statuses.

```python
class ComponentsModel:
    # Status variable
    status: linopy.Variable  # (component, time, ...)

    # Status features (via StatusHelpers)
    startup: linopy.Variable
    shutdown: linopy.Variable
    # ...
```

#### ConvertersModel (`elements.py`)

Handles CONVERSION constraints for LinearConverter.

```python
class ConvertersModel:
    # Linear conversion
    def create_linear_constraints(self)
        # sum(flow_rate * coefficient * sign) == 0

    # Piecewise conversion
    def create_piecewise_variables(self)
        # inside_piece, lambda0, lambda1

    def create_piecewise_constraints(self)
        # lambda_sum, single_segment, coupling
```

## Variable Storage

Variables are stored in model classes with a consistent pattern:

```python
class TypeModel:
    _variables: dict[str, linopy.Variable]

    @cached_property
    def some_variable(self) -> linopy.Variable:
        var = self.model.add_variables(...)
        self._variables['some_variable'] = var
        return var

    def get_variable(self, name: str, element_id: str = None):
        """Access variable, optionally selecting specific element."""
        var = self._variables.get(name)
        if element_id:
            return var.sel({self.dim_name: element_id})
        return var
```

**Storage locations:**

| Variable Type | Stored In | Dimension |
|---------------|-----------|-----------|
| Flow rate | `FlowsModel._variables['rate']` | `(flow, time, ...)` |
| Flow status | `FlowsModel._variables['status']` | `(flow, time, ...)` |
| Flow size | `FlowsModel._variables['size']` | `(flow, period, scenario)` |
| Storage charge | `StoragesModel._variables['charge_state']` | `(storage, time, ...)` |
| Storage size | `StoragesModel._variables['size']` | `(storage, period, scenario)` |
| Component status | `ComponentsModel._variables['status']` | `(component, time, ...)` |
| Effect totals | `EffectsModel._variables` | `(effect, ...)` |

## Data Flow

### Flow Rate Bounds Example

```
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

```
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

## Future Development

### 1. Migration of Per-Element Operations

Currently, individual element classes handle three main operations:

| Operation | Method | Purpose |
|-----------|--------|---------|
| Linking | `link_to_flow_system()` | Propagate FlowSystem reference to nested objects |
| Transformation | `transform_data()` | Convert user inputs to `xr.DataArray` |
| Validation | `_plausibility_checks()` | Validate parameter consistency |

#### Current Implementation (Per-Element)

```python
class Flow(Element):
    def link_to_flow_system(self, flow_system, prefix: str = '') -> None:
        """Propagate flow_system reference to nested Interface objects."""
        super().link_to_flow_system(flow_system, self.label_full)
        if self.status_parameters is not None:
            self.status_parameters.link_to_flow_system(flow_system, ...)
        if isinstance(self.size, InvestParameters):
            self.size.link_to_flow_system(flow_system, ...)

    def transform_data(self) -> None:
        """Convert user inputs to xr.DataArray with proper dimensions."""
        self.relative_minimum = self._fit_coords(..., self.relative_minimum)
        self.relative_maximum = self._fit_coords(..., self.relative_maximum)
        self.effects_per_flow_hour = self._fit_effect_coords(...)
        # ... many more fields
        if self.status_parameters is not None:
            self.status_parameters.transform_data()

    def _plausibility_checks(self) -> None:
        """Validate parameter consistency."""
        if self.size is None and self.status_parameters is not None:
            raise PlausibilityError(
                f'Flow "{self.label_full}" has status_parameters but no size.'
            )
        if self.size is None and np.any(self.relative_minimum > 0):
            raise PlausibilityError(
                f'Flow "{self.label_full}" has relative_minimum > 0 but no size.'
            )
        # ... many more checks
```

**Problems with current approach:**
- Fail-fast: First error stops validation, hiding other issues
- Repeated iteration: Each element validated separately
- Scattered logic: Related validations spread across classes
- Hard to test: Validation tightly coupled to element construction

#### Migration Strategy

##### Phase 1: Validation in *Data Classes

Move validation to `*Data` classes, collecting all errors before raising:

```python
@dataclass
class ValidationError:
    element_id: str
    field: str
    message: str
    severity: Literal['error', 'warning'] = 'error'


class FlowsData:
    def validate(self) -> list[ValidationError]:
        """Validate all flows, returning all errors at once."""
        errors = []
        errors.extend(self._validate_size_requirements())
        errors.extend(self._validate_bounds_consistency())
        errors.extend(self._validate_status_parameters())
        return errors

    def _validate_size_requirements(self) -> list[ValidationError]:
        """Check that size-dependent features have size defined."""
        errors = []
        missing_size = set(self.without_size)

        # Flows with status_parameters need size (for big-M)
        for fid in self.with_status:
            if fid in missing_size:
                errors.append(ValidationError(
                    element_id=fid,
                    field='size',
                    message='status_parameters requires size for big-M constraints'
                ))

        # Flows with relative_minimum > 0 need size
        if self.relative_lower_bounds is not None:
            has_nonzero_min = (self.relative_lower_bounds > 0).any(dim='time')
            for fid in has_nonzero_min.coords['flow'].values:
                if bool(has_nonzero_min.sel(flow=fid)) and fid in missing_size:
                    errors.append(ValidationError(
                        element_id=fid,
                        field='size',
                        message='relative_minimum > 0 requires size'
                    ))

        return errors

    def _validate_bounds_consistency(self) -> list[ValidationError]:
        """Check that lower bounds <= upper bounds."""
        errors = []
        if self.relative_lower_bounds is None or self.relative_upper_bounds is None:
            return errors

        # Batched comparison across all flows
        invalid = self.relative_lower_bounds > self.relative_upper_bounds
        if invalid.any():
            for fid in invalid.coords['flow'].values:
                if invalid.sel(flow=fid).any():
                    errors.append(ValidationError(
                        element_id=fid,
                        field='relative_bounds',
                        message='relative_minimum > relative_maximum'
                    ))

        return errors

    def raise_if_invalid(self) -> None:
        """Validate and raise if any errors found."""
        errors = self.validate()
        if errors:
            error_msgs = [f"  - {e.element_id}: {e.message}" for e in errors if e.severity == 'error']
            warning_msgs = [f"  - {e.element_id}: {e.message}" for e in errors if e.severity == 'warning']

            for msg in warning_msgs:
                logger.warning(msg)

            if error_msgs:
                raise PlausibilityError(
                    f"Validation failed with {len(error_msgs)} error(s):\n" +
                    "\n".join(error_msgs)
                )
```

**Benefits:**
- All errors reported at once
- Batched checks using xarray operations
- Clear categorization of validation types
- Warnings vs errors distinguished
- Testable in isolation

##### Phase 2: Data Transformation in *Data Classes

Move coordinate fitting to data classes, applied during batching:

```python
class FlowsData:
    def __init__(self, flows: dict[str, Flow], flow_system: FlowSystem):
        self._flows = flows
        self._fs = flow_system
        # Transformation happens here, not in individual Flow objects

    @cached_property
    def relative_lower_bounds(self) -> xr.DataArray:
        """Build batched relative_minimum, fitting coords during construction."""
        arrays = []
        for fid, flow in self._flows.items():
            # Fit coords here instead of in Flow.transform_data()
            arr = self._fit_to_coords(
                flow.relative_minimum,
                dims=['time', 'period', 'scenario']
            )
            arrays.append(arr.expand_dims({self._dim: [fid]}))
        return xr.concat(arrays, dim=self._dim)
```

**Note:** This requires careful consideration of when transformation happens:
- Currently: During `FlowSystem.add_elements()` → `transform_data()`
- Future: During `FlowsData` construction (lazy, on first access)

##### Phase 3: Linking in *Data Classes

The `link_to_flow_system` pattern could be simplified:

```python
class FlowsData:
    def __init__(self, flows: dict[str, Flow], flow_system: FlowSystem):
        self._fs = flow_system

        # Set flow_system reference on all nested objects
        for flow in flows.values():
            if flow.status_parameters is not None:
                flow.status_parameters._flow_system = flow_system
            if isinstance(flow.size, InvestParameters):
                flow.size._flow_system = flow_system
```

Or, better, have `*Data` classes own the reference and provide it when needed:

```python
class StatusData:
    def __init__(self, params: dict[str, StatusParameters], flow_system: FlowSystem):
        self._params = params
        self._fs = flow_system  # StatusData owns the reference

    @cached_property
    def effects_per_active_hour(self) -> xr.DataArray | None:
        # Uses self._fs.effects directly, no linking needed
        effect_ids = list(self._fs.effects.keys())
        return self._build_effects('effects_per_active_hour', effect_ids)
```

#### Validation Categories

Organize validation by category for clarity:

| Category | Example Checks | Location |
|----------|----------------|----------|
| **Structural** | Size required for status | `FlowsData._validate_size_requirements()` |
| **Bounds** | min <= max | `FlowsData._validate_bounds_consistency()` |
| **Cross-element** | Bus balance possible | `BusesData._validate_connectivity()` |
| **Temporal** | Previous state length matches | `StatusData._validate_previous_states()` |
| **Effects** | Effect IDs exist | `InvestmentData._validate_effect_references()` |

#### Example: StatusData Validation

```python
class StatusData:
    def validate(self) -> list[ValidationError]:
        errors = []

        # Uptime bounds consistency
        if self.uptime_bounds is not None:
            min_up, max_up = self.uptime_bounds
            invalid = min_up > max_up
            for eid in invalid.coords[self._dim].values:
                if bool(invalid.sel({self._dim: eid})):
                    errors.append(ValidationError(
                        element_id=eid,
                        field='uptime',
                        message=f'minimum_uptime ({min_up.sel({self._dim: eid}).item()}) > '
                                f'maximum_uptime ({max_up.sel({self._dim: eid}).item()})'
                    ))

        # Previous state length
        if self.previous_uptime is not None:
            for eid in self.with_uptime_tracking:
                prev = self._params[eid].previous_uptime
                min_up = self._params[eid].minimum_uptime or 0
                if prev is not None and prev < min_up:
                    errors.append(ValidationError(
                        element_id=eid,
                        field='previous_uptime',
                        message=f'previous_uptime ({prev}) < minimum_uptime ({min_up}), '
                                f'constraint will be violated at t=0'
                    ))

        return errors
```

#### Example: InvestmentData Validation

```python
class InvestmentData:
    def validate(self) -> list[ValidationError]:
        errors = []

        # Size bounds consistency
        invalid = self.size_minimum > self.size_maximum
        for eid in invalid.coords[self._dim].values:
            if bool(invalid.sel({self._dim: eid})):
                errors.append(ValidationError(
                    element_id=eid,
                    field='size',
                    message='minimum_size > maximum_size'
                ))

        # Effect references exist
        for eid in self.with_effects_per_size:
            effects = self._params[eid].effects_of_investment_per_size
            for effect_name in effects.keys():
                if effect_name not in self._effect_ids:
                    errors.append(ValidationError(
                        element_id=eid,
                        field='effects_of_investment_per_size',
                        message=f'Unknown effect "{effect_name}"'
                    ))

        return errors
```

#### Integration with Model Building

Validation runs automatically when accessing data:

```python
class FlowsData:
    _validated: bool = False

    def _ensure_validated(self) -> None:
        if not self._validated:
            self.raise_if_invalid()
            self._validated = True

    @cached_property
    def absolute_lower_bounds(self) -> xr.DataArray:
        self._ensure_validated()  # Validate on first data access
        return self._build_absolute_bounds('lower')
```

Or explicitly during model creation:

```python
class FlowSystemModel:
    def __init__(self, flow_system: FlowSystem):
        # Validate all data before building model
        self._validate_all_data()

    def _validate_all_data(self) -> None:
        all_errors = []
        all_errors.extend(self.flow_system.batched.flows.validate())
        all_errors.extend(self.flow_system.batched.buses.validate())
        # ... other data classes

        if any(e.severity == 'error' for e in all_errors):
            raise PlausibilityError(self._format_errors(all_errors))

### 2. StatusData for Components

**Current:** ComponentsModel builds status data inline.

**Future:** Create `ComponentStatusData` similar to flow's `StatusData`:

```python
class ComponentStatusData:
    """Batched status data for components."""

    @cached_property
    def uptime_bounds(self) -> tuple[xr.DataArray, xr.DataArray] | None:
        """(component,) bounds for components with uptime tracking."""
        ...
```

### 3. Unified Effect Collection

**Current:** Effects are collected separately for flows, storages, and components.

**Future:** Unified `EffectsData` that aggregates all effect contributions:

```python
class EffectsData:
    """Batched effect data from all sources."""

    @cached_property
    def all_temporal_effects(self) -> xr.DataArray:
        """(source, effect, time, ...) - all temporal effect contributions."""
        sources = []
        if self._flows_data.effects_per_flow_hour is not None:
            sources.append(('flows', self._flows_data.effects_per_flow_hour))
        # ... storages, components
        return xr.concat(...)
```

### 4. Lazy Data Building

**Current:** All data properties are built eagerly on first access.

**Future:** Consider lazy building with explicit `prepare()` step:

```python
class FlowsData:
    def prepare(self, categories: list[str] = None):
        """Pre-build specified data categories."""
        if categories is None or 'bounds' in categories:
            _ = self.absolute_lower_bounds
            _ = self.absolute_upper_bounds
        if categories is None or 'status' in categories:
            _ = self._status_data
```

### 5. Serialization Support

**Future:** Add serialization for data classes to support:
- Caching computed data between runs
- Debugging data preparation issues
- Parallel model building

```python
class FlowsData:
    def to_dataset(self) -> xr.Dataset:
        """Export all batched data as xr.Dataset."""
        ...

    @classmethod
    def from_dataset(cls, ds: xr.Dataset, flows: dict[str, Flow]) -> FlowsData:
        """Reconstruct from serialized dataset."""
        ...
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

## Migration Roadmap

### Current State (v1.0)

| Component | Data Class | Model Class | Validation | Notes |
|-----------|------------|-------------|------------|-------|
| Flows | `FlowsData` | `FlowsModel` | Per-element | Fully batched |
| Status (flows) | `StatusData` | `FlowsModel` | Per-element | Delegates from FlowsData |
| Investment (flows) | `InvestmentData` | `FlowsModel` | Per-element | Delegates from FlowsData |
| Storages | - | `StoragesModel` | Per-element | Uses InvestmentData |
| Components | - | `ComponentsModel` | Per-element | Status only |
| Converters | - | `ConvertersModel` | Per-element | Linear + piecewise |
| Buses | - | `BusesModel` | Per-element | Balance constraints |
| Effects | - | `EffectsModel` | Per-element | Aggregation |

### Target State (v2.0)

| Component | Data Class | Validation | Migration Priority |
|-----------|------------|------------|-------------------|
| Flows | `FlowsData` | `FlowsData.validate()` | High |
| Status | `StatusData` | `StatusData.validate()` | High |
| Investment | `InvestmentData` | `InvestmentData.validate()` | High |
| Storages | `StoragesData` | `StoragesData.validate()` | Medium |
| Components | `ComponentsData` | `ComponentsData.validate()` | Medium |
| Converters | `ConvertersData` | `ConvertersData.validate()` | Low |
| Buses | `BusesData` | `BusesData.validate()` | Low |
| Effects | `EffectsData` | `EffectsData.validate()` | Low |

### Migration Steps

#### Step 1: Add Validation to Existing *Data Classes

```python
# StatusData.validate() - already has data, add validation
# InvestmentData.validate() - already has data, add validation
# FlowsData.validate() - delegates to nested + own checks
```

#### Step 2: Create Missing *Data Classes

```python
class StoragesData:
    """Batched data for storages."""
    _investment_data: InvestmentData | None

class ComponentsData:
    """Batched data for components with status."""
    _status_data: StatusData | None

class ConvertersData:
    """Batched data for converters."""
    # Linear conversion coefficients
    # Piecewise breakpoints
```

#### Step 3: Migrate transform_data()

Move coordinate fitting from elements to data classes:

```python
# Before (in Flow.__init__ or transform_data)
self.relative_minimum = self._fit_coords(...)

# After (in FlowsData property)
@cached_property
def relative_lower_bounds(self) -> xr.DataArray:
    return self._batch_and_fit([f.relative_minimum for f in self._flows.values()])
```

#### Step 4: Simplify link_to_flow_system()

Remove need for explicit linking by having *Data classes own FlowSystem reference:

```python
# Before
flow.link_to_flow_system(flow_system, prefix)
flow.status_parameters.link_to_flow_system(...)

# After
# FlowsData receives flow_system in __init__
# StatusData receives it via FlowsData
# No explicit linking needed
```

## Testing Strategy

### Unit Testing *Data Classes

```python
class TestFlowsData:
    def test_categorizations(self, sample_flows):
        data = FlowsData(sample_flows, mock_flow_system)
        assert data.with_status == ['flow_with_status']
        assert data.with_investment == ['flow_with_invest']

    def test_bounds_batching(self, sample_flows):
        data = FlowsData(sample_flows, mock_flow_system)
        bounds = data.absolute_lower_bounds
        assert bounds.dims == ('flow', 'time')
        assert bounds.sel(flow='flow1').values == pytest.approx([0, 0, 0])

    def test_validation_size_required(self):
        flows = {'bad': Flow('bad', status_parameters=StatusParameters(), size=None)}
        data = FlowsData(flows, mock_flow_system)
        errors = data.validate()
        assert len(errors) == 1
        assert 'size' in errors[0].message

    def test_validation_all_errors_collected(self):
        """Verify all errors are returned, not just first."""
        flows = {
            'bad1': Flow('bad1', status_parameters=StatusParameters(), size=None),
            'bad2': Flow('bad2', relative_minimum=0.5, size=None),
        }
        data = FlowsData(flows, mock_flow_system)
        errors = data.validate()
        assert len(errors) == 2  # Both errors reported
```

### Integration Testing

```python
class TestDataModelIntegration:
    def test_flows_data_to_model(self, flow_system):
        """Verify FlowsData properties are correctly used by FlowsModel."""
        model = FlowSystemModel(flow_system)
        flows_model = model._flows_model

        # Data layer provides correct bounds
        assert flows_model.data.absolute_lower_bounds is not None

        # Model layer uses them correctly
        rate_var = flows_model.rate
        assert rate_var.lower.equals(flows_model.data.absolute_lower_bounds)

    def test_validation_before_model(self, invalid_flow_system):
        """Verify validation runs before model building."""
        with pytest.raises(PlausibilityError) as exc_info:
            FlowSystemModel(invalid_flow_system)
        assert 'Validation failed' in str(exc_info.value)
```

## Summary

The batched modeling architecture provides:

1. **Clear separation**: Data preparation vs. optimization logic
2. **Efficient batching**: Single-pass array building with caching
3. **Consistent patterns**: All `*Model` classes follow similar structure
4. **Extensibility**: New element types can follow established patterns
5. **Testability**: Data classes can be tested independently
6. **Better validation**: All errors reported at once, batched checks

Key classes and their responsibilities:

| Class | Layer | Responsibility |
|-------|-------|----------------|
| `FlowsData` | Data | Batch flow parameters, validation |
| `StatusData` | Data | Batch status parameters |
| `InvestmentData` | Data | Batch investment parameters |
| `FlowsModel` | Model | Flow variables and constraints |
| `StoragesModel` | Model | Storage variables and constraints |
| `ComponentsModel` | Model | Component status features |
| `ConvertersModel` | Model | Conversion constraints |
| `EffectsModel` | Model | Effect aggregation |

### Design Principles

1. **Data classes batch, Model classes optimize**: Clear responsibility split
2. **Delegation for nested parameters**: StatusData/InvestmentData reusable
3. **Cached properties**: Compute once, access many times
4. **Validation collects all errors**: User sees complete picture
5. **xarray for everything**: Consistent labeled array interface
