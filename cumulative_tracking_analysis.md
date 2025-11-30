# Cumulative Tracking Analysis and Recommendations

## Current State Analysis

### 1. **Startup Count (StatusParameters)**
**Current Implementation:**
```python
# Line 218-224 in features.py
count = self.add_variables(
    lower=0,
    upper=self.parameters.startup_limit,
    coords=self._model.get_coords(('period', 'scenario')),
    short_name='startup_count',
)
self.add_constraints(count == self.startup.sum('time'), short_name='startup_count')
```

**Problem:**
- Only creates a single scalar variable per period/scenario
- Constraint: `startup_count == sum(startup[t])`
- **Not cumulative over time** - just tracks the total

**Desired Behavior:**
- Make `startup_count[t]` cumulative: `startup_count[t] = sum(startup[0:t+1])`
- This would allow constraints like:
  - "Maximum 10 startups in first 100 hours"
  - "At most 5 startups per week"
  - "Limit startup rate in certain periods"

---

### 2. **Active Hours (StatusParameters)**
**Current Implementation:**
```python
# Line 191-200 in features.py
ModelingPrimitives.expression_tracking_variable(
    self,
    tracked_expression=(self.status * self._model.hours_per_step).sum('time'),
    bounds=(
        self.parameters.active_hours_min if self.parameters.active_hours_min is not None else 0,
        self.parameters.active_hours_max if self.parameters.active_hours_max is not None else None,
    ),
    short_name='active_hours',
    coords=['period', 'scenario'],
)
```

**Problem:**
- Only tracks total over entire time horizon per period
- Single variable per period/scenario

**Would Benefit From Cumulative:**
- Track cumulative operating hours over time: `active_hours[t]`
- Enable constraints like:
  - "Must run at least 100 hours within first month"
  - "Cannot exceed 500 hours in any rolling 30-day window"
  - "Ramp up constraints on total usage"

---

### 3. **Flow Hours (Flow)**
**Current Implementation:**
```python
# Line 691-701 in elements.py
ModelingPrimitives.expression_tracking_variable(
    model=self,
    name=f'{self.label_full}|total_flow_hours',
    tracked_expression=(self.flow_rate * self._model.hours_per_step).sum('time'),
    bounds=(
        self.element.flow_hours_min if self.element.flow_hours_min is not None else 0,
        self.element.flow_hours_max if self.element.flow_hours_max is not None else None,
    ),
    coords=['period', 'scenario'],
    short_name='total_flow_hours',
)
```

**Problem:**
- Same as active_hours - only total per period

**Would Benefit From Cumulative:**
- Track cumulative energy delivered: `cumulative_flow_hours[t]`
- Enable constraints like:
  - "Deliver at least X MWh by end of Q1"
  - "Maximum energy budget per month"
  - "Staged delivery requirements"

---

## Other Areas That Would Benefit

### 4. **Effect Accumulation**
Currently effects are just summed at the end, but cumulative tracking would enable:
- **Budget constraints over time**: "Don't exceed €1M in first quarter"
- **Emissions tracking**: "Stay under CO2 limit for any 7-day period"
- **Resource consumption**: "Fuel usage cannot exceed X tons by mid-year"

### 5. **Investment Decisions**
For multi-period investment planning:
- **Cumulative installed capacity**: Track total capacity installed up to time t
- **Phased deployment**: "Must have 100 MW installed by year 3"
- **Budget staging**: "Spend at most €50M in years 1-2"

### 6. **Storage Charge State** (Already cumulative!)
Storage already has this pattern:
```python
charge[t] = charge[t-1] + inflow[t] - outflow[t]
```
This is the model to follow!

---

## Design Pattern: Cumulative Variable Tracking

### Proposed Pattern
```python
def cumulative_variable_tracking(
    model: Submodel,
    cumulated_expression: linopy.expressions.LinearExpression,  # What to accumulate
    bounds: tuple[Numeric_TPS | None, Numeric_TPS | None] = (None, None),
    initial_value: Numeric_PS = 0,
    short_name: str,
    coords: list[str] | None = None,
) -> linopy.Variable:
    """
    Create a cumulative variable that tracks the running sum of an expression.

    Creates: cumulative[t] = cumulative[t-1] + cumulated_expression[t]

    Args:
        model: The model to add variables to
        cumulated_expression: Expression to accumulate (must have 'time' dimension)
        bounds: (lower, upper) bounds for cumulative variable at each time step
        initial_value: Starting value at t=0
        short_name: Name for the variable
        coords: Coordinate dimensions (default: same as expression)

    Returns:
        The cumulative variable

    Example:
        # Cumulative startup count
        cumulative_startups = cumulative_variable_tracking(
            model=self,
            cumulated_expression=self.startup,  # Binary variable
            bounds=(0, None),  # Non-negative
            initial_value=0,
            short_name='cumulative_startups',
        )

        # Now you can add flexible constraints:
        # Max 10 startups in first 100 hours
        model.add_constraints(
            cumulative_startups.sel(time=slice(0, 100)) <= 10,
            short_name='startup_limit_early'
        )
    """
    # Implementation would be similar to storage charge state tracking
    pass
```

---

## Flexible Constraint Patterns Enabled

### Pattern 1: Rolling Window Constraints
```python
# Maximum X events in any Y-hour window
for t in range(len(time) - window_size):
    model.add_constraints(
        cumulative[t + window_size] - cumulative[t] <= max_in_window,
        short_name=f'rolling_window_{t}'
    )
```

### Pattern 2: Staged Milestones
```python
# Must achieve certain cumulative values by specific times
model.add_constraints(
    cumulative.sel(time='2025-03-31') >= 1000,  # Q1 target
    short_name='q1_milestone'
)
model.add_constraints(
    cumulative.sel(time='2025-06-30') >= 2500,  # H1 target
    short_name='h1_milestone'
)
```

### Pattern 3: Rate Limiting
```python
# Limit rate of accumulation in certain periods
critical_period = time.sel(time=slice('2025-01', '2025-03'))
model.add_constraints(
    cumulative.sel(time=critical_period[-1]) - cumulative.sel(time=critical_period[0]) <= limit,
    short_name='rate_limit_q1'
)
```

### Pattern 4: Budget Allocation
```python
# Different limits for different phases
model.add_constraints(
    cumulative.sel(time='2025-12-31') <= phase1_budget,
    short_name='phase1_budget'
)
model.add_constraints(
    cumulative.sel(time='2026-12-31') <= phase1_budget + phase2_budget,
    short_name='phase2_budget'
)
```

---

## Implementation Recommendations

### Priority 1: Add Cumulative Startup Count
**Impact:** HIGH - Enables sophisticated maintenance scheduling, warranty management
```python
# In StatusParameters
cumulative_startup_limit: Numeric_TPS | None = None  # Time-varying limit
cumulative_startup_limit_rolling_window: tuple[int, Numeric_PS] | None = None  # (hours, max_count)
```

### Priority 2: Add Cumulative Active Hours
**Impact:** HIGH - Enables progressive usage limits, maintenance windows
```python
# In StatusParameters
cumulative_active_hours_max: Numeric_TPS | None = None  # Max cumulative at each time
cumulative_active_hours_min: Numeric_TPS | None = None  # Min cumulative at each time
```

### Priority 3: Add Cumulative Flow Hours
**Impact:** MEDIUM-HIGH - Enables staged delivery, energy quotas
```python
# In Flow
cumulative_flow_hours_max: Numeric_TPS | None = None
cumulative_flow_hours_min: Numeric_TPS | None = None
```

### Priority 4: Add Cumulative Effects
**Impact:** MEDIUM - Enables budget tracking, emissions monitoring
```python
# In Effect
cumulative_maximum: Numeric_TPS | None = None  # Max cumulative at each time
cumulative_minimum: Numeric_TPS | None = None  # Min cumulative at each time
```

---

## Example Use Cases

### Use Case 1: Gas Turbine Maintenance Scheduling
```python
# Limited starts per maintenance interval
gas_turbine = fx.Flow(
    label='GT_power',
    bus='electricity',
    size=100,
    status_parameters=fx.StatusParameters(
        effects_per_startup={'maintenance_cost': 5000},
        min_uptime=4,
        # NEW: Cumulative constraint
        cumulative_startup_limit=xr.DataArray(
            [10, 20, 30, 40, 50],  # Progressive limits
            coords=[maintenance_schedule]  # Every 1000 hours
        )
    )
)
```

### Use Case 2: Contract Energy Delivery
```python
# Must deliver minimum energy by certain dates
supply_contract = fx.Flow(
    label='contracted_supply',
    bus='electricity',
    size=50,
    # NEW: Cumulative flow hours
    cumulative_flow_hours_min=xr.DataArray(
        [1000, 3000, 6000, 10000],  # MWh by end of each quarter
        coords=[quarter_ends]
    )
)
```

### Use Case 3: Emissions Budget
```python
# Annual emissions budget with monthly checkpoints
boiler = fx.linear_converters.Boiler(
    label='boiler',
    thermal_efficiency=0.85,
    thermal_flow=fx.Flow(
        'heat',
        bus='heat_bus',
        size=100,
        effects_per_flow_hour={'CO2': 0.2}  # kg CO2 / kWh
    ),
    fuel_flow=fx.Flow('fuel', bus='gas'),
)

# NEW: Add cumulative effect constraint to the effect
CO2 = fx.Effect(
    'CO2',
    unit='kg',
    cumulative_maximum=monthly_co2_budget  # Array with progressive limits
)
```

---

## Implementation Approach

### Step 1: Create Modeling Primitive
Add to `modeling.py`:
```python
@staticmethod
def cumulative_variable_tracking(
    model: Submodel,
    cumulated_expression: linopy.expressions.LinearExpression,
    bounds: tuple[Numeric_TPS | None, Numeric_TPS | None] = (None, None),
    initial_value: Numeric_PS = 0,
    short_name: str,
) -> linopy.Variable:
    """Create cumulative tracking variable."""
    # Similar to consecutive_duration_tracking but simpler
    # cumulative[t] = cumulative[t-1] + expression[t]
```

### Step 2: Update StatusParameters
Add cumulative options to `StatusParameters.__init__()`:
```python
cumulative_startup_limit: Numeric_TPS | None = None
cumulative_active_hours_max: Numeric_TPS | None = None
cumulative_active_hours_min: Numeric_TPS | None = None
```

### Step 3: Update StatusModel
In `StatusModel._do_modeling()`, create cumulative variables when specified

### Step 4: Extend to Flow and Effect
Apply same pattern to Flow (cumulative_flow_hours) and Effect (cumulative effects)

---

## Benefits

1. **Much More Expressive Modeling**
   - Progressive limits
   - Staged delivery requirements
   - Rolling window constraints
   - Rate limiting

2. **Real-World Constraints**
   - Maintenance schedules
   - Contract compliance
   - Budget tracking
   - Emissions monitoring

3. **Flexibility**
   - User defines limits as time-series
   - Can be constant, stepped, or continuous
   - Works with periods and scenarios

4. **Minimal Breaking Changes**
   - All new optional parameters
   - Existing code continues to work
   - Backwards compatible

---

## Conclusion

Making these variables **cumulative over time** (like storage charge state) would enable a whole new class of optimization problems and constraints. The pattern is well-established (storage), and extending it to startup counts, active hours, flow hours, and effects would be natural and powerful.

**Recommended Next Steps:**
1. Implement `cumulative_variable_tracking()` primitive in modeling.py
2. Add to StatusParameters (startup_count, active_hours) as proof of concept
3. Get user feedback on the API
4. Extend to Flow and Effect if successful
