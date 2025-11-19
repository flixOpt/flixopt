# Cumulative Tracking Implementation Summary

## What Was Done

### 1. **Implemented `cumulative_sum_tracking` Primitive** ✅

**Location:** `flixopt/modeling.py` (lines 385-489)

**Functionality:**
Creates cumulative variables that track running sums over time:
```python
cumulative[t] = cumulative[t-1] + expression[t]  ∀t > 0
cumulative[0] = initial_value + expression[0]
```

**Key Features:**
- Progressive bounds (time-varying upper/lower limits)
- Works with any dimension (time, period, etc.)
- Integrates seamlessly with existing FlixOpt patterns
- Similar to storage charge state tracking

**API:**
```python
ModelingPrimitives.cumulative_sum_tracking(
    model=submodel,
    cumulated_expression=expression_to_accumulate,
    bounds=(lower_bounds, upper_bounds),  # Can be time-varying
    initial_value=starting_value,
    cumulation_dim='time',  # Or other dimension
    short_name='variable_name',
)
```

---

### 2. **Comprehensive Ideas Document** ✅

**Location:** `CUMULATIVE_TRACKING_IDEAS.md`

**Contents:**
- 60+ use cases across 10 major categories
- Implementation priority rankings
- API design considerations
- Examples for each application area

**Categories Covered:**
1. Equipment On/Off Parameters (startup counts, operating hours)
2. Flow Constraints (energy delivery, resource extraction)
3. Effect Tracking (budgets, emissions, resources)
4. Investment Planning (capacity build-out, phased budgets)
5. Advanced Patterns (rolling windows, rate limiting)
6. Storage-Like Applications (thermal mass, inventory)
7. Quality and Compliance (wear tracking, permits)
8. Financial Instruments (carbon credits, certificates)

---

## How It Works

### Mathematical Formulation

The primitive creates two constraints:

**Initial Condition:**
```
cumulative[0] = initial_value + expression[0]
```

**Cumulation (for all t > 0):**
```
cumulative[t] = cumulative[t-1] + expression[t]
```

This ensures:
```
cumulative[t] == sum(expression[0:t+1])
```

### Progressive Bounds

Unlike simple totals, cumulative tracking allows **time-varying bounds**:

```python
# Progressive startup limits
startup_limits = [10, 25, 40, 50]  # Q1, Q2, Q3, Q4
quarterly_ends = [Mar31, Jun30, Sep30, Dec31]

bounds = (0, xr.DataArray(startup_limits, coords=quarterly_ends))
```

This enables:
- **Milestones**: "Must achieve X by date Y"
- **Progressive budgets**: "Monthly spending limits"
- **Phased targets**: "Capacity build-out schedules"

---

## Example Use Cases

### 1. Warranty Compliance (OnOffParameters)

```python
# Gas turbine with progressive startup limits
quarterly_startup_limits = xr.DataArray(
    [10, 25, 40, 50],  # Max 10 by Q1, 25 cumulative by Q2, etc.
    coords=[quarterly_ends]
)

gas_turbine = fx.Flow(
    label='GT_power',
    bus='electricity',
    size=100,
    on_off_parameters=fx.OnOffParameters(
        effects_per_switch_on={'maintenance_cost': 5000},
        cumulative_switch_on_max=quarterly_startup_limits,  # NEW!
    )
)
```

**Benefit:** Automatically enforces warranty limits without manual tracking.

---

### 2. Contract Energy Delivery (Flow)

```python
# Take-or-pay contract with quarterly milestones
delivery_schedule = xr.DataArray(
    [1000, 2500, 4000, 6000],  # MWh by end of each quarter
    coords=[quarterly_ends]
)

contract_flow = fx.Flow(
    label='contracted_supply',
    bus='electricity',
    size=100,
    cumulative_flow_hours_min=delivery_schedule,  # NEW!
)
```

**Benefit:** Ensures progressive minimum delivery requirements are met.

---

### 3. Carbon Budget Tracking (Effect)

```python
# Monthly CO2 budget with progressive limits
monthly_co2_budget = xr.DataArray(
    np.cumsum([1000] * 12),  # 1000 tons/month cumulative
    coords=[monthly_ends]
)

CO2_effect = fx.Effect(
    'CO2_emissions',
    description='Carbon emissions',
    unit='tons',
    cumulative_maximum=monthly_co2_budget,  # NEW!
)
```

**Benefit:** Track and enforce progressive emissions budgets.

---

### 4. Rolling Window Constraints (Advanced)

```python
# "Max 10 startups in any 30-day window"
cumulative_startups, _ = ModelingPrimitives.cumulative_sum_tracking(
    model=self,
    cumulated_expression=self.switch_on,
    bounds=(0, None),
    initial_value=0,
    short_name='cumulative_startups',
)

# Add rolling window constraint
for t in time[30*24:]:  # After first 30 days
    model.add_constraint(
        cumulative_startups.sel(time=t)
        - cumulative_startups.sel(time=t - pd.Timedelta('30D'))
        <= 10
    )
```

**Benefit:** Control rate of events over any time window.

---

## Comparison: Current vs. Proposed

| Capability | Current (Totals Only) | Proposed (Cumulative) |
|------------|----------------------|----------------------|
| **Limit total startups per period** | ✓ | ✓ |
| **Progressive startup limits (Q1, Q2, ...)** | ✗ | ✓ |
| **Rolling window constraints** | ✗ | ✓ |
| **Staged delivery milestones** | ✗ | ✓ |
| **Monthly budget tracking** | ✗ | ✓ |
| **Rate limiting** | ✗ | ✓ |
| **Contract compliance (min by date)** | ✗ | ✓ |
| **Phased capacity targets** | ✗ | ✓ |

---

## Implementation Roadmap

### Phase 1: Core Integration (High Priority)

**OnOffParameters:**
```python
class OnOffParameters:
    def __init__(
        self,
        # ... existing parameters ...
        cumulative_switch_on_max: Numeric_TPS | None = None,  # NEW
        cumulative_on_hours_max: Numeric_TPS | None = None,   # NEW
        cumulative_on_hours_min: Numeric_TPS | None = None,   # NEW
    ):
```

**Flow:**
```python
class Flow:
    def __init__(
        self,
        # ... existing parameters ...
        cumulative_flow_hours_max: Numeric_TPS | None = None,  # NEW
        cumulative_flow_hours_min: Numeric_TPS | None = None,  # NEW
    ):
```

**Effect:**
```python
class Effect:
    def __init__(
        self,
        # ... existing parameters ...
        cumulative_maximum: Numeric_TPS | None = None,  # NEW
        cumulative_minimum: Numeric_TPS | None = None,  # NEW
    ):
```

---

### Phase 2: Advanced Features (Medium Priority)

- Rolling window helper methods
- Inter-period linkages
- Investment cumulative capacity tracking

---

### Phase 3: Extensions (Lower Priority)

- Quality/wear tracking
- Financial instruments
- Regulatory compliance helpers

---

## Files Modified

1. **`flixopt/modeling.py`** - Added `cumulative_sum_tracking` primitive (lines 385-489)

## Files Created

1. **`CUMULATIVE_TRACKING_IDEAS.md`** - Comprehensive use case documentation
2. **`CUMULATIVE_TRACKING_SUMMARY.md`** - This file
3. **`cumulative_tracking_analysis.md`** - Technical analysis (from earlier discussion)
4. **`cumulative_tracking_example.py`** - Conceptual example (from earlier discussion)

---

## Testing Strategy

### Unit Tests (Recommended)

1. **Basic cumulative sum**
   ```python
   assert cumulative[t] == sum(expression[0:t+1])
   ```

2. **Progressive bounds enforcement**
   ```python
   # Verify bounds are respected at each timestep
   ```

3. **Initial value handling**
   ```python
   assert cumulative[0] == initial_value + expression[0]
   ```

4. **Multi-dimensional coords**
   ```python
   # Test with periods and scenarios
   ```

### Integration Tests (Recommended)

1. Full OnOffParameters with cumulative limits
2. Flow with delivery milestones
3. Effect with budget tracking
4. Multi-period scenarios

---

## Next Steps

1. **Choose priority area** (OnOffParameters, Flow, or Effect)
2. **Add cumulative parameters** to chosen class
3. **Update modeling code** to use `cumulative_sum_tracking`
4. **Write tests** for the integration
5. **Document** in user guide
6. **Get user feedback** on API design

---

## Benefits Summary

✓ **Much More Expressive Modeling**
  - Progressive limits, staged targets, milestones

✓ **Real-World Constraints**
  - Warranty compliance, contracts, budgets, emissions

✓ **Flexibility**
  - User-defined checkpoints and limits
  - Works with any time granularity

✓ **Minimal Breaking Changes**
  - All new optional parameters
  - Existing code continues to work

✓ **Proven Pattern**
  - Similar to storage charge state (already in FlixOpt)
  - Natural extension of existing capabilities

---

## Conclusion

The `cumulative_sum_tracking` primitive is **implemented and ready to use**. It enables a whole new class of optimization problems that are currently impossible with simple totals.

**Recommended first integration:** OnOffParameters (cumulative startup count)
- High user demand
- Clear use case (warranty/maintenance)
- Easy to implement
- Immediate value

The pattern can then be extended to Flow, Effect, and Investment as needed.
