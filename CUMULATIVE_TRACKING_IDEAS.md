# Cumulative Tracking Applications and Ideas

This document collects ideas and use cases for the new `cumulative_sum_tracking` primitive implemented in `modeling.py`.

---

## Overview

The `cumulative_sum_tracking` primitive creates variables that track running sums over time:
```
cumulative[t] = cumulative[t-1] + expression[t]
```

This enables **progressive constraints**, **milestones**, **budgets**, and **rolling windows** - much more flexible than simple totals.

---

## 1. Equipment On/Off Parameters

### 1.1 Cumulative Startup Count

**Current Limitation:**
- Only tracks total startups per period (scalar)
- Cannot limit startups in sub-periods or progressive phases

**With Cumulative Tracking:**
```python
# Progressive startup limits for warranty compliance
startup_limits = xr.DataArray(
    [10, 25, 40, 50],  # Cumulative max by each quarter
    coords=[pd.DatetimeIndex(['2025-03-31', '2025-06-30', '2025-09-30', '2025-12-31'])]
)

# In OnOffParameters
self.cumulative_startup_count, _ = ModelingPrimitives.cumulative_sum_tracking(
    model=self,
    cumulated_expression=self.switch_on,  # Binary startup variable
    bounds=(0, startup_limits),
    initial_value=0,
    short_name='cumulative_startups',
)
```

**Use Cases:**
- **Warranty compliance**: "Max 10 starts in first quarter, 25 cumulative by mid-year"
- **Maintenance scheduling**: "Service required after 50 cumulative starts"
- **Equipment degradation**: Progressive limits based on wear accumulation
- **Rolling windows**: "Max 5 starts per 30-day window" (difference between cumulative values)

**API Extension:**
```python
OnOffParameters(
    effects_per_switch_on={'cost': 1000},
    switch_on_max=50,  # OLD: Total limit per period
    cumulative_switch_on_max=startup_limits,  # NEW: Progressive limits
)
```

---

### 1.2 Cumulative Operating Hours

**Current Limitation:**
- Only tracks total operating hours per period
- Cannot enforce progressive usage requirements

**With Cumulative Tracking:**
```python
# Minimum operating hour milestones (e.g., maintenance requirements)
min_operating_hours = xr.DataArray(
    [100, 500, 1200, 2000],  # Min cumulative by quarter
    coords=[quarterly_ends]
)

self.cumulative_on_hours, _ = ModelingPrimitives.cumulative_sum_tracking(
    model=self,
    cumulated_expression=self.on * hours_per_step,
    bounds=(min_operating_hours, None),
    initial_value=0,
    short_name='cumulative_on_hours',
)
```

**Use Cases:**
- **Contract compliance**: "Must run at least 100 hours per month"
- **Testing requirements**: Progressive testing schedules
- **Phased usage**: Ramp-up/ramp-down constraints
- **Maintenance windows**: "Maximum 500 hours before service"

**API Extension:**
```python
OnOffParameters(
    on_hours_min=2000,  # OLD: Total per period
    cumulative_on_hours_min=min_hours_progressive,  # NEW: Progressive mins
    cumulative_on_hours_max=max_hours_progressive,  # NEW: Progressive maxs
)
```

---

## 2. Flow Constraints

### 2.1 Cumulative Energy Delivery

**Current Limitation:**
- Only total flow_hours per period
- Cannot model staged delivery or take-or-pay contracts

**With Cumulative Tracking:**
```python
# Contract delivery milestones
delivery_schedule = xr.DataArray(
    [1000, 2500, 4000, 6000],  # MWh by end of each quarter
    coords=[quarterly_ends]
)

# In Flow model
self.cumulative_flow_hours, _ = ModelingPrimitives.cumulative_sum_tracking(
    model=self,
    cumulated_expression=self.flow_rate * self._model.hours_per_step,
    bounds=(delivery_schedule, None),  # Minimum delivery by date
    initial_value=0,
    short_name='cumulative_flow_hours',
)
```

**Use Cases:**
- **Contract milestones**: "Deliver min 1000 MWh by Q1, 2500 by Q2..."
- **Take-or-pay contracts**: Progressive minimum delivery requirements
- **Quotas**: Maximum energy delivery by specific dates
- **Load curves**: Ensure progressive consumption patterns

**API Extension:**
```python
Flow(
    label='contracted_supply',
    bus='electricity',
    size=100,
    flow_hours_min=6000,  # OLD: Total per period
    cumulative_flow_hours_min=delivery_milestones,  # NEW: Progressive mins
    cumulative_flow_hours_max=quota_limits,  # NEW: Progressive maxs
)
```

---

### 2.2 Cumulative Resource Extraction

**For natural resource management:**
```python
# Annual extraction quota with monthly checkpoints
monthly_quotas = xr.DataArray(
    np.arange(1, 13) * 100,  # 100, 200, ..., 1200 tons cumulative
    coords=[monthly_ends]
)

extraction_flow = fx.Flow(
    label='mining_extraction',
    bus='ore',
    size=50,
    cumulative_flow_hours_max=monthly_quotas,  # Progressive extraction limit
)
```

**Use Cases:**
- **Mining quotas**: Progressive monthly/annual limits
- **Water rights**: Cumulative withdrawal limits
- **Fishing/Forestry**: Sustainable harvest schedules
- **Well production**: Cumulative production targets

---

## 3. Effect Tracking (Costs, Emissions, Resources)

### 3.1 Cumulative Budget Tracking

**Current Limitation:**
- Effects only sum at the end
- Cannot track spending progress or enforce monthly budgets

**With Cumulative Tracking:**
```python
# Monthly budget limits
monthly_budget = xr.DataArray(
    np.cumsum([100_000] * 12),  # €100k per month cumulative
    coords=[monthly_ends]
)

# In Effect model
cumulative_costs, _ = ModelingPrimitives.cumulative_sum_tracking(
    model=self,
    cumulated_expression=total_cost_per_timestep,
    bounds=(None, monthly_budget),  # Progressive budget limit
    initial_value=0,
    short_name='cumulative_costs',
)
```

**Use Cases:**
- **Budget management**: Monthly/quarterly spending limits
- **Cash flow**: Ensure progressive payment capabilities
- **Phased projects**: Budget allocation per phase
- **Cost control**: Early warning of budget overruns

**API Extension:**
```python
Cost_Effect = fx.Effect(
    'operational_costs',
    unit='€',
    maximum_total=1_200_000,  # OLD: Total per period
    cumulative_maximum=monthly_budget,  # NEW: Progressive budget
)
```

---

### 3.2 Cumulative Emissions Tracking

**Current Limitation:**
- Only total emissions per period
- Cannot enforce monthly/quarterly emissions limits

**With Cumulative Tracking:**
```python
# Monthly CO2 budget (e.g., 1000 tons/month)
monthly_co2_budget = xr.DataArray(
    np.arange(1, 13) * 1000,  # Progressive monthly budget
    coords=[monthly_ends]
)

CO2_effect = fx.Effect(
    'CO2_emissions',
    unit='tons',
    cumulative_maximum=monthly_co2_budget,  # Progressive emissions budget
)
```

**Use Cases:**
- **Carbon budgets**: Monthly/quarterly emission allowances
- **Compliance**: Progressive limits for environmental permits
- **Trading**: Track progress toward emission caps
- **NOx/SOx/PM**: Progressive limits for air quality

**API Extension:**
```python
Effect(
    label='CO2',
    unit='tons',
    maximum_total=12_000,  # OLD: Annual total
    cumulative_maximum=monthly_co2_limits,  # NEW: Monthly budgets
    cumulative_minimum=min_emissions_for_carbon_credits,  # NEW: Minimums too
)
```

---

### 3.3 Cumulative Resource Consumption

**For tracking scarce resources:**
```python
# Fuel quota with progressive limits
fuel_quota = xr.DataArray(
    [500, 1200, 2000, 3000],  # GJ by quarter
    coords=[quarterly_ends]
)

Fuel_effect = fx.Effect(
    'fuel_consumption',
    unit='GJ',
    cumulative_maximum=fuel_quota,
)
```

**Use Cases:**
- **Fuel quotas**: Progressive consumption limits
- **Water usage**: Cumulative consumption tracking
- **Material consumption**: Progressive inventory draw-down
- **Spare parts budget**: Track cumulative usage

---

## 4. Investment Planning

### 4.1 Cumulative Installed Capacity

**For multi-period investment:**
```python
# Progressive capacity build-out targets
capacity_targets = xr.DataArray(
    [100, 300, 600, 1000],  # MW by year
    coords=[yearly_milestones]
)

# Track cumulative investment decisions
cumulative_capacity, _ = ModelingPrimitives.cumulative_sum_tracking(
    model=self,
    cumulated_expression=investment_decision * component_size,
    bounds=(capacity_targets, None),  # Must meet progressive targets
    initial_value=existing_capacity,
    short_name='cumulative_capacity',
)
```

**Use Cases:**
- **Renewable targets**: "500 MW solar by 2030, 1000 MW by 2040"
- **Grid expansion**: Progressive infrastructure build-out
- **Retirement schedules**: Track cumulative capacity changes
- **Regulatory compliance**: Meet progressive capacity requirements

---

### 4.2 Cumulative Investment Budget

**For phased capital deployment:**
```python
# Investment budget by phase
investment_budget = xr.DataArray(
    [50_000_000, 150_000_000, 300_000_000],  # €50M, €100M, €150M per phase
    coords=[phase_ends]
)

cumulative_investment, _ = ModelingPrimitives.cumulative_sum_tracking(
    model=self,
    cumulated_expression=investment_costs,
    bounds=(None, investment_budget),  # Progressive budget limits
    initial_value=0,
    short_name='cumulative_capex',
)
```

**Use Cases:**
- **Phased deployment**: Budget allocation per phase
- **Financing constraints**: Match investment to funding availability
- **Risk management**: Limit early-stage capital exposure
- **Cash flow**: Align investment with revenue generation

---

## 5. Advanced Constraint Patterns

### 5.1 Rolling Window Constraints

**Using differences between cumulative values:**
```python
# Max 10 startups in any 30-day window
for t in time_index[30*24:]:  # Skip first 30 days
    model.add_constraint(
        cumulative_startups.sel(time=t)
        - cumulative_startups.sel(time=t - pd.Timedelta('30D'))
        <= 10,
        name=f'rolling_startup_limit_{t}'
    )
```

**Use Cases:**
- **Startup rate limiting**: Control cycling frequency
- **Emissions**: "Max 1000 tons CO2 per month" (rolling)
- **Usage caps**: "Max 500 hours operation per quarter" (rolling)
- **Budget**: "Max €100k spending per 30 days" (rolling)

---

### 5.2 Rate Limiting

**Control accumulation speed:**
```python
# Limit rate of change in certain periods
critical_months = time.sel(time=slice('2025-01', '2025-03'))
model.add_constraint(
    cumulative.sel(time=critical_months[-1])
    - cumulative.sel(time=critical_months[0])
    <= rate_limit,
    name='rate_limit_q1'
)
```

**Use Cases:**
- **Gradual ramp-up**: Prevent sudden capacity changes
- **Market impact**: Limit speed of trading/production
- **Resource protection**: Control extraction/consumption rate
- **System stability**: Gradual operational changes

---

### 5.3 Inter-Period Linkages

**Link cumulative values across periods:**
```python
# Must start next period where previous ended
model.add_constraint(
    cumulative.sel(period=2025, time=0)
    == cumulative.sel(period=2024, time=-1),
    name='carry_over_2024_2025'
)
```

**Use Cases:**
- **Multi-year tracking**: Carry-over of cumulative metrics
- **Long-term degradation**: Equipment wear across years
- **Debt/Credit**: Carry-over of financial positions
- **Storage reserves**: Multi-year reservoir management

---

## 6. Storage-Like Applications

### 6.1 Thermal Mass / Building Energy

**Track accumulated heat in building mass:**
```python
# Building thermal storage
cumulative_heat, _ = ModelingPrimitives.cumulative_sum_tracking(
    model=self,
    cumulated_expression=heating_input - cooling_loss - comfort_demand,
    bounds=(min_comfort_level, max_comfort_level),  # Temperature bounds
    initial_value=initial_temperature,
    short_name='building_thermal_mass',
)
```

**Use Cases:**
- **Building thermal mass**: Track temperature/enthalpy
- **Soil temperature**: Underground thermal storage
- **Industrial processes**: Heat accumulation in furnaces
- **Phase change materials**: Latent heat storage

---

### 6.2 Inventory / Stock Management

**Track material inventory:**
```python
# Warehouse inventory
cumulative_stock, _ = ModelingPrimitives.cumulative_sum_tracking(
    model=self,
    cumulated_expression=inflow - outflow,
    bounds=(safety_stock, warehouse_capacity),
    initial_value=initial_stock,
    short_name='inventory_level',
)
```

**Use Cases:**
- **Inventory management**: Track stock levels
- **Just-in-time**: Progressive material delivery
- **Seasonal storage**: Accumulate/deplete inventory
- **Buffer management**: Safety stock requirements

---

### 6.3 Financial Instruments

**Track cumulative financial positions:**
```python
# Carbon credit accumulation/usage
cumulative_credits, _ = ModelingPrimitives.cumulative_sum_tracking(
    model=self,
    cumulated_expression=credits_earned - credits_used,
    bounds=(0, None),  # Cannot go negative
    initial_value=existing_credits,
    short_name='carbon_credit_balance',
)
```

**Use Cases:**
- **Carbon credits**: Earn/spend over time
- **Renewable certificates**: Track REC accumulation
- **Financial options**: Progressive position building
- **Cash balance**: Track operating cash flow

---

## 7. Quality and Compliance

### 7.1 Cumulative Quality Metrics

**Track quality degradation:**
```python
# Equipment reliability tracking
cumulative_wear, _ = ModelingPrimitives.cumulative_sum_tracking(
    model=self,
    cumulated_expression=operating_stress_per_hour,
    bounds=(None, maintenance_threshold),
    initial_value=current_wear_level,
    short_name='cumulative_wear',
)
```

**Use Cases:**
- **Equipment wear**: Progressive degradation tracking
- **Reliability**: Cumulative failure probability
- **Product quality**: Process drift monitoring
- **Calibration**: Track time since last calibration

---

### 7.2 Regulatory Compliance

**Track permit limits:**
```python
# Annual operating permit with monthly checkpoints
monthly_permit_hours = xr.DataArray(
    np.cumsum([100] * 12),  # 100 hours/month cumulative
    coords=[monthly_ends]
)

cumulative_operation, _ = ModelingPrimitives.cumulative_sum_tracking(
    model=self,
    cumulated_expression=operating_hours,
    bounds=(None, monthly_permit_hours),
    initial_value=0,
    short_name='permitted_operation',
)
```

**Use Cases:**
- **Operating permits**: Progressive hour/production limits
- **Environmental limits**: Staged compliance requirements
- **Safety metrics**: Cumulative exposure tracking
- **Noise/Vibration**: Progressive disturbance limits

---

## 8. Implementation Priority

### High Priority (Immediate Value)

1. **OnOffParameters - Cumulative Startups**
   - High impact for maintenance scheduling
   - Clear user demand
   - Easy to implement

2. **Flow - Cumulative Flow Hours**
   - Contract compliance use cases
   - Energy delivery milestones
   - Resource quotas

3. **Effect - Cumulative Budget/Emissions**
   - Budget tracking
   - Emissions compliance
   - Resource management

### Medium Priority (Valuable Extensions)

4. **Investment - Cumulative Capacity**
   - Multi-period planning
   - Regulatory targets
   - Phased deployment

5. **OnOffParameters - Cumulative Operating Hours**
   - Usage tracking
   - Maintenance windows
   - Contract requirements

### Lower Priority (Advanced Use Cases)

6. **Rolling window constraints**
   - More complex to implement
   - Requires additional helper methods
   - Advanced use cases

7. **Inter-period linkages**
   - Complex multi-year modeling
   - Requires careful design
   - Niche applications

---

## 9. API Design Considerations

### Option A: Time-Varying Bounds (Flexible)
```python
# User provides checkpoints and values
startup_limits = pd.Series([10, 25, 40, 50], index=quarterly_ends)

OnOffParameters(
    cumulative_switch_on_max=startup_limits,  # Progressive limits
)
```

### Option B: Rolling Windows (Simpler)
```python
# User provides window size and limit
OnOffParameters(
    switch_on_max_per_window=(30*24, 10),  # Max 10 in any 30-day window
)
```

### Option C: Hybrid (Best of both)
```python
OnOffParameters(
    # Progressive limits at specific times
    cumulative_switch_on_max=quarterly_limits,

    # Rolling window constraints
    switch_on_max_rolling_window=(hours=720, limit=10),
)
```

---

## 10. Testing and Validation

### Unit Tests Needed:
1. Basic cumulative tracking (sum equals total)
2. Progressive bounds enforcement
3. Initial value handling
4. Multi-dimensional coords (period, scenario)
5. Edge cases (empty arrays, single timestep)

### Integration Tests:
1. Full OnOffParameters with cumulative limits
2. Flow with delivery milestones
3. Effect with budget tracking
4. Multi-period investment scenarios

### Validation:
1. Compare cumulative[t] == sum(expression[0:t+1])
2. Verify bound violations raise errors
3. Check initial conditions
4. Test with periods and scenarios

---

## Conclusion

The `cumulative_sum_tracking` primitive unlocks a vast array of new modeling capabilities. Priority should be:

1. **Implement in OnOffParameters** (cumulative startups, operating hours)
2. **Extend to Flow** (cumulative flow hours, delivery milestones)
3. **Add to Effect** (budgets, emissions)
4. **Consider Investment** (phased capacity)

This will enable real-world constraint modeling that is currently impossible with simple totals!
