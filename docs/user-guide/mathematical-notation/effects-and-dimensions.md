# Effects & Dimensions

Effects track metrics (costs, CO₂, energy). Dimensions define the structure over which effects aggregate.

## Defining Effects

```python
costs = fx.Effect(label='costs', unit='€', is_objective=True)
co2 = fx.Effect(label='co2', unit='kg')

flow_system.add_elements(costs, co2)
```

One effect is the **objective** (minimized). Others are tracked or constrained.

---

## Effect Types

=== "Temporal"

    Accumulated over timesteps — operational costs, emissions, energy:

    $E_{temp}(t) = \text{value}(t) \cdot c \cdot \Delta t$

    ```python
    fx.Flow(..., effects_per_flow_hour={'costs': 50})  # €50/MWh
    ```

=== "Periodic"

    Time-independent — investment costs, fixed fees:

    $E_{per} = P \cdot c_{inv}$

    ```python
    fx.InvestParameters(effects_of_investment_per_size={'costs': 200})  # €200/kW
    ```

=== "Total"

    Sum of periodic and temporal components.

---

## Where Effects Are Contributed

=== "Flow"

    ```python
    fx.Flow(
        effects_per_flow_hour={'costs': 50, 'co2': 0.2},  # Per MWh
    )
    ```

=== "Status"

    ```python
    fx.StatusParameters(
        effects_per_startup={'costs': 1000},      # Per startup event
        effects_per_active_hour={'costs': 10},    # Per hour while running
    )
    ```

=== "Investment"

    ```python
    fx.InvestParameters(
        effects_of_investment={'costs': 50000},           # Fixed if investing
        effects_of_investment_per_size={'costs': 800},    # Per kW installed
        effects_of_retirement={'costs': 10000},           # If NOT investing
    )
    ```

=== "Bus"

    ```python
    fx.Bus(
        excess_penalty_per_flow_hour=1e6,    # Penalty for excess
        shortage_penalty_per_flow_hour=1e6,  # Penalty for shortage
    )
    ```

---

## Dimensions

The model operates across three dimensions:

=== "Timesteps"

    The basic time resolution — always required:

    ```python
    flow_system = fx.FlowSystem(
        timesteps=pd.date_range('2024-01-01', periods=8760, freq='h'),
    )
    ```

    All variables and constraints are indexed by time. Temporal effects sum over timesteps.

=== "Scenarios"

    Represent uncertainty (weather, prices). Operations vary per scenario, investments are shared:

    ```python
    flow_system = fx.FlowSystem(
        timesteps=pd.date_range('2024-01-01', periods=8760, freq='h'),
        scenarios=pd.Index(['sunny_year', 'cloudy_year']),
        scenario_weights=[0.7, 0.3],
    )
    ```

    Scenarios are independent — no energy or information exchange between them.

=== "Periods"

    Sequential time blocks (years) for multi-period planning:

    ```python
    flow_system = fx.FlowSystem(
        timesteps=pd.date_range('2024-01-01', periods=8760, freq='h'),
        periods=pd.Index([2025, 2030]),
    )
    ```

    Periods are independent — each has its own investment decisions.

---

## Objective Function

The objective aggregates effects across all dimensions with weights:

=== "Basic"

    Single period, no scenarios:

    $$\min \quad E_{per} + \sum_t E_{temp}(t)$$

=== "With Scenarios"

    Investment decided once, operations weighted by probability:

    $$\min \quad E_{per} + \sum_s w_s \cdot \sum_t E_{temp}(t, s)$$

    - $w_s$ — scenario weight (probability)

=== "With Periods"

    Multi-year planning with discounting:

    $$\min \quad \sum_y w_y \cdot \left( E_{per}(y) + \sum_t E_{temp}(t, y) \right)$$

    - $w_y$ — period weight (duration or discount factor)

=== "Full"

    Periods × Scenarios:

    $$\min \quad \sum_y w_y \cdot \left( E_{per}(y) + \sum_s w_s \cdot \sum_t E_{temp}(t, y, s) \right)$$

The penalty effect is always included: $\min \quad E_{objective} + E_{penalty}$

---

## Weights

=== "Scenario Weights"

    Provided explicitly — typically probabilities:

    ```python
    scenario_weights=[0.6, 0.4]
    ```

    Default: equal weights, normalized to sum to 1.

=== "Period Weights"

    Computed automatically from period index (interval sizes):

    ```python
    periods = pd.Index([2020, 2025, 2030])
    # → weights: [5, 5, 5] (5-year intervals)
    ```

=== "Combined"

    When both present:

    $w_{y,s} = w_y \cdot w_s$

---

## Constraints on Effects

=== "Total Limit"

    Bound on aggregated effect (temporal + periodic) per period:

    ```python
    fx.Effect(label='co2', unit='kg', maximum_total=100_000)
    ```

=== "Per-Timestep Limit"

    Bound at each timestep:

    ```python
    fx.Effect(label='peak', unit='kW', maximum_per_hour=500)
    ```

=== "Periodic Limit"

    Bound on periodic component only:

    ```python
    fx.Effect(label='capex', unit='€', maximum_periodic=1_000_000)
    ```

=== "Temporal Limit"

    Bound on temporal component only:

    ```python
    fx.Effect(label='opex', unit='€', maximum_temporal=500_000)
    ```

=== "Over All Periods"

    Bound across all periods (weighted sum):

    ```python
    fx.Effect(label='co2', unit='kg', maximum_over_periods=1_000_000)
    ```

---

## Cross-Effects

Effects can contribute to each other (e.g., carbon pricing):

```python
co2 = fx.Effect(label='co2', unit='kg')

costs = fx.Effect(
    label='costs', unit='€', is_objective=True,
    share_from_temporal={'co2': 0.08},  # €80/tonne
)
```

---

## Penalty Effect

A built-in `Penalty` effect enables soft constraints and prevents infeasibility:

```python
fx.StatusParameters(effects_per_startup={'Penalty': 1})
fx.Bus(label='heat', excess_penalty_per_flow_hour=1e5)
```

Penalty is weighted identically to the objective effect across all dimensions.

---

## Shared vs Independent Decisions

=== "Investments (Sizes)"

    By default, investment decisions are **shared across scenarios** within a period:

    - Build capacity once → operate differently per scenario
    - Reflects real-world investment under uncertainty

    $$P(y) \quad \text{(one decision per period, used in all scenarios)}$$

=== "Operations (Flows)"

    By default, operational decisions are **independent per scenario**:

    $$p(t, y, s) \quad \text{(different for each scenario)}$$

---

## Use Cases

=== "Carbon Budget"

    Limit total CO₂ emissions across all years:

    ```python
    co2 = fx.Effect(
        label='co2', unit='kg',
        maximum_over_periods=1_000_000,  # 1000 tonnes total
    )

    # Contribute emissions from gas consumption
    gas_flow = fx.Flow(
        label='gas', bus=gas_bus,
        effects_per_flow_hour={'co2': 0.2},  # 0.2 kg/kWh
    )
    ```

=== "Investment Budget"

    Cap annual investment spending:

    ```python
    capex = fx.Effect(
        label='capex', unit='€',
        maximum_periodic=5_000_000,  # €5M per period
    )

    battery = fx.Storage(
        ...,
        capacity=fx.InvestParameters(
            effects_of_investment_per_size={'capex': 600},  # €600/kWh
        ),
    )
    ```

=== "Peak Demand Charge"

    Track and limit peak power:

    ```python
    peak = fx.Effect(
        label='peak', unit='kW',
        maximum_per_hour=1000,  # Grid connection limit
    )

    grid_import = fx.Flow(
        label='import', bus=elec_bus,
        effects_per_flow_hour={'peak': 1},  # Track instantaneous power
    )
    ```

=== "Carbon Pricing"

    Add CO₂ cost to objective automatically:

    ```python
    co2 = fx.Effect(label='co2', unit='kg')

    costs = fx.Effect(
        label='costs', unit='€', is_objective=True,
        share_from_temporal={'co2': 0.08},  # €80/tonne carbon price
    )

    # Now any CO₂ contribution automatically adds to costs
    ```

=== "Land Use Constraint"

    Limit total land area for installations:

    ```python
    land = fx.Effect(
        label='land', unit='m²',
        maximum_periodic=50_000,  # 5 hectares max
    )

    pv = fx.Source(
        ...,
        output=fx.Flow(
            ...,
            invest_parameters=fx.InvestParameters(
                effects_of_investment_per_size={'land': 5},  # 5 m²/kWp
            ),
        ),
    )
    ```

=== "Multi-Criteria Optimization"

    Track multiple objectives, optimize one:

    ```python
    costs = fx.Effect(label='costs', unit='€', is_objective=True)
    co2 = fx.Effect(label='co2', unit='kg')
    primary_energy = fx.Effect(label='PE', unit='kWh')

    # All are tracked, costs is minimized
    # Use maximum_total on co2 for ε-constraint method
    ```

---

## Reference

| Variable | Description |
|----------|-------------|
| $E_{temp}(t)$ | Temporal effect at timestep $t$ |
| $E_{per}$ | Periodic effect |
| $w_s$ | Scenario weight |
| $w_y$ | Period weight |

| Constraint | Python | Scope |
|-----------|--------|-------|
| Total limit | `maximum_total` | Per period |
| Timestep limit | `maximum_per_hour` | Each timestep |
| Periodic limit | `maximum_periodic` | Per period (periodic only) |
| Temporal limit | `maximum_temporal` | Per period (temporal only) |
| Global limit | `maximum_over_periods` | Across all periods |

**Classes:** [`Effect`][flixopt.effects.Effect], [`EffectCollection`][flixopt.effects.EffectCollection]
