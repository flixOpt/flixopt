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

    Accumulated over timesteps — operational costs, emissions, energy - per timestep:

    $E_{temp}(t) = \text{flow}(t) \cdot c \cdot \Delta t$

    ```python
    fx.Flow(..., effects_per_flow_hour={'costs': 50})  # €50/MWh
    ```

=== "Periodic"

    Time-independent — investment costs, fixed fees - per period:

    $E_{per} = P \cdot c_{inv}$

    ```python
    fx.InvestParameters(specific_effects={'costs': 200})  # €200/kW
    ```

=== "Total"

    Sum of periodic and temporal components.

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
    scenario_weights={'base': 0.6, 'high_demand': 0.4}
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

    Bound on aggregated effect:

    ```python
    fx.Effect(label='co2', unit='kg', maximum_total=100_000)
    ```

=== "Per-Timestep Limit"

    Bound at each timestep:

    ```python
    fx.Effect(label='peak', unit='kW', maximum_per_hour=500)
    ```

=== "Periodic Limit"

    Bound on periodic component:

    ```python
    fx.Effect(label='capex', unit='€', maximum_periodic=1_000_000)
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

## Reference

| Variable | Description |
|----------|-------------|
| $E_{temp}(t)$ | Temporal effect at timestep $t$ |
| $E_{per}$ | Periodic effect |
| $w_s$ | Scenario weight |
| $w_y$ | Period weight |

| Parameter | Python | Description |
|-----------|--------|-------------|
| Objective | `is_objective=True` | Minimize this effect |
| Total limit | `maximum_total` | Upper bound on total |
| Timestep limit | `maximum_per_hour` | Upper bound per timestep |
| Periodic limit | `maximum_periodic` | Upper bound on periodic |
| Cross-effect | `share_from_temporal` | Link from other effect |

**Classes:** [`Effect`][flixopt.effects.Effect], [`EffectCollection`][flixopt.effects.EffectCollection]
