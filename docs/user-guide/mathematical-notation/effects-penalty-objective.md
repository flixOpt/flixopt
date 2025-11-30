# Effects & Objective

Effects track metrics (costs, CO₂, energy) and define what you optimize.

## Basic: Defining Effects

```python
costs = fx.Effect(label='costs', unit='€', is_objective=True)
co2 = fx.Effect(label='co2', unit='kg')

flow_system.add_elements(costs, co2)
```

One effect is the **objective** (minimized). Others are tracked or constrained.

---

## Temporal vs Periodic

Effects have two components with different time behavior:

=== "Temporal (Operational)"

    Accumulated over timesteps — fuel costs, emissions, energy use:

    $E_{temp} = \sum_t s(t) \cdot \Delta t$

    ```python
    gas = fx.Flow(
        ...,
        effects_per_flow_hour={'costs': 50},  # €50/MWh
    )
    ```

=== "Periodic (Investment)"

    Time-independent — incurred once per period:

    $E_{per} = P \cdot c_{inv}$

    ```python
    fx.InvestParameters(
        specific_effects={'costs': 200},  # €200/kW
    )
    ```

=== "Total"

    $E = E_{per} + E_{temp}$

---

## Dimensions & Weights

Effects aggregate across dimensions with weights:

=== "Single Period"

    $$\min \quad E_{per} + \sum_t E_{temp}(t)$$

=== "With Scenarios"

    Scenarios represent uncertainty (weather, prices). Temporal effects are scenario-specific, periodic effects are shared:

    $$\min \quad E_{per} + \sum_s w_s \cdot \sum_t E_{temp}(t, s)$$

    Investment decided once, operations vary by scenario.

=== "With Periods"

    Periods represent sequential time blocks (years). Each has independent effects:

    $$\min \quad \sum_y w_y \cdot \left( E_{per}(y) + \sum_t E_{temp}(t, y) \right)$$

=== "Full (Periods + Scenarios)"

    $$\min \quad \sum_y w_y \cdot \left( E_{per}(y) + \sum_s w_s \cdot \sum_t E_{temp}(t, y, s) \right)$$

    - $w_y$ — period weight (e.g., discount factor)
    - $w_s$ — scenario weight (e.g., probability)

---

## Constraints on Effects

=== "Total Limit"

    ```python
    co2 = fx.Effect(label='co2', unit='kg', maximum_total=100_000)
    ```

    $E_{co2} \leq 100{,}000$

=== "Per-Timestep Limit"

    ```python
    peak = fx.Effect(label='peak', unit='kW', maximum_per_hour=500)
    ```

    $E_{peak}(t) \leq 500 \quad \forall t$

=== "Periodic Limit"

    ```python
    capex = fx.Effect(label='capex', unit='€', maximum_periodic=1_000_000)
    ```

    $E_{capex,per} \leq 1{,}000{,}000$

---

## Cross-Effects

Effects can contribute to each other (e.g., carbon pricing):

```python
co2 = fx.Effect(label='co2', unit='kg')

costs = fx.Effect(
    label='costs', unit='€', is_objective=True,
    share_from_temporal={'co2': 0.08},  # €0.08/kg = €80/tonne
)
```

CO₂ emissions automatically add to costs.

---

## Penalty Effect

A built-in `Penalty` effect prevents infeasibility and allows soft biases:

```python
# Bias against startups without affecting cost tracking
fx.StatusParameters(effects_per_startup={'Penalty': 1})

# Bus imbalance penalty
fx.Bus(label='heat', excess_penalty_per_flow_hour=1e5)
```

The objective is always: $\min \quad E_{objective} + E_{penalty}$

---

## Reference

| Component | Description |
|-----------|-------------|
| $E_{temp}(t)$ | Temporal effect at timestep $t$ |
| $E_{per}$ | Periodic effect (time-independent) |
| $E$ | Total = periodic + sum of temporal |
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
