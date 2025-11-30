# Flow

A Flow is the primary optimization variable — the solver decides how much flows at each timestep.

## Basic: Bounded Flow Rate

Every flow has a **size** $P$ (capacity) and a **flow rate** $p(t)$ (what the solver optimizes):

$$
P \cdot p_{rel}^{min} \leq p(t) \leq P \cdot p_{rel}^{max}
$$

```python
# 100 kW boiler, minimum 30% when running
heat = fx.Flow(label='heat', bus=heat_bus, size=100, relative_minimum=0.3)
# → 30 ≤ p(t) ≤ 100
```

!!! warning "Cannot be zero"
    With `relative_minimum > 0`, the flow cannot be zero. Use `status_parameters` to allow shutdown.

---

## Adding Features

=== "Status"

    Allow the flow to be zero with `status_parameters`:

    $s(t) \cdot P \cdot p_{rel}^{min} \leq p(t) \leq s(t) \cdot P \cdot p_{rel}^{max}$

    Where $s(t) \in \{0, 1\}$: inactive or active.

    ```python
    generator = fx.Flow(
        label='power', bus=elec_bus, size=50,
        relative_minimum=0.4,
        status_parameters=fx.StatusParameters(
            effects_per_startup={'costs': 500},
            min_uptime=2,
        ),
    )
    ```

    See [StatusParameters](../features/StatusParameters.md).

=== "Variable Size"

    Optimize the capacity with `InvestParameters`:

    $P^{min} \leq P \leq P^{max}$

    ```python
    battery = fx.Flow(
        label='power', bus=elec_bus,
        size=fx.InvestParameters(
            minimum_size=0,
            maximum_size=1000,
            specific_effects={'costs': 100_000},
        ),
    )
    ```

    See [InvestParameters](../features/InvestParameters.md).

=== "Flow Effects"

    Add effects per energy (flow hours) moved:

    ```python
    gas = fx.Flow(
        label='gas', bus=gas_bus, size=150,
        effects_per_flow_hour={'costs': 50},  # €50/MWh
    )
    ```

    Flow hours: $h(t) = p(t) \cdot \Delta t$


=== "Fixed Profile"

    Lock the flow to a time series (demands, renewables):

    $p(t) = P \cdot \pi(t)$

    ```python
    demand = fx.Flow(
        label='demand', bus=heat_bus, size=100,
        fixed_relative_profile=[0.5, 0.8, 1.0, 0.6]  # π(t)
    )
    ```

---

## Optional Constraints

=== "Load Factor"

    Constrain average utilization:

    $\lambda_{min} \leq \frac{\sum_t p(t)}{P \cdot n_t} \leq \lambda_{max}$

    ```python
    fx.Flow(..., load_factor_min=0.5, load_factor_max=0.9)
    ```

=== "Flow Hours"

    Constrain total energy:

    $h_{min} \leq \sum_t p(t) \cdot \Delta t \leq h_{max}$

    ```python
    fx.Flow(..., flow_hours_min=1000, flow_hours_max=5000)
    ```

---

## Reference

| Symbol | Type | Description |
|--------|------|-------------|
| $p(t)$ | $\mathbb{R}_{\geq 0}$ | Flow rate at timestep $t$ |
| $P$ | $\mathbb{R}_{\geq 0}$ | Size (capacity) — fixed or optimized |
| $s(t)$ | $\{0, 1\}$ | Binary status (with `status_parameters`) |
| $p_{rel}^{min}$ | $[0, 1]$ | Minimum relative flow (`relative_minimum`) |
| $p_{rel}^{max}$ | $[0, 1]$ | Maximum relative flow (`relative_maximum`) |
| $\pi(t)$ | $[0, 1]$ | Fixed profile (`fixed_relative_profile`) |
| $\Delta t$ | $\mathbb{R}_{> 0}$ | Timestep duration (hours) |

**Classes:** [`Flow`][flixopt.elements.Flow], [`FlowModel`][flixopt.elements.FlowModel]
