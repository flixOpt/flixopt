# Flow

A Flow represents the movement of energy or material between a component and a bus. It's the primary optimization variable — the solver decides how much flows at each timestep.

!!! example "Real-world examples"
    - Heat output from a boiler to the heat bus
    - Electricity import from the grid
    - Gas consumption by a CHP unit
    - Charging power into a battery

## Core Concept: Size and Flow Rate

Every flow has two key quantities:

- **Size** ($P$) — The capacity or maximum possible flow rate
- **Flow Rate** ($p(t)$) — The actual flow at each timestep (optimization variable)

## Flow Rate Bounds

=== "Fixed Size"

    The flow rate is bounded by the size:

    $$
    P \cdot p_{rel}^{min}(t) \leq p(t) \leq P \cdot p_{rel}^{max}(t)
    $$

    The flow **cannot be zero** if $p_{rel}^{min} > 0$.

    !!! example "100 kW boiler, min 30%"
        - $P = 100$, $p_{rel}^{min} = 0.3$, $p_{rel}^{max} = 1$
        - Constraint: $30 \leq p(t) \leq 100$

=== "Fixed Size + Status"

    When `status_parameters` is specified, the flow can also be zero:

    $$
    s(t) \cdot P \cdot p_{rel}^{min}(t) \leq p(t) \leq s(t) \cdot P \cdot p_{rel}^{max}(t)
    $$

    Where $s(t) \in \{0, 1\}$ is the binary status.

    - When $s(t) = 0$: $p(t) = 0$ (inactive)
    - When $s(t) = 1$: $P \cdot p_{rel}^{min} \leq p(t) \leq P \cdot p_{rel}^{max}$ (active)

    See [StatusParameters](../features/StatusParameters.md) for details.

=== "Variable Size"

    When `size` is `InvestParameters`, the capacity $P$ becomes a variable:

    $$
    P^{min} \leq P \leq P^{max}
    $$

    $$
    P \cdot p_{rel}^{min}(t) \leq p(t) \leq P \cdot p_{rel}^{max}(t)
    $$

    See [InvestParameters](../features/InvestParameters.md) for details.

=== "Variable Size + Status"

    !!! warning "Work in Progress"
        This section needs review. The linearization constraints below may not be accurate.

    When both `size` is `InvestParameters` and `status_parameters` is set:

    $$
    P^{min} \leq P \leq P^{max}
    $$

    $$
    s(t) \cdot P \cdot p_{rel}^{min}(t) \leq p(t) \leq s(t) \cdot P \cdot p_{rel}^{max}(t)
    $$

    This creates a bilinear term $s(t) \cdot P$. flixOpt linearizes this using big-M constraints:

    $$
    p(t) \leq P^{max} \cdot s(t)
    $$

    $$
    p(t) \leq P - P^{min} \cdot (1 - s(t))
    $$

    $$
    p(t) \geq P - P^{max} \cdot (1 - s(t))
    $$

=== "Fixed Profile"

    When `fixed_relative_profile` is set, the flow rate is fixed:

    $$
    p(t) = P \cdot \pi(t)
    $$

    No optimization freedom — used for demands and renewable generation.

## Flow Hours: Energy vs Power

**Flow rate** is power (kW, MW). **Flow hours** is energy (kWh, MWh):

$$
h_f(t) = p(t) \cdot \Delta t
$$

| Flow Rate | Timestep | Flow Hours |
|-----------|----------|------------|
| 100 kW | 1 hour | 100 kWh |
| 100 kW | 15 min | 25 kWh |

This matters for costs: `effects_per_flow_hour` is cost per energy (€/MWh).

## Additional Constraints

=== "Load Factor"

    Constrain average utilization (`load_factor_min/max`):

    $$
    \lambda_{min} \cdot P \cdot n_t \leq \sum_t p(t) \leq \lambda_{max} \cdot P \cdot n_t
    $$

    Where $n_t$ is the number of timesteps.

=== "Flow Hours Limits"

    Constrain total energy (`flow_hours_min/max`):

    $$
    h_{min} \leq \sum_t p(t) \cdot \Delta t \leq h_{max}
    $$

## Variables

| Symbol | Python Name | Description | When Created |
|--------|-------------|-------------|--------------|
| $p(t)$ | `flow_rate` | Flow rate at timestep $t$ | Always |
| $P$ | `size` | Capacity (variable) | `size` is `InvestParameters` |
| $s(t)$ | `status` | Binary status | `status_parameters` set |
| $s^{start}(t)$ | `startup` | Startup indicator | `status_parameters` set |
| $s^{stop}(t)$ | `shutdown` | Shutdown indicator | `status_parameters` set |

## Parameters

| Symbol | Python Name | Description | Default |
|--------|-------------|-------------|---------|
| $P$ | `size` | Flow capacity | Required |
| $p_{rel}^{min}(t)$ | `relative_minimum` | Min as fraction of size | 0 |
| $p_{rel}^{max}(t)$ | `relative_maximum` | Max as fraction of size | 1 |
| $\pi(t)$ | `fixed_relative_profile` | Fixed profile | None |
| $\lambda_{min}$ | `load_factor_min` | Min average utilization | None |
| $\lambda_{max}$ | `load_factor_max` | Max average utilization | None |
| $h_{min}$ | `flow_hours_min` | Min total energy | None |
| $h_{max}$ | `flow_hours_max` | Max total energy | None |

## Usage Examples

### Basic Flow

```python
heat_output = fx.Flow(
    label='heat_out',
    bus=heat_bus,
    size=100,  # 100 kW capacity
    relative_minimum=0.3,  # Min 30 kW when operating
)
```

### Flow with Costs

```python
gas_input = fx.Flow(
    label='gas_in',
    bus=gas_bus,
    size=150,
    effects_per_flow_hour={'costs': 50},  # €50/MWh
)
```

### Fixed Profile (Solar)

```python
solar = fx.Flow(
    label='solar',
    bus=electricity_bus,
    size=500,  # 500 kW peak
    fixed_relative_profile=[0, 0.1, 0.4, 0.8, 0.9, 0.6, 0.2, 0],
)
```

### With Status Operation

```python
generator = fx.Flow(
    label='power',
    bus=electricity_bus,
    size=50,
    relative_minimum=0.4,  # 40% min when active, but can be inactive
    status_parameters=fx.StatusParameters(
        effects_per_startup={'costs': 500},
        min_uptime=2,
    ),
)
```

### Investment Decision

```python
battery_flow = fx.Flow(
    label='power',
    bus=electricity_bus,
    size=fx.InvestParameters(
        minimum_size=0,
        maximum_size=1000,
        specific_effects={'costs': 100_000},
    ),
)
```

## Implementation Details

- **Element Class:** [`Flow`][flixopt.elements.Flow]
- **Model Class:** [`FlowModel`][flixopt.elements.FlowModel]

## See Also

- [Bus](Bus.md) — Where flows connect
- [LinearConverter](LinearConverter.md) — Components using flows
- [StatusParameters](../features/StatusParameters.md) — Binary operation
- [InvestParameters](../features/InvestParameters.md) — Capacity optimization
