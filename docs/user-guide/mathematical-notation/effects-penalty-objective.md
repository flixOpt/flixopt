# Effects & Objective

Effects are how you track and optimize metrics in your system. One effect is your **objective** (what you minimize), while others can be **constraints** or just tracked for reporting.

!!! example "Common effects"
    - **Costs** — minimize total costs (objective)
    - **CO₂ emissions** — track or constrain to meet targets
    - **Primary energy** — report for efficiency analysis
    - **Peak power** — constrain maximum grid import

## Core Concept: Aggregating Contributions

Every element in your model can contribute to effects. These contributions are aggregated into totals:

$$
E_{total} = \sum_{elements} contributions
$$

For example, total costs might come from:

- Gas consumption × gas price
- Electricity import × electricity price
- Equipment startup costs
- Investment costs (annualized)

## The Objective Function

flixOpt minimizes one effect (plus any penalties):

$$
\min \quad E_{objective} + penalty
$$

The penalty term comes from bus balance violations (see [Bus](elements/Bus.md)).

## Two Types of Effects

### Temporal Effects (Operational)

Vary with time — accumulated over all timesteps:

$$
E_{temporal} = \sum_t contribution(t) \cdot \Delta t
$$

!!! example "Fuel costs"
    ```python
    gas_flow = fx.Flow(
        label='gas',
        bus=gas_bus,
        size=100,
        effects_per_flow_hour={'costs': 50},  # €50/MWh
    )
    ```
    Contribution: $50 \cdot p_{gas}(t) \cdot \Delta t$ at each timestep

### Periodic Effects (Investment)

Time-independent — incurred once per period:

$$
E_{periodic} = \sum_{investments} size \cdot specific\_cost
$$

!!! example "Battery investment"
    ```python
    battery = fx.Storage(
        ...,
        capacity_in_flow_hours=InvestParameters(
            maximum_size=1000,
            specific_effects={'costs': 200},  # €200/kWh annualized
        ),
    )
    ```
    Contribution: $200 \cdot capacity$ (once, not per timestep)

### Total Effect

$$
E_{total} = E_{periodic} + E_{temporal}
$$

## Cross-Effects: Linking Metrics

Effects can contribute to each other. This enables carbon pricing, multi-criteria optimization, and complex cost structures.

!!! example "Carbon pricing"
    CO₂ emissions cost €80/tonne:
    ```python
    co2 = fx.Effect(label='co2', unit='kg')

    costs = fx.Effect(
        label='costs',
        unit='€',
        is_objective=True,
        share_from_temporal={'co2': 0.08},  # €0.08/kg = €80/tonne
    )
    ```

    Now CO₂ emissions automatically contribute to costs.

## Effect Constraints

Besides optimizing one effect, you can constrain others:

### Total Limit

```python
co2 = fx.Effect(
    label='co2',
    unit='kg',
    maximum_total=100_000,  # Max 100 tonnes total
)
```

$$
E_{co2,total} \leq 100{,}000
$$

### Per-Timestep Limit

```python
peak_power = fx.Effect(
    label='peak_power',
    unit='kW',
    maximum_per_hour=500,  # Max 500 kW at any timestep
)
```

$$
E_{peak}(t) \leq 500 \quad \forall t
$$

### Investment Budget

```python
capex = fx.Effect(
    label='capex',
    unit='€',
    maximum_periodic=5_000_000,  # €5M investment budget
)
```

$$
E_{capex,periodic} \leq 5{,}000{,}000
$$

## Variables

| Variable | Description | When Created |
|----------|-------------|--------------|
| $E_{e,temporal}(t)$ | Temporal effect at timestep $t$ | Always |
| $E_{e,periodic}$ | Periodic (investment) effect | Always |
| $E_{e,total}$ | Total effect | Always |
| $penalty$ | Sum of all penalty contributions | Always |

## Parameters

| Parameter | Python Name | Description |
|-----------|-------------|-------------|
| - | `is_objective` | If True, this effect is minimized |
| - | `is_standard` | If True, allows shorthand effect syntax |
| - | `maximum_total` | Upper bound on total effect |
| - | `minimum_total` | Lower bound on total effect |
| - | `maximum_per_hour` | Upper bound per timestep |
| - | `maximum_periodic` | Upper bound on periodic (investment) part |
| - | `share_from_temporal` | Cross-effect contributions (temporal) |
| - | `share_from_periodic` | Cross-effect contributions (periodic) |

## Usage Examples

### Basic Cost Minimization

```python
costs = fx.Effect(
    label='costs',
    unit='€',
    is_objective=True,
    is_standard=True,  # Allows shorthand: effects_per_flow_hour={'costs': 50}
)

# Elements contribute via effects_per_flow_hour
gas_flow = fx.Flow(
    label='gas',
    bus=gas_bus,
    size=100,
    effects_per_flow_hour={'costs': 50},
)
```

### Cost + CO₂ Constraint (ε-Constraint Method)

```python
costs = fx.Effect(label='costs', unit='€', is_objective=True)
co2 = fx.Effect(label='co2', unit='kg', maximum_total=50_000)

# Generator contributes to both
generator_flow = fx.Flow(
    label='power',
    bus=electricity_bus,
    size=100,
    effects_per_flow_hour={
        'costs': 45,      # €45/MWh
        'co2': 0.4,       # 0.4 kg/kWh = 400 kg/MWh
    },
)
```

Minimize costs while staying under 50 tonnes CO₂.

### Carbon Pricing (Weighted Sum Method)

```python
co2 = fx.Effect(label='co2', unit='kg')

costs = fx.Effect(
    label='costs',
    unit='€',
    is_objective=True,
    share_from_temporal={'co2': 0.08},  # €80/tonne
)

# Generator only specifies CO₂
generator_flow = fx.Flow(
    label='power',
    bus=electricity_bus,
    size=100,
    effects_per_flow_hour={'co2': 0.4},
)
```

CO₂ automatically converted to costs. No need to specify cost contribution separately.

### Peak Power Constraint

```python
peak = fx.Effect(
    label='grid_peak',
    unit='kW',
    maximum_per_hour=500,
)

grid_import = fx.Flow(
    label='grid',
    bus=electricity_bus,
    size=1000,
    effects_per_flow_hour={'grid_peak': 1},  # 1 kW per kW flow
)
```

Grid import limited to 500 kW at any timestep.

### Investment Budget

```python
capex = fx.Effect(label='capex', unit='€', maximum_periodic=1_000_000)
opex = fx.Effect(label='opex', unit='€')
total_costs = fx.Effect(label='total', unit='€', is_objective=True)

# Link them
total_costs.share_from_periodic = {'capex': 1}
total_costs.share_from_temporal = {'opex': 1}

# Battery investment
battery = fx.Storage(
    ...,
    capacity_in_flow_hours=InvestParameters(
        maximum_size=1000,
        specific_effects={'capex': 500},  # €500/kWh
    ),
)
```

Optimize total costs while respecting €1M investment budget.

## Multi-Dimensional Effects

When using periods and scenarios (see [Dimensions](dimensions.md)), effects aggregate with weights:

$$
\min \sum_{periods} w_{period} \cdot \sum_{scenarios} w_{scenario} \cdot E(period, scenario)
$$

**Key principle:** Investment decisions (periodic effects) are shared across scenarios within a period, while operational effects are scenario-specific.

## Implementation Details

- **Element Class:** [`Effect`][flixopt.effects.Effect]
- **Model Class:** [`EffectModel`][flixopt.effects.EffectModel]
- **Collection Class:** [`EffectCollection`][flixopt.effects.EffectCollection]

## See Also

- [Bus](elements/Bus.md) — Penalty contributions from balance violations
- [Flow](elements/Flow.md) — Temporal contributions via `effects_per_flow_hour`
- [InvestParameters](features/InvestParameters.md) — Periodic contributions from investments
- [Dimensions](dimensions.md) — Multi-period and scenario handling
- [Core Concepts: Effects](../core-concepts.md#effects-what-youre-tracking) — High-level overview
