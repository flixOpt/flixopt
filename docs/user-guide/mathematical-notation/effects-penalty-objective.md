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
E_e = \sum_{l \in \mathcal{L}} s_{l \rightarrow e}
$$

Where $s_{l \rightarrow e}$ is the share from element $l$ to effect $e$.

## The Objective Function

flixOpt minimizes one effect (plus any penalties):

$$
\min \quad E_\Omega + \Phi
$$

Where $E_\Omega$ is the objective effect and $\Phi$ is the penalty from bus violations.

## Two Types of Effects

=== "Temporal (Operational)"

    Accumulated over timesteps:

    $$
    E_{e,temp} = \sum_t s_{l \rightarrow e}(t) \cdot \Delta t
    $$

    !!! example "Fuel costs"
        ```python
        gas_flow = fx.Flow(
            label='gas', bus=gas_bus, size=100,
            effects_per_flow_hour={'costs': 50},  # €50/MWh
        )
        ```
        Contribution: $50 \cdot p(t) \cdot \Delta t$

=== "Periodic (Investment)"

    Time-independent — incurred once:

    $$
    E_{e,per} = \sum_{inv} P \cdot c_{inv}
    $$

    !!! example "Battery investment"
        ```python
        capacity=fx.InvestParameters(
            maximum_size=1000,
            specific_effects={'costs': 200},  # €200/kWh
        )
        ```
        Contribution: $200 \cdot C$

=== "Total"

    $$
    E_e = E_{e,per} + E_{e,temp}
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
Every FlixOpt model includes a special **Penalty Effect** $E_\Phi$ to:

- Prevent infeasible problems
- Allow introducing a bias without influencing effects, simplifying results analysis

**Key Feature:** Penalty is implemented as a standard Effect (labeled `Penalty`), so you can **add penalty contributions anywhere effects are used**:

```python
import flixopt as fx

# Add penalty contributions just like any other effect
on_off = fx.OnOffParameters(
    effects_per_switch_on={'Penalty': 1}  # Add bias against switching on this component, without adding costs
)
```

```python
capex = fx.Effect(
    label='capex',
    unit='€',
    maximum_periodic=5_000_000,  # €5M investment budget
)
```
**Optionally Define Custom Penalty:**
Users can define their own Penalty effect with custom properties (unit, constraints, etc.):

$$
E_{capex,periodic} \leq 5{,}000{,}000
```python
# Define custom penalty effect (must use fx.PENALTY_EFFECT_LABEL)
custom_penalty = fx.Effect(
    fx.PENALTY_EFFECT_LABEL,  # Always use this constant: 'Penalty'
    unit='€',
    description='Penalty costs for constraint violations',
    maximum_total=1e6,  # Limit total penalty for debugging
)
flow_system.add_elements(custom_penalty)
```

If not user-defined, the Penalty effect is automatically created during modeling with default settings.

**Periodic penalty shares** (time-independent):
$$ \label{eq:Penalty_periodic}
E_{\Phi, \text{per}} = \sum_{l \in \mathcal{L}} s_{l \rightarrow \Phi,\text{per}}
$$

**Temporal penalty shares** (time-dependent):
$$ \label{eq:Penalty_temporal}
E_{\Phi, \text{temp}}(\text{t}_{i}) = \sum_{l \in \mathcal{L}} s_{l \rightarrow \Phi, \text{temp}}(\text{t}_i)
$$

**Total penalty** (combining both domains):
$$ \label{eq:Penalty_total}
E_{\Phi} = E_{\Phi,\text{per}} + \sum_{\text{t}_i \in \mathcal{T}} E_{\Phi, \text{temp}}(\text{t}_{i})
$$

## Variables

| Symbol | Python Name | Description | When Created |
|--------|-------------|-------------|--------------|
| $E_{e,temp}(t)$ | `(temporal)\|total` | Temporal effect at $t$ | Always |
| $E_{e,per}$ | `(periodic)\|total` | Periodic effect | Always |
| $E_e$ | `total` | Total effect | Always |
| $\Phi$ | `penalty` | Sum of penalties | Always |
- $\mathcal{L}$ is the set of all elements
- $\mathcal{T}$ is the set of all timesteps
- $s_{l \rightarrow \Phi, \text{per}}$ is the periodic penalty share from element $l$
- $s_{l \rightarrow \Phi, \text{temp}}(\text{t}_i)$ is the temporal penalty share from element $l$ at timestep $\text{t}_i$

## Parameters
**Primary usage:** Penalties occur in [Buses](elements/Bus.md) via the `excess_penalty_per_flow_hour` parameter, which allows nodal imbalances at a high cost, and in time series aggregation to allow period flexibility.

**Key properties:**
- Penalty shares are added via `add_share_to_effects(name, expressions={fx.PENALTY_EFFECT_LABEL: ...}, target='temporal'/'periodic')`
- Like other effects, penalty can be constrained (e.g., `maximum_total` for debugging)
- Results include breakdown: temporal, periodic, and total penalty contributions
- Penalty is always added to the objective function (cannot be disabled)
- Access via `flow_system.effects.penalty_effect` or `flow_system.effects[fx.PENALTY_EFFECT_LABEL]`
- **Scenario weighting**: Penalty is weighted identically to the objective effect—see [Time + Scenario](#time--scenario) for details

| Symbol | Python Name | Description |
|--------|-------------|-------------|
| - | `is_objective` | Minimize this effect |
| - | `is_standard` | Allow shorthand syntax |
| $E_e^{max}$ | `maximum_total` | Upper bound on total |
| $E_e^{min}$ | `minimum_total` | Lower bound on total |
| $E_{e,temp}^{max}(t)$ | `maximum_per_hour` | Upper bound per timestep |
| $E_{e,per}^{max}$ | `maximum_periodic` | Upper bound on periodic |
| $r_{x \rightarrow e}$ | `share_from_temporal` | Cross-effect factor (temporal) |
| $r_{x \rightarrow e}$ | `share_from_periodic` | Cross-effect factor (periodic) |
---

## Penalty

Every FlixOpt model includes a special **Penalty Effect** $E_\Phi$ to:

- Prevent infeasible problems
- Allow introducing a bias without influencing effects, simplifying results analysis

**Key Feature:** Penalty is implemented as a standard Effect (labeled `Penalty`), so you can **add penalty contributions anywhere effects are used**:

```python
import flixopt as fx

# Add penalty contributions just like any other effect
on_off = fx.OnOffParameters(
    effects_per_switch_on={'Penalty': 1}  # Add bias against switching on this component, without adding costs
)
```

**Optionally Define Custom Penalty:**
Users can define their own Penalty effect with custom properties (unit, constraints, etc.):

```python
# Define custom penalty effect (must use fx.PENALTY_EFFECT_LABEL)
custom_penalty = fx.Effect(
    fx.PENALTY_EFFECT_LABEL,  # Always use this constant: 'Penalty'
    unit='€',
    description='Penalty costs for constraint violations',
    maximum_total=1e6,  # Limit total penalty for debugging
)
flow_system.add_elements(custom_penalty)
```

If not user-defined, the Penalty effect is automatically created during modeling with default settings.

**Periodic penalty shares** (time-independent):
$$ \label{eq:Penalty_periodic}
E_{\Phi, \text{per}} = \sum_{l \in \mathcal{L}} s_{l \rightarrow \Phi,\text{per}}
$$

**Temporal penalty shares** (time-dependent):
$$ \label{eq:Penalty_temporal}
E_{\Phi, \text{temp}}(\text{t}_{i}) = \sum_{l \in \mathcal{L}} s_{l \rightarrow \Phi, \text{temp}}(\text{t}_i)
$$

**Total penalty** (combining both domains):
$$ \label{eq:Penalty_total}
E_{\Phi} = E_{\Phi,\text{per}} + \sum_{\text{t}_i \in \mathcal{T}} E_{\Phi, \text{temp}}(\text{t}_{i})
$$

Where:

- $\mathcal{L}$ is the set of all elements
- $\mathcal{T}$ is the set of all timesteps
- $s_{l \rightarrow \Phi, \text{per}}$ is the periodic penalty share from element $l$
- $s_{l \rightarrow \Phi, \text{temp}}(\text{t}_i)$ is the temporal penalty share from element $l$ at timestep $\text{t}_i$

**Primary usage:** Penalties occur in [Buses](elements/Bus.md) via the `excess_penalty_per_flow_hour` parameter, which allows nodal imbalances at a high cost, and in time series aggregation to allow period flexibility.

**Key properties:**
- Penalty shares are added via `add_share_to_effects(name, expressions={fx.PENALTY_EFFECT_LABEL: ...}, target='temporal'/'periodic')`
- Like other effects, penalty can be constrained (e.g., `maximum_total` for debugging)
- Results include breakdown: temporal, periodic, and total penalty contributions
- Penalty is always added to the objective function (cannot be disabled)
- Access via `flow_system.effects.penalty_effect` or `flow_system.effects[fx.PENALTY_EFFECT_LABEL]`
- **Scenario weighting**: Penalty is weighted identically to the objective effect—see [Time + Scenario](#time--scenario) for details

---

## Objective Function

The optimization objective minimizes the chosen effect plus the penalty effect:

$$ \label{eq:Objective}
\min \left( E_{\Omega} + E_{\Phi} \right)
$$

Where:

- $E_{\Omega}$ is the chosen **objective effect** (see $\eqref{eq:Effect_Total}$)
- $E_{\Phi}$ is the [penalty effect](#penalty) (see $\eqref{eq:Penalty_total}$)

One effect must be designated as the objective via `is_objective=True`. The penalty effect is automatically created and always added to the objective.


## Usage Examples

### Basic Cost Minimization
- $E_{\Omega}$ is the chosen **objective effect** (see $\eqref{eq:Effect_Total}$)
- $E_{\Phi}$ is the [penalty effect](#penalty) (see $\eqref{eq:Penalty_total}$)

```python
costs = fx.Effect(
    label='costs',
    unit='€',
    is_objective=True,
    is_standard=True,  # Allows shorthand: effects_per_flow_hour={'costs': 50}
)
One effect must be designated as the objective via `is_objective=True`. The penalty effect is automatically created and always added to the objective.

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
$$
\min \quad E_{\Omega} + E_{\Phi} = \sum_{\text{t}_i \in \mathcal{T}} E_{\Omega,\text{temp}}(\text{t}_i) + E_{\Omega,\text{per}} + E_{\Phi,\text{per}} + \sum_{\text{t}_i \in \mathcal{T}} E_{\Phi,\text{temp}}(\text{t}_i)
$$

CO₂ automatically converted to costs. No need to specify cost contribution separately.
Where:
- Temporal effects sum over time: $\sum_{\text{t}_i} E_{\Omega,\text{temp}}(\text{t}_i)$ and $\sum_{\text{t}_i} E_{\Phi,\text{temp}}(\text{t}_i)$
- Periodic effects are constant: $E_{\Omega,\text{per}}$ and $E_{\Phi,\text{per}}$

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
$$
\min \quad \sum_{s \in \mathcal{S}} w_s \cdot \left( E_{\Omega}(s) + E_{\Phi}(s) \right)
$$

Grid import limited to 500 kW at any timestep.
Where:
- $\mathcal{S}$ is the set of scenarios
- $w_s$ is the weight for scenario $s$ (typically scenario probability)
- Periodic effects are **shared across scenarios**: $E_{\Omega,\text{per}}$ and $E_{\Phi,\text{per}}$ (same for all $s$)
- Temporal effects are **scenario-specific**: $E_{\Omega,\text{temp}}(s) = \sum_{\text{t}_i} E_{\Omega,\text{temp}}(\text{t}_i, s)$ and $E_{\Phi,\text{temp}}(s) = \sum_{\text{t}_i} E_{\Phi,\text{temp}}(\text{t}_i, s)$

### Investment Budget
**Interpretation:**
- Investment decisions (periodic) made once, used across all scenarios
- Operations (temporal) differ by scenario
- Objective balances expected value across scenarios
- **Both $E_{\Omega}$ (objective effect) and $E_{\Phi}$ (penalty) are weighted identically by $w_s$**

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
$$
\min \quad \sum_{y \in \mathcal{Y}} w_y \cdot \left( E_{\Omega}(y) + E_{\Phi}(y) \right)
$$

Optimize total costs while respecting €1M investment budget.
Where:
- $\mathcal{Y}$ is the set of periods (e.g., years)
- $w_y$ is the weight for period $y$ (typically annual discount factor)
- Each period $y$ has **independent** periodic and temporal effects (including penalty)
- Each period $y$ has **independent** investment and operational decisions
- **Both $E_{\Omega}$ (objective effect) and $E_{\Phi}$ (penalty) are weighted identically by $w_y$**

## Multi-Dimensional Effects

When using periods and scenarios (see [Dimensions](dimensions.md)), effects aggregate with weights:

$$
\min \sum_{periods} w_{period} \cdot \sum_{scenarios} w_{scenario} \cdot E(period, scenario)
\min \quad \sum_{y \in \mathcal{Y}} \left[ w_y \cdot \left( E_{\Omega,\text{per}}(y) + E_{\Phi,\text{per}}(y) \right) + \sum_{s \in \mathcal{S}} w_{y,s} \cdot \left( E_{\Omega,\text{temp}}(y,s) + E_{\Phi,\text{temp}}(y,s) \right) \right]
$$

**Key principle:** Investment decisions (periodic effects) are shared across scenarios within a period, while operational effects are scenario-specific.
Where:
- $\mathcal{S}$ is the set of scenarios
- $\mathcal{Y}$ is the set of periods
- $w_y$ is the period weight (for periodic effects)
- $w_{y,s}$ is the combined period-scenario weight (for temporal effects)
- **Periodic effects** $E_{\Omega,\text{per}}(y)$ and $E_{\Phi,\text{per}}(y)$ are period-specific but **scenario-independent**
- **Temporal effects** $E_{\Omega,\text{temp}}(y,s) = \sum_{\text{t}_i} E_{\Omega,\text{temp}}(\text{t}_i, y, s)$ and $E_{\Phi,\text{temp}}(y,s) = \sum_{\text{t}_i} E_{\Phi,\text{temp}}(\text{t}_i, y, s)$ are **fully indexed**

**Key Principle:**
- Scenarios and periods are **operationally independent** (no energy/resource exchange)
- Coupled **only through the weighted objective function**
- **Periodic effects within a period are shared across all scenarios** (investment made once per period)
- **Temporal effects are independent per scenario** (different operations under different conditions)
- **Both $E_{\Omega}$ (objective effect) and $E_{\Phi}$ (penalty) use identical weighting** ($w_y$ for periodic, $w_{y,s}$ for temporal)

---

## Implementation Details

- **Element Class:** [`Effect`][flixopt.effects.Effect]
- **Model Class:** [`EffectModel`][flixopt.effects.EffectModel]
- **Collection Class:** [`EffectCollection`][flixopt.effects.EffectCollection]
| Concept | Formulation | Time Dependency | Dimension Indexing |
|---------|-------------|-----------------|-------------------|
| **Temporal share** | $s_{l \rightarrow e, \text{temp}}(\text{t}_i)$ | Time-dependent | $(t, y, s)$ when present |
| **Periodic share** | $s_{l \rightarrow e, \text{per}}$ | Time-independent | $(y)$ when periods present |
| **Total temporal effect** | $E_{e,\text{temp},\text{tot}} = \sum_{\text{t}_i} E_{e,\text{temp}}(\text{t}_i)$ | Sum over time | Depends on dimensions |
| **Total periodic effect** | $E_{e,\text{per}}$ | Constant | $(y)$ when periods present |
| **Total effect** | $E_e = E_{e,\text{per}} + E_{e,\text{temp},\text{tot}}$ | Combined | Depends on dimensions |
| **Penalty effect** | $E_\Phi = E_{\Phi,\text{per}} + E_{\Phi,\text{temp},\text{tot}}$ | Combined (same as effects) | **Weighted identically to objective effect** |
| **Objective** | $\min(E_{\Omega} + E_{\Phi})$ | With weights when multi-dimensional | See formulations above |

---

## See Also

- [Bus](elements/Bus.md) — Penalty contributions from balance violations
- [Flow](elements/Flow.md) — Temporal contributions via `effects_per_flow_hour`
- [InvestParameters](features/InvestParameters.md) — Periodic contributions from investments
- [Dimensions](dimensions.md) — Multi-period and scenario handling
- [Core Concepts: Effects](../core-concepts.md#effects-what-youre-tracking) — High-level overview
