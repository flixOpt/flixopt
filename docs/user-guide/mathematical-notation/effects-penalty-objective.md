# Effects, Penalty & Objective

Effects quantify system-wide impacts like costs, emissions, or resource consumption, aggregating contributions from Elements across the FlowSystem. One Effect serves as the optimization objective, while others can be tracked and constrained.

=== "Variables"

    | Symbol | Python Name | Description | Domain | Created When |
    |--------|-------------|-------------|--------|--------------|
    | $E_{e,\text{per}}$ | `(periodic)\|total` | Total periodic effect (time-independent) | $\mathbb{R}$ | Always (per Effect) |
    | $E_{e,\text{temp}}(\text{t}_i)$ | `(temporal)\|total` | Temporal effect at time $\text{t}_i$ | $\mathbb{R}$ | Always (per Effect) |
    | $E_e$ | `total` | Total effect (periodic + temporal combined) | $\mathbb{R}$ | Always (per Effect) |
    | $\Phi$ | `penalty` | Penalty term for constraint violations | $\mathbb{R}_+$ | Always (system-wide) |
    | - | `total_over_periods` | Weighted sum of total effect across all periods | $\mathbb{R}$ | `minimum_over_periods` or `maximum_over_periods` specified |

=== "Constraints"

    **Element shares to periodic effects**:

    $$\label{eq:Share_periodic}
    s_{l \rightarrow e, \text{per}} = \sum_{v \in \mathcal{V}_{l, \text{per}}} v \cdot \text{a}_{v \rightarrow e}
    $$

    Where $\mathcal{V}_{l, \text{per}}$ are periodic (investment-related) variables of element $l$, and $\text{a}_{v \rightarrow e}$ is the effect factor (e.g., €/kW).

    ---

    **Element shares to temporal effects**:

    $$\label{eq:Share_temporal}
    s_{l \rightarrow e, \text{temp}}(\text{t}_i) = \sum_{v \in \mathcal{V}_{l,\text{temp}}} v(\text{t}_i) \cdot \text{a}_{v \rightarrow e}(\text{t}_i)
    $$

    Where $\mathcal{V}_{l, \text{temp}}$ are temporal (operational) variables, and $\text{a}_{v \rightarrow e}(\text{t}_i)$ is the time-varying effect factor (e.g., €/kWh).

    ---

    **Total periodic effect**:

    $$\label{eq:Effect_periodic}
    E_{e, \text{per}} =
    \sum_{l \in \mathcal{L}} s_{l \rightarrow e,\text{per}} +
    \sum_{x \in \mathcal{E}\backslash e} E_{x, \text{per}}  \cdot \text{r}_{x \rightarrow  e,\text{per}}
    $$

    Aggregates element periodic shares plus cross-effect contributions (e.g., CO₂ → costs via carbon pricing).

    ---

    **Total temporal effect**:

    $$\label{eq:Effect_temporal}
    E_{e, \text{temp}}(\text{t}_{i}) =
    \sum_{l \in \mathcal{L}} s_{l \rightarrow e, \text{temp}}(\text{t}_i) +
    \sum_{x \in \mathcal{E}\backslash e} E_{x, \text{temp}}(\text{t}_i) \cdot \text{r}_{x \rightarrow {e},\text{temp}}(\text{t}_i)
    $$

    Aggregates element temporal shares plus cross-effect contributions at each timestep.

    ---

    **Total temporal effect (sum over time)**:

    $$\label{eq:Effect_temporal_total}
    E_{e,\text{temp},\text{tot}} = \sum_{i=1}^n  E_{e,\text{temp}}(\text{t}_{i})
    $$

    ---

    **Total combined effect**:

    $$\label{eq:Effect_Total}
    E_{e} = E_{e,\text{per}} + E_{e,\text{temp},\text{tot}}
    $$

    ---

    **Penalty term**:

    $$\label{eq:Penalty}
    \Phi = \sum_{l \in \mathcal{L}} \left( s_{l \rightarrow \Phi}  +\sum_{\text{t}_i \in \mathcal{T}} s_{l \rightarrow \Phi}(\text{t}_{i}) \right)
    $$

    Accumulates penalty shares from elements (primarily from [Bus](elements/Bus.md) via `excess_penalty_per_flow_hour`).

    ---

    **Objective function**:

    $$\label{eq:Objective}
    \min \left( E_{\Omega} + \Phi \right)
    $$

    Where $E_{\Omega}$ is the chosen objective effect (designated with `is_objective=True`).

    ---

    **Effect bounds - Temporal total** (when `minimum_temporal` or `maximum_temporal` specified):

    $$\label{eq:Bounds_Temporal_Total}
    E_{e,\text{temp}}^\text{L} \leq E_{e,\text{temp},\text{tot}} \leq E_{e,\text{temp}}^\text{U}
    $$

    ---

    **Effect bounds - Temporal per timestep** (when `minimum_per_hour` or `maximum_per_hour` specified):

    $$\label{eq:Bounds_Timestep}
    E_{e,\text{temp}}^\text{L}(\text{t}_i) \leq E_{e,\text{temp}}(\text{t}_i) \leq E_{e,\text{temp}}^\text{U}(\text{t}_i)
    $$

    ---

    **Effect bounds - Periodic** (when `minimum_periodic` or `maximum_periodic` specified):

    $$\label{eq:Bounds_Periodic}
    E_{e,\text{per}}^\text{L} \leq E_{e,\text{per}} \leq E_{e,\text{per}}^\text{U}
    $$

    ---

    **Effect bounds - Total combined** (when `minimum_total` or `maximum_total` specified):

    $$\label{eq:Bounds_Total}
    E_e^\text{L} \leq E_e \leq E_e^\text{U}
    $$

    ---

    **Effect bounds - Across all periods** (when `minimum_over_periods` or `maximum_over_periods` specified):

    $$\label{eq:Bounds_Over_Periods}
    E_e^\text{L,periods} \leq \sum_{y \in \mathcal{Y}} w_y \cdot E_e(y) \leq E_e^\text{U,periods}
    $$

    **Mathematical Patterns:** Effect Aggregation, Weighted Objectives

=== "Parameters"

    | Symbol | Python Parameter | Description | Default |
    |--------|------------------|-------------|---------|
    | $\mathcal{E}$ | - | Set of all effects in the system | From registered Effects |
    | $\mathcal{L}$ | - | Set of all elements in the FlowSystem | From all Elements |
    | $\mathcal{T}$ | - | Set of all timesteps | From system time index |
    | $\mathcal{V}_{l, \text{per}}$ | - | Subset of periodic (investment) variables for element $l$ | Element-dependent |
    | $\mathcal{V}_{l, \text{temp}}$ | - | Subset of temporal (operational) variables for element $l$ | Element-dependent |
    | $\mathcal{Y}$ | - | Set of periods | From system dimensions |
    | $\text{a}_{v \rightarrow e}$ | - | Effect factor (e.g., €/kW for investment) | Element-specific |
    | $\text{a}_{v \rightarrow e}(\text{t}_i)$ | - | Time-varying effect factor (e.g., €/kWh) | Element-specific |
    | $\text{r}_{x \rightarrow e, \text{per}}$ | `share_from_periodic` | Periodic conversion factor from effect $x$ to $e$ | None |
    | $\text{r}_{x \rightarrow e, \text{temp}}(\text{t}_i)$ | `share_from_temporal` | Temporal conversion factor from effect $x$ to $e$ | None |
    | $w_y$ | `weights` | Period weights (e.g., discount factors) | From FlowSystem or Effect-specific |
    | - | `description` | Descriptive name for the effect | None |
    | - | `is_objective` | If True, this effect is the optimization objective | False |
    | - | `is_standard` | If True, allows direct value input without effect dictionaries | False |
    | - | `maximum_over_periods` | Maximum weighted sum across all periods | None |
    | - | `maximum_per_hour` | Maximum contribution per timestep | None |
    | - | `maximum_periodic` | Maximum periodic contribution | None |
    | - | `maximum_temporal` | Maximum total temporal contribution | None |
    | - | `maximum_total` | Maximum total effect (periodic + temporal) | None |
    | - | `minimum_over_periods` | Minimum weighted sum across all periods | None |
    | - | `minimum_per_hour` | Minimum contribution per timestep | None |
    | - | `minimum_periodic` | Minimum periodic contribution | None |
    | - | `minimum_temporal` | Minimum total temporal contribution | None |
    | - | `minimum_total` | Minimum total effect (periodic + temporal) | None |
    | - | `unit` | Unit of the effect (informative only) | None |

=== "Use Cases"

    ## Basic Cost Objective

    ```python
    from flixopt import Effect, Flow

    # Define cost effect
    cost = Effect(
        label='system_costs',
        unit='€',
        is_objective=True,  # This is what we minimize
    )

    # Elements contribute via effects parameters
    generator = Flow(
        label='electricity_out',
        bus='grid',
        size=100,
        effects_per_flow_hour={'system_costs': 45},  # €45/MWh operational cost
    )
    ```

    **Variables:** $E_{\text{costs},\text{per}}$, $E_{\text{costs},\text{temp}}(\text{t}_i)$, $E_{\text{costs}}$

    **Constraints:** $\eqref{eq:Share_temporal}$ accumulating operational costs, $\eqref{eq:Effect_temporal}$ aggregating across elements, $\eqref{eq:Effect_Total}$ combining domains, $\eqref{eq:Objective}$ minimizing total cost

    ---

    ## Multi-Criteria: Cost with Emission Constraint

    ```python
    from flixopt import Effect, Flow

    # Cost objective
    cost = Effect(
        label='costs',
        unit='€',
        is_objective=True,
    )

    # CO₂ emissions as constraint
    co2 = Effect(
        label='co2_emissions',
        unit='kg',
        maximum_temporal=100000,  # Max 100 tons CO₂ per period
    )

    # Generator with both effects
    generator = Flow(
        label='electricity_out',
        bus='grid',
        size=100,
        effects_per_flow_hour={
            'costs': 45,  # €45/MWh
            'co2_emissions': 0.8,  # 0.8 kg CO₂/kWh
        },
    )
    ```

    **Variables:** $E_{\text{costs}}$, $E_{\text{CO}_2}$, $\Phi$

    **Constraints:** $\eqref{eq:Objective}$ minimizing costs, $\eqref{eq:Bounds_Temporal_Total}$ with $E_{\text{CO}_2,\text{temp},\text{tot}} \leq 100{,}000$ kg

    **Behavior:** ε-constraint method - optimize one objective while constraining another

    ---

    ## Cross-Effect: Carbon Pricing

    ```python
    from flixopt import Effect

    # CO₂ emissions
    co2 = Effect(
        label='co2_emissions',
        unit='kg',
    )

    # Cost with carbon pricing
    cost = Effect(
        label='costs',
        unit='€',
        is_objective=True,
        share_from_temporal={'co2_emissions': 0.05},  # €50/ton CO₂
    )
    ```

    **Variables:** $E_{\text{CO}_2,\text{temp}}(\text{t}_i)$, $E_{\text{costs},\text{temp}}(\text{t}_i)$

    **Constraints:** $\eqref{eq:Effect_temporal}$ with cross-effect term: $E_{\text{costs},\text{temp}}(\text{t}_i) = s_{\text{costs}}(\text{t}_i) + E_{\text{CO}_2,\text{temp}}(\text{t}_i) \cdot 0.05$

    **Behavior:** Weighted sum method - CO₂ emissions automatically converted to costs

    ---

    ## Investment Constraints

    ```python
    from flixopt import Effect, Flow, InvestParameters

    # Investment budget constraint
    capex = Effect(
        label='capital_costs',
        unit='€',
        maximum_periodic=5_000_000,  # €5M investment budget
    )

    cost = Effect(
        label='total_costs',
        unit='€',
        is_objective=True,
    )

    # Battery with investment
    battery_flow = Flow(
        label='storage_capacity',
        bus='electricity',
        size=InvestParameters(
            minimum_size=0,
            maximum_size=1000,
            effects_of_investment_per_size={'capital_costs': 600},  # €600/kWh
            effects_of_investment_per_size={'total_costs': 80},  # Annualized: €80/kWh/year
        ),
    )
    ```

    **Variables:** $E_{\text{capex},\text{per}}$, $E_{\text{costs},\text{per}}$, $E_{\text{costs},\text{temp}}(\text{t}_i)$

    **Constraints:** $\eqref{eq:Share_periodic}$ from investment, $\eqref{eq:Bounds_Periodic}$ with $E_{\text{capex},\text{per}} \leq 5{,}000{,}000$ €, $\eqref{eq:Objective}$ minimizing total costs

    ---

    ## Hourly Peak Constraint

    ```python
    from flixopt import Effect, Flow

    # Peak power limit
    peak_power = Effect(
        label='grid_power',
        unit='kW',
        maximum_per_hour=500,  # Max 500 kW at any timestep
    )

    grid_import = Flow(
        label='grid_import',
        bus='electricity',
        size=1000,
        effects_per_flow_hour={'grid_power': 1},  # 1 kW per kW flow
    )
    ```

    **Variables:** $E_{\text{peak},\text{temp}}(\text{t}_i)$

    **Constraints:** $\eqref{eq:Bounds_Timestep}$ with $E_{\text{peak},\text{temp}}(\text{t}_i) \leq 500$ kW for all $\text{t}_i$

---

## Multi-Dimensional Objective

When the FlowSystem includes **periods** and/or **scenarios** (see [Dimensions](dimensions.md)), the objective aggregates effects using weights.

### Time Only (Base Case)

$$
\min \quad E_{\Omega} + \Phi = \sum_{\text{t}_i \in \mathcal{T}} E_{\Omega,\text{temp}}(\text{t}_i) + E_{\Omega,\text{per}} + \Phi
$$

### Time + Scenario

$$
\min \quad \sum_{s \in \mathcal{S}} w_s \cdot \left( E_{\Omega}(s) + \Phi(s) \right)
$$

Where:
- Periodic effects are **shared across scenarios**: $E_{\Omega,\text{per}}$
- Temporal effects are **scenario-specific**: $E_{\Omega,\text{temp}}(s) = \sum_{\text{t}_i} E_{\Omega,\text{temp}}(\text{t}_i, s)$
- $w_s$ is the scenario weight (typically probability)

### Time + Period

$$
\min \quad \sum_{y \in \mathcal{Y}} w_y \cdot \left( E_{\Omega}(y) + \Phi(y) \right)
$$

Where $w_y$ is the period weight (typically annual discount factor).

### Time + Period + Scenario (Full)

$$
\min \quad \sum_{y \in \mathcal{Y}} \left[ w_y \cdot E_{\Omega,\text{per}}(y) + \sum_{s \in \mathcal{S}} w_{y,s} \cdot \left( E_{\Omega,\text{temp}}(y,s) + \Phi(y,s) \right) \right]
$$

**Key Principle:**
- **Periodic effects within a period are shared across all scenarios** (investment made once)
- **Temporal effects are independent per scenario** (different operations)
- Scenarios and periods are **operationally independent**, coupled only through the weighted objective

---

## Summary Table

| Concept | Formulation | Time Dependency | Dimension Indexing |
|---------|-------------|-----------------|-------------------|
| **Temporal share** | $s_{l \rightarrow e, \text{temp}}(\text{t}_i)$ | Time-dependent | $(t, y, s)$ when present |
| **Periodic share** | $s_{l \rightarrow e, \text{per}}$ | Time-independent | $(y)$ when periods present |
| **Total temporal effect** | $E_{e,\text{temp},\text{tot}} = \sum_{\text{t}_i} E_{e,\text{temp}}(\text{t}_i)$ | Sum over time | Depends on dimensions |
| **Total periodic effect** | $E_{e,\text{per}}$ | Constant | $(y)$ when periods present |
| **Total effect** | $E_e = E_{e,\text{per}} + E_{e,\text{temp},\text{tot}}$ | Combined | Depends on dimensions |
| **Objective** | $\min(E_{\Omega} + \Phi)$ | With weights when multi-dimensional | See formulations above |

---

## Implementation

- **Element Class:** [`Effect`][flixopt.effects.Effect]
- **Model Class:** [`EffectModel`][flixopt.effects.EffectModel]
- **Collection Class:** [`EffectCollection`][flixopt.effects.EffectCollection]

## See Also

- **System-Level:** [Dimensions](dimensions.md) - Multi-dimensional modeling with periods and scenarios
- **Elements:** [Flow](elements/Flow.md) - Temporal contributions via `effects_per_flow_hour`
- **Features:** [InvestParameters](features/InvestParameters.md) - Periodic contributions via investment effects
- **Components:** [Bus](elements/Bus.md) - Penalty contributions via `excess_penalty_per_flow_hour`
