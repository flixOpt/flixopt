# Effects, Penalty & Objective

## Effects

[`Effects`][flixopt.effects.Effect] are used to quantify system-wide impacts like costs, emissions, or resource consumption. These arise from **shares** contributed by **Elements** such as [Flows](elements/Flow.md), [Storage](elements/Storage.md), and other components.

**Example:**

[`Flows`][flixopt.elements.Flow] have an attribute `effects_per_flow_hour` that defines the effect contribution per flow-hour:
- Costs (€/kWh)
- Emissions (kg CO₂/kWh)
- Primary energy consumption (kWh_primary/kWh)

Effects are categorized into two domains:

1. **Temporal effects** - Time-dependent contributions (e.g., operational costs, hourly emissions)
2. **Periodic effects** - Time-independent contributions (e.g., investment costs, fixed annual fees)

### Multi-Dimensional Effects

**The formulations below are written with time index $\text{t}_i$ only, but automatically expand when periods and/or scenarios are present.**

When the FlowSystem has additional dimensions (see [Dimensions](dimensions.md)):

- **Temporal effects** are indexed by all present dimensions: $E_{e,\text{temp}}(\text{t}_i, y, s)$
- **Periodic effects** are indexed by period only (scenario-independent within a period): $E_{e,\text{per}}(y)$
- Effects are aggregated with dimension weights in the objective function

For complete details on how dimensions affect effects and the objective, see [Dimensions](dimensions.md).

---

## Effect Formulation

### Shares from Elements

Each element $l$ contributes shares to effect $e$ in both temporal and periodic domains:

**Periodic shares** (time-independent):
$$ \label{eq:Share_periodic}
s_{l \rightarrow e, \text{per}} = \sum_{v \in \mathcal{V}_{l, \text{per}}} v \cdot \text{a}_{v \rightarrow e}
$$

**Temporal shares** (time-dependent):
$$ \label{eq:Share_temporal}
s_{l \rightarrow e, \text{temp}}(\text{t}_i) = \sum_{v \in \mathcal{V}_{l,\text{temp}}} v(\text{t}_i) \cdot \text{a}_{v \rightarrow e}(\text{t}_i)
$$

Where:

- $\text{t}_i$ is the time step
- $\mathcal{V}_l$ is the set of all optimization variables of element $l$
- $\mathcal{V}_{l, \text{per}}$ is the subset of periodic (investment-related) variables
- $\mathcal{V}_{l, \text{temp}}$ is the subset of temporal (operational) variables
- $v$ is an optimization variable
- $v(\text{t}_i)$ is the variable value at timestep $\text{t}_i$
- $\text{a}_{v \rightarrow e}$ is the effect factor (e.g., €/kW for investment, €/kWh for operation)
- $s_{l \rightarrow e, \text{per}}$ is the periodic share of element $l$ to effect $e$
- $s_{l \rightarrow e, \text{temp}}(\text{t}_i)$ is the temporal share of element $l$ to effect $e$

**Examples:**
- **Periodic share**: Investment cost = $\text{size} \cdot \text{specific\_cost}$ (€/kW)
- **Temporal share**: Operational cost = $\text{flow\_rate}(\text{t}_i) \cdot \text{price}(\text{t}_i)$ (€/kWh)

---

### Cross-Effect Contributions

Effects can contribute shares to other effects, enabling relationships like carbon pricing or resource accounting.

An effect $x$ can contribute to another effect $e \in \mathcal{E}\backslash x$ via conversion factors:

**Example:** CO₂ emissions (kg) → Monetary costs (€)
- Effect $x$: "CO₂ emissions" (unit: kg)
- Effect $e$: "costs" (unit: €)
- Factor $\text{r}_{x \rightarrow e}$: CO₂ price (€/kg)

**Note:** Circular references must be avoided.

### Total Effect Calculation

**Periodic effects** aggregate element shares and cross-effect contributions:

$$ \label{eq:Effect_periodic}
E_{e, \text{per}} =
\sum_{l \in \mathcal{L}} s_{l \rightarrow e,\text{per}} +
\sum_{x \in \mathcal{E}\backslash e} E_{x, \text{per}}  \cdot \text{r}_{x \rightarrow  e,\text{per}}
$$

**Temporal effects** at each timestep:

$$ \label{eq:Effect_temporal}
E_{e, \text{temp}}(\text{t}_{i}) =
\sum_{l \in \mathcal{L}} s_{l \rightarrow e, \text{temp}}(\text{t}_i) +
\sum_{x \in \mathcal{E}\backslash e} E_{x, \text{temp}}(\text{t}_i) \cdot \text{r}_{x \rightarrow {e},\text{temp}}(\text{t}_i)
$$

**Total temporal effects** (sum over all timesteps):

$$\label{eq:Effect_temporal_total}
E_{e,\text{temp},\text{tot}} = \sum_{i=1}^n  E_{e,\text{temp}}(\text{t}_{i})
$$

**Total effect** (combining both domains):

$$ \label{eq:Effect_Total}
E_{e} = E_{e,\text{per}} + E_{e,\text{temp},\text{tot}}
$$

Where:

- $\mathcal{L}$ is the set of all elements in the FlowSystem
- $\mathcal{E}$ is the set of all effects
- $\text{r}_{x \rightarrow e, \text{per}}$ is the periodic conversion factor from effect $x$ to effect $e$
- $\text{r}_{x \rightarrow e, \text{temp}}(\text{t}_i)$ is the temporal conversion factor

---

### Constraining Effects

Effects can be bounded to enforce limits on costs, emissions, or other impacts:

**Total bounds** (apply to $E_{e,\text{per}}$, $E_{e,\text{temp},\text{tot}}$, or $E_e$):

$$ \label{eq:Bounds_Total}
E^\text{L} \leq E \leq E^\text{U}
$$

**Temporal bounds per timestep:**

$$ \label{eq:Bounds_Timestep}
E_{e,\text{temp}}^\text{L}(\text{t}_i) \leq E_{e,\text{temp}}(\text{t}_i) \leq E_{e,\text{temp}}^\text{U}(\text{t}_i)
$$

**Implementation:** See [`Effect`][flixopt.effects.Effect] parameters:
- `minimum_temporal`, `maximum_temporal` - Total temporal bounds
- `minimum_per_hour`, `maximum_per_hour` - Hourly temporal bounds
- `minimum_periodic`, `maximum_periodic` - Periodic bounds
- `minimum_total`, `maximum_total` - Combined total bounds

---

## Penalty

Every FlixOpt model includes a special **Penalty Effect** $E_\Phi$ to:
- Prevent infeasible problems
- Simplify troubleshooting by allowing constraint violations with high cost

**Key Feature:** Penalty is implemented as a standard Effect (labeled `Penalty`), so you can **add penalty contributions anywhere effects are used**:

```python
import flixopt as fx

# Add penalty contributions just like any other effect
on_off = fx.OnOffParameters(
    effects_per_switch_on={'Penalty': 1}  #Instead of costs, just add a metric to switching on operations
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

### Multi-Criteria Optimization

This formulation supports multiple optimization approaches:

**1. Weighted Sum Method**
- The objective effect can incorporate other effects via cross-effect factors
- Example: Minimize costs while including carbon pricing: $\text{CO}_2 \rightarrow \text{costs}$

**2. ε-Constraint Method**
- Optimize one effect while constraining others
- Example: Minimize costs subject to $\text{CO}_2 \leq 1000$ kg

---

## Objective with Multiple Dimensions

When the FlowSystem includes **periods** and/or **scenarios** (see [Dimensions](dimensions.md)), the objective aggregates effects across all dimensions using weights.

### Time Only (Base Case)

$$
\min \quad E_{\Omega} + E_{\Phi} = \sum_{\text{t}_i \in \mathcal{T}} E_{\Omega,\text{temp}}(\text{t}_i) + E_{\Omega,\text{per}} + E_{\Phi,\text{per}} + \sum_{\text{t}_i \in \mathcal{T}} E_{\Phi,\text{temp}}(\text{t}_i)
$$

Where:
- Temporal effects sum over time: $\sum_{\text{t}_i} E_{\Omega,\text{temp}}(\text{t}_i)$ and $\sum_{\text{t}_i} E_{\Phi,\text{temp}}(\text{t}_i)$
- Periodic effects are constant: $E_{\Omega,\text{per}}$ and $E_{\Phi,\text{per}}$

---

### Time + Scenario

$$
\min \quad \sum_{s \in \mathcal{S}} w_s \cdot \left( E_{\Omega}(s) + E_{\Phi}(s) \right)
$$

Where:
- $\mathcal{S}$ is the set of scenarios
- $w_s$ is the weight for scenario $s$ (typically scenario probability)
- Periodic effects are **shared across scenarios**: $E_{\Omega,\text{per}}$ and $E_{\Phi,\text{per}}$ (same for all $s$)
- Temporal effects are **scenario-specific**: $E_{\Omega,\text{temp}}(s) = \sum_{\text{t}_i} E_{\Omega,\text{temp}}(\text{t}_i, s)$ and $E_{\Phi,\text{temp}}(s) = \sum_{\text{t}_i} E_{\Phi,\text{temp}}(\text{t}_i, s)$

**Interpretation:**
- Investment decisions (periodic) made once, used across all scenarios
- Operations (temporal) differ by scenario
- Objective balances expected value across scenarios
- **Both $E_{\Omega}$ (objective effect) and $E_{\Phi}$ (penalty) are weighted identically by $w_s$**

---

### Time + Period

$$
\min \quad \sum_{y \in \mathcal{Y}} w_y \cdot \left( E_{\Omega}(y) + E_{\Phi}(y) \right)
$$

Where:
- $\mathcal{Y}$ is the set of periods (e.g., years)
- $w_y$ is the weight for period $y$ (typically annual discount factor)
- Each period $y$ has **independent** periodic and temporal effects (including penalty)
- Each period $y$ has **independent** investment and operational decisions
- **Both $E_{\Omega}$ (objective effect) and $E_{\Phi}$ (penalty) are weighted identically by $w_y$**

---

### Time + Period + Scenario (Full Multi-Dimensional)

$$
\min \quad \sum_{y \in \mathcal{Y}} \left[ w_y \cdot \left( E_{\Omega,\text{per}}(y) + E_{\Phi,\text{per}}(y) \right) + \sum_{s \in \mathcal{S}} w_{y,s} \cdot \left( E_{\Omega,\text{temp}}(y,s) + E_{\Phi,\text{temp}}(y,s) \right) \right]
$$

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

## Summary

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

- [Dimensions](dimensions.md) - Complete explanation of multi-dimensional modeling
- [Flow](elements/Flow.md) - Temporal effect contributions via `effects_per_flow_hour`
- [InvestParameters](features/InvestParameters.md) - Periodic effect contributions via investment
- [Effect API][flixopt.effects.Effect] - Implementation details and parameters
