# Dimensions

FlixOpt's `FlowSystem` supports multiple dimensions for modeling optimization problems. Understanding these dimensions is crucial for interpreting the mathematical formulations presented in this documentation.

## The Three Dimensions

FlixOpt models can have up to three dimensions:

1. **Time (`time`)** - **MANDATORY**
    - Represents the temporal evolution of the system
    - Defined via `pd.DatetimeIndex`
    - Must contain at least 2 timesteps
    - All optimization variables and constraints evolve over time
2. **Period (`period`)** - **OPTIONAL**
    - Represents independent planning periods (e.g., years 2020, 2021, 2022)
    - Defined via `pd.Index` with integer values
    - Used for multi-period optimization such as investment planning across years
    - Each period is independent with its own time series
3. **Scenario (`scenario`)** - **OPTIONAL**
    - Represents alternative futures or uncertainty realizations (e.g., "Base Case", "High Demand")
    - Defined via `pd.Index` with any labels
    - Scenarios within the same period share the same time dimension
    - Used for stochastic optimization or scenario comparison

---

## Dimensional Structure

**Coordinate System:**

```python
FlowSystemDimensions = Literal['time', 'period', 'scenario']

coords = {
    'time': pd.DatetimeIndex,      # Always present
    'period': pd.Index | None,      # Optional
    'scenario': pd.Index | None     # Optional
}
```

**Example:**
```python
import pandas as pd
import numpy as np
import flixopt as fx

timesteps = pd.date_range('2020-01-01', periods=24, freq='h')
scenarios = pd.Index(['Base Case', 'High Demand'])
periods = pd.Index([2020, 2021, 2022])

flow_system = fx.FlowSystem(
    timesteps=timesteps,
    periods=periods,
    scenarios=scenarios,
    scenario_weights=np.array([0.5, 0.5])  # Scenario weights
)
```

This creates a system with:
- 24 time steps per scenario per period
- 2 scenarios with equal weights (0.5 each)
- 3 periods (years)
- **Total decision space:** 24 × 2 × 3 = 144 time-scenario-period combinations

---

## Independence of Formulations

**All mathematical formulations in this documentation are independent of whether periods or scenarios are present.**

The equations shown throughout this documentation (for [Flow](elements/Flow.md), [Storage](elements/Storage.md), [Bus](elements/Bus.md), etc.) are written with only the time index $\text{t}_i$. When periods and/or scenarios are added, **the same equations apply** - they are simply expanded to additional dimensions.

### How Dimensions Expand Formulations

**Flow rate bounds** (from [Flow](elements/Flow.md)):

$$
\text{P} \cdot \text{p}^{\text{L}}_{\text{rel}}(\text{t}_{i}) \leq p(\text{t}_{i}) \leq \text{P} \cdot \text{p}^{\text{U}}_{\text{rel}}(\text{t}_{i})
$$

This equation remains valid regardless of dimensions:

| Dimensions Present | Variable Indexing | Interpretation |
|-------------------|-------------------|----------------|
| Time only | $p(\text{t}_i)$ | Flow rate at time $\text{t}_i$ |
| Time + Scenario | $p(\text{t}_i, s)$ | Flow rate at time $\text{t}_i$ in scenario $s$ |
| Time + Period | $p(\text{t}_i, y)$ | Flow rate at time $\text{t}_i$ in period $y$ |
| Time + Period + Scenario | $p(\text{t}_i, y, s)$ | Flow rate at time $\text{t}_i$ in period $y$, scenario $s$ |

**The mathematical relationship remains identical** - only the indexing expands.

---

## Independence Between Scenarios and Periods

**There is no interconnection between scenarios and periods, except for shared investment decisions within a period.**

### Scenario Independence

Scenarios within a period are **operationally independent**:

- Each scenario has its own operational variables: $p(\text{t}_i, s_1)$ and $p(\text{t}_i, s_2)$ are independent
- Scenarios cannot exchange energy, information, or resources
- Storage states are separate: $c(\text{t}_i, s_1) \neq c(\text{t}_i, s_2)$
- Binary states (on/off) are independent: $s(\text{t}_i, s_1)$ vs $s(\text{t}_i, s_2)$

Scenarios are connected **only through the objective function** via weights:

$$
\min \quad \sum_{s \in \mathcal{S}} w_s \cdot \text{Objective}_s
$$

Where:
- $\mathcal{S}$ is the set of scenarios
- $w_s$ is the weight for scenario $s$
- The optimizer balances performance across scenarios according to their weights
- **Both the objective effect and Penalty effect are weighted by $w_s$** (see [Penalty weighting](effects-penalty-objective.md#penalty))

### Period Independence

Periods are **completely independent** optimization problems:

- Each period has separate operational variables
- Each period has separate investment decisions
- No temporal coupling between periods (e.g., storage state at end of period $y$ does not affect period $y+1$)
- Periods cannot exchange resources or information

Periods are connected **only through weighted aggregation** in the objective:

$$
\min \quad \sum_{y \in \mathcal{Y}} w_y \cdot \text{Objective}_y
$$

Where **both the objective effect and Penalty effect are weighted by $w_y$** (see [Penalty weighting](effects-penalty-objective.md#penalty))

### Shared Periodic Decisions: The Exception

**Investment decisions (sizes) can be shared across all scenarios:**

By default, sizes (e.g., Storage capacity, Thermal power, ...) are **scenario-independent** but **flow_rates are scenario-specific**.

**Example - Flow with investment:**

$$
v_\text{invest}(y) = s_\text{invest}(y) \cdot \text{size}_\text{fixed} \quad \text{(one decision per period)}
$$

$$
p(\text{t}_i, y, s) \leq v_\text{invest}(y) \cdot \text{rel}_\text{upper} \quad \forall s \in \mathcal{S} \quad \text{(same capacity for all scenarios)}
$$

**Interpretation:**
- "We decide once in period $y$ how much capacity to build" (periodic decision)
- "This capacity is then operated differently in each scenario $s$ within period $y$" (temporal decisions)
- "Periodic effects (investment) are incurred once per period, temporal effects (operational) are weighted across scenarios"

This reflects real-world investment under uncertainty: you build capacity once (periodic/investment decision), but it operates under different conditions (temporal/operational decisions per scenario).

**Mathematical Flexibility:**

Variables can be either scenario-independent or scenario-specific:

| Variable Type | Scenario-Independent | Scenario-Specific |
|---------------|---------------------|-------------------|
| **Sizes** (e.g., $\text{P}$) | $\text{P}(y)$ - Single value per period | $\text{P}(y, s)$ - Different per scenario |
| **Flow rates** (e.g., $p(\text{t}_i)$) | $p(\text{t}_i, y)$ - Same across scenarios | $p(\text{t}_i, y, s)$ - Different per scenario |

**Use Cases:**

*Investment problems (with InvestParameters):*
- **Sizes shared** (default): Investment under uncertainty - build capacity that performs well across all scenarios
- **Sizes vary**: Scenario-specific capacity planning where different investments can be made for each future
- **Selected sizes shared**: Mix of shared critical infrastructure and scenario-specific optional/flexible capacity

*Dispatch problems (fixed sizes, no investments):*
- **Flow rates shared**: Robust dispatch - find a single operational strategy that works across all forecast scenarios (e.g., day-ahead unit commitment under demand/weather uncertainty)
- **Flow rates vary** (default): Scenario-adaptive dispatch - optimize operations for each scenario's specific conditions (demand, weather, prices)

For implementation details on controlling scenario independence, see the [`FlowSystem`][flixopt.flow_system.FlowSystem] API reference.

---

## Dimensional Impact on Objective Function

The objective function aggregates effects across all dimensions with weights:

### Time Only
$$
\min \quad \sum_{\text{t}_i \in \mathcal{T}} \sum_{e \in \mathcal{E}} s_{e}(\text{t}_i)
$$

### Time + Scenario
$$
\min \quad \sum_{s \in \mathcal{S}} w_s \cdot \left( \sum_{\text{t}_i \in \mathcal{T}} \sum_{e \in \mathcal{E}} s_{e}(\text{t}_i, s) \right)
$$

### Time + Period
$$
\min \quad \sum_{y \in \mathcal{Y}} w_y \cdot \left( \sum_{\text{t}_i \in \mathcal{T}} \sum_{e \in \mathcal{E}} s_{e}(\text{t}_i, y) \right)
$$

### Time + Period + Scenario (Full Multi-Dimensional)
$$
\min \quad \sum_{y \in \mathcal{Y}} \sum_{s \in \mathcal{S}} w_{y,s} \cdot \left( \sum_{\text{t}_i \in \mathcal{T}} \sum_{e \in \mathcal{E}} s_{e}(\text{t}_i, y, s) \right)
$$

Where:
- $\mathcal{T}$ is the set of time steps
- $\mathcal{E}$ is the set of effects (including the Penalty effect $E_\Phi$)
- $\mathcal{S}$ is the set of scenarios
- $\mathcal{Y}$ is the set of periods
- $s_{e}(\cdots)$ are the effect contributions (costs, emissions, etc.)
- $w_s, w_y, w_{y,s}$ are the dimension weights
- **Penalty effect is weighted identically to other effects**

**See [Effects, Penalty & Objective](effects-penalty-objective.md) for complete formulations including:**
- How temporal and periodic effects expand with dimensions
- Detailed objective function for each dimensional case
- Periodic (investment) vs temporal (operational) effect handling
- Explicit Penalty weighting formulations

---

## Weights

Weights determine the relative importance of scenarios and periods in the objective function.

### Scenario Weights

You provide scenario weights explicitly via the `scenario_weights` parameter:

```python
flow_system = fx.FlowSystem(
    timesteps=timesteps,
    scenarios=scenarios,
    scenario_weights=np.array([0.3, 0.7])  # Scenario probabilities
)
```

**Default:** If not specified, all scenarios have equal weight (normalized to sum to 1).

### Period Weights

Period weights are **automatically computed** from the period index (similar to how `hours_per_timestep` is computed from the time index):

```python
# Period weights are computed from the differences between period values
periods = pd.Index([2020, 2025, 2030, 2035])
# → period_weights = [5, 5, 5, 5] (representing 5-year intervals)

flow_system = fx.FlowSystem(
    timesteps=timesteps,
    periods=periods,
    # No need to specify period weights - they're computed automatically
)
```

**How period weights are computed:**
- For periods `[2020, 2025, 2030, 2035]`, the weights are `[5, 5, 5, 5]` (the interval sizes)
- This ensures that when you use `.sel()` to select a subset of periods, the weights are correctly recalculated
- You can specify `weight_of_last_period` if the last period weight cannot be inferred from the index

### Combined Weights

When both periods and scenarios are present, the combined `weights` array (accessible via `flow_system.model.objective_weights`) is computed as:

$$
w_{y,s} = w_y \times \frac{w_s}{\sum_{s \in \mathcal{S}} w_s}
$$

Where:
- $w_y$ are the period weights (computed from period index)
- $w_s$ are the scenario weights (user-specified)
- $\mathcal{S}$ is the set of all scenarios
- The scenario weights are normalized to sum to 1 before multiplication

**Example:**
```python
periods = pd.Index([2020, 2030, 2040])  # → period_weights = [10, 10, 10]
scenarios = pd.Index(['Base', 'High'])
scenario_weights = np.array([0.6, 0.4])

flow_system = fx.FlowSystem(
    timesteps=timesteps,
    periods=periods,
    scenarios=scenarios,
    scenario_weights=scenario_weights
)

# Combined weights shape: (3 periods, 2 scenarios)
# [[6.0, 4.0],   # 2020: 10 × [0.6, 0.4]
#  [6.0, 4.0],   # 2030: 10 × [0.6, 0.4]
#  [6.0, 4.0]]   # 2040: 10 × [0.6, 0.4]
```

**Normalization:** Set `normalize_weights=False` in `Optimization` to turn off the normalization.

---

## Summary Table

| Dimension | Required? | Independence | Typical Use Case |
|-----------|-----------|--------------|------------------|
| **time** | ✅ Yes | Variables evolve over time via constraints (e.g., storage balance) | All optimization problems |
| **scenario** | ❌ No | Fully independent operations; shared investments within period | Uncertainty modeling, risk assessment |
| **period** | ❌ No | Fully independent; no coupling between periods | Multi-year planning, long-term investment |

**Key Principle:** All constraints and formulations operate **within** each (period, scenario) combination independently. Only the objective function couples them via weighted aggregation.

---

## See Also

- [Effects, Penalty & Objective](effects-penalty-objective.md) - How dimensions affect the objective function
- [InvestParameters](features/InvestParameters.md) - Investment decisions across scenarios
- [FlowSystem API][flixopt.flow_system.FlowSystem] - Creating multi-dimensional systems
