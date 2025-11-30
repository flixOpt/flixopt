# Mathematical Notation

This section provides the detailed mathematical formulations behind flixOpt. It expands on the concepts introduced in [Core Concepts](../core-concepts.md) with precise equations, variables, and constraints.

!!! tip "When to read this"
    You don't need this section to use flixOpt effectively. It's here for:

    - Understanding exactly what the solver is optimizing
    - Debugging unexpected model behavior
    - Extending flixOpt with custom constraints
    - Academic work requiring formal notation

## Structure

The documentation follows the same structure as Core Concepts:

| Core Concept | Mathematical Details |
|--------------|---------------------|
| **Buses** — where things connect | [Bus](elements/Bus.md) — balance equations, penalty terms |
| **Flows** — what moves | [Flow](elements/Flow.md) — capacity bounds, load factors, profiles |
| **Converters** — transform things | [LinearConverter](elements/LinearConverter.md) — conversion ratios |
| **Storages** — save for later | [Storage](elements/Storage.md) — charge dynamics, efficiency losses |
| **Effects** — what you track | [Effects & Objective](effects-penalty-objective.md) — cost aggregation, constraints |

Additional sections cover:

- **[Features](features/InvestParameters.md)** — Investment decisions, on/off operation, piecewise linearization
- **[Dimensions](dimensions.md)** — Time, periods, and scenarios
- **[Modeling Patterns](modeling-patterns/index.md)** — Internal implementation details (advanced)

## Notation Conventions

### Variables (What the optimizer decides)

Optimization variables are shown in *italic*:

| Symbol | Meaning | Example |
|--------|---------|---------|
| $p(t)$ | Flow rate at time $t$ | Heat output of a boiler |
| $c(t)$ | Charge state at time $t$ | Energy stored in a battery |
| $P$ | Size/capacity (when optimized) | Installed capacity of a heat pump |
| $s(t)$ | Binary on/off state | Whether a generator is running |

### Parameters (What you provide)

Parameters and constants are shown in upright text:

| Symbol | Meaning | Example |
|--------|---------|---------|
| $\eta$ | Efficiency | Boiler thermal efficiency (0.9) |
| $\Delta t$ | Timestep duration | 1 hour |
| $p_{min}$, $p_{max}$ | Flow bounds | Min/max operating power |

### Sets and Indices

| Symbol | Meaning |
|--------|---------|
| $t \in \mathcal{T}$ | Time steps |
| $f \in \mathcal{F}$ | Flows |
| $e \in \mathcal{E}$ | Effects |

## The Optimization Problem

At its core, flixOpt solves:

$$
\min \quad objective + penalty
$$

**Subject to:**

- Balance constraints at each bus
- Capacity bounds on each flow
- Storage dynamics over time
- Conversion relationships in converters
- Any additional effect constraints

The following pages detail each of these components.

## Quick Example

Consider a simple system: a gas boiler connected to a heat bus serving a demand.

**Variables:**

- $p_{gas}(t)$ — gas consumption at each timestep
- $p_{heat}(t)$ — heat production at each timestep

**Constraints:**

1. **Conversion** (boiler efficiency 90%):
   $$p_{heat}(t) = 0.9 \cdot p_{gas}(t)$$

2. **Capacity bounds** (boiler max 100 kW):
   $$0 \leq p_{heat}(t) \leq 100$$

3. **Balance** (meet demand):
   $$p_{heat}(t) = demand(t)$$

**Objective** (minimize gas cost at €50/MWh):
$$\min \sum_t p_{gas}(t) \cdot \Delta t \cdot 50$$

This simple example shows how the concepts combine. Real models have many more components, but the principles remain the same.

## Next Steps

Start with the element that's most relevant to your question:

- **Why isn't my demand being met?** → [Bus](elements/Bus.md) (balance constraints)
- **Why is my component not running?** → [Flow](elements/Flow.md) (capacity bounds)
- **How does storage charge/discharge?** → [Storage](elements/Storage.md) (charge dynamics)
- **How are efficiencies handled?** → [LinearConverter](elements/LinearConverter.md) (conversion)
- **How are costs calculated?** → [Effects & Objective](effects-penalty-objective.md)
