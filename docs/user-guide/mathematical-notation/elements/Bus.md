# Bus

A **Bus** represents a nodal balance between incoming and outgoing flow rates.

!!! info "Quick Reference"

    | Aspect | Description | Key Equation |
    |--------|-------------|--------------|
    | **Nodal Balance** | Sum of inputs equals sum of outputs | $\sum_{f \in \mathcal{F}_\text{in}} p_f = \sum_{f \in \mathcal{F}_\text{out}} p_f$ |
    | **Excess Penalty** | Optional soft constraint with penalty | See [penalty formulation](#excess-penalty-optional) |

---

## Mathematical Formulation

### Nodal Balance

Basic balance equation between incoming and outgoing flows:

$$ \label{eq:bus_balance}
  \sum_{f_\text{in} \in \mathcal{F}_\text{in}} p_{f_\text{in}}(\text{t}_i) =
  \sum_{f_\text{out} \in \mathcal{F}_\text{out}} p_{f_\text{out}}(\text{t}_i)
$$

??? note "Variable Definitions"

    - $\mathcal{F}_\text{in}$, $\mathcal{F}_\text{out}$ - Sets of incoming and outgoing flows
    - $p_{f_\text{in}}(\text{t}_i)$, $p_{f_\text{out}}(\text{t}_i)$ - Flow rates (see [Flow](Flow.md))
    - $\text{t}_i$ - Time step (see [notation reference](../notation-reference.md))

### Excess Penalty (Optional)

Optionally, a Bus can have an `excess_penalty_per_flow_hour` parameter, which penalizes imbalance. This is useful for handling potential infeasibility gracefully.

**Modified Balance:**

$$ \label{eq:bus_balance_excess}
  \sum_{f_\text{in} \in \mathcal{F}_\text{in}} p_{f_\text{in}}(\text{t}_i) + \phi_\text{in}(\text{t}_i) =
  \sum_{f_\text{out} \in \mathcal{F}_\text{out}} p_{f_\text{out}}(\text{t}_i) + \phi_\text{out}(\text{t}_i)
$$

**Penalty Term:**

$$ \label{eq:bus_penalty}
  s_{b \rightarrow \Phi}(\text{t}_i) =
      \text{a}_{b \rightarrow \Phi}(\text{t}_i) \cdot \Delta \text{t}_i
      \cdot [ \phi_\text{in}(\text{t}_i) + \phi_\text{out}(\text{t}_i) ]
$$

??? note "Variable Definitions"

    - $\phi_\text{in}(\text{t}_i)$, $\phi_\text{out}(\text{t}_i)$ - Missing/excess flow rates (variables)
    - $s_{b \rightarrow \Phi}(\text{t}_i)$ - Penalty term
    - $\text{a}_{b \rightarrow \Phi}(\text{t}_i)$ - Penalty coefficient (`excess_penalty_per_flow_hour`)
    - $\Delta \text{t}_i$ - Time step duration

---

## Implementation

[:octicons-code-24: `Bus`][flixopt.elements.Bus]{ .md-button .md-button--primary }

### Key Parameters

| Parameter | Mathematical Symbol | Description |
|-----------|---------------------|-------------|
| - | $\mathcal{F}_\text{in}$, $\mathcal{F}_\text{out}$ | Flows connected to bus (implicit) |
| `excess_penalty_per_flow_hour` | $\text{a}_{b \rightarrow \Phi}$ | Penalty coefficient (optional) |

---

## See Also

- [Flow](Flow.md) - Definition of flow rates in the balance
- [Effects, Penalty & Objective](../effects-penalty-objective.md) - How penalties are included in the objective function
- [Modeling Patterns](../modeling-patterns/index.md) - Mathematical building blocks
