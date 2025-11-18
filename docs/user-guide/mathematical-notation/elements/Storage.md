# Storages

**Storages** have one incoming and one outgoing **[Flow](Flow.md)** with charging and discharging efficiency.

!!! info "Quick Reference"

    | Aspect | Description | Key Equation |
    |--------|-------------|--------------|
    | **Charge Bounds** | State of charge limited by size and relative bounds | $\text{C} \cdot \text{c}^{\text{L}}_{\text{rel}} \leq c(\text{t}_i) \leq \text{C} \cdot \text{c}^{\text{U}}_{\text{rel}}$ |
    | **Balance Equation** | Charge evolution with losses and flows | $c(\text{t}_{i+1}) = c(\text{t}_i)(1-\dot{\text{c}}_\text{rel,loss})^{\Delta t} + p_{in}\Delta t \cdot \eta_{in} - p_{out}\Delta t \cdot \eta_{out}$ |

---

## Mathematical Formulation

### Charge State Bounds

A storage has a state of charge $c(\text{t}_i)$ which is limited by its size $\text{C}$ and relative bounds:

$$ \label{eq:Storage_Bounds}
    \text{C} \cdot \text{c}^{\text{L}}_{\text{rel}}(\text{t}_i)
    \leq c(\text{t}_i) \leq
    \text{C} \cdot \text{c}^{\text{U}}_{\text{rel}}(\text{t}_i)
$$

??? note "Variable Definitions"

    - $\text{C}$ - Storage size (capacity)
    - $c(\text{t}_i)$ - State of charge at time $\text{t}_i$
    - $\text{c}^{\text{L}}_{\text{rel}}(\text{t}_i)$ - Relative lower bound (typically 0)
    - $\text{c}^{\text{U}}_{\text{rel}}(\text{t}_i)$ - Relative upper bound (typically 1)

    See [notation reference](../notation-reference.md) for time variable conventions.

!!! example "Typical Case"

    With $\text{c}^{\text{L}}_{\text{rel}} = 0$ and $\text{c}^{\text{U}}_{\text{rel}} = 1$, equation $\eqref{eq:Storage_Bounds}$ simplifies to:

    $$ 0 \leq c(\text{t}_i) \leq \text{C} $$

### Storage Balance

The state of charge $c(\text{t}_i)$ decreases by a fraction of the prior state of charge. The parameter $\dot{\text{c}}_\text{rel,loss}(\text{t}_i)$ expresses the "loss fraction per hour". The storage balance from $\text{t}_i$ to $\text{t}_{i+1}$ is:

$$
\begin{align*}
    c(\text{t}_{i+1}) &= c(\text{t}_i) \cdot (1-\dot{\text{c}}_\text{rel,loss}(\text{t}_i))^{\Delta \text{t}_i} \\
    &\quad + p_{f_\text{in}}(\text{t}_i) \cdot \Delta \text{t}_i \cdot \eta_\text{in}(\text{t}_i) \\
    &\quad - p_{f_\text{out}}(\text{t}_i) \cdot \Delta \text{t}_i \cdot \eta_\text{out}(\text{t}_i)
    \tag{3}
\end{align*}
$$

??? note "Variable Definitions"

    - $c(\text{t}_{i+1})$, $c(\text{t}_i)$ - State of charge at time $\text{t}_{i+1}$ and $\text{t}_i$
    - $\dot{\text{c}}_\text{rel,loss}(\text{t}_i)$ - Relative loss rate (self-discharge) per hour
    - $\Delta \text{t}_i$ - Time step duration in hours
    - $p_{f_\text{in}}(\text{t}_i)$, $p_{f_\text{out}}(\text{t}_i)$ - Input/output flow rates (see [Flow](Flow.md))
    - $\eta_\text{in}(\text{t}_i)$, $\eta_\text{out}(\text{t}_i)$ - Charging/discharging efficiencies

---

## Implementation

[:octicons-code-24: `Storage`][flixopt.components.Storage]{ .md-button .md-button--primary }

### Key Parameters

| Parameter | Mathematical Symbol | Description |
|-----------|---------------------|-------------|
| `capacity_in_flow_hours` | $\text{C}$ | Storage capacity |
| `relative_loss_per_hour` | $\dot{\text{c}}_\text{rel,loss}$ | Self-discharge rate |
| `initial_charge_state` | $c(\text{t}_0)$ | Initial charge |
| `minimal_final_charge_state` | $c(\text{t}_\text{end})$ | Minimum final charge (optional) |
| `maximal_final_charge_state` | $c(\text{t}_\text{end})$ | Maximum final charge (optional) |
| `eta_charge` | $\eta_\text{in}$ | Charging efficiency |
| `eta_discharge` | $\eta_\text{out}$ | Discharging efficiency |

### Mathematical Patterns Used

!!! abstract "Modeling Patterns"

    Storage formulation uses the following patterns:

    | Pattern | Application | Reference |
    |---------|-------------|-----------|
    | **[Basic Bounds](../modeling-patterns/bounds-and-states.md#basic-bounds)** | Charge state bounds | Equation $\eqref{eq:Storage_Bounds}$ |
    | **[Scaled Bounds](../modeling-patterns/bounds-and-states.md#scaled-bounds)** | Flow rate bounds relative to storage size | - |
    | **[Bounds with State](../modeling-patterns/bounds-and-states.md#bounds-with-state)** | With investment parameters | See [InvestParameters](../features/InvestParameters.md) |

---

## See Also

- [Flow](Flow.md) - Input/output flow definitions
- [InvestParameters](../features/InvestParameters.md) - Variable storage sizing
- [Modeling Patterns](../modeling-patterns/index.md) - Mathematical building blocks
