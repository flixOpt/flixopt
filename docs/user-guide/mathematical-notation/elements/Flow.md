# Flow

The **flow rate** is the main optimization variable of a Flow, limited by its size and relative bounds.

!!! info "Quick Reference"

    | Aspect | Description | Key Equation |
    |--------|-------------|--------------|
    | **Flow Rate Bounds** | Rate limited by size and relative bounds | $\text{P} \cdot \text{p}^{\text{L}}_{\text{rel}} \leq p(\text{t}_i) \leq \text{P} \cdot \text{p}^{\text{U}}_{\text{rel}}$ |
    | **Extensions** | Can add status or investment | [StatusParameters](../features/StatusParameters.md), [InvestParameters](../features/InvestParameters.md) |

---

## Mathematical Formulation

### Flow Rate Bounds

The flow rate $p(\text{t}_i)$ is bounded by the Flow size $\text{P}$ and relative bounds:

$$ \label{eq:flow_rate}
    \text{P} \cdot \text{p}^{\text{L}}_{\text{rel}}(\text{t}_i)
    \leq p(\text{t}_i) \leq
    \text{P} \cdot \text{p}^{\text{U}}_{\text{rel}}(\text{t}_i)
$$

??? note "Variable Definitions"

    - $\text{P}$ - Flow size (capacity)
    - $p(\text{t}_i)$ - Flow rate at time $\text{t}_i$ (variable)
    - $\text{p}^{\text{L}}_{\text{rel}}(\text{t}_i)$ - Relative lower bound (typically 0)
    - $\text{p}^{\text{U}}_{\text{rel}}(\text{t}_i)$ - Relative upper bound (typically 1)

    See [notation reference](../notation-reference.md) for common symbols.

!!! example "Typical Case"

    With $\text{p}^{\text{L}}_{\text{rel}} = 0$ and $\text{p}^{\text{U}}_{\text{rel}} = 1$, equation $\eqref{eq:flow_rate}$ simplifies to:

    $$ 0 \leq p(\text{t}_i) \leq \text{P} $$

### Extensions

This formulation can be extended by:

- **[StatusParameters](../features/StatusParameters.md)** - Define active/inactive state of the Flow
- **[InvestParameters](../features/InvestParameters.md)** - Change size from constant to optimization variable

---

## Implementation

[:octicons-code-24: `Flow`][flixopt.elements.Flow]{ .md-button .md-button--primary }

### Key Parameters

| Parameter | Mathematical Symbol | Description |
|-----------|---------------------|-------------|
| `size` | $\text{P}$ | Flow size (capacity) |
| `relative_minimum` | $\text{p}^{\text{L}}_{\text{rel}}$ | Relative lower bound |
| `relative_maximum` | $\text{p}^{\text{U}}_{\text{rel}}$ | Relative upper bound |
| `effects_per_flow_hour` | - | Operational effects (costs, emissions) |
| `invest_parameters` | - | Optional investment modeling |
| `status_parameters` | - | Optional active/inactive operation |

### Mathematical Patterns Used

!!! abstract "Modeling Patterns"

    Flow formulation uses the following patterns:

    | Pattern | Application | Reference |
    |---------|-------------|-----------|
    | **[Scaled Bounds](../modeling-patterns/bounds-and-states.md#scaled-bounds)** | Basic flow rate bounds | Equation $\eqref{eq:flow_rate}$ |
    | **[Scaled Bounds with State](../modeling-patterns/bounds-and-states.md#scaled-bounds-with-state)** | With StatusParameters | See [StatusParameters](../features/StatusParameters.md) |
    | **[Bounds with State](../modeling-patterns/bounds-and-states.md#bounds-with-state)** | With InvestParameters | See [InvestParameters](../features/InvestParameters.md) |

---

## See Also

- [StatusParameters](../features/StatusParameters.md) - Binary active/inactive operation
- [InvestParameters](../features/InvestParameters.md) - Variable flow sizing
- [Bus](Bus.md) - Flow balance constraints
- [LinearConverter](LinearConverter.md) - Flow ratio constraints
- [Storage](Storage.md) - Flow integration over time
- [Modeling Patterns](../modeling-patterns/index.md) - Mathematical building blocks
