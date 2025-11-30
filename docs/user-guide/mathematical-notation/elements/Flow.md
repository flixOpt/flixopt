# Flow

The flow_rate is the main optimization variable of the Flow. It's limited by the size of the Flow and relative bounds \eqref{eq:flow_rate}.

$$ \label{eq:flow_rate}
    \text P \cdot \text p^{\text{L}}_{\text{rel}}(\text{t}_{i})
    \leq p(\text{t}_{i}) \leq
    \text P \cdot \text p^{\text{U}}_{\text{rel}}(\text{t}_{i})
$$

With:

- $\text P$ being the size of the Flow
- $p(\text{t}_{i})$ being the flow-rate at time $\text{t}_{i}$
- $\text p^{\text{L}}_{\text{rel}}(\text{t}_{i})$ being the relative lower bound (typically 0)
- $\text p^{\text{U}}_{\text{rel}}(\text{t}_{i})$ being the relative upper bound (typically 1)

With $\text p^{\text{L}}_{\text{rel}}(\text{t}_{i}) = 0$ and $\text p^{\text{U}}_{\text{rel}}(\text{t}_{i}) = 1$,
equation \eqref{eq:flow_rate} simplifies to

$$
    0 \leq p(\text{t}_{i}) \leq \text P
$$


This mathematical formulation can be extended by using [StatusParameters](../features/StatusParameters.md)
to define the active/inactive state of the Flow, or by using [InvestParameters](../features/InvestParameters.md)
to change the size of the Flow from a constant to an optimization variable.

---

## Mathematical Patterns Used

Flow formulation uses the following modeling patterns:

- **[Scaled Bounds](../modeling-patterns/bounds-and-states.md#scaled-bounds)** - Basic flow rate bounds (equation $\eqref{eq:flow_rate}$)
- **[Scaled Bounds with State](../modeling-patterns/bounds-and-states.md#scaled-bounds-with-state)** - When combined with [StatusParameters](../features/StatusParameters.md)
- **[Bounds with State](../modeling-patterns/bounds-and-states.md#bounds-with-state)** - Investment decisions with [InvestParameters](../features/InvestParameters.md)

---

## Implementation

**Python Class:** [`Flow`][flixopt.elements.Flow]

**Key Parameters:**

- `size`: Flow size $\text{P}$ (can be fixed or variable with InvestParameters)
- `relative_minimum`, `relative_maximum`: Relative bounds $\text{p}^{\text{L}}_{\text{rel}}, \text{p}^{\text{U}}_{\text{rel}}$
- `effects_per_flow_hour`: Operational effects (costs, emissions, etc.)
- `invest_parameters`: Optional investment modeling (see [InvestParameters](../features/InvestParameters.md))
- `status_parameters`: Optional active/inactive operation (see [StatusParameters](../features/StatusParameters.md))

See the [`Flow`][flixopt.elements.Flow] API documentation for complete parameter list and usage examples.

---

## See Also

- [StatusParameters](../features/StatusParameters.md) - Binary active/inactive operation
- [InvestParameters](../features/InvestParameters.md) - Variable flow sizing
- [Bus](../elements/Bus.md) - Flow balance constraints
- [LinearConverter](../elements/LinearConverter.md) - Flow ratio constraints
- [Storage](../elements/Storage.md) - Flow integration over time
- [Modeling Patterns](../modeling-patterns/index.md) - Mathematical building blocks
