# Linear Converter

**Linear Converters** define a linear ratio between incoming and outgoing [Flows](Flow.md).

!!! info "Quick Reference"

    | Aspect | Description | Key Equation |
    |--------|-------------|--------------|
    | **General Ratio** | Multi-flow weighted balance | $\sum_{f \in \mathcal{F}_{in}} \text{a}_f \cdot p_f = \sum_{f \in \mathcal{F}_{out}} \text{b}_f \cdot p_f$ |
    | **Simple Ratio** | Single input/output | $\text{a} \cdot p_{in} = p_{out}$ |

---

## Mathematical Formulation

### General Conversion Ratio

For multiple incoming and outgoing flows:

$$ \label{eq:Linear-Transformer-Ratio}
    \sum_{f_{\text{in}} \in \mathcal{F}_{in}} \text{a}_{f_{\text{in}}}(\text{t}_i) \cdot p_{f_\text{in}}(\text{t}_i) =
    \sum_{f_{\text{out}} \in \mathcal{F}_{out}} \text{b}_{f_\text{out}}(\text{t}_i) \cdot p_{f_\text{out}}(\text{t}_i)
$$

??? note "Variable Definitions"

    - $\mathcal{F}_{in}$, $\mathcal{F}_{out}$ - Sets of incoming and outgoing flows
    - $p_{f_\text{in}}(\text{t}_i)$, $p_{f_\text{out}}(\text{t}_i)$ - Flow rates (see [Flow](Flow.md))
    - $\text{a}_{f_\text{in}}(\text{t}_i)$, $\text{b}_{f_\text{out}}(\text{t}_i)$ - Conversion ratios (parameters)
    - $\text{t}_i$ - Time step

### Simplified Ratio

With one incoming and one outgoing flow, this simplifies to:

$$ \label{eq:Linear-Transformer-Ratio-simple}
    \text{a}(\text{t}_i) \cdot p_{f_\text{in}}(\text{t}_i) = p_{f_\text{out}}(\text{t}_i)
$$

where $\text{a}$ can be interpreted as the **conversion efficiency** of the LinearConverter.

!!! tip "Piecewise Conversion Factors"

    Conversion efficiency can be defined as a piecewise linear approximation for non-linear behavior. See [Piecewise](../features/Piecewise.md) for details.

---

## Implementation

[:octicons-code-24: `LinearConverter`][flixopt.components.LinearConverter]{ .md-button .md-button--primary }

### Specialized Linear Converters

FlixOpt provides specialized linear converter classes for common applications:

| Class | Application | Mathematical Basis |
|-------|-------------|-------------------|
| [:octicons-code-16: `HeatPump`][flixopt.linear_converters.HeatPump] | Heat pump systems | Coefficient of Performance (COP) |
| [:octicons-code-16: `Power2Heat`][flixopt.linear_converters.Power2Heat] | Electric heating | Efficiency ≤ 1 |
| [:octicons-code-16: `CHP`][flixopt.linear_converters.CHP] | Combined heat & power | Multi-output conversion |
| [:octicons-code-16: `Boiler`][flixopt.linear_converters.Boiler] | Fuel to heat | Combustion efficiency |

These classes handle the mathematical formulation automatically based on physical relationships.

### Key Parameters

| Parameter | Mathematical Symbol | Description |
|-----------|---------------------|-------------|
| Input/output flows | $\mathcal{F}_{in}$, $\mathcal{F}_{out}$ | Flows connected to converter |
| Conversion ratios | $\text{a}_f$, $\text{b}_f$ | Flow-specific ratios |
| `piecewise` | - | Optional piecewise linear efficiency |

---

## See Also

- [Flow](Flow.md) - Definition of flow rates
- [Piecewise](../features/Piecewise.md) - Non-linear conversion efficiency modeling
- [InvestParameters](../features/InvestParameters.md) - Variable converter sizing
- [Modeling Patterns](../modeling-patterns/index.md) - Mathematical building blocks
