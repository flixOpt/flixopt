[`LinearConverters`][flixopt.components.LinearConverter] define a ratio between incoming and outgoing [Flows](../elements/Flow.md).

$$ \label{eq:Linear-Transformer-Ratio}
    \sum_{f_{\text{in}} \in \mathcal F_{in}} \text a_{f_{\text{in}}}(\text{t}_i) \cdot p_{f_\text{in}}(\text{t}_i) = \sum_{f_{\text{out}} \in \mathcal F_{out}}  \text b_{f_\text{out}}(\text{t}_i) \cdot p_{f_\text{out}}(\text{t}_i)
$$

With:

- $\mathcal F_{in}$ and $\mathcal F_{out}$ being the set of all incoming and outgoing flows
- $p_{f_\text{in}}(\text{t}_i)$ and $p_{f_\text{out}}(\text{t}_i)$ being the flow-rate at time $\text{t}_i$ for flow $f_\text{in}$ and $f_\text{out}$, respectively
- $\text a_{f_\text{in}}(\text{t}_i)$ and $\text b_{f_\text{out}}(\text{t}_i)$ being the ratio of the flow-rate at time $\text{t}_i$ for flow $f_\text{in}$ and $f_\text{out}$, respectively

With one incoming **Flow** and one outgoing **Flow**, this can be simplified to:

$$ \label{eq:Linear-Transformer-Ratio-simple}
    \text a(\text{t}_i) \cdot p_{f_\text{in}}(\text{t}_i) = p_{f_\text{out}}(\text{t}_i)
$$

where $\text a$ can be interpreted as the conversion efficiency of the **LinearConverter**.

#### Piecewise Conversion factors
The conversion efficiency can be defined as a piecewise linear approximation. See [Piecewise](../features/Piecewise.md) for more details.

---

## Implementation

**Class:** [`LinearConverter`][flixopt.components.LinearConverter]

**Location:** `flixopt/components.py:37`

**Key Constraints:**
- Linear conversion equation (eq. $\eqref{eq:Linear-Transformer-Ratio}$): Created in component-specific modeling methods

**Parameters:**
- Conversion factors $\text{a}_{f_\text{in}}$ and $\text{b}_{f_\text{out}}$ are defined through flow connections
- For simple converters with one input and one output, efficiency $\text{a}$ is specified

**Specialized Linear Converters:**

FlixOpt provides specialized linear converter classes for common applications:

- **[`HeatPump`][flixopt.linear_converters.HeatPump]** - Coefficient of Performance (COP) based conversion
- **[`Power2Heat`][flixopt.linear_converters.Power2Heat]** - Electric heating with efficiency ≤ 1
- **[`CHP`][flixopt.linear_converters.CHP]** - Combined heat and power generation
- **[`Boiler`][flixopt.linear_converters.Boiler]** - Fuel to heat conversion

These classes handle the mathematical formulation automatically based on physical relationships.

---

## See Also

- [Flow](../elements/Flow.md) - Definition of flow rates
- [Piecewise](../features/Piecewise.md) - Non-linear conversion efficiency modeling
- [InvestParameters](../features/InvestParameters.md) - Variable converter sizing
- [Modeling Patterns](../modeling-patterns/index.md) - Mathematical building blocks
