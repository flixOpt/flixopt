# Piecewise Linear Approximation

**Piecewise** enables modeling non-linear relationships using linear segments, maintaining problem linearity.

!!! info "Quick Reference"

    | Aspect | Description | Use Case |
    |--------|-------------|----------|
    | **Purpose** | Approximate non-linear functions with linear segments | Efficiency curves, cost functions |
    | **Mathematical Basis** | Convex combination of segment endpoints | See [formulation](#mathematical-formulation) |

---

## Mathematical Formulation

### Basic Piecewise Formulation

A Piecewise consists of multiple [`Pieces`][flixopt.interface.Piece], each defining a valid range for a variable $v$.

**Active Piece Selection:**

$$ \label{eq:active_piece}
    \beta_k = \lambda_{0,k} + \lambda_{1,k}
$$

**Variable Value within Piece:**

$$ \label{eq:piece}
    v_k = \lambda_{0,k} \cdot \text{v}_{\text{start},k} + \lambda_{1,k} \cdot \text{v}_{\text{end},k}
$$

**Single Piece Active:**

$$ \label{eq:piecewise_in_pieces}
    \sum_{k=1}^K \beta_k = 1
$$

??? note "Variable Definitions"

    - $v$ - Variable defined by the Piecewise
    - $\text{v}_{\text{start},k}$, $\text{v}_{\text{end},k}$ - Start/end points of piece $k$ (parameters)
    - $\beta_k \in \{0, 1\}$ - Binary indicator: piece $k$ is active
    - $\lambda_{0,k}, \lambda_{1,k} \in [0, 1]$ - Convex combination weights for start/end points
    - $K$ - Total number of pieces

This formulation represents: $v \in [\text{v}_{\text{start}}, \text{v}_{\text{end}}]$

### Including Zero Values

To allow $v = 0$ in addition to the piece ranges, equation $\eqref{eq:piecewise_in_pieces}$ is modified:

$$ \label{eq:piecewise_in_pieces_zero}
    \sum_{k=1}^K \beta_k = \beta_{\text{zero}}
$$

where $\beta_{\text{zero}} \in \{0, 1\}$.

This formulation represents: $v \in \{0\} \cup [\text{v}_{\text{start}_k}, \text{v}_{\text{end}_k}]$

### Combining Multiple Piecewises

!!! abstract "Multi-Dimensional Non-Linearity"

    Piecewise approximation allows representing multi-variable non-linear relationships while maintaining linearity.

    **Key Constraint:** All Piecewises in a group must have the **same number of pieces** $K$.

**Variable Sharing:**

- Binary variables $\beta_k$ and weights $\lambda_{0,k}, \lambda_{1,k}$ are shared across all Piecewises
- Each Piecewise has its own constraint $\eqref{eq:piece}$ with specific endpoints
- This ensures **consistent segmentation** across multiple variables

**Example Application:**

- Heat pump efficiency: Power input vs. heat output (non-linear COP curve)
- Piece 1: Low power regime with endpoints $(P_1, H_1)$
- Piece 2: Medium power regime with endpoints $(P_2, H_2)$
- Piece 3: High power regime with endpoints $(P_3, H_3)$

---

## Implementation

[:octicons-code-24: `Piecewise`][flixopt.interface.Piecewise]{ .md-button .md-button--primary }
[:octicons-code-24: `Piece`][flixopt.interface.Piece]{ .md-button }

### Key Concepts

| Concept | Description | Mathematical Representation |
|---------|-------------|---------------------------|
| **Piece** | Single linear segment | $v \in [\text{v}_{\text{start}}, \text{v}_{\text{end}}]$ |
| **Piecewise** | Collection of pieces | Union of segment ranges |
| **Convex Combination** | Interpolation within segment | $\lambda_0 + \lambda_1 = 1$ if active |
| **Binary Selection** | Only one piece active at a time | $\sum_k \beta_k = 1$ |

### Common Use Cases

- **Efficiency curves** - Non-linear conversion efficiency (e.g., heat pumps, turbines)
- **Cost functions** - Tiered pricing, economies of scale
- **Operating ranges** - Equipment with distinct operating modes
- **Degradation** - Performance degradation over operating range

---

## See Also

- [LinearConverter](../elements/LinearConverter.md) - Using piecewise for conversion efficiency
- [InvestParameters](InvestParameters.md) - Piecewise investment costs
- [Modeling Patterns](../modeling-patterns/index.md) - Mathematical building blocks
