# Bounds and States

Mathematical formulations for variable bounding patterns used throughout FlixOpt. These patterns define how optimization variables are constrained, both with and without state control.

!!! info "Quick Reference"

    | Pattern | Key Equation | Use Case |
    |---------|--------------|----------|
    | **Basic Bounds** | $\text{lower} \leq v \leq \text{upper}$ | Simple variable constraints |
    | **Bounds with State** | $s \cdot \max(\varepsilon, \text{lower}) \leq v \leq s \cdot \text{upper}$ | Active/inactive operation |
    | **Scaled Bounds** | $v_\text{scale} \cdot \text{rel}_\text{lower} \leq v \leq v_\text{scale} \cdot \text{rel}_\text{upper}$ | Size-dependent bounds |
    | **Scaled + State** | Combined scaling and state control | Investment with status |

---

## Basic Bounds

The simplest bounding pattern constrains a variable between lower and upper bounds.

$$\label{eq:basic_bounds}
\text{lower} \leq v \leq \text{upper}
$$

??? note "Variable Definitions"

    - $v$ - Optimization variable
    - $\text{lower}$ - Lower bound (constant or time-dependent)
    - $\text{upper}$ - Upper bound (constant or time-dependent)

**Used in:**

- Storage charge state bounds (see [Storage](../elements/Storage.md))
- Flow rate absolute bounds

[:octicons-code-16: `BoundingPatterns.basic_bounds()`][flixopt.modeling.BoundingPatterns.basic_bounds]{ .md-button }

---

## Bounds with State

When a variable should only be non-zero if a binary state variable is active (e.g., active/inactive operation, investment decisions), the bounds are controlled by the state:

$$\label{eq:bounds_with_state}
s \cdot \max(\varepsilon, \text{lower}) \leq v \leq s \cdot \text{upper}
$$

??? note "Variable Definitions"

    - $v$ - Optimization variable
    - $s \in \{0, 1\}$ - Binary state variable
    - $\text{lower}$ - Lower bound when active
    - $\text{upper}$ - Upper bound when active
    - $\varepsilon$ - Small positive number for numerical stability

**Behavior:**

- When $s = 0$: Variable forced to zero ($0 \leq v \leq 0$)
- When $s = 1$: Variable can take values in $[\text{lower}, \text{upper}]$

**Used in:**

- Flow rates with active/inactive operation (see [StatusParameters](../features/StatusParameters.md))
- Investment size decisions (see [InvestParameters](../features/InvestParameters.md))

[:octicons-code-16: `BoundingPatterns.bounds_with_state()`][flixopt.modeling.BoundingPatterns.bounds_with_state]{ .md-button }

---

## Scaled Bounds

When a variable's bounds depend on another variable (e.g., flow rate scaled by component size), scaled bounds are used:

$$\label{eq:scaled_bounds}
v_\text{scale} \cdot \text{rel}_\text{lower} \leq v \leq v_\text{scale} \cdot \text{rel}_\text{upper}
$$

??? note "Variable Definitions"

    - $v$ - Optimization variable (e.g., flow rate)
    - $v_\text{scale}$ - Scaling variable (e.g., component size)
    - $\text{rel}_\text{lower}$ - Relative lower bound factor (typically 0)
    - $\text{rel}_\text{upper}$ - Relative upper bound factor (typically 1)

!!! example "Flow Rate Bounds"

    If $v_\text{scale} = P$ (flow size) and $\text{rel}_\text{upper} = 1$:

    $$0 \leq p(t_i) \leq P$$

    See [Flow](../elements/Flow.md) for complete formulation.

**Used in:**

- Flow rate constraints (see [Flow](../elements/Flow.md))
- Storage charge state constraints (see [Storage](../elements/Storage.md))

[:octicons-code-16: `BoundingPatterns.scaled_bounds()`][flixopt.modeling.BoundingPatterns.scaled_bounds]{ .md-button }

---

## Scaled Bounds with State

Combining scaled bounds with binary state control requires a Big-M formulation to handle both the scaling and the active/inactive behavior:

$$\label{eq:scaled_bounds_with_state_1}
(s - 1) \cdot M_\text{misc} + v_\text{scale} \cdot \text{rel}_\text{lower} \leq v \leq v_\text{scale} \cdot \text{rel}_\text{upper}
$$

$$\label{eq:scaled_bounds_with_state_2}
s \cdot M_\text{lower} \leq v \leq s \cdot M_\text{upper}
$$

??? note "Variable Definitions"

    - $v$ - Optimization variable
    - $v_\text{scale}$ - Scaling variable
    - $s \in \{0, 1\}$ - Binary state variable
    - $\text{rel}_\text{lower}$ - Relative lower bound factor
    - $\text{rel}_\text{upper}$ - Relative upper bound factor
    - $M_\text{misc} = v_\text{scale,max} \cdot \text{rel}_\text{lower}$
    - $M_\text{upper} = v_\text{scale,max} \cdot \text{rel}_\text{upper}$
    - $M_\text{lower} = \max(\varepsilon, v_\text{scale,min} \cdot \text{rel}_\text{lower})$

    Where $v_\text{scale,max}$ and $v_\text{scale,min}$ are the maximum and minimum possible values of the scaling variable.

**Behavior:**

- When $s = 0$: Variable forced to zero
- When $s = 1$: Variable follows scaled bounds $v_\text{scale} \cdot \text{rel}_\text{lower} \leq v \leq v_\text{scale} \cdot \text{rel}_\text{upper}$

**Used in:**

- Flow rates with active/inactive operation and investment sizing
- Components combining [StatusParameters](../features/StatusParameters.md) and [InvestParameters](../features/InvestParameters.md)

[:octicons-code-16: `BoundingPatterns.scaled_bounds_with_state()`][flixopt.modeling.BoundingPatterns.scaled_bounds_with_state]{ .md-button }

---

## Expression Tracking

Sometimes it's necessary to create an auxiliary variable that equals an expression:

$$\label{eq:expression_tracking}
v_\text{tracker} = \text{expression}
$$

With optional bounds:

$$\label{eq:expression_tracking_bounds}
\text{lower} \leq v_\text{tracker} \leq \text{upper}
$$

??? note "Variable Definitions"

    - $v_\text{tracker}$ - Auxiliary tracking variable
    - $\text{expression}$ - Linear expression of other variables
    - $\text{lower}, \text{upper}$ - Optional bounds on the tracker

**Use cases:**

- Creating named variables for complex expressions
- Bounding intermediate results
- Simplifying constraint formulations

[:octicons-code-16: `ModelingPrimitives.expression_tracking_variable()`][flixopt.modeling.ModelingPrimitives.expression_tracking_variable]{ .md-button }

---

## Mutual Exclusivity

When multiple binary variables should not be active simultaneously (at most one can be 1):

$$\label{eq:mutual_exclusivity}
\sum_{i} s_i(t) \leq \text{tolerance} \quad \forall t
$$

??? note "Variable Definitions"

    - $s_i(t) \in \{0, 1\}$ - Binary state variables
    - $\text{tolerance}$ - Maximum number of simultaneously active states (typically 1)
    - $t$ - Time index

**Use cases:**

- Ensuring only one operating mode is active
- Mutual exclusion of operation and maintenance states
- Enforcing single-choice decisions

**Used in:**

- Operating mode selection
- Piecewise linear function segments (see [Piecewise](../features/Piecewise.md))

[:octicons-code-16: `ModelingPrimitives.mutual_exclusivity_constraint()`][flixopt.modeling.ModelingPrimitives.mutual_exclusivity_constraint]{ .md-button }

---

## See Also

- [Flow](../elements/Flow.md) - Scaled bounds application
- [Storage](../elements/Storage.md) - Basic and scaled bounds
- [StatusParameters](../features/StatusParameters.md) - Bounds with state for operation
- [InvestParameters](../features/InvestParameters.md) - Bounds with state for investment
- [Piecewise](../features/Piecewise.md) - Mutual exclusivity in piecewise segments
