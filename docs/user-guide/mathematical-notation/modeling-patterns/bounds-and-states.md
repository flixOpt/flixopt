# Bounds and States

This document describes the mathematical formulations for variable bounding patterns used throughout FlixOpt. These patterns define how optimization variables are constrained, both with and without state control.

## Basic Bounds

The simplest bounding pattern constrains a variable between lower and upper bounds.

$$\label{eq:basic_bounds}
\text{lower} \leq v \leq \text{upper}
$$

With:

- $v$ being the optimization variable
- $\text{lower}$ being the lower bound (constant or time-dependent)
- $\text{upper}$ being the upper bound (constant or time-dependent)

**Implementation:** [`BoundingPatterns.basic_bounds()`][flixopt.modeling.BoundingPatterns.basic_bounds]

**Used in:**
- Storage charge state bounds (see [Storage](../elements/Storage.md))
- Flow rate absolute bounds

---

## Bounds with State

When a variable should only be non-zero if a binary state variable is active (e.g., active/inactive operation, investment decisions), the bounds are controlled by the state:

$$\label{eq:bounds_with_state}
s \cdot \max(\varepsilon, \text{lower}) \leq v \leq s \cdot \text{upper}
$$

With:

- $v$ being the optimization variable
- $s \in \{0, 1\}$ being the binary state variable
- $\text{lower}$ being the lower bound when active
- $\text{upper}$ being the upper bound when active
- $\varepsilon$ being a small positive number to ensure numerical stability

**Behavior:**
- When $s = 0$: variable is forced to zero ($0 \leq v \leq 0$)
- When $s = 1$: variable can take values in $[\text{lower}, \text{upper}]$

**Implementation:** [`BoundingPatterns.bounds_with_state()`][flixopt.modeling.BoundingPatterns.bounds_with_state]

**Used in:**
- Flow rates with active/inactive operation (see [StatusParameters](../features/StatusParameters.md))
- Investment size decisions (see [InvestParameters](../features/InvestParameters.md))

---

## Scaled Bounds

When a variable's bounds depend on another variable (e.g., flow rate scaled by component size), scaled bounds are used:

$$\label{eq:scaled_bounds}
v_\text{scale} \cdot \text{rel}_\text{lower} \leq v \leq v_\text{scale} \cdot \text{rel}_\text{upper}
$$

With:

- $v$ being the optimization variable (e.g., flow rate)
- $v_\text{scale}$ being the scaling variable (e.g., component size)
- $\text{rel}_\text{lower}$ being the relative lower bound factor (typically 0)
- $\text{rel}_\text{upper}$ being the relative upper bound factor (typically 1)

**Example:** Flow rate bounds
- If $v_\text{scale} = P$ (flow size) and $\text{rel}_\text{upper} = 1$
- Then: $0 \leq p(t_i) \leq P$ (see [Flow](../elements/Flow.md))

**Implementation:** [`BoundingPatterns.scaled_bounds()`][flixopt.modeling.BoundingPatterns.scaled_bounds]

**Used in:**
- Flow rate constraints (see [Flow](../elements/Flow.md) equation 1)
- Storage charge state constraints (see [Storage](../elements/Storage.md) equation 1)

---

## Scaled Bounds with State

Combining scaled bounds with binary state control requires a Big-M formulation to handle both the scaling and the active/inactive behavior:

$$\label{eq:scaled_bounds_with_state_1}
(s - 1) \cdot M_\text{misc} + v_\text{scale} \cdot \text{rel}_\text{lower} \leq v \leq v_\text{scale} \cdot \text{rel}_\text{upper}
$$

$$\label{eq:scaled_bounds_with_state_2}
s \cdot M_\text{lower} \leq v \leq s \cdot M_\text{upper}
$$

With:

- $v$ being the optimization variable
- $v_\text{scale}$ being the scaling variable
- $s \in \{0, 1\}$ being the binary state variable
- $\text{rel}_\text{lower}$ being the relative lower bound factor
- $\text{rel}_\text{upper}$ being the relative upper bound factor
- $M_\text{misc} = v_\text{scale,max} \cdot \text{rel}_\text{lower}$
- $M_\text{upper} = v_\text{scale,max} \cdot \text{rel}_\text{upper}$
- $M_\text{lower} = \max(\varepsilon, v_\text{scale,min} \cdot \text{rel}_\text{lower})$

Where $v_\text{scale,max}$ and $v_\text{scale,min}$ are the maximum and minimum possible values of the scaling variable.

**Behavior:**
- When $s = 0$: variable is forced to zero
- When $s = 1$: variable follows scaled bounds $v_\text{scale} \cdot \text{rel}_\text{lower} \leq v \leq v_\text{scale} \cdot \text{rel}_\text{upper}$

**Implementation:** [`BoundingPatterns.scaled_bounds_with_state()`][flixopt.modeling.BoundingPatterns.scaled_bounds_with_state]

**Used in:**
- Flow rates with active/inactive operation and investment sizing
- Components combining [StatusParameters](../features/StatusParameters.md) and [InvestParameters](../features/InvestParameters.md)

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

With:

- $v_\text{tracker}$ being the auxiliary tracking variable
- $\text{expression}$ being a linear expression of other variables
- $\text{lower}, \text{upper}$ being optional bounds on the tracker

**Use cases:**
- Creating named variables for complex expressions
- Bounding intermediate results
- Simplifying constraint formulations

**Implementation:** [`ModelingPrimitives.expression_tracking_variable()`][flixopt.modeling.ModelingPrimitives.expression_tracking_variable]

---

## Mutual Exclusivity

When multiple binary variables should not be active simultaneously (at most one can be 1):

$$\label{eq:mutual_exclusivity}
\sum_{i} s_i(t) \leq \text{tolerance} \quad \forall t
$$

With:

- $s_i(t) \in \{0, 1\}$ being binary state variables
- $\text{tolerance}$ being the maximum number of simultaneously active states (typically 1)
- $t$ being the time index

**Use cases:**
- Ensuring only one operating mode is active
- Mutual exclusion of operation and maintenance states
- Enforcing single-choice decisions

**Implementation:** [`ModelingPrimitives.mutual_exclusivity_constraint()`][flixopt.modeling.ModelingPrimitives.mutual_exclusivity_constraint]

**Used in:**
- Operating mode selection
- Piecewise linear function segments (see [Piecewise](../features/Piecewise.md))
