# State Transitions

State transition patterns model changes between discrete states and link them to continuous variables. These patterns are essential for modeling startup/shutdown events, switching behavior, and controlled changes in system operation.

!!! info "Quick Reference"

    | Pattern | Key Equation | Use Case |
    |---------|--------------|----------|
    | **Binary Transitions** | $s^{\text{on}} - s^{\text{off}} = s(t) - s(t-1)$ | Track startups/shutdowns |
    | **Continuous Transitions** | Change only when switching | Variable changes with state |
    | **Level Changes** | $\ell(t) = \ell(t-1) + \ell^{\text{inc}} - \ell^{\text{dec}}$ | Controlled level adjustments |

---

## Mathematical Formulation

### Binary State Transitions

For a binary state variable $s(t) \in \{0, 1\}$, state transitions track when the state switches on or off.

**Switch Variables:**

- $s^{\text{on}}(t) \in \{0, 1\}$ - Equals 1 when switching from off to on
- $s^{\text{off}}(t) \in \{0, 1\}$ - Equals 1 when switching from on to off

### Transition Tracking

The state change equals the difference between switch-on and switch-off:

$$\label{eq:state_transition}
s^{\text{on}}(t) - s^{\text{off}}(t) = s(t) - s(t-1) \quad \forall t > 0
$$

$$\label{eq:state_transition_initial}
s^{\text{on}}(0) - s^{\text{off}}(0) = s(0) - s_\text{prev}
$$

??? note "Variable Definitions"

    - $s(t)$ - Binary state variable
    - $s_\text{prev}$ - State before the optimization period
    - $s^{\text{on}}(t), s^{\text{off}}(t)$ - Switch variables

**Behavior:**

- Off → On ($s(t-1)=0, s(t)=1$): $s^{\text{on}}(t)=1, s^{\text{off}}(t)=0$
- On → Off ($s(t-1)=1, s(t)=0$): $s^{\text{on}}(t)=0, s^{\text{off}}(t)=1$
- No change: $s^{\text{on}}(t)=0, s^{\text{off}}(t)=0$

### Mutual Exclusivity of Switches

A state cannot switch on and off simultaneously:

$$\label{eq:switch_exclusivity}
s^{\text{on}}(t) + s^{\text{off}}(t) \leq 1 \quad \forall t
$$

This ensures:

- At most one switch event per time step
- No simultaneous active/inactive switching

### Complete State Transition Formulation

??? note "Combined Constraints"

    $$
    \begin{align}
    s^{\text{on}}(t) - s^{\text{off}}(t) &= s(t) - s(t-1) && \forall t > 0 \label{eq:transition_complete_1} \\
    s^{\text{on}}(0) - s^{\text{off}}(0) &= s(0) - s_\text{prev} && \label{eq:transition_complete_2} \\
    s^{\text{on}}(t) + s^{\text{off}}(t) &\leq 1 && \forall t \label{eq:transition_complete_3} \\
    s^{\text{on}}(t), s^{\text{off}}(t) &\in \{0, 1\} && \forall t \label{eq:transition_complete_4}
    \end{align}
    $$

[:octicons-code-16: `BoundingPatterns.state_transition_bounds()`][flixopt.modeling.BoundingPatterns.state_transition_bounds]{ .md-button }

---

### Continuous Transitions

When a continuous variable should only change when certain switch events occur, continuous transition bounds link the variable changes to binary switches.

**Change Bounds with Switches:**

$$\label{eq:continuous_transition}
-\Delta v^{\text{max}} \cdot (s^{\text{on}}(t) + s^{\text{off}}(t)) \leq v(t) - v(t-1) \leq \Delta v^{\text{max}} \cdot (s^{\text{on}}(t) + s^{\text{off}}(t)) \quad \forall t > 0
$$

$$\label{eq:continuous_transition_initial}
-\Delta v^{\text{max}} \cdot (s^{\text{on}}(0) + s^{\text{off}}(0)) \leq v(0) - v_\text{prev} \leq \Delta v^{\text{max}} \cdot (s^{\text{on}}(0) + s^{\text{off}}(0))
$$

??? note "Variable Definitions"

    - $v(t)$ - Continuous variable
    - $v_\text{prev}$ - Value before the optimization period
    - $\Delta v^{\text{max}}$ - Maximum allowed change
    - $s^{\text{on}}(t), s^{\text{off}}(t) \in \{0, 1\}$ - Switch binary variables

**Behavior:**

- When $s^{\text{on}}(t) = 0$ and $s^{\text{off}}(t) = 0$: Forces $v(t) = v(t-1)$ (no change)
- When $s^{\text{on}}(t) = 1$ or $s^{\text{off}}(t) = 1$: Allows change up to $\pm \Delta v^{\text{max}}$

[:octicons-code-16: `BoundingPatterns.continuous_transition_bounds()`][flixopt.modeling.BoundingPatterns.continuous_transition_bounds]{ .md-button }

---

### Level Changes with Binaries

This pattern models a level variable that can increase or decrease, with changes controlled by binary variables. This is useful for inventory management, capacity adjustments, or gradual state changes.

**Level Evolution:**

$$\label{eq:level_initial}
\ell(0) = \ell_\text{init} + \ell^{\text{inc}}(0) - \ell^{\text{dec}}(0)
$$

$$\label{eq:level_evolution}
\ell(t) = \ell(t-1) + \ell^{\text{inc}}(t) - \ell^{\text{dec}}(t) \quad \forall t > 0
$$

??? note "Variable Definitions"

    - $\ell(t)$ - Level variable
    - $\ell_\text{init}$ - Initial level
    - $\ell^{\text{inc}}(t)$ - Increase in level at time $t$ (non-negative)
    - $\ell^{\text{dec}}(t)$ - Decrease in level at time $t$ (non-negative)

**Change Bounds with Binary Control:**

$$\label{eq:increase_bound}
\ell^{\text{inc}}(t) \leq \Delta \ell^{\text{max}} \cdot b^{\text{inc}}(t) \quad \forall t
$$

$$\label{eq:decrease_bound}
\ell^{\text{dec}}(t) \leq \Delta \ell^{\text{max}} \cdot b^{\text{dec}}(t) \quad \forall t
$$

where $\Delta \ell^{\text{max}}$ is the maximum change per time step, and $b^{\text{inc}}(t), b^{\text{dec}}(t) \in \{0, 1\}$ are binary control variables.

**Mutual Exclusivity of Changes:**

$$\label{eq:change_exclusivity}
b^{\text{inc}}(t) + b^{\text{dec}}(t) \leq 1 \quad \forall t
$$

This ensures level can only increase OR decrease (or stay constant) in each time step.

??? note "Complete Formulation"

    $$
    \begin{align}
    \ell(0) &= \ell_\text{init} + \ell^{\text{inc}}(0) - \ell^{\text{dec}}(0) && \label{eq:level_complete_1} \\
    \ell(t) &= \ell(t-1) + \ell^{\text{inc}}(t) - \ell^{\text{dec}}(t) && \forall t > 0 \label{eq:level_complete_2} \\
    \ell^{\text{inc}}(t) &\leq \Delta \ell^{\text{max}} \cdot b^{\text{inc}}(t) && \forall t \label{eq:level_complete_3} \\
    \ell^{\text{dec}}(t) &\leq \Delta \ell^{\text{max}} \cdot b^{\text{dec}}(t) && \forall t \label{eq:level_complete_4} \\
    b^{\text{inc}}(t) + b^{\text{dec}}(t) &\leq 1 && \forall t \label{eq:level_complete_5} \\
    b^{\text{inc}}(t), b^{\text{dec}}(t) &\in \{0, 1\} && \forall t \label{eq:level_complete_6}
    \end{align}
    $$

[:octicons-code-16: `BoundingPatterns.link_changes_to_level_with_binaries()`][flixopt.modeling.BoundingPatterns.link_changes_to_level_with_binaries]{ .md-button }

---

## Use Cases

=== "Startup/Shutdown Costs"

    Track startup and shutdown events to apply costs:

    ```python
    # Create switch variables
    startup, shutdown = modeling.state_transition_bounds(
        state=on_state,
        previous_state=previous_on_state
    )

    # Apply costs to switches
    startup_cost = startup * startup_cost_per_event
    shutdown_cost = shutdown * shutdown_cost_per_event
    ```

    **Application:** Equipment with startup/shutdown costs, wear from cycling

=== "Limited Switching"

    Restrict the number of state changes:

    ```python
    # Track all switches
    startup, shutdown = modeling.state_transition_bounds(
        state=on_state
    )

    # Limit total switches
    model.add_constraint(
        (startup + shutdown).sum() <= max_switches
    )
    ```

    **Application:** Equipment life management, operational complexity reduction

=== "Gradual Capacity Changes"

    Model systems where capacity can be incrementally adjusted:

    ```python
    # Level represents installed capacity
    level_var, increase, decrease, inc_binary, dec_binary = \
        modeling.link_changes_to_level_with_binaries(
            initial_level=current_capacity,
            max_change=max_capacity_change_per_period
        )

    # Constrain total increases
    model.add_constraint(increase.sum() <= max_total_expansion)
    ```

    **Application:** Staged capacity expansion, inventory management

---

## Used In

These patterns are used in:

- [StatusParameters](../features/StatusParameters.md) - Startup/shutdown tracking and costs
- Operating mode switching with transition costs
- Investment planning with staged capacity additions
- Inventory management with controlled stock changes

---

## See Also

- [StatusParameters](../features/StatusParameters.md) - Practical application of state transitions
- [Duration Tracking](duration-tracking.md) - Complementary pattern for duration constraints
- [Bounds and States](bounds-and-states.md) - Related bounding patterns
