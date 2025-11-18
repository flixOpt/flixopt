# Duration Tracking

Duration tracking allows monitoring how long a binary state has been consecutively active. This is essential for modeling minimum run times, ramp-up periods, and similar time-dependent constraints.

!!! info "Quick Reference"

    | Aspect | Key Equation | Purpose |
    |--------|--------------|---------|
    | **Upper Bound** | $d(t) \leq s(t) \cdot M$ | Force zero when inactive |
    | **Accumulation** | $d(t+1) = d(t) + \Delta d(t)$ when active | Increment duration |
    | **Minimum Duration** | $d(t) \geq (s(t-1) - s(t)) \cdot d_\text{min}$ | Enforce min runtime |

---

## Mathematical Formulation

### Consecutive Duration Tracking

For a binary state variable $s(t) \in \{0, 1\}$, the consecutive duration $d(t)$ tracks how long the state has been continuously active.

### Duration Upper Bound

The duration cannot exceed zero when the state is inactive:

$$\label{eq:duration_upper}
d(t) \leq s(t) \cdot M \quad \forall t
$$

??? note "Variable Definitions"

    - $d(t)$ - Duration variable (continuous, non-negative)
    - $s(t) \in \{0, 1\}$ - Binary state variable
    - $M$ - Sufficiently large constant (big-M)

**Behavior:**

- When $s(t) = 0$: Forces $d(t) \leq 0$, thus $d(t) = 0$
- When $s(t) = 1$: Allows $d(t)$ to be positive

### Duration Accumulation

While the state is active, the duration increases by the time step size:

$$\label{eq:duration_accumulation_upper}
d(t+1) \leq d(t) + \Delta d(t) \quad \forall t
$$

$$\label{eq:duration_accumulation_lower}
d(t+1) \geq d(t) + \Delta d(t) + (s(t+1) - 1) \cdot M \quad \forall t
$$

??? note "Variable Definitions"

    - $\Delta d(t)$ - Duration increment for time step $t$ (typically $\Delta t_i$ from the time series)
    - $M$ - Sufficiently large constant

**Behavior:**

- When $s(t+1) = 1$: Both inequalities enforce $d(t+1) = d(t) + \Delta d(t)$
- When $s(t+1) = 0$: Only upper bound applies, and $d(t+1) = 0$ (from equation $\eqref{eq:duration_upper}$)

### Initial Duration

The duration at the first time step depends on both the state and any previous duration:

$$\label{eq:duration_initial}
d(0) = (\Delta d(0) + d_\text{prev}) \cdot s(0)
$$

??? note "Variable Definitions"

    - $d_\text{prev}$ - Duration from before the optimization period
    - $\Delta d(0)$ - Duration increment for the first time step

**Behavior:**

- When $s(0) = 1$: Duration continues from previous period
- When $s(0) = 0$: Duration resets to zero

### Complete Formulation

??? note "Combined Constraints"

    Combining all constraints:

    $$
    \begin{align}
    d(t) &\leq s(t) \cdot M && \forall t \label{eq:duration_complete_1} \\
    d(t+1) &\leq d(t) + \Delta d(t) && \forall t \label{eq:duration_complete_2} \\
    d(t+1) &\geq d(t) + \Delta d(t) + (s(t+1) - 1) \cdot M && \forall t \label{eq:duration_complete_3} \\
    d(0) &= (\Delta d(0) + d_\text{prev}) \cdot s(0) && \label{eq:duration_complete_4}
    \end{align}
    $$

---

### Minimum Duration Constraints

To enforce a minimum consecutive duration (e.g., minimum run time), an additional constraint links the duration to state changes:

$$\label{eq:minimum_duration}
d(t) \geq (s(t-1) - s(t)) \cdot d_\text{min}(t-1) \quad \forall t > 0
$$

where $d_\text{min}(t)$ is the required minimum duration at time $t$.

**Behavior:**

- When shutting down ($s(t-1) = 1, s(t) = 0$): Enforces $d(t-1) \geq d_\text{min}(t-1)$
- This ensures the state was active for at least $d_\text{min}$ before turning off
- When state is constant or turning on: Constraint is non-binding

---

## Implementation

[:octicons-code-24: `ModelingPrimitives.consecutive_duration_tracking()`][flixopt.modeling.ModelingPrimitives.consecutive_duration_tracking]{ .md-button .md-button--primary }

---

## Use Cases

=== "Minimum Run Time"

    Ensuring equipment runs for a minimum duration once started:

    ```python
    # State: 1 when running, 0 when off
    # Require at least 2 hours of operation
    duration = modeling.consecutive_duration_tracking(
        state=on_state,
        duration_per_step=time_step_hours,
        minimum_duration=2.0
    )
    ```

    **Application:** Power plants, manufacturing processes, HVAC systems

=== "Ramp-Up Tracking"

    Tracking time since startup for gradual ramp-up constraints:

    ```python
    # Track startup duration
    startup_duration = modeling.consecutive_duration_tracking(
        state=on_state,
        duration_per_step=time_step_hours
    )
    # Constrain output based on startup duration
    # (additional constraints would link output to startup_duration)
    ```

    **Application:** Thermal power plants, industrial furnaces

=== "Cooldown Requirements"

    Tracking time in a state before allowing transitions:

    ```python
    # Track maintenance duration
    maintenance_duration = modeling.consecutive_duration_tracking(
        state=maintenance_state,
        duration_per_step=time_step_hours,
        minimum_duration=scheduled_maintenance_hours
    )
    ```

    **Application:** Maintenance windows, cooldown periods, safety interlocks

---

## Used In

This pattern is used in:

- [StatusParameters](../features/StatusParameters.md) - Minimum active/inactive times
- Operating mode constraints with minimum durations
- Startup/shutdown sequence modeling

---

## See Also

- [StatusParameters](../features/StatusParameters.md) - Practical application of duration tracking
- [State Transitions](state-transitions.md) - Related pattern for tracking state changes
- [Bounds and States](bounds-and-states.md) - Complementary bounding patterns
