# Duration Tracking

Duration tracking allows monitoring how long a binary state has been consecutively active. This is essential for modeling minimum run times, ramp-up periods, and similar time-dependent constraints.

## Consecutive Duration Tracking

For a binary state variable $s(t) \in \{0, 1\}$, the consecutive duration $d(t)$ tracks how long the state has been continuously active.

### Duration Upper Bound

The duration cannot exceed zero when the state is inactive:

$$\label{eq:duration_upper}
d(t) \leq s(t) \cdot M \quad \forall t
$$

With:

- $d(t)$ being the duration variable (continuous, non-negative)
- $s(t) \in \{0, 1\}$ being the binary state variable
- $M$ being a sufficiently large constant (big-M)

**Behavior:**
- When $s(t) = 0$: forces $d(t) \leq 0$, thus $d(t) = 0$
- When $s(t) = 1$: allows $d(t)$ to be positive

---

### Duration Accumulation

While the state is active, the duration increases by the time step size:

$$\label{eq:duration_accumulation_upper}
d(t+1) \leq d(t) + \Delta d(t) \quad \forall t
$$

$$\label{eq:duration_accumulation_lower}
d(t+1) \geq d(t) + \Delta d(t) + (s(t+1) - 1) \cdot M \quad \forall t
$$

With:

- $\Delta d(t)$ being the duration increment for time step $t$ (typically $\Delta t_i$ from the time series)
- $M$ being a sufficiently large constant

**Behavior:**
- When $s(t+1) = 1$: both inequalities enforce $d(t+1) = d(t) + \Delta d(t)$
- When $s(t+1) = 0$: only the upper bound applies, and $d(t+1) = 0$ (from equation $\eqref{eq:duration_upper}$)

---

### Initial Duration

The duration at the first time step depends on both the state and any previous duration:

$$\label{eq:duration_initial}
d(0) = (\Delta d(0) + d_\text{prev}) \cdot s(0)
$$

With:

- $d_\text{prev}$ being the duration from before the optimization period
- $\Delta d(0)$ being the duration increment for the first time step

**Behavior:**
- When $s(0) = 1$: duration continues from previous period
- When $s(0) = 0$: duration resets to zero

---

### Complete Formulation

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

## Minimum Duration Constraints

To enforce a minimum consecutive duration (e.g., minimum run time), an additional constraint links the duration to state changes:

$$\label{eq:minimum_duration}
d(t) \geq (s(t-1) - s(t)) \cdot d_\text{min}(t-1) \quad \forall t > 0
$$

With:

- $d_\text{min}(t)$ being the required minimum duration at time $t$

**Behavior:**
- When shutting down ($s(t-1) = 1, s(t) = 0$): enforces $d(t-1) \geq d_\text{min}(t-1)$
- This ensures the state was active for at least $d_\text{min}$ before turning off
- When state is constant or turning on: constraint is non-binding

---

## Implementation

**Function:** [`ModelingPrimitives.consecutive_duration_tracking()`][flixopt.modeling.ModelingPrimitives.consecutive_duration_tracking]

See the API documentation for complete parameter list and usage details.

---

## Use Cases

### Minimum Run Time

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

### Ramp-Up Tracking

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

### Cooldown Requirements

Tracking time in a state before allowing transitions:

```python
# Track maintenance duration
maintenance_duration = modeling.consecutive_duration_tracking(
    state=maintenance_state,
    duration_per_step=time_step_hours,
    minimum_duration=scheduled_maintenance_hours
)
```

---

## Used In

This pattern is used in:

- [`StatusParameters`](../features/StatusParameters.md) - Minimum active/inactive times
- Operating mode constraints with minimum durations
- Startup/shutdown sequence modeling
