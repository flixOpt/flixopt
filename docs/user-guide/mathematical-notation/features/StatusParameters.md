# StatusParameters

**StatusParameters** model equipment operating in discrete active/inactive states, capturing realistic operational constraints including startup costs, minimum run times, and cycling limitations.

!!! info "Quick Reference"

    | Aspect | Key Equation | Description |
    |--------|--------------|-------------|
    | **Binary State** | $s(t) \in \{0, 1\}$ | Active (1) or inactive (0) |
    | **State Transitions** | $s^{\text{startup}} - s^{\text{shutdown}} = s(t) - s(t-1)$ | Track startups/shutdowns |
    | **Flow Coupling** | $s(t) \cdot P \cdot \text{rel}_\text{lower} \leq p(t) \leq s(t) \cdot P \cdot \text{rel}_\text{upper}$ | Flow forced to zero when inactive |

---

## Mathematical Formulation

### Binary State Variable

Equipment operation is modeled using a binary state variable:

$$\label{eq:status_state}
s(t) \in \{0, 1\} \quad \forall t
$$

- $s(t) = 1$: Equipment operating (active)
- $s(t) = 0$: Equipment shutdown (inactive)

This state variable controls operational constraints and modifies flow bounds using the [bounds with state](../modeling-patterns/bounds-and-states.md#bounds-with-state) pattern.

### State Transitions

State transitions are tracked using switch variables (see [State Transitions](../modeling-patterns/state-transitions.md)):

$$\label{eq:status_transitions}
s^{\text{startup}}(t) - s^{\text{shutdown}}(t) = s(t) - s(t-1) \quad \forall t > 0
$$

$$\label{eq:status_switch_exclusivity}
s^{\text{startup}}(t) + s^{\text{shutdown}}(t) \leq 1 \quad \forall t
$$

??? note "Variable Definitions"

    - $s^{\text{startup}}(t) \in \{0, 1\}$ - Equals 1 when switching from inactive to active
    - $s^{\text{shutdown}}(t) \in \{0, 1\}$ - Equals 1 when switching from active to inactive

    **Behavior:**

    - Inactive → Active: $s^{\text{startup}}(t) = 1$, $s^{\text{shutdown}}(t) = 0$
    - Active → Inactive: $s^{\text{startup}}(t) = 0$, $s^{\text{shutdown}}(t) = 1$
    - No change: $s^{\text{startup}}(t) = 0$, $s^{\text{shutdown}}(t) = 0$

### Effects and Costs

**Startup Effects:**

Effects incurred when equipment starts up:

$$\label{eq:status_switch_effects}
E_{e,\text{switch}} = \sum_{t} s^{\text{startup}}(t) \cdot \text{effect}_{e,\text{switch}}
$$

where $\text{effect}_{e,\text{switch}}$ is the effect value per startup event.

**Examples:** Startup fuel, wear costs, labor, inrush power demands

**Running Effects:**

Effects incurred while equipment is operating:

$$\label{eq:status_running_effects}
E_{e,\text{run}} = \sum_{t} s(t) \cdot \Delta t \cdot \text{effect}_{e,\text{run}}
$$

where $\text{effect}_{e,\text{run}}$ is the effect rate per operating hour.

**Examples:** Fixed O&M costs, auxiliary power, consumables, emissions

### Operating Hour Constraints

??? abstract "Total Operating Hours"

    Bounds on total operating time across planning horizon:

    $$\label{eq:status_total_hours}
    h_\text{min} \leq \sum_{t} s(t) \cdot \Delta t \leq h_\text{max}
    $$

    - $h_\text{min}$, $h_\text{max}$ - Minimum/maximum total operating hours

    **Use cases:** Contract requirements, fuel availability limits, equipment life

??? abstract "Consecutive Operating Hours"

    **Minimum Consecutive Uptime:**

    Enforces minimum runtime once started (see [Duration Tracking](../modeling-patterns/duration-tracking.md)):

    $$\label{eq:status_min_uptime}
    d^{\text{uptime}}(t) \geq (s(t-1) - s(t)) \cdot h^{\text{uptime}}_\text{min} \quad \forall t > 0
    $$

    - $d^{\text{uptime}}(t)$ - Consecutive uptime duration at time $t$
    - $h^{\text{uptime}}_\text{min}$ - Minimum required uptime

    Prevents short cycling and frequent startups.

    **Maximum Consecutive Uptime:**

    Limits continuous operation before requiring shutdown:

    $$\label{eq:status_max_uptime}
    d^{\text{uptime}}(t) \leq h^{\text{uptime}}_\text{max} \quad \forall t
    $$

    **Use cases:** Maintenance intervals, batch time limits, thermal cycling

??? abstract "Consecutive Shutdown Hours"

    **Minimum Consecutive Downtime:**

    Enforces minimum shutdown duration before restarting:

    $$\label{eq:status_min_downtime}
    d^{\text{downtime}}(t) \geq (s(t) - s(t-1)) \cdot h^{\text{downtime}}_\text{min} \quad \forall t > 0
    $$

    - $d^{\text{downtime}}(t)$ - Consecutive downtime duration
    - $h^{\text{downtime}}_\text{min}$ - Minimum required downtime

    **Use cases:** Cooling periods, maintenance, process stabilization

    **Maximum Consecutive Downtime:**

    Limits shutdown duration before mandatory restart:

    $$\label{eq:status_max_downtime}
    d^{\text{downtime}}(t) \leq h^{\text{downtime}}_\text{max} \quad \forall t
    $$

    **Use cases:** Equipment preservation, process stability, contractual activity levels

### Cycling Limits

Maximum number of startups across planning horizon:

$$\label{eq:status_max_switches}
\sum_{t} s^{\text{startup}}(t) \leq n_\text{max}
$$

where $n_\text{max}$ is the maximum allowed number of startups.

**Use cases:** Equipment wear prevention, grid stability, operational complexity, maintenance budget

### Integration with Flow Bounds

StatusParameters modify flow rate bounds by coupling them to the active/inactive state.

**Without StatusParameters** (continuous operation):
$$
P \cdot \text{rel}_\text{lower} \leq p(t) \leq P \cdot \text{rel}_\text{upper}
$$

**With StatusParameters** (binary operation):
$$
s(t) \cdot P \cdot \max(\varepsilon, \text{rel}_\text{lower}) \leq p(t) \leq s(t) \cdot P \cdot \text{rel}_\text{upper}
$$

Using the [bounds with state](../modeling-patterns/bounds-and-states.md#bounds-with-state) pattern.

**Behavior:**

- When $s(t) = 0$: Flow forced to zero
- When $s(t) = 1$: Flow follows normal bounds

??? note "Complete Formulation Summary"

    For equipment with StatusParameters, the complete system includes:

    1. **State variable:** $s(t) \in \{0, 1\}$
    2. **Switch tracking:** $s^{\text{startup}}(t) - s^{\text{shutdown}}(t) = s(t) - s(t-1)$
    3. **Switch exclusivity:** $s^{\text{startup}}(t) + s^{\text{shutdown}}(t) \leq 1$
    4. **Duration tracking:** $d^{\text{uptime}}(t)$, $d^{\text{downtime}}(t)$ following [duration tracking pattern](../modeling-patterns/duration-tracking.md)
    5. **Minimum uptime:** $d^{\text{uptime}}(t) \geq (s(t-1) - s(t)) \cdot h^{\text{uptime}}_\text{min}$
    6. **Maximum uptime:** $d^{\text{uptime}}(t) \leq h^{\text{uptime}}_\text{max}$
    7. **Minimum downtime:** $d^{\text{downtime}}(t) \geq (s(t) - s(t-1)) \cdot h^{\text{downtime}}_\text{min}$
    8. **Maximum downtime:** $d^{\text{downtime}}(t) \leq h^{\text{downtime}}_\text{max}$
    9. **Total hours:** $h_\text{min} \leq \sum_t s(t) \cdot \Delta t \leq h_\text{max}$
    10. **Cycling limit:** $\sum_t s^{\text{startup}}(t) \leq n_\text{max}$
    11. **Flow bounds:** $s(t) \cdot P \cdot \max(\varepsilon, \text{rel}_\text{lower}) \leq p(t) \leq s(t) \cdot P \cdot \text{rel}_\text{upper}$

---

## Implementation

[:octicons-code-24: `StatusParameters`][flixopt.interface.StatusParameters]{ .md-button .md-button--primary }

### Key Parameters

| Parameter | Mathematical Symbol | Description |
|-----------|---------------------|-------------|
| `effects_per_startup` | $\text{effect}_{e,\text{switch}}$ | Costs per startup event |
| `effects_per_active_hour` | $\text{effect}_{e,\text{run}}$ | Costs per hour of operation |
| `active_hours_min`, `active_hours_max` | $h_\text{min}$, $h_\text{max}$ | Total runtime bounds |
| `min_uptime`, `max_uptime` | $h^{\text{uptime}}_\text{min}$, $h^{\text{uptime}}_\text{max}$ | Consecutive runtime bounds |
| `min_downtime`, `max_downtime` | $h^{\text{downtime}}_\text{min}$, $h^{\text{downtime}}_\text{max}$ | Consecutive shutdown bounds |
| `startup_limit` | $n_\text{max}$ | Maximum number of startups |
| `force_startup_tracking` | - | Create switch variables for tracking |

### Mathematical Patterns Used

!!! abstract "Modeling Patterns"

    | Pattern | Application | Reference |
    |---------|-------------|-----------|
    | **[State Transitions](../modeling-patterns/state-transitions.md)** | Switch tracking | Equations $\eqref{eq:status_transitions}$, $\eqref{eq:status_switch_exclusivity}$ |
    | **[Duration Tracking](../modeling-patterns/duration-tracking.md)** | Consecutive time constraints | Uptime/downtime tracking |
    | **[Bounds with State](../modeling-patterns/bounds-and-states.md#bounds-with-state)** | Flow control | Flow coupling to state |

### Used In

- [`Flow`][flixopt.elements.Flow] - Active/inactive operation for flows
- All components supporting discrete operational states

!!! note "Time Series Boundary"

    Final time period constraints for min/max uptime and min/max downtime are not enforced at the end of the planning horizon. This allows optimization to end with ongoing campaigns that may extend beyond the modeled period.

---

## Examples

=== "Power Plant"

    ```python
    power_plant = StatusParameters(
        effects_per_startup={'startup_cost': 25000},  # €25k per startup
        effects_per_active_hour={'fixed_om': 125},    # €125/hour while running
        min_uptime=8,          # Minimum 8-hour run
        min_downtime=4,        # 4-hour cooling period
        active_hours_max=6000, # Annual limit
    )
    ```

    **Use case:** Large thermal power plant with significant startup costs and minimum runtime requirements.

=== "Batch Process"

    ```python
    batch_reactor = StatusParameters(
        effects_per_startup={'setup_cost': 1500},
        min_uptime=12,      # 12-hour minimum batch
        max_uptime=24,      # 24-hour maximum batch
        min_downtime=6,     # Cleaning time
        startup_limit=200,  # Max 200 batches
    )
    ```

    **Use case:** Chemical batch reactor with fixed batch durations and cleaning requirements.

=== "HVAC System"

    ```python
    hvac = StatusParameters(
        effects_per_startup={'compressor_wear': 0.5},
        min_uptime=1,        # Prevent short cycling
        min_downtime=0.5,    # 30-min minimum off
        startup_limit=2000,  # Limit compressor starts
    )
    ```

    **Use case:** HVAC system with compressor wear prevention and short-cycling avoidance.

=== "Backup Generator"

    ```python
    backup_gen = StatusParameters(
        effects_per_startup={'fuel_priming': 50},  # L diesel
        min_uptime=0.5,       # 30-min test duration
        max_downtime=720,     # Test every 30 days
        active_hours_min=26,  # Weekly testing requirement
    )
    ```

    **Use case:** Backup generator with mandatory periodic testing requirements.

---

## See Also

- [Flow](../elements/Flow.md) - Using StatusParameters with flows
- [InvestParameters](InvestParameters.md) - Combining with investment decisions
- [State Transitions](../modeling-patterns/state-transitions.md) - Mathematical pattern details
- [Duration Tracking](../modeling-patterns/duration-tracking.md) - Consecutive duration constraints
- [Bounds with State](../modeling-patterns/bounds-and-states.md#bounds-with-state) - Flow coupling pattern
