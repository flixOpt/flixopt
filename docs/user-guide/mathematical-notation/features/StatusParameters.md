# StatusParameters

[`StatusParameters`][flixopt.interface.StatusParameters] model equipment that operates in discrete active/inactive states rather than continuous operation. This captures realistic operational constraints including startup costs, minimum run times, cycling limitations, and maintenance scheduling.

## Binary State Variable

Equipment operation is modeled using a binary state variable:

$$\label{eq:status_state}
s(t) \in \{0, 1\} \quad \forall t
$$

With:

- $s(t) = 1$: equipment is operating (active state)
- $s(t) = 0$: equipment is shutdown (inactive state)

This state variable controls the equipment's operational constraints and modifies flow bounds using the **bounds with state** pattern from [Bounds and States](../modeling-patterns/bounds-and-states.md#bounds-with-state).

---

## State Transitions and Switching

State transitions are tracked using switch variables (see [State Transitions](../modeling-patterns/state-transitions.md#binary-state-transitions)):

$$\label{eq:status_transitions}
s^\text{startup}(t) - s^\text{shutdown}(t) = s(t) - s(t-1) \quad \forall t > 0
$$

$$\label{eq:status_switch_exclusivity}
s^\text{startup}(t) + s^\text{shutdown}(t) \leq 1 \quad \forall t
$$

With:

- $s^\text{startup}(t) \in \{0, 1\}$: equals 1 when switching from inactive to active (startup)
- $s^\text{shutdown}(t) \in \{0, 1\}$: equals 1 when switching from active to inactive (shutdown)

**Behavior:**
- Inactive → Active: $s^\text{startup}(t) = 1, s^\text{shutdown}(t) = 0$
- Active → Inactive: $s^\text{startup}(t) = 0, s^\text{shutdown}(t) = 1$
- No change: $s^\text{startup}(t) = 0, s^\text{shutdown}(t) = 0$

---

## Effects and Costs

### Startup Effects

Effects incurred when equipment starts up:

$$\label{eq:status_switch_effects}
E_{e,\text{switch}} = \sum_{t} s^\text{startup}(t) \cdot \text{effect}_{e,\text{switch}}
$$

With:

- $\text{effect}_{e,\text{switch}}$ being the effect value per startup event

**Examples:**
- Startup fuel consumption
- Wear and tear costs
- Labor costs for startup procedures
- Inrush power demands

---

### Running Effects

Effects incurred while equipment is operating:

$$\label{eq:status_running_effects}
E_{e,\text{run}} = \sum_{t} s(t) \cdot \Delta t \cdot \text{effect}_{e,\text{run}}
$$

With:

- $\text{effect}_{e,\text{run}}$ being the effect rate per operating hour
- $\Delta t$ being the time step duration

**Examples:**
- Fixed operating and maintenance costs
- Auxiliary power consumption
- Consumable materials
- Emissions while running

---

## Operating Hour Constraints

### Total Operating Hours

Bounds on total operating time across the planning horizon:

$$\label{eq:status_total_hours}
h_\text{min} \leq \sum_{t} s(t) \cdot \Delta t \leq h_\text{max}
$$

With:

- $h_\text{min}$ being the minimum total operating hours
- $h_\text{max}$ being the maximum total operating hours

**Use cases:**
- Minimum runtime requirements (contracts, maintenance)
- Maximum runtime limits (fuel availability, permits, equipment life)

---

### Consecutive Operating Hours

**Minimum Consecutive Uptime:**

Enforces minimum runtime once started using duration tracking (see [Duration Tracking](../modeling-patterns/duration-tracking.md#minimum-duration-constraints)):

$$\label{eq:status_min_uptime}
d^\text{uptime}(t) \geq (s(t-1) - s(t)) \cdot h^\text{uptime}_\text{min} \quad \forall t > 0
$$

With:

- $d^\text{uptime}(t)$ being the consecutive uptime duration at time $t$
- $h^\text{uptime}_\text{min}$ being the minimum required uptime

**Behavior:**
- When shutting down at time $t$: enforces equipment was on for at least $h^\text{uptime}_\text{min}$ prior to the switch
- Prevents short cycling and frequent startups

**Maximum Consecutive Uptime:**

Limits continuous operation before requiring shutdown:

$$\label{eq:status_max_uptime}
d^\text{uptime}(t) \leq h^\text{uptime}_\text{max} \quad \forall t
$$

**Use cases:**
- Mandatory maintenance intervals
- Process batch time limits
- Thermal cycling requirements

---

### Consecutive Shutdown Hours

**Minimum Consecutive Downtime:**

Enforces minimum shutdown duration before restarting:

$$\label{eq:status_min_downtime}
d^\text{downtime}(t) \geq (s(t) - s(t-1)) \cdot h^\text{downtime}_\text{min} \quad \forall t > 0
$$

With:

- $d^\text{downtime}(t)$ being the consecutive downtime duration at time $t$
- $h^\text{downtime}_\text{min}$ being the minimum required downtime

**Use cases:**
- Cooling periods
- Maintenance requirements
- Process stabilization

**Maximum Consecutive Downtime:**

Limits shutdown duration before mandatory restart:

$$\label{eq:status_max_downtime}
d^\text{downtime}(t) \leq h^\text{downtime}_\text{max} \quad \forall t
$$

**Use cases:**
- Equipment preservation requirements
- Process stability needs
- Contractual minimum activity levels

---

## Cycling Limits

Maximum number of startups across the planning horizon:

$$\label{eq:status_max_switches}
\sum_{t} s^\text{startup}(t) \leq n_\text{max}
$$

With:

- $n_\text{max}$ being the maximum allowed number of startups

**Use cases:**
- Preventing excessive equipment wear
- Grid stability requirements
- Operational complexity limits
- Maintenance budget constraints

---

## Integration with Flow Bounds

StatusParameters modify flow rate bounds by coupling them to the active/inactive state.

**Without StatusParameters** (continuous operation):
$$
P \cdot \text{rel}_\text{lower} \leq p(t) \leq P \cdot \text{rel}_\text{upper}
$$

**With StatusParameters** (binary operation):
$$
s(t) \cdot P \cdot \max(\varepsilon, \text{rel}_\text{lower}) \leq p(t) \leq s(t) \cdot P \cdot \text{rel}_\text{upper}
$$

Using the **bounds with state** pattern from [Bounds and States](../modeling-patterns/bounds-and-states.md#bounds-with-state).

**Behavior:**
- When $s(t) = 0$: flow is forced to zero
- When $s(t) = 1$: flow follows normal bounds

---

## Complete Formulation Summary

For equipment with StatusParameters, the complete constraint system includes:

1. **State variable:** $s(t) \in \{0, 1\}$
2. **Switch tracking:** $s^\text{startup}(t) - s^\text{shutdown}(t) = s(t) - s(t-1)$
3. **Switch exclusivity:** $s^\text{startup}(t) + s^\text{shutdown}(t) \leq 1$
4. **Duration tracking:**

    - On-duration: $d^\text{uptime}(t)$ following duration tracking pattern
    - Off-duration: $d^\text{downtime}(t)$ following duration tracking pattern
5. **Minimum uptime:** $d^\text{uptime}(t) \geq (s(t-1) - s(t)) \cdot h^\text{uptime}_\text{min}$
6. **Maximum uptime:** $d^\text{uptime}(t) \leq h^\text{uptime}_\text{max}$
7. **Minimum downtime:** $d^\text{downtime}(t) \geq (s(t) - s(t-1)) \cdot h^\text{downtime}_\text{min}$
8. **Maximum downtime:** $d^\text{downtime}(t) \leq h^\text{downtime}_\text{max}$
9. **Total hours:** $h_\text{min} \leq \sum_t s(t) \cdot \Delta t \leq h_\text{max}$
10. **Cycling limit:** $\sum_t s^\text{startup}(t) \leq n_\text{max}$
11. **Flow bounds:** $s(t) \cdot P \cdot \max(\varepsilon, \text{rel}_\text{lower}) \leq p(t) \leq s(t) \cdot P \cdot \text{rel}_\text{upper}$

---

## Implementation

**Python Class:** [`StatusParameters`][flixopt.interface.StatusParameters]

**Key Parameters:**

- `effects_per_startup`: Costs per startup event
- `effects_per_active_hour`: Costs per hour of operation
- `active_hours_min`, `active_hours_max`: Total runtime bounds
- `min_uptime`, `max_uptime`: Consecutive runtime bounds
- `min_downtime`, `max_downtime`: Consecutive shutdown bounds
- `startup_limit`: Maximum number of startups
- `force_startup_tracking`: Create switch variables even without limits (for tracking)

See the [`StatusParameters`][flixopt.interface.StatusParameters] API documentation for complete parameter list and usage examples.

**Mathematical Patterns Used:**
- [State Transitions](../modeling-patterns/state-transitions.md#binary-state-transitions) - Switch tracking
- [Duration Tracking](../modeling-patterns/duration-tracking.md) - Consecutive time constraints
- [Bounds with State](../modeling-patterns/bounds-and-states.md#bounds-with-state) - Flow control

**Used in:**
- [`Flow`][flixopt.elements.Flow] - Active/inactive operation for flows
- All components supporting discrete operational states

---

## Examples

### Power Plant with Startup Costs
```python
power_plant = StatusParameters(
    effects_per_startup={'startup_cost': 25000},  # €25k per startup
    effects_per_active_hour={'fixed_om': 125},  # €125/hour while running
    min_uptime=8,  # Minimum 8-hour run
    min_downtime=4,  # 4-hour cooling period
    active_hours_max=6000,  # Annual limit
)
```

### Batch Process with Cycling Limits
```python
batch_reactor = StatusParameters(
    effects_per_startup={'setup_cost': 1500},
    min_uptime=12,  # 12-hour minimum batch
    max_uptime=24,  # 24-hour maximum batch
    min_downtime=6,  # Cleaning time
    startup_limit=200,  # Max 200 batches
)
```

### HVAC with Cycle Prevention
```python
hvac = StatusParameters(
    effects_per_startup={'compressor_wear': 0.5},
    min_uptime=1,  # Prevent short cycling
    min_downtime=0.5,  # 30-min minimum off
    startup_limit=2000,  # Limit compressor starts
)
```

### Backup Generator with Testing Requirements
```python
backup_gen = StatusParameters(
    effects_per_startup={'fuel_priming': 50},  # L diesel
    min_uptime=0.5,  # 30-min test duration
    max_downtime=720,  # Test every 30 days
    active_hours_min=26,  # Weekly testing requirement
)
```

---

## Notes

**Time Series Boundary:** The final time period constraints for min_uptime/max and min_downtime/max are not enforced at the end of the planning horizon. This allows optimization to end with ongoing campaigns that may be shorter/longer than specified, as they extend beyond the modeled period.
