# OnOffParameters

[`OnOffParameters`][flixopt.interface.OnOffParameters] model equipment that operates in discrete on/off states rather than continuous operation. This captures realistic operational constraints including startup costs, minimum run times, cycling limitations, and maintenance scheduling.

## Binary State Variable

Equipment operation is modeled using a binary state variable:

$$\label{eq:onoff_state}
s(t) \in \{0, 1\} \quad \forall t
$$

With:
- $s(t) = 1$: equipment is operating (on state)
- $s(t) = 0$: equipment is shutdown (off state)

This state variable controls the equipment's operational constraints and modifies flow bounds using the **bounds with state** pattern from [Bounds and States](../modeling-patterns/bounds-and-states.md#bounds-with-state).

---

## State Transitions and Switching

State transitions are tracked using switch variables (see [State Transitions](../modeling-patterns/state-transitions.md#binary-state-transitions)):

$$\label{eq:onoff_transitions}
s^\text{on}(t) - s^\text{off}(t) = s(t) - s(t-1) \quad \forall t > 0
$$

$$\label{eq:onoff_switch_exclusivity}
s^\text{on}(t) + s^\text{off}(t) \leq 1 \quad \forall t
$$

With:
- $s^\text{on}(t) \in \{0, 1\}$: equals 1 when switching from off to on (startup)
- $s^\text{off}(t) \in \{0, 1\}$: equals 1 when switching from on to off (shutdown)

**Behavior:**
- Off → On: $s^\text{on}(t) = 1, s^\text{off}(t) = 0$
- On → Off: $s^\text{on}(t) = 0, s^\text{off}(t) = 1$
- No change: $s^\text{on}(t) = 0, s^\text{off}(t) = 0$

---

## Effects and Costs

### Switching Effects

Effects incurred when equipment starts up:

$$\label{eq:onoff_switch_effects}
E_{e,\text{switch}} = \sum_{t} s^\text{on}(t) \cdot \text{effect}_{e,\text{switch}}
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

$$\label{eq:onoff_running_effects}
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

$$\label{eq:onoff_total_hours}
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

**Minimum Consecutive On-Time:**

Enforces minimum runtime once started using duration tracking (see [Duration Tracking](../modeling-patterns/duration-tracking.md#minimum-duration-constraints)):

$$\label{eq:onoff_min_on_duration}
d^\text{on}(t) \geq (s(t-1) - s(t)) \cdot h^\text{on}_\text{min} \quad \forall t > 0
$$

With:
- $d^\text{on}(t)$ being the consecutive on-time duration at time $t$
- $h^\text{on}_\text{min}$ being the minimum required on-time

**Behavior:**
- When shutting down at time $t$: enforces equipment was on for at least $h^\text{on}_\text{min}$ prior to the switch
- Prevents short cycling and frequent startups

**Maximum Consecutive On-Time:**

Limits continuous operation before requiring shutdown:

$$\label{eq:onoff_max_on_duration}
d^\text{on}(t) \leq h^\text{on}_\text{max} \quad \forall t
$$

**Use cases:**
- Mandatory maintenance intervals
- Process batch time limits
- Thermal cycling requirements

---

### Consecutive Shutdown Hours

**Minimum Consecutive Off-Time:**

Enforces minimum shutdown duration before restarting:

$$\label{eq:onoff_min_off_duration}
d^\text{off}(t) \geq (s(t) - s(t-1)) \cdot h^\text{off}_\text{min} \quad \forall t > 0
$$

With:
- $d^\text{off}(t)$ being the consecutive off-time duration at time $t$
- $h^\text{off}_\text{min}$ being the minimum required off-time

**Use cases:**
- Cooling periods
- Maintenance requirements
- Process stabilization

**Maximum Consecutive Off-Time:**

Limits shutdown duration before mandatory restart:

$$\label{eq:onoff_max_off_duration}
d^\text{off}(t) \leq h^\text{off}_\text{max} \quad \forall t
$$

**Use cases:**
- Equipment preservation requirements
- Process stability needs
- Contractual minimum activity levels

---

## Cycling Limits

Maximum number of startups across the planning horizon:

$$\label{eq:onoff_max_switches}
\sum_{t} s^\text{on}(t) \leq n_\text{max}
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

OnOffParameters modify flow rate bounds by coupling them to the on/off state.

**Without OnOffParameters** (continuous operation):
$$
P \cdot \text{rel}_\text{lower} \leq p(t) \leq P \cdot \text{rel}_\text{upper}
$$

**With OnOffParameters** (binary operation):
$$
s(t) \cdot P \cdot \max(\varepsilon, \text{rel}_\text{lower}) \leq p(t) \leq s(t) \cdot P \cdot \text{rel}_\text{upper}
$$

Using the **bounds with state** pattern from [Bounds and States](../modeling-patterns/bounds-and-states.md#bounds-with-state).

**Behavior:**
- When $s(t) = 0$: flow is forced to zero
- When $s(t) = 1$: flow follows normal bounds

---

## Complete Formulation Summary

For equipment with OnOffParameters, the complete constraint system includes:

1. **State variable:** $s(t) \in \{0, 1\}$
2. **Switch tracking:** $s^\text{on}(t) - s^\text{off}(t) = s(t) - s(t-1)$
3. **Switch exclusivity:** $s^\text{on}(t) + s^\text{off}(t) \leq 1$
4. **Duration tracking:**
    - On-duration: $d^\text{on}(t)$ following duration tracking pattern
    - Off-duration: $d^\text{off}(t)$ following duration tracking pattern
5. **Minimum on-time:** $d^\text{on}(t) \geq (s(t-1) - s(t)) \cdot h^\text{on}_\text{min}$
6. **Maximum on-time:** $d^\text{on}(t) \leq h^\text{on}_\text{max}$
7. **Minimum off-time:** $d^\text{off}(t) \geq (s(t) - s(t-1)) \cdot h^\text{off}_\text{min}$
8. **Maximum off-time:** $d^\text{off}(t) \leq h^\text{off}_\text{max}$
9. **Total hours:** $h_\text{min} \leq \sum_t s(t) \cdot \Delta t \leq h_\text{max}$
10. **Cycling limit:** $\sum_t s^\text{on}(t) \leq n_\text{max}$
11. **Flow bounds:** $s(t) \cdot P \cdot \text{rel}_\text{lower} \leq p(t) \leq s(t) \cdot P \cdot \text{rel}_\text{upper}$

---

## Implementation

**Python Class:** [`OnOffParameters`][flixopt.interface.OnOffParameters]

**Key Parameters:**
- `effects_per_switch_on`: Costs per startup event
- `effects_per_running_hour`: Costs per hour of operation
- `on_hours_min`, `on_hours_max`: Total runtime bounds
- `consecutive_on_hours_min`, `consecutive_on_hours_max`: Consecutive runtime bounds
- `consecutive_off_hours_min`, `consecutive_off_hours_max`: Consecutive shutdown bounds
- `switch_on_max`: Maximum number of startups
- `force_switch_on`: Create switch variables even without limits (for tracking)

See the [`OnOffParameters`][flixopt.interface.OnOffParameters] API documentation for complete parameter list and usage examples.

**Mathematical Patterns Used:**
- [State Transitions](../modeling-patterns/state-transitions.md#binary-state-transitions) - Switch tracking
- [Duration Tracking](../modeling-patterns/duration-tracking.md) - Consecutive time constraints
- [Bounds with State](../modeling-patterns/bounds-and-states.md#bounds-with-state) - Flow control

**Used in:**
- [`Flow`][flixopt.elements.Flow] - On/off operation for flows
- All components supporting discrete operational states

---

## Examples

### Power Plant with Startup Costs
```python
power_plant = OnOffParameters(
    effects_per_switch_on={'startup_cost': 25000},  # €25k per startup
    effects_per_running_hour={'fixed_om': 125},  # €125/hour while running
    consecutive_on_hours_min=8,  # Minimum 8-hour run
    consecutive_off_hours_min=4,  # 4-hour cooling period
    on_hours_max=6000,  # Annual limit
)
```

### Batch Process with Cycling Limits
```python
batch_reactor = OnOffParameters(
    effects_per_switch_on={'setup_cost': 1500},
    consecutive_on_hours_min=12,  # 12-hour minimum batch
    consecutive_on_hours_max=24,  # 24-hour maximum batch
    consecutive_off_hours_min=6,  # Cleaning time
    switch_on_max=200,  # Max 200 batches
)
```

### HVAC with Cycle Prevention
```python
hvac = OnOffParameters(
    effects_per_switch_on={'compressor_wear': 0.5},
    consecutive_on_hours_min=1,  # Prevent short cycling
    consecutive_off_hours_min=0.5,  # 30-min minimum off
    switch_on_max=2000,  # Limit compressor starts
)
```

### Backup Generator with Testing Requirements
```python
backup_gen = OnOffParameters(
    effects_per_switch_on={'fuel_priming': 50},  # L diesel
    consecutive_on_hours_min=0.5,  # 30-min test duration
    consecutive_off_hours_max=720,  # Test every 30 days
    on_hours_min=26,  # Weekly testing requirement
)
```

---

## Notes

**Time Series Boundary:** The final time period constraints for consecutive_on_hours_min/max and consecutive_off_hours_min/max are not enforced at the end of the planning horizon. This allows optimization to end with ongoing campaigns that may be shorter/longer than specified, as they extend beyond the modeled period.
