# StatusParameters

StatusParameters define a binary status variable with operational constraints and effects.

!!! example "Real-world examples"
    - **Power plant** — Startup costs, minimum run time
    - **Batch reactor** — Must run complete cycles
    - **Chiller** — Maximum operating hours per year

## Core Concept: Binary Status Variable

StatusParameters create a binary status variable $s(t) \in \{0, 1\}$:

- $s(t) = 0$: Inactive
- $s(t) = 1$: Active

!!! note "Connection to continuous variables"
    How this binary status connects to continuous variables (like flow rate) is defined where `status_parameters` is used. See [Flow](../elements/Flow.md) for how flows use the status to modify their bounds.

## State Transitions

=== "Startup Detection"

    Track when status changes from inactive to active:

    $$
    s^{start}(t) - s^{stop}(t) = s(t) - s(t-1)
    $$

    Where:

    - $s^{start}(t) = 1$ when starting up (0 → 1)
    - $s^{stop}(t) = 1$ when shutting down (1 → 0)

=== "Startup Effects"

    Effects incurred each time equipment starts:

    $$
    E_{startup} = \sum_t s^{start}(t) \cdot c_{startup}
    $$

=== "Active Hour Effects"

    Effects while status is active:

    $$
    E_{active} = \sum_t s(t) \cdot \Delta t \cdot c_{active}
    $$

## Duration Constraints

=== "Min Uptime"

    Once active, must stay active for minimum duration:

    $$
    s^{start}(t) = 1 \Rightarrow \sum_{j=t}^{t+k} s(j) \cdot \Delta t \geq T_{up}^{min}
    $$

=== "Min Downtime"

    Once inactive, must stay inactive for minimum duration:

    $$
    s^{stop}(t) = 1 \Rightarrow \sum_{j=t}^{t+k} (1-s(j)) \cdot \Delta t \geq T_{down}^{min}
    $$

=== "Max Uptime"

    Cannot stay active continuously beyond limit:

    $$
    \sum_{j=t}^{t+k} s(j) \cdot \Delta t \leq T_{up}^{max}
    $$

=== "Active Hours"

    Constrain total active hours per period:

    $$
    H^{min} \leq \sum_t s(t) \cdot \Delta t \leq H^{max}
    $$

=== "Startup Limit"

    Limit number of startups per period:

    $$
    \sum_t s^{start}(t) \leq N_{start}^{max}
    $$

## Variables

| Symbol | Python Name | Description | When Created |
|--------|-------------|-------------|--------------|
| $s(t)$ | `status` | Binary status | Always |
| $s^{start}(t)$ | `startup` | Startup indicator | Startup effects or constraints |
| $s^{stop}(t)$ | `shutdown` | Shutdown indicator | Startup effects or constraints |

## Parameters

| Symbol | Python Name | Description | Default |
|--------|-------------|-------------|---------|
| $c_{startup}$ | `effects_per_startup` | Effects per startup | None |
| $c_{active}$ | `effects_per_active_hour` | Effects while active | None |
| $T_{up}^{min}$ | `min_uptime` | Min consecutive uptime | None |
| $T_{up}^{max}$ | `max_uptime` | Max consecutive uptime | None |
| $T_{down}^{min}$ | `min_downtime` | Min consecutive downtime | None |
| $T_{down}^{max}$ | `max_downtime` | Max consecutive downtime | None |
| $H^{min}$ | `active_hours_min` | Min total active hours | None |
| $H^{max}$ | `active_hours_max` | Max total active hours | None |
| $N_{start}^{max}$ | `startup_limit` | Max startups | None |

## Usage Examples

### Power Plant with Startup Costs

```python
generator = fx.Flow(
    label='power',
    bus=elec_bus,
    size=100,
    relative_minimum=0.4,  # 40% min when active
    status_parameters=fx.StatusParameters(
        effects_per_startup={'costs': 25000},  # €25k startup
        min_uptime=8,   # Must run 8+ hours
        min_downtime=4,  # Must stay off 4+ hours
    ),
)
```

### Batch Process with Cycle Limits

```python
reactor = fx.Flow(
    label='output',
    bus=prod_bus,
    size=50,
    status_parameters=fx.StatusParameters(
        effects_per_startup={'costs': 1500},
        effects_per_active_hour={'costs': 200},
        min_uptime=12,  # 12h batch
        startup_limit=20,  # Max 20 batches
    ),
)
```

### HVAC with Operating Limits

```python
chiller = fx.Flow(
    label='cooling',
    bus=cool_bus,
    size=500,
    status_parameters=fx.StatusParameters(
        active_hours_min=2000,  # Min 2000h/year
        active_hours_max=5000,  # Max 5000h/year
        max_uptime=18,  # Max 18h continuous
    ),
)
```

## Implementation Details

- **Feature Class:** [`StatusParameters`][flixopt.interface.StatusParameters]
- **Model Class:** [`StatusModel`][flixopt.features.StatusModel]

## See Also

- [Flow](../elements/Flow.md) — How flows use status
- [InvestParameters](InvestParameters.md) — Combining with investment
- [Effects & Objective](../effects-penalty-objective.md) — How effects are tracked
