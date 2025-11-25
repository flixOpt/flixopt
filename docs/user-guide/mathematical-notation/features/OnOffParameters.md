# OnOffParameters

OnOffParameters add binary on/off behavior — equipment that can be completely off or operating within its capacity range.

!!! example "Real-world examples"
    - **Power plant** — Startup costs, minimum run time
    - **Batch reactor** — Must run complete cycles
    - **Chiller** — Maximum operating hours per year

## Core Concept: Binary State

The on/off state $s(t) \in \{0, 1\}$ determines whether equipment operates:

- $s(t) = 0$: Off — flow rate must be zero
- $s(t) = 1$: On — flow rate within capacity bounds

This modifies the flow bounds (see [Flow](../elements/Flow.md)):

$$
s(t) \cdot P \cdot p_{rel}^{min} \leq p(t) \leq s(t) \cdot P \cdot p_{rel}^{max}
$$

## State Transitions

=== "Switch Detection"

    Track when equipment turns on/off:

    $$
    s^{on}(t) - s^{off}(t) = s(t) - s(t-1)
    $$

    Where:

    - $s^{on}(t) = 1$ when switching on (was off, now on)
    - $s^{off}(t) = 1$ when switching off (was on, now off)

=== "Startup Costs"

    Effects incurred each time equipment starts:

    $$
    E_{e,switch} = \sum_t s^{on}(t) \cdot c_{switch}
    $$

=== "Running Costs"

    Effects while equipment operates:

    $$
    E_{e,run} = \sum_t s(t) \cdot \Delta t \cdot c_{run}
    $$

## Duration Constraints

=== "Min Run Time"

    Once started, must run for minimum duration:

    $$
    \text{If } s^{on}(t) = 1 \Rightarrow \sum_{j=t}^{t+k} s(j) \cdot \Delta t \geq T_{on}^{min}
    $$

=== "Min Off Time"

    Once stopped, must stay off for minimum duration:

    $$
    \text{If } s^{off}(t) = 1 \Rightarrow \sum_{j=t}^{t+k} (1-s(j)) \cdot \Delta t \geq T_{off}^{min}
    $$

=== "Max Run Time"

    Cannot run continuously beyond limit:

    $$
    \sum_{j=t}^{t+k} s(j) \cdot \Delta t \leq T_{on}^{max}
    $$

=== "Total Hours"

    Constrain total operating hours per period:

    $$
    H^{min} \leq \sum_t s(t) \cdot \Delta t \leq H^{max}
    $$

=== "Max Startups"

    Limit number of startups per period:

    $$
    \sum_t s^{on}(t) \leq N_{switch}^{max}
    $$

## Variables

| Symbol | Python Name | Description | When Created |
|--------|-------------|-------------|--------------|
| $s(t)$ | `on` | Binary on/off state | Always |
| $s^{on}(t)$ | `switch_on` | Startup indicator | Switch effects or constraints |
| $s^{off}(t)$ | `switch_off` | Shutdown indicator | Switch effects or constraints |

## Parameters

| Symbol | Python Name | Description | Default |
|--------|-------------|-------------|---------|
| $c_{switch}$ | `effects_per_switch_on` | Startup effects | None |
| $c_{run}$ | `effects_per_running_hour` | Running effects | None |
| $T_{on}^{min}$ | `consecutive_on_hours_min` | Min run time | None |
| $T_{on}^{max}$ | `consecutive_on_hours_max` | Max run time | None |
| $T_{off}^{min}$ | `consecutive_off_hours_min` | Min off time | None |
| $T_{off}^{max}$ | `consecutive_off_hours_max` | Max off time | None |
| $H^{min}$ | `on_hours_min` | Min total hours | None |
| $H^{max}$ | `on_hours_max` | Max total hours | None |
| $N_{switch}^{max}$ | `switch_on_max` | Max startups | None |

## Usage Examples

### Power Plant with Startup Costs

```python
generator = fx.Flow(
    label='power',
    bus=elec_bus,
    size=100,
    relative_minimum=0.4,  # 40% min when ON
    on_off_parameters=fx.OnOffParameters(
        effects_per_switch_on={'costs': 25000},  # €25k startup
        consecutive_on_hours_min=8,   # Must run 8+ hours
        consecutive_off_hours_min=4,  # Must stay off 4+ hours
    ),
)
```

### Batch Process with Cycle Limits

```python
reactor = fx.Flow(
    label='output',
    bus=prod_bus,
    size=50,
    on_off_parameters=fx.OnOffParameters(
        effects_per_switch_on={'costs': 1500},
        effects_per_running_hour={'costs': 200},
        consecutive_on_hours_min=12,  # 12h batch
        switch_on_max=20,  # Max 20 batches
    ),
)
```

### HVAC with Operating Limits

```python
chiller = fx.Flow(
    label='cooling',
    bus=cool_bus,
    size=500,
    on_off_parameters=fx.OnOffParameters(
        on_hours_min=2000,  # Min 2000h/year
        on_hours_max=5000,  # Max 5000h/year
        consecutive_on_hours_max=18,  # Max 18h continuous
    ),
)
```

## Implementation Details

- **Feature Class:** [`OnOffParameters`][flixopt.interface.OnOffParameters]
- **Model Class:** [`OnOffModel`][flixopt.features.OnOffModel]

## See Also

- [Flow](../elements/Flow.md) — How on/off affects flow bounds
- [InvestParameters](InvestParameters.md) — Combining with investment
- [Effects & Objective](../effects-penalty-objective.md) — How costs are tracked
