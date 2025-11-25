# OnOffParameters

OnOffParameters define a binary on/off state variable with associated constraints and effects.

!!! example "Real-world examples"
    - **Power plant** — Startup costs, minimum run time
    - **Batch reactor** — Must run complete cycles
    - **Chiller** — Maximum operating hours per year

## Core Concept: Binary State Variable

OnOffParameters create a binary state variable $s(t) \in \{0, 1\}$:

- $s(t) = 0$: Off
- $s(t) = 1$: On

!!! note "Connection to continuous variables"
    How this binary state connects to continuous variables (like flow rate) is defined where `on_off_parameters` is used. See [Flow](../elements/Flow.md) for how flows use the on/off state to modify their bounds.

## State Transitions

=== "Switch Detection"

    Track when state changes:

    $$
    s^{on}(t) - s^{off}(t) = s(t) - s(t-1)
    $$

    Where:

    - $s^{on}(t) = 1$ when switching on (0 → 1)
    - $s^{off}(t) = 1$ when switching off (1 → 0)

=== "Startup Effects"

    Effects incurred each time state switches on:

    $$
    E_{e,switch} = \sum_t s^{on}(t) \cdot c_{switch}
    $$

=== "Running Effects"

    Effects while state is on:

    $$
    E_{e,run} = \sum_t s(t) \cdot \Delta t \cdot c_{run}
    $$

## Duration Constraints

=== "Min On Time"

    Once on, must stay on for minimum duration:

    $$
    s^{on}(t) = 1 \Rightarrow \sum_{j=t}^{t+k} s(j) \cdot \Delta t \geq T_{on}^{min}
    $$

=== "Min Off Time"

    Once off, must stay off for minimum duration:

    $$
    s^{off}(t) = 1 \Rightarrow \sum_{j=t}^{t+k} (1-s(j)) \cdot \Delta t \geq T_{off}^{min}
    $$

=== "Max On Time"

    Cannot stay on continuously beyond limit:

    $$
    \sum_{j=t}^{t+k} s(j) \cdot \Delta t \leq T_{on}^{max}
    $$

=== "Total On Hours"

    Constrain total on-time per period:

    $$
    H^{min} \leq \sum_t s(t) \cdot \Delta t \leq H^{max}
    $$

=== "Max Switches"

    Limit number of switch-ons per period:

    $$
    \sum_t s^{on}(t) \leq N_{switch}^{max}
    $$

## Variables

| Symbol | Python Name | Description | When Created |
|--------|-------------|-------------|--------------|
| $s(t)$ | `on` | Binary state | Always |
| $s^{on}(t)$ | `switch_on` | Switch-on indicator | Switch effects or constraints |
| $s^{off}(t)$ | `switch_off` | Switch-off indicator | Switch effects or constraints |

## Parameters

| Symbol | Python Name | Description | Default |
|--------|-------------|-------------|---------|
| $c_{switch}$ | `effects_per_switch_on` | Effects per switch-on | None |
| $c_{run}$ | `effects_per_running_hour` | Effects while on | None |
| $T_{on}^{min}$ | `consecutive_on_hours_min` | Min consecutive on | None |
| $T_{on}^{max}$ | `consecutive_on_hours_max` | Max consecutive on | None |
| $T_{off}^{min}$ | `consecutive_off_hours_min` | Min consecutive off | None |
| $T_{off}^{max}$ | `consecutive_off_hours_max` | Max consecutive off | None |
| $H^{min}$ | `on_hours_min` | Min total on hours | None |
| $H^{max}$ | `on_hours_max` | Max total on hours | None |
| $N_{switch}^{max}$ | `switch_on_max` | Max switch-ons | None |

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

- [Flow](../elements/Flow.md) — How flows use on/off state
- [InvestParameters](InvestParameters.md) — Combining with investment
- [Effects & Objective](../effects-penalty-objective.md) — How effects are tracked
