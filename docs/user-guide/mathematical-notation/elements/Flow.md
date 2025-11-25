# Flow

A Flow represents the movement of energy or material between a component and a bus. It's the primary optimization variable — the solver decides how much flows at each timestep.

!!! example "Real-world examples"
    - Heat output from a boiler to the heat bus
    - Electricity import from the grid
    - Gas consumption by a CHP unit
    - Charging power into a battery

## Core Concept: Size and Flow Rate

Every flow has two key quantities:

- **Size** ($P$) — The capacity or maximum possible flow rate. Think of it as "how big is the pipe?"
- **Flow Rate** ($p(t)$) — The actual flow at each timestep. This is what the optimizer decides.

The fundamental constraint:

$$
p_{min}(t) \leq p(t) \leq p_{max}(t)
$$

Usually, bounds are defined *relative* to the size:

$$
P \cdot relative\_min(t) \leq p(t) \leq P \cdot relative\_max(t)
$$

!!! example "A 100 kW boiler that can modulate down to 30%"
    - Size: $P = 100$ kW
    - Relative minimum: 0.3
    - Relative maximum: 1.0
    - Constraint: $30 \leq p(t) \leq 100$ kW

## Flow Hours: Energy vs Power

**Flow rate** is power (kW, MW). **Flow hours** is energy (kWh, MWh) — flow rate times time:

$$
flow\_hours(t) = p(t) \cdot \Delta t
$$

| Flow Rate | Timestep | Flow Hours |
|-----------|----------|------------|
| 100 kW | 1 hour | 100 kWh |
| 100 kW | 15 min | 25 kWh |
| 50 MW | 1 hour | 50 MWh |

This matters for costs: `effects_per_flow_hour` is cost per energy (€/MWh), not per power.

## Constraints

### Capacity Bounds (Always Active)

$$
P \cdot relative\_min(t) \leq p(t) \leq P \cdot relative\_max(t)
$$

### Fixed Profile (Renewable Generation, Demands)

When you have a known profile (solar irradiance, demand curve):

$$
p(t) = P \cdot profile(t)
$$

The flow rate is fixed to the profile — no optimization freedom.

### Load Factor Limits

Constrain average utilization over the period:

$$
LF_{min} \cdot P \cdot N_t \leq \sum_t p(t) \leq LF_{max} \cdot P \cdot N_t
$$

Where $N_t$ is the number of timesteps.

!!! example "Baseload plant must run at least 70% average"
    - Size: 200 MW
    - Load factor minimum: 0.7
    - Over 8760 hours: must produce at least $200 \times 0.7 \times 8760 = 1{,}226{,}400$ MWh

### Flow Hours Limits

Constrain total energy over the period:

$$
FH_{min} \leq \sum_t p(t) \cdot \Delta t \leq FH_{max}
$$

!!! example "Annual gas limit of 10,000 MWh"
    Sets $FH_{max} = 10{,}000$ MWh.

## Variables

| Variable | Python Name | Description | When Created |
|----------|-------------|-------------|--------------|
| $p(t)$ | `flow_rate` | Flow rate at each timestep | Always |
| $P$ | `size` | Capacity (decision variable) | When `size` is `InvestParameters` |
| $s(t)$ | `on_off_state` | Binary on/off state | When `on_off_parameters` set |

## Parameters

| Parameter | Python Name | Description | Default |
|-----------|-------------|-------------|---------|
| $P$ | `size` | Flow capacity | Required |
| $relative\_min$ | `relative_minimum` | Min as fraction of size | 0 |
| $relative\_max$ | `relative_maximum` | Max as fraction of size | 1 |
| $profile$ | `fixed_relative_profile` | Fixed relative profile | None |
| $LF_{min}$ | `load_factor_min` | Minimum average utilization | None |
| $LF_{max}$ | `load_factor_max` | Maximum average utilization | None |
| $FH_{min}$ | `flow_hours_min` | Minimum total energy | None |
| $FH_{max}$ | `flow_hours_max` | Maximum total energy | None |

## Usage Examples

### Basic Flow with Fixed Capacity

```python
heat_output = fx.Flow(
    label='heat_out',
    bus=heat_bus,
    size=100,  # 100 kW capacity
    relative_minimum=0.3,  # Can't go below 30 kW
)
```

### Flow with Costs

```python
gas_input = fx.Flow(
    label='gas_in',
    bus=gas_bus,
    size=150,
    effects_per_flow_hour={'costs': 50},  # €50/MWh gas price
)
```

### Fixed Profile (Solar PV)

```python
solar_profile = [0, 0, 0.1, 0.4, 0.8, 1.0, 0.9, 0.6, 0.2, 0, 0, 0]  # Relative to peak

solar_output = fx.Flow(
    label='solar_out',
    bus=electricity_bus,
    size=500,  # 500 kW peak
    fixed_relative_profile=solar_profile,
)
```

### Investment Decision (Optimized Size)

```python
from flixopt import InvestParameters

battery_flow = fx.Flow(
    label='battery_power',
    bus=electricity_bus,
    size=InvestParameters(
        minimum_size=0,
        maximum_size=1000,  # Up to 1 MW
        specific_effects={'costs': 100_000},  # €100k/MW/year
    ),
)
```

See [InvestParameters](../features/InvestParameters.md) for details.

### On/Off Operation

```python
from flixopt import OnOffParameters

generator_output = fx.Flow(
    label='power_out',
    bus=electricity_bus,
    size=50,
    relative_minimum=0.4,  # 40% minimum when ON
    on_off_parameters=OnOffParameters(
        effects_per_switch_on={'costs': 500},  # €500 startup cost
        consecutive_on_hours_min=2,  # Must run at least 2 hours
    ),
)
```

See [OnOffParameters](../features/OnOffParameters.md) for details.

## How Flows Connect to Components

Flows are always part of a component:

```python
boiler = fx.linear_converters.Boiler(
    label='boiler',
    eta=0.9,
    # These flows are created automatically:
    # - inputs: gas flow from gas_bus
    # - outputs: heat flow to heat_bus
    Q_th=fx.Flow(label='heat', bus=heat_bus, size=100),
    Q_fu=fx.Flow(label='fuel', bus=gas_bus, size=111),  # 100/0.9
)
```

The component defines how input and output flows relate (conversion equations).

## Implementation Details

- **Element Class:** [`Flow`][flixopt.elements.Flow]
- **Model Class:** [`FlowModel`][flixopt.elements.FlowModel]

## See Also

- [Bus](Bus.md) — Where flows connect
- [LinearConverter](LinearConverter.md) — Components that use flows
- [InvestParameters](../features/InvestParameters.md) — Optimizing flow capacity
- [OnOffParameters](../features/OnOffParameters.md) — Binary on/off operation
- [Core Concepts: Flows](../../core-concepts.md#flows-what-moves-between-elements) — High-level overview
