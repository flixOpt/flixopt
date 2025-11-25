# LinearConverter

A LinearConverter transforms inputs into outputs with defined conversion ratios. It's the workhorse for modeling any equipment that converts one form of energy or material into another.

!!! example "Real-world examples"
    - **Gas boiler** — gas → heat (90% efficiency)
    - **Heat pump** — electricity → heat (COP 3.5 = 350% "efficiency")
    - **CHP** — gas → electricity + heat (35% electrical, 50% thermal)
    - **Electrolyzer** — electricity → hydrogen (65% efficiency)

## Core Concept: Conversion Factors

The fundamental equation links inputs and outputs:

$$
\sum_{f \in inputs} a_f \cdot p_f(t) = \sum_{f \in outputs} b_f \cdot p_f(t)
$$

The conversion factors $a$ and $b$ define the relationship between flows.

### Simple Case: One Input, One Output

For a gas boiler with 90% efficiency:

$$
0.9 \cdot p_{gas}(t) = 1 \cdot p_{heat}(t)
$$

Or equivalently: $p_{heat}(t) = 0.9 \cdot p_{gas}(t)$

!!! note "Direction of efficiency"
    The factor is on the **input** side. If you put in 100 kW of gas, you get out 90 kW of heat.

### Multiple Outputs: CHP

A CHP unit produces both electricity and heat from fuel:

$$
0.35 \cdot p_{fuel}(t) = p_{electricity}(t)
$$

$$
0.50 \cdot p_{fuel}(t) = p_{heat}(t)
$$

These are two separate conversion equations from the same input.

### Heat Pump: COP > 1

A heat pump has COP (Coefficient of Performance) of 3.5:

$$
3.5 \cdot p_{electricity}(t) = p_{heat}(t)
$$

The factor is greater than 1 because the heat pump also extracts energy from the environment.

## Variables

| Variable | Python Name | Description | When Created |
|----------|-------------|-------------|--------------|
| $p_{in}(t)$ | (from input Flows) | Input flow rates | Always |
| $p_{out}(t)$ | (from output Flows) | Output flow rates | Always |

The converter itself doesn't create new variables — it creates constraints linking the flow variables.

## Parameters

| Parameter | Python Name | Description |
|-----------|-------------|-------------|
| Input flows | `inputs` | List of input Flows |
| Output flows | `outputs` | List of output Flows |
| Conversion factors | `conversion_factors` | List of factor dictionaries |

## Usage Examples

### Gas Boiler (Simple)

```python
boiler = fx.LinearConverter(
    label='gas_boiler',
    inputs=[fx.Flow(label='gas', bus=gas_bus, size=111)],
    outputs=[fx.Flow(label='heat', bus=heat_bus, size=100)],
    conversion_factors=[{
        'gas': 0.9,   # 90% of gas input...
        'heat': 1,    # ...becomes heat output
    }],
)
```

**Constraint:** $0.9 \cdot p_{gas}(t) = p_{heat}(t)$

### CHP Unit (One Input, Two Outputs)

```python
chp = fx.LinearConverter(
    label='chp',
    inputs=[fx.Flow(label='fuel', bus=gas_bus, size=100)],
    outputs=[
        fx.Flow(label='electricity', bus=electricity_bus, size=35),
        fx.Flow(label='heat', bus=heat_bus, size=50),
    ],
    conversion_factors=[
        {'fuel': 0.35, 'electricity': 1},  # 35% electrical efficiency
        {'fuel': 0.50, 'heat': 1},         # 50% thermal efficiency
    ],
)
```

**Constraints:**

- $0.35 \cdot p_{fuel}(t) = p_{electricity}(t)$
- $0.50 \cdot p_{fuel}(t) = p_{heat}(t)$

Total efficiency: 85%

### Heat Pump

```python
heat_pump = fx.LinearConverter(
    label='heat_pump',
    inputs=[fx.Flow(label='electricity', bus=electricity_bus, size=100)],
    outputs=[fx.Flow(label='heat', bus=heat_bus, size=350)],
    conversion_factors=[{
        'electricity': 3.5,  # COP of 3.5
        'heat': 1,
    }],
)
```

**Constraint:** $3.5 \cdot p_{electricity}(t) = p_{heat}(t)$

### Time-Varying Efficiency

Efficiency can vary with time (e.g., heat pump COP depends on outside temperature):

```python
cop_profile = [3.0, 3.2, 3.5, 4.0, 3.8, 3.5, ...]  # COP varies by timestep

heat_pump = fx.LinearConverter(
    label='heat_pump',
    inputs=[fx.Flow(label='electricity', bus=electricity_bus, size=100)],
    outputs=[fx.Flow(label='heat', bus=heat_bus, size=400)],
    conversion_factors=[{
        'electricity': cop_profile,
        'heat': 1,
    }],
)
```

## Pre-Built Specialized Components

flixOpt provides convenience classes that set up conversion factors automatically:

### Boiler

```python
boiler = fx.linear_converters.Boiler(
    label='boiler',
    eta=0.9,  # Thermal efficiency
    Q_th=fx.Flow(label='heat', bus=heat_bus, size=100),
    Q_fu=fx.Flow(label='fuel', bus=gas_bus),  # Size calculated automatically
)
```

### HeatPump

```python
heat_pump = fx.linear_converters.HeatPump(
    label='heat_pump',
    COP=3.5,
    P_el=fx.Flow(label='electricity', bus=electricity_bus, size=100),
    Q_th=fx.Flow(label='heat', bus=heat_bus),  # Size = 350 kW
)
```

### CHP

```python
chp = fx.linear_converters.CHP(
    label='chp',
    eta_el=0.35,
    eta_th=0.50,
    P_el=fx.Flow(label='electricity', bus=electricity_bus, size=35),
    Q_th=fx.Flow(label='heat', bus=heat_bus, size=50),
    Q_fu=fx.Flow(label='fuel', bus=gas_bus, size=100),
)
```

## Advanced: Piecewise Linear Conversion

For non-linear relationships (e.g., part-load efficiency curves), use piecewise linearization:

```python
from flixopt import Piecewise, Piece, PiecewiseConversion

# Efficiency varies with load
efficiency_curve = Piecewise([
    Piece(start=(0, 0), end=(50, 40)),     # 80% efficiency at low load
    Piece(start=(50, 40), end=(100, 90)),  # 90% efficiency at high load
])

boiler = fx.LinearConverter(
    label='boiler',
    inputs=[fx.Flow(label='gas', bus=gas_bus, size=100)],
    outputs=[fx.Flow(label='heat', bus=heat_bus, size=90)],
    piecewise_conversion=PiecewiseConversion(
        origin_flow='gas',
        piecewise_shares={'heat': efficiency_curve},
    ),
)
```

See [Piecewise Linearization](../features/Piecewise.md) for details.

## On/Off Operation

Add startup costs and minimum run times:

```python
from flixopt import OnOffParameters

generator = fx.LinearConverter(
    label='generator',
    inputs=[fx.Flow(label='fuel', bus=fuel_bus, size=100)],
    outputs=[fx.Flow(label='power', bus=electricity_bus, size=40, relative_minimum=0.4)],
    conversion_factors=[{'fuel': 0.4, 'power': 1}],
    on_off_parameters=OnOffParameters(
        effects_per_switch_on={'costs': 1000},  # €1000 startup cost
        consecutive_on_hours_min=4,             # Must run at least 4 hours
    ),
)
```

See [On/Off Operation](../features/OnOffParameters.md) for details.

## Implementation Details

- **Component Class:** [`LinearConverter`][flixopt.components.LinearConverter]
- **Model Class:** [`LinearConverterModel`][flixopt.components.LinearConverterModel]
- **Specialized Classes:** [`Boiler`][flixopt.linear_converters.Boiler], [`HeatPump`][flixopt.linear_converters.HeatPump], [`CHP`][flixopt.linear_converters.CHP]

## See Also

- [Flow](Flow.md) — Input and output flows
- [Bus](Bus.md) — Where converters connect
- [Piecewise Linearization](../features/Piecewise.md) — Non-linear efficiency curves
- [On/Off Operation](../features/OnOffParameters.md) — Binary operation
- [Core Concepts: Converters](../../core-concepts.md#converters-transform-one-thing-into-another) — High-level overview
