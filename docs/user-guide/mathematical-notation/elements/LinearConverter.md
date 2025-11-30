# LinearConverter

A LinearConverter transforms inputs into outputs with defined conversion ratios. It's the workhorse for modeling equipment that converts energy or material.

!!! example "Real-world examples"
    - **Gas boiler** — gas → heat (η = 90%)
    - **Heat pump** — electricity → heat (COP = 3.5)
    - **CHP** — gas → electricity + heat
    - **Electrolyzer** — electricity → hydrogen

## Core Concept: Conversion Factors

The fundamental equation links inputs and outputs:

$$
\sum_{f \in \mathcal{F}_{in}} a_f \cdot p_f(t) = \sum_{f \in \mathcal{F}_{out}} b_f \cdot p_f(t)
$$

The conversion factors $a_f$ and $b_f$ define the relationship.

=== "Single Input/Output"

    For a boiler with 90% efficiency:

    $$
    0.9 \cdot p_{gas}(t) = 1 \cdot p_{heat}(t)
    $$

    Or: $p_{heat}(t) = 0.9 \cdot p_{gas}(t)$

=== "Multiple Outputs (CHP)"

    CHP produces electricity and heat from fuel:

    $$
    0.35 \cdot p_{fuel}(t) = p_{el}(t)
    $$

    $$
    0.50 \cdot p_{fuel}(t) = p_{th}(t)
    $$

    Total efficiency: 85%

=== "COP > 1 (Heat Pump)"

    Heat pump with COP = 3.5:

    $$
    3.5 \cdot p_{el}(t) = p_{th}(t)
    $$

    Factor > 1 because it extracts heat from environment.

=== "Time-Varying"

    Efficiency can vary with time (e.g., COP depends on temperature):

    $$
    \eta(t) \cdot p_{in}(t) = p_{out}(t)
    $$

## Variables

| Symbol | Python Name | Description | When Created |
|--------|-------------|-------------|--------------|
| $p_{in}(t)$ | (from input Flows) | Input flow rates | Always |
| $p_{out}(t)$ | (from output Flows) | Output flow rates | Always |

The converter creates **constraints** linking flow variables, not new variables.

## Parameters

| Symbol | Python Name | Description |
|--------|-------------|-------------|
| $\mathcal{F}_{in}$ | `inputs` | List of input Flows |
| $\mathcal{F}_{out}$ | `outputs` | List of output Flows |
| $a_f$, $b_f$ | `conversion_factors` | Factor dictionaries |

## Usage Examples

### Gas Boiler

```python
boiler = fx.LinearConverter(
    label='boiler',
    inputs=[fx.Flow(label='gas', bus=gas_bus, size=111)],
    outputs=[fx.Flow(label='heat', bus=heat_bus, size=100)],
    conversion_factors=[{'gas': 0.9, 'heat': 1}],
)
```

### CHP Unit

```python
chp = fx.LinearConverter(
    label='chp',
    inputs=[fx.Flow(label='fuel', bus=gas_bus, size=100)],
    outputs=[
        fx.Flow(label='el', bus=elec_bus, size=35),
        fx.Flow(label='heat', bus=heat_bus, size=50),
    ],
    conversion_factors=[
        {'fuel': 0.35, 'el': 1},
        {'fuel': 0.50, 'heat': 1},
    ],
)
```

### Heat Pump

```python
hp = fx.LinearConverter(
    label='hp',
    inputs=[fx.Flow(label='el', bus=elec_bus, size=100)],
    outputs=[fx.Flow(label='heat', bus=heat_bus, size=350)],
    conversion_factors=[{'el': 3.5, 'heat': 1}],
)
```

### Time-Varying COP

```python
cop = [3.0, 3.2, 3.5, 4.0, 3.8, ...]  # Varies by timestep

hp = fx.LinearConverter(
    label='hp',
    inputs=[fx.Flow(label='el', bus=elec_bus, size=100)],
    outputs=[fx.Flow(label='heat', bus=heat_bus, size=400)],
    conversion_factors=[{'el': cop, 'heat': 1}],
)
```

## Specialized Components

flixOpt provides convenience classes:

```python
# Boiler
boiler = fx.linear_converters.Boiler(
    label='boiler', eta=0.9,
    Q_th=fx.Flow(label='heat', bus=heat_bus, size=100),
    Q_fu=fx.Flow(label='fuel', bus=gas_bus),
)

# Heat Pump
hp = fx.linear_converters.HeatPump(
    label='hp', COP=3.5,
    P_el=fx.Flow(label='el', bus=elec_bus, size=100),
    Q_th=fx.Flow(label='heat', bus=heat_bus),
)

# CHP
chp = fx.linear_converters.CHP(
    label='chp', eta_el=0.35, eta_th=0.50,
    P_el=fx.Flow(label='el', bus=elec_bus, size=35),
    Q_th=fx.Flow(label='heat', bus=heat_bus, size=50),
    Q_fu=fx.Flow(label='fuel', bus=gas_bus, size=100),
)
```

## Advanced Features

=== "Piecewise Linear"

    For non-linear efficiency curves:

    ```python
    from flixopt import Piecewise, Piece, PiecewiseConversion

    curve = Piecewise([
        Piece(start=(0, 0), end=(50, 40)),    # 80% at low load
        Piece(start=(50, 40), end=(100, 90)), # 90% at high load
    ])

    boiler = fx.LinearConverter(
        ...,
        piecewise_conversion=PiecewiseConversion(
            origin_flow='gas',
            piecewise_shares={'heat': curve},
        ),
    )
    ```

    See [Piecewise](../features/Piecewise.md).

=== "On/Off Operation"

    Add startup costs and minimum run times:

    ```python
    gen = fx.LinearConverter(
        label='gen',
        inputs=[fx.Flow(label='fuel', bus=fuel_bus, size=100)],
        outputs=[fx.Flow(label='el', bus=elec_bus, size=40, relative_minimum=0.4)],
        conversion_factors=[{'fuel': 0.4, 'el': 1}],
        status_parameters=fx.StatusParameters(
            effects_per_startup={'costs': 1000},
            min_uptime=4,
        ),
    )
    ```

    See [StatusParameters](../features/StatusParameters.md).

## Implementation Details

- **Component Class:** [`LinearConverter`][flixopt.components.LinearConverter]
- **Model Class:** [`LinearConverterModel`][flixopt.components.LinearConverterModel]

## See Also

- [Flow](Flow.md) — Input and output flows
- [Bus](Bus.md) — Where converters connect
- [Piecewise](../features/Piecewise.md) — Non-linear efficiency
- [StatusParameters](../features/StatusParameters.md) — Binary operation
