# LinearConverter

A LinearConverter transforms inputs into outputs with fixed ratios.

## Basic: Conversion Equation

$$
\sum_{in} a_f \cdot p_f(t) = \sum_{out} b_f \cdot p_f(t)
$$

=== "Boiler (η = 90%)"

    $0.9 \cdot p_{gas}(t) = p_{heat}(t)$

    ```python
    boiler = fx.LinearConverter(
        label='boiler',
        inputs=[fx.Flow(label='gas', bus=gas_bus, size=111)],
        outputs=[fx.Flow(label='heat', bus=heat_bus, size=100)],
        conversion_factors=[{'gas': 0.9, 'heat': 1}],
    )
    ```

=== "Heat Pump (COP = 3.5)"

    $3.5 \cdot p_{el}(t) = p_{heat}(t)$

    ```python
    hp = fx.LinearConverter(
        label='hp',
        inputs=[fx.Flow(label='el', bus=elec_bus, size=100)],
        outputs=[fx.Flow(label='heat', bus=heat_bus, size=350)],
        conversion_factors=[{'el': 3.5, 'heat': 1}],
    )
    ```

=== "CHP (35% el, 50% th)"

    Two constraints linking fuel to outputs:

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

---

## Time-Varying Efficiency

Pass a list for time-dependent conversion:

```python
cop = [3.0, 3.2, 3.5, 4.0, 3.8, ...]  # Varies with ambient temperature

hp = fx.LinearConverter(
    ...,
    conversion_factors=[{'el': cop, 'heat': 1}],
)
```

---

## Convenience Classes

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
    P_el=fx.Flow(...), Q_th=fx.Flow(...), Q_fu=fx.Flow(...),
)
```

---

## Adding Features

=== "On/Off Operation"

    ```python
    gen = fx.LinearConverter(
        ...,
        status_parameters=fx.StatusParameters(
            effects_per_startup={'costs': 1000},
            min_uptime=4,
        ),
    )
    ```

    See [StatusParameters](../features/StatusParameters.md).

=== "Piecewise Efficiency"

    For non-linear efficiency curves:

    ```python
    curve = fx.Piecewise([
        fx.Piece(start=(0, 0), end=(50, 40)),    # 80% at low load
        fx.Piece(start=(50, 40), end=(100, 90)), # 100% at high load
    ])

    boiler = fx.LinearConverter(
        ...,
        piecewise_conversion=fx.PiecewiseConversion(
            origin_flow='gas',
            piecewise_shares={'heat': curve},
        ),
    )
    ```

    See [Piecewise](../features/Piecewise.md).

---

## Reference

The converter creates **constraints** linking flows, not new variables.

| Parameter | Python | Description |
|-----------|--------|-------------|
| Inputs | `inputs` | List of input Flows |
| Outputs | `outputs` | List of output Flows |
| Ratios | `conversion_factors` | List of factor dicts |

**Classes:** [`LinearConverter`][flixopt.components.LinearConverter], [`LinearConverterModel`][flixopt.components.LinearConverterModel]
