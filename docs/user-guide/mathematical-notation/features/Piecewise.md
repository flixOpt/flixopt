# Piecewise

Piecewise linearization models non-linear relationships using connected linear segments — keeping the problem linear while capturing complex behavior.

!!! example "Real-world examples"
    - **Part-load efficiency** — Boiler efficiency varies with load
    - **Economies of scale** — Cost per kW decreases with size
    - **Forbidden regions** — Turbine can't operate between 0-40%

## Core Concept: Linear Segments

A piecewise function is defined by segments (pieces), each with start and end points:

```
    y
    │        ╱─── Piece 3
    │      ╱
    │    ╱───── Piece 2
    │  ╱
    │╱──────── Piece 1
    └───────────── x
```

The variable can be anywhere along one of these segments.

## Mathematical Formulation

Each piece $k$ has:

- Start point: $(x_k^{start}, y_k^{start})$
- End point: $(x_k^{end}, y_k^{end})$

The value is a weighted combination:

$$
x = \lambda_0 \cdot x^{start} + \lambda_1 \cdot x^{end}
$$

$$
y = \lambda_0 \cdot y^{start} + \lambda_1 \cdot y^{end}
$$

Where $\lambda_0, \lambda_1 \geq 0$ and $\lambda_0 + \lambda_1 = \beta_k$ (piece is active).

=== "Single Piece Active"

    Exactly one piece must be active:

    $$
    \sum_k \beta_k = 1
    $$

=== "With Zero Point"

    Allow all variables to be zero (equipment off):

    $$
    \sum_k \beta_k = \beta_{zero}
    $$

    - $\beta_{zero} = 0$: All off, $x = y = 0$
    - $\beta_{zero} = 1$: One piece active

## Piece Patterns

=== "Continuous (Touching)"

    Pieces share boundary points — smooth function:

    ```python
    curve = fx.Piecewise([
        fx.Piece((0, 0), (50, 45)),     # Low load
        fx.Piece((50, 45), (100, 90)),  # High load (touches at 50)
    ])
    ```

    Operation anywhere from 0-100.

=== "Gap (Forbidden Region)"

    Non-contiguous pieces — forbidden operating range:

    ```python
    curve = fx.Piecewise([
        fx.Piece((0, 0), (0, 0)),       # Off (point)
        fx.Piece((40, 36), (100, 90)),  # Operating (gap: 0-40 forbidden)
    ])
    ```

    Must be off (0) or operating (40-100).

=== "Zero Point"

    Explicitly allow zero without a zero piece:

    ```python
    curve = fx.Piecewise(
        pieces=[
            fx.Piece((10, 9), (50, 45)),
            fx.Piece((50, 45), (100, 90)),
        ],
        zero_point=True,  # Can also be completely off
    )
    ```

    Either off (0) or operating (10-100).

## Variables

| Symbol | Python Name | Description | When Created |
|--------|-------------|-------------|--------------|
| $\beta_k$ | `beta` | Piece $k$ active | Always |
| $\lambda_{0,k}$ | `lambda0` | Weight on start point | Always |
| $\lambda_{1,k}$ | `lambda1` | Weight on end point | Always |
| $\beta_{zero}$ | `zero_point` | Allow zero | `zero_point=True` |

## Parameters

| Symbol | Python Name | Description |
|--------|-------------|-------------|
| $(x_k^{start}, y_k^{start})$ | `Piece.start` | Start point of piece $k$ |
| $(x_k^{end}, y_k^{end})$ | `Piece.end` | End point of piece $k$ |
| - | `zero_point` | Allow all variables = 0 |

## Usage Examples

### Variable COP Heat Pump

```python
# COP varies: 2.5 at low load, 4.0 at high load
elec_to_heat = fx.Piecewise([
    fx.Piece((0, 0), (50, 125)),      # COP ~2.5
    fx.Piece((50, 125), (100, 350)),  # COP ~3.5-4.5
])

hp = fx.LinearConverter(
    label='hp',
    inputs=[fx.Flow(label='el', bus=elec_bus, size=100)],
    outputs=[fx.Flow(label='heat', bus=heat_bus, size=350)],
    piecewise_conversion=fx.PiecewiseConversion(
        origin_flow='el',
        piecewise_shares={'heat': elec_to_heat},
    ),
)
```

### Part-Load Efficiency Boiler

```python
# Efficiency: 80% at low load, 92% at high load
gas_to_heat = fx.Piecewise([
    fx.Piece((0, 0), (30, 24)),       # 80% at 0-30%
    fx.Piece((30, 24), (100, 92)),    # 92% at 30-100%
])

boiler = fx.LinearConverter(
    label='boiler',
    inputs=[fx.Flow(label='gas', bus=gas_bus, size=100)],
    outputs=[fx.Flow(label='heat', bus=heat_bus, size=92)],
    piecewise_conversion=fx.PiecewiseConversion(
        origin_flow='gas',
        piecewise_shares={'heat': gas_to_heat},
    ),
)
```

### Economies of Scale (Investment)

```python
# Cost per kWh decreases with size
battery = fx.InvestParameters(
    minimum_size=10,
    maximum_size=1000,
    piecewise_effects_of_investment=fx.PiecewiseEffects(
        piecewise_origin=fx.Piecewise([
            fx.Piece((0, 0), (100, 100)),
            fx.Piece((100, 100), (500, 500)),
            fx.Piece((500, 500), (1000, 1000)),
        ]),
        piecewise_shares={
            'costs': fx.Piecewise([
                fx.Piece((0, 0), (100, 80000)),     # €800/kWh
                fx.Piece((100, 80000), (500, 350000)),  # €675/kWh avg
                fx.Piece((500, 350000), (1000, 750000)), # €500/kWh
            ])
        },
    ),
)
```

### Forbidden Operating Region

```python
# Turbine: off or 40-100%, not in between
turbine_curve = fx.Piecewise([
    fx.Piece((0, 0), (0, 0)),        # Off
    fx.Piece((40, 40), (100, 100)),  # Operating range
])
```

## Implementation Details

- **Feature Class:** [`Piecewise`][flixopt.interface.Piecewise]
- **Helper Class:** [`Piece`][flixopt.interface.Piece]
- **Model Class:** [`PiecewiseModel`][flixopt.features.PiecewiseModel]

## See Also

- [LinearConverter](../elements/LinearConverter.md) — Using piecewise conversion
- [InvestParameters](InvestParameters.md) — Piecewise investment costs
