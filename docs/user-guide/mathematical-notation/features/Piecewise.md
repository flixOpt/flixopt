# Piecewise

Piecewise linearization models non-linear relationships with connected linear segments.

## Basic: Linear Segments

```plotly
{
  "data": [
    {"x": [0, 50], "y": [0, 45], "mode": "lines+markers", "name": "Piece 1", "line": {"color": "#009688", "width": 3}, "marker": {"size": 10}},
    {"x": [50, 100], "y": [45, 90], "mode": "lines+markers", "name": "Piece 2", "line": {"color": "#00bcd4", "width": 3}, "marker": {"size": 10}}
  ],
  "layout": {"xaxis": {"title": "Input"}, "yaxis": {"title": "Output"}, "height": 250, "margin": {"t": 20}}
}
```

```python
curve = fx.Piecewise([
    fx.Piece((0, 0), (50, 45)),     # Slope 0.9
    fx.Piece((50, 45), (100, 90)),  # Slope 0.9
])
```

The optimizer picks a point along exactly one segment.

---

## Use Cases

=== "Part-Load Efficiency"

    Boiler: 80% at low load, 92% at high load:

    ```python
    gas_to_heat = fx.Piecewise([
        fx.Piece((0, 0), (30, 24)),     # 80%
        fx.Piece((30, 24), (100, 92)),  # 92%
    ])

    boiler = fx.LinearConverter(
        ...,
        piecewise_conversion=fx.PiecewiseConversion(
            origin_flow='gas',
            piecewise_shares={'heat': gas_to_heat},
        ),
    )
    ```

=== "Variable COP"

    Heat pump COP varies with load:

    ```python
    elec_to_heat = fx.Piecewise([
        fx.Piece((0, 0), (50, 125)),      # COP ~2.5
        fx.Piece((50, 125), (100, 350)),  # COP ~4.5
    ])

    hp = fx.LinearConverter(
        ...,
        piecewise_conversion=fx.PiecewiseConversion(
            origin_flow='el',
            piecewise_shares={'heat': elec_to_heat},
        ),
    )
    ```

=== "Forbidden Region"

    Turbine: off or 40-100%, nothing in between:

    ```python
    curve = fx.Piecewise([
        fx.Piece((0, 0), (0, 0)),        # Off (point)
        fx.Piece((40, 40), (100, 100)),  # Operating
    ])
    ```

=== "Economies of Scale"

    Investment cost decreases with size:

    ```python
    fx.InvestParameters(
        piecewise_effects_of_investment=fx.PiecewiseEffects(
            piecewise_origin=fx.Piecewise([...]),
            piecewise_shares={'costs': fx.Piecewise([
                fx.Piece((0, 0), (100, 80000)),      # €800/kWh
                fx.Piece((100, 80000), (500, 350000)),  # €675/kWh
            ])},
        ),
    )
    ```

---

## Zero Point

Allow equipment to be completely off without a zero piece:

```python
curve = fx.Piecewise(
    pieces=[
        fx.Piece((10, 9), (100, 90)),  # Operating range 10-100
    ],
    zero_point=True,  # Can also be off (0, 0)
)
```

---

## Reference

| Variable | Description |
|----------|-------------|
| $\beta_k$ | Piece $k$ is active |
| $\lambda_0, \lambda_1$ | Weights on start/end points |

| Parameter | Description |
|-----------|-------------|
| `Piece(start, end)` | Define segment from start to end point |
| `zero_point` | Allow all variables = 0 |

**Classes:** [`Piecewise`][flixopt.interface.Piecewise], [`Piece`][flixopt.interface.Piece]
