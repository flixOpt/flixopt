# Piecewise

Piecewise linearization approximates non-linear relationships using connected linear segments.

## Mathematical Formulation

A piecewise linear function with $n$ segments uses SOS2 (Special Ordered Set Type 2) variables:

$$
x = \sum_{i=0}^{n} \lambda_i \cdot x_i \quad \text{and} \quad y = \sum_{i=0}^{n} \lambda_i \cdot y_i
$$

Where:

- $(x_i, y_i)$ are the breakpoints defining the segments
- $\lambda_i \geq 0$ are interpolation weights with $\sum_i \lambda_i = 1$
- At most two adjacent $\lambda_i$ can be non-zero (SOS2 constraint)

This ensures $y$ follows the piecewise linear path defined by the breakpoints.

---

## Building Blocks

=== "Piece"

    A linear segment from start to end value:

    ```python
    fx.Piece(start=10, end=50)  # Linear from 10 to 50
    ```

    Values can be time-varying:

    ```python
    fx.Piece(
        start=np.linspace(5, 6, n_timesteps),
        end=np.linspace(30, 35, n_timesteps)
    )
    ```

=== "Piecewise"

    Multiple connected pieces forming a piecewise linear function:

    ```python
    fx.Piecewise([
        fx.Piece(0, 30),   # Segment 1: 0 → 30
        fx.Piece(30, 60),  # Segment 2: 30 → 60
    ])
    ```

=== "PiecewiseConversion"

    Synchronizes multiple flows — all interpolate at the same relative position:

    ```python
    fx.PiecewiseConversion({
        'input_flow':  fx.Piecewise([...]),
        'output_flow': fx.Piecewise([...]),
    })
    ```

    All piecewise functions must have the same number of segments.

=== "PiecewiseEffects"

    Maps a size/capacity variable to effects (costs, emissions):

    ```python
    fx.PiecewiseEffects(
        piecewise_origin=fx.Piecewise([...]),  # Size segments
        piecewise_shares={'costs': fx.Piecewise([...])},  # Effect segments
    )
    ```

---

## Usage

=== "Variable Efficiency"

    Converter efficiency that varies with load:

    ```python
    chp = fx.LinearConverter(
        ...,
        piecewise_conversion=fx.PiecewiseConversion({
            'el':   fx.Piecewise([fx.Piece(5, 30), fx.Piece(40, 60)]),
            'heat': fx.Piecewise([fx.Piece(6, 35), fx.Piece(45, 100)]),
            'fuel': fx.Piecewise([fx.Piece(12, 70), fx.Piece(90, 200)]),
        }),
    )
    ```

=== "Economies of Scale"

    Investment cost per unit decreases with size:

    ```python
    fx.InvestParameters(
        piecewise_effects_of_investment=fx.PiecewiseEffects(
            piecewise_origin=fx.Piecewise([
                fx.Piece(0, 100),
                fx.Piece(100, 500),
            ]),
            piecewise_shares={
                'costs': fx.Piecewise([
                    fx.Piece(0, 80_000),
                    fx.Piece(80_000, 280_000),
                ])
            },
        ),
    )
    ```

=== "Forbidden Operating Region"

    Equipment cannot operate in certain ranges:

    ```python
    fx.PiecewiseConversion({
        'fuel':  fx.Piecewise([fx.Piece(0, 0), fx.Piece(40, 100)]),
        'power': fx.Piecewise([fx.Piece(0, 0), fx.Piece(35, 95)]),
    })
    # Either off (0,0) or operating above 40%
    ```

---

## Reference

| Variable | Description |
|----------|-------------|
| $\lambda_i$ | SOS2 interpolation weight for breakpoint $i$ |
| $(x_i, y_i)$ | Breakpoint coordinates |

| Class | Description |
|-------|-------------|
| `Piece(start, end)` | Linear segment |
| `Piecewise([pieces])` | Collection of segments |
| `PiecewiseConversion({flow: piecewise})` | Synchronized multi-flow conversion |
| `PiecewiseEffects(origin, shares)` | Size-to-effect mapping |

**Classes:** [`Piecewise`][flixopt.interface.Piecewise], [`Piece`][flixopt.interface.Piece], [`PiecewiseConversion`][flixopt.interface.PiecewiseConversion], [`PiecewiseEffects`][flixopt.interface.PiecewiseEffects]
