# Piecewise

Piecewise linearization approximates non-linear relationships using connected linear segments.

## Mathematical Formulation

A piecewise linear function with $n$ segments uses per-segment interpolation:

$$
x = \sum_{i=1}^{n} \left( \lambda_i^0 \cdot x_i^{start} + \lambda_i^1 \cdot x_i^{end} \right)
$$

Each segment $i$ has:

- $s_i \in \{0, 1\}$ — binary indicating if segment is active
- $\lambda_i^0, \lambda_i^1 \geq 0$ — interpolation weights for segment endpoints

Constraints ensure valid interpolation:

$$
\lambda_i^0 + \lambda_i^1 = s_i \quad \forall i
$$

$$
\sum_{i=1}^{n} s_i \leq 1
$$

When segment $i$ is active ($s_i = 1$), the lambdas interpolate between $x_i^{start}$ and $x_i^{end}$. When inactive ($s_i = 0$), both lambdas are zero.

!!! note "Implementation Note"
    This formulation is an explicit binary reformulation of SOS2 (Special Ordered Set Type 2) constraints. It produces identical results but uses more variables. We will migrate to native SOS2 constraints once [linopy](https://github.com/PyPSA/linopy) supports them.

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

    Multiple segments forming a piecewise linear function:

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

| Symbol | Type | Description |
|--------|------|-------------|
| $x$ | $\mathbb{R}$ | Interpolated variable value |
| $s_i$ | $\{0, 1\}$ | Binary: segment $i$ is active |
| $\lambda_i^0$ | $\mathbb{R}_{\geq 0}$ | Interpolation weight for segment start |
| $\lambda_i^1$ | $\mathbb{R}_{\geq 0}$ | Interpolation weight for segment end |
| $x_i^{start}$ | $\mathbb{R}$ | Start value of segment $i$ |
| $x_i^{end}$ | $\mathbb{R}$ | End value of segment $i$ |
| $n$ | $\mathbb{Z}_{> 0}$ | Number of segments |

**Classes:** [`Piecewise`][flixopt.interface.Piecewise], [`Piece`][flixopt.interface.Piece], [`PiecewiseConversion`][flixopt.interface.PiecewiseConversion], [`PiecewiseEffects`][flixopt.interface.PiecewiseEffects]
