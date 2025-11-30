# SizingParameters

SizingParameters make capacity a decision variable — should we build this? How big?

!!! note "Naming Change"
    `SizingParameters` replaces the deprecated `InvestParameters`.
    For investment timing with fixed lifetime (when to invest), see [InvestmentParameters](InvestmentParameters.md).

## Basic: Size as Variable

$$
P^{min} \leq P \leq P^{max}
$$

```python
battery = fx.Storage(
    ...,
    capacity_in_flow_hours=fx.SizingParameters(
        minimum_size=10,
        maximum_size=1000,
        effects_per_size={'costs': 600},  # €600/kWh
    ),
)
```

---

## Sizing Modes

By default, sizing is **optional** — the optimizer can choose $P = 0$ (don't build).

=== "Continuous"

    Choose size within range (or zero):

    ```python
    fx.SizingParameters(
        minimum_size=10,
        maximum_size=1000,
    )
    # → P = 0  OR  10 ≤ P ≤ 1000
    ```

=== "Binary"

    Fixed size or nothing:

    ```python
    fx.SizingParameters(
        fixed_size=100,  # 100 kW or 0
    )
    # → P ∈ {0, 100}
    ```

=== "Mandatory"

    Force sizing with `mandatory=True` — zero not allowed:

    ```python
    fx.SizingParameters(
        minimum_size=50,
        maximum_size=200,
        mandatory=True,
    )
    # → 50 ≤ P ≤ 200 (no zero option)
    ```

---

## Sizing Effects

=== "Per-Size Cost"

    Cost proportional to capacity (€/kW):

    $E = P \cdot c_{spec}$

    ```python
    fx.SizingParameters(
        effects_per_size={'costs': 1200},  # €1200/kW
    )
    ```

=== "Fixed Cost"

    One-time cost if sizing:

    $E = s_{sized} \cdot c_{fix}$

    ```python
    fx.SizingParameters(
        effects_of_size={'costs': 25000},  # €25k
    )
    ```

=== "Retirement Cost"

    Cost if NOT sizing (demolition, opportunity cost):

    $E = (1 - s_{sized}) \cdot c_{ret}$

    ```python
    fx.SizingParameters(
        effects_of_retirement={'costs': 8000},  # Demolition
    )
    ```

=== "Piecewise Cost"

    Non-linear cost curves (e.g., economies of scale):

    $E = f_{piecewise}(P)$

    ```python
    fx.SizingParameters(
        piecewise_effects_per_size=fx.PiecewiseEffects(
            piecewise_origin=fx.Piecewise([
                fx.Piece(0, 100),
                fx.Piece(100, 500),
            ]),
            piecewise_shares={
                'costs': fx.Piecewise([
                    fx.Piece(0, 80_000),      # €800/kW for 0-100
                    fx.Piece(80_000, 280_000), # €500/kW for 100-500
                ])
            },
        ),
    )
    ```

    See [Piecewise](Piecewise.md) for details on the formulation.

---

## Reference

| Symbol | Type | Description |
|--------|------|-------------|
| $P$ | $\mathbb{R}_{\geq 0}$ | Size (capacity) |
| $s_{sized}$ | $\{0, 1\}$ | Binary sizing decision (0=no, 1=yes) |
| $P^{min}$ | $\mathbb{R}_{\geq 0}$ | Minimum size (`minimum_size`) |
| $P^{max}$ | $\mathbb{R}_{\geq 0}$ | Maximum size (`maximum_size`) |
| $c_{spec}$ | $\mathbb{R}$ | Per-size effect (`effects_per_size`) |
| $c_{fix}$ | $\mathbb{R}$ | Fixed effect (`effects_of_size`) |
| $c_{ret}$ | $\mathbb{R}$ | Retirement effect (`effects_of_retirement`) |

**Classes:** [`SizingParameters`][flixopt.interface.SizingParameters], [`SizingModel`][flixopt.features.SizingModel]
