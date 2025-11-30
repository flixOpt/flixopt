# InvestParameters

InvestParameters make capacity a decision variable — should we build this? How big?

## Basic: Size as Variable

$$
P^{min} \leq P \leq P^{max}
$$

```python
battery = fx.Storage(
    ...,
    capacity_in_flow_hours=fx.InvestParameters(
        minimum_size=10,
        maximum_size=1000,
        specific_effects={'costs': 600},  # €600/kWh
    ),
)
```

---

## Investment Modes

By default, investment is **optional** — the optimizer can choose $P = 0$ (don't invest).

=== "Continuous"

    Choose size within range (or zero):

    ```python
    fx.InvestParameters(
        minimum_size=10,
        maximum_size=1000,
    )
    # → P = 0  OR  10 ≤ P ≤ 1000
    ```

=== "Binary"

    Fixed size or nothing:

    ```python
    fx.InvestParameters(
        fixed_size=100,  # 100 kW or 0
    )
    # → P ∈ {0, 100}
    ```

=== "Mandatory"

    Force investment with `mandatory=True` — zero not allowed:

    ```python
    fx.InvestParameters(
        minimum_size=50,
        maximum_size=200,
        mandatory=True,
    )
    # → 50 ≤ P ≤ 200 (no zero option)
    ```

---

## Investment Effects

=== "Per-Size Cost"

    Cost proportional to capacity (€/kW):

    $E = P \cdot c_{spec}$

    ```python
    fx.InvestParameters(
        specific_effects={'costs': 1200},  # €1200/kW
    )
    ```

=== "Fixed Cost"

    One-time cost if investing:

    $E = s_{inv} \cdot c_{fix}$

    ```python
    fx.InvestParameters(
        effects_of_investment={'costs': 25000},  # €25k
    )
    ```

=== "Retirement Cost"

    Cost if NOT investing:

    $E = (1 - s_{inv}) \cdot c_{ret}$

    ```python
    fx.InvestParameters(
        effects_of_retirement={'costs': 8000},  # Demolition
    )
    ```

=== "Piecewise Cost"

    Non-linear cost curves (e.g., economies of scale):

    $E = f_{piecewise}(P)$

    ```python
    fx.InvestParameters(
        piecewise_effects_of_investment=fx.PiecewiseEffects(
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
| $P$ | $\mathbb{R}_{\geq 0}$ | Investment size (capacity) |
| $s_{inv}$ | $\{0, 1\}$ | Binary investment decision (0=no, 1=yes) |
| $P^{min}$ | $\mathbb{R}_{\geq 0}$ | Minimum size (`minimum_size`) |
| $P^{max}$ | $\mathbb{R}_{\geq 0}$ | Maximum size (`maximum_size`) |
| $c_{spec}$ | $\mathbb{R}$ | Per-size effect (`effects_of_investment_per_size`) |
| $c_{fix}$ | $\mathbb{R}$ | Fixed effect (`effects_of_investment`) |
| $c_{ret}$ | $\mathbb{R}$ | Retirement effect (`effects_of_retirement`) |

**Classes:** [`InvestParameters`][flixopt.interface.InvestParameters], [`InvestmentModel`][flixopt.features.InvestmentModel]
