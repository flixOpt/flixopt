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

=== "Continuous"

    Choose size within range:

    ```python
    fx.InvestParameters(
        minimum_size=10,
        maximum_size=1000,
    )
    # → 10 ≤ P ≤ 1000
    ```

=== "Binary"

    Fixed size or nothing:

    ```python
    fx.InvestParameters(
        fixed_size=100,  # 100 kW or 0
    )
    # → P ∈ {0, 100}
    ```

=== "Optional"

    Can choose not to invest ($P = 0$ allowed):

    ```python
    fx.InvestParameters(
        minimum_size=0,  # Zero allowed
        maximum_size=1000,
    )
    ```

=== "Mandatory"

    Must invest (no binary variable):

    ```python
    fx.InvestParameters(
        minimum_size=50,
        maximum_size=200,
        mandatory=True,
    )
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

---

## Reference

| Variable | Description |
|----------|-------------|
| $P$ | Investment size |
| $s_{inv}$ | Binary decision (0=no, 1=yes) |

| Parameter | Python | Description |
|-----------|--------|-------------|
| Min size | `minimum_size` | Lower bound |
| Max size | `maximum_size` | Upper bound |
| Fixed size | `fixed_size` | Binary: this or nothing |
| Per-size cost | `specific_effects` | €/unit |
| Fixed cost | `effects_of_investment` | One-time if investing |
| Retirement | `effects_of_retirement` | Cost if not investing |
| Force invest | `mandatory` | No binary variable |

**Classes:** [`InvestParameters`][flixopt.interface.InvestParameters], [`InvestmentModel`][flixopt.features.InvestmentModel]
