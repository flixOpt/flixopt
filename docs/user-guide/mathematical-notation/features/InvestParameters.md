# InvestParameters

InvestParameters enable investment decisions — should we build this? If so, how big?

!!! example "Real-world examples"
    - **Solar PV** — Install 100 kW system or not (binary)
    - **Battery** — Choose capacity between 10-1000 kWh (continuous)
    - **Boiler replacement** — Upgrade or pay demolition costs

## Core Concept: Size as a Variable

When `size` is an `InvestParameters`, the capacity $P$ becomes an optimization variable instead of a fixed parameter.

=== "Binary (Fixed Size)"

    Build a predefined size or nothing:

    $$
    P = s_{inv} \cdot P_{fixed}
    $$

    Where $s_{inv} \in \{0, 1\}$ is the binary investment decision.

    - $s_{inv} = 0$: Don't invest, $P = 0$
    - $s_{inv} = 1$: Invest, $P = P_{fixed}$

=== "Continuous (Size Range)"

    Choose size within bounds:

    $$
    s_{inv} \cdot P^{min} \leq P \leq s_{inv} \cdot P^{max}
    $$

    - $s_{inv} = 0$: Don't invest, $P = 0$
    - $s_{inv} = 1$: Invest, $P \in [P^{min}, P^{max}]$

=== "Mandatory"

    Must invest, only choose size:

    $$
    P^{min} \leq P \leq P^{max}
    $$

    No binary variable — investment is required.

## Investment Effects (Costs)

=== "Fixed Effects"

    One-time costs if investing (permits, grid connection):

    $$
    E_{e,fix} = s_{inv} \cdot c_{fix}
    $$

=== "Specific Effects"

    Costs proportional to size (€/kW):

    $$
    E_{e,spec} = P \cdot c_{spec}
    $$

=== "Retirement Effects"

    Costs if NOT investing (demolition):

    $$
    E_{e,ret} = (1 - s_{inv}) \cdot c_{ret}
    $$

=== "Piecewise Effects"

    Non-linear costs (economies of scale):

    $$
    E_{e,pw} = f_{pw}(P)
    $$

    See [Piecewise](Piecewise.md) for details.

**Total investment effects:**

$$
E_{e,inv} = E_{e,fix} + E_{e,spec} + E_{e,ret} + E_{e,pw}
$$

## Variables

| Symbol | Python Name | Description | When Created |
|--------|-------------|-------------|--------------|
| $P$ | `size` | Investment size | Always |
| $s_{inv}$ | `invested` | Binary decision | `mandatory=False` |

## Parameters

| Symbol | Python Name | Description | Default |
|--------|-------------|-------------|---------|
| $P_{fixed}$ | `fixed_size` | Fixed size (binary decision) | None |
| $P^{min}$ | `minimum_size` | Minimum size if investing | ε |
| $P^{max}$ | `maximum_size` | Maximum size | Big M |
| $c_{fix}$ | `effects_of_investment` | Fixed effects | None |
| $c_{spec}$ | `effects_of_investment_per_size` | Per-unit effects | None |
| $c_{ret}$ | `effects_of_retirement` | Effects if not investing | None |
| - | `mandatory` | Force investment | False |
| - | `piecewise_effects_of_investment` | Non-linear effects | None |

## Usage Examples

### Binary Investment (Solar)

```python
solar = fx.Flow(
    label='solar',
    bus=elec_bus,
    size=fx.InvestParameters(
        fixed_size=100,  # 100 kW or nothing
        effects_of_investment={'costs': 25000},      # Fixed: €25k
        effects_of_investment_per_size={'costs': 1200},  # €1200/kW
    ),
)
```

### Continuous Sizing (Battery)

```python
battery = fx.Storage(
    ...,
    capacity_in_flow_hours=fx.InvestParameters(
        minimum_size=10,   # At least 10 kWh
        maximum_size=1000, # At most 1 MWh
        effects_of_investment={'costs': 5000},       # Grid connection
        effects_of_investment_per_size={'costs': 600},  # €600/kWh
    ),
)
```

### With Retirement Costs

```python
boiler = fx.LinearConverter(
    ...,
    inputs=[fx.Flow(
        label='gas',
        bus=gas_bus,
        size=fx.InvestParameters(
            minimum_size=50,
            maximum_size=200,
            effects_of_investment_per_size={'costs': 400},
            effects_of_retirement={'costs': 8000},  # Demolition if not replaced
        ),
    )],
    ...
)
```

### Mandatory Investment

```python
upgrade = fx.Flow(
    label='grid',
    bus=elec_bus,
    size=fx.InvestParameters(
        minimum_size=100,
        maximum_size=500,
        mandatory=True,  # Must invest
        effects_of_investment_per_size={'costs': 1000},
    ),
)
```

## Cost Annualization

Investment costs must be annualized to match the optimization time horizon:

$$
c_{annual} = \frac{c_{capital} \cdot r}{1 - (1 + r)^{-n}}
$$

Where $r$ is the discount rate and $n$ is the lifetime (years).

!!! example "€1M equipment, 20 years, 5% discount"
    $c_{annual} = \frac{1{,}000{,}000 \cdot 0.05}{1 - 1.05^{-20}} \approx €80{,}243/year$

## Implementation Details

- **Feature Class:** [`InvestParameters`][flixopt.interface.InvestParameters]
- **Model Class:** [`InvestmentModel`][flixopt.features.InvestmentModel]

## See Also

- [Flow](../elements/Flow.md) — Using investment in flows
- [Storage](../elements/Storage.md) — Storage capacity investment
- [Piecewise](Piecewise.md) — Non-linear cost structures
- [Effects & Objective](../effects-penalty-objective.md) — How costs are tracked
