# LinearConverter

A LinearConverter defines linear relationships (ratios) between incoming and outgoing flows, representing energy/material conversion processes with fixed or variable efficiencies.

=== "Variables"

    | Symbol | Python Name | Description | Domain | Created When |
    |--------|-------------|-------------|--------|--------------|
    | $p_{f_\text{in}}(\text{t}_i)$ | - | Flow rate of incoming flow $f_\text{in}$ at time $\text{t}_i$ | $\mathbb{R}$ | Always (from input Flows) |
    | $p_{f_\text{out}}(\text{t}_i)$ | - | Flow rate of outgoing flow $f_\text{out}$ at time $\text{t}_i$ | $\mathbb{R}$ | Always (from output Flows) |
    | $\lambda_k$ | - | Piecewise lambda variables | $\mathbb{R}_+$ | `piecewise_conversion` is specified |

=== "Constraints"

    **General linear ratio constraint** (when `conversion_factors` specified):

    $$\label{eq:Linear-Transformer-Ratio}
    \sum_{f_{\text{in}} \in \mathcal F_{in}} \text a_{f_{\text{in}}}(\text{t}_i) \cdot p_{f_\text{in}}(\text{t}_i) = \sum_{f_{\text{out}} \in \mathcal F_{out}}  \text b_{f_\text{out}}(\text{t}_i) \cdot p_{f_\text{out}}(\text{t}_i)
    $$

    ---

    **Simplified single-input single-output** (special case of above):

    $$\label{eq:Linear-Transformer-Ratio-simple}
    \text a(\text{t}_i) \cdot p_{f_\text{in}}(\text{t}_i) = p_{f_\text{out}}(\text{t}_i)
    $$

    Where $\text a$ represents the conversion efficiency or conversion ratio (COP for heat pumps, thermal efficiency for boilers, electrical efficiency for generators).

    ---

    **Piecewise linear conversion** (when `piecewise_conversion` specified):

    See [Piecewise](../features/Piecewise.md) for the detailed mathematical formulation of piecewise linear relationships between flows.

    **Mathematical Patterns:** [Linear Equality Constraints](../modeling-patterns/bounds-and-states.md), [Piecewise Linear Approximations](../features/Piecewise.md)

=== "Parameters"

    | Symbol | Python Parameter | Description | Default |
    |--------|------------------|-------------|---------|
    | $\mathcal F_{in}$ | `inputs` | Set of all incoming flows | Required |
    | $\mathcal F_{out}$ | `outputs` | Set of all outgoing flows | Required |
    | $\text a_{f_\text{in}}(\text{t}_i)$ | `conversion_factors` | Conversion factor for incoming flow (weight/efficiency coefficient) | None |
    | $\text b_{f_\text{out}}(\text{t}_i)$ | `conversion_factors` | Conversion factor for outgoing flow (weight/efficiency coefficient) | None |
    | - | `on_off_parameters` | OnOffParameters for on/off operation | None |
    | - | `piecewise_conversion` | PiecewiseConversion for non-linear behavior | None |

=== "Use Cases"

    ## Simple Boiler (Single Input/Output)

    ```python
    from flixopt import LinearConverter, Flow

    boiler = LinearConverter(
        label='gas_boiler',
        inputs=[Flow(label='gas_in', bus='natural_gas', size=100)],
        outputs=[Flow(label='heat_out', bus='heating', size=90)],
        conversion_factors=[{'gas_in': 0.9, 'heat_out': 1}],  # 90% efficiency
    )
    ```

    **Variables:** Flow rates from input and output flows

    **Constraints:** $\eqref{eq:Linear-Transformer-Ratio-simple}$ with $0.9 \cdot p_\text{gas}(t) = p_\text{heat}(t)$, representing 90% thermal efficiency

    ---

    ## CHP Plant (One Input, Two Outputs)

    ```python
    from flixopt import LinearConverter, Flow

    chp = LinearConverter(
        label='chp_unit',
        inputs=[Flow(label='fuel_in', bus='natural_gas', size=100)],
        outputs=[
            Flow(label='electricity_out', bus='electricity', size=35),
            Flow(label='heat_out', bus='heating', size=55),
        ],
        conversion_factors=[
            {'fuel_in': 0.35, 'electricity_out': 1},  # 35% electrical efficiency
            {'fuel_in': 0.55, 'heat_out': 1},  # 55% thermal efficiency
        ],
    )
    ```

    **Variables:** Flow rates from one input and two output flows

    **Constraints:** Two instances of $\eqref{eq:Linear-Transformer-Ratio}$:
    - $0.35 \cdot p_\text{fuel}(t) = p_\text{elec}(t)$
    - $0.55 \cdot p_\text{fuel}(t) = p_\text{heat}(t)$

    ---

    ## Heat Pump with Temperature-Dependent COP

    ```python
    from flixopt import LinearConverter, Flow, Piecewise, Piece, PiecewiseConversion

    # COP varies from 2.5 (cold) to 4.0 (warm)
    cop_curve = Piecewise(
        [
            Piece((0, 0), (100, 250)),     # 0-100 kW elec → 0-250 kW heat (COP 2.5)
            Piece((100, 250), (200, 600)),  # 100-200 kW elec → 250-600 kW heat (COP 3.5)
        ]
    )

    heat_pump = LinearConverter(
        label='heat_pump',
        inputs=[Flow(label='electricity_in', bus='electricity', size=200)],
        outputs=[Flow(label='heat_out', bus='heating', size=600)],
        piecewise_conversion=PiecewiseConversion(
            origin_flow='electricity_in',
            piecewise_shares={'heat_out': cop_curve},
        ),
    )
    ```

    **Variables:** Flow rates + piecewise lambda variables $\lambda_k$

    **Constraints:** Piecewise constraints linking electricity input to heat output with variable COP (see [Piecewise](../features/Piecewise.md))

    ---

    ## Converter with Investment Decision

    ```python
    from flixopt import LinearConverter, Flow, InvestParameters

    electrolyzer = LinearConverter(
        label='hydrogen_electrolyzer',
        inputs=[Flow(
            label='electricity_in',
            bus='electricity',
            size=InvestParameters(
                minimum_size=0,
                maximum_size=1000,  # Up to 1 MW
                specific_effects={'cost': 800},  # €800/kW annualized
            ),
        )],
        outputs=[Flow(label='h2_out', bus='hydrogen', size=1)],  # Sized by ratio
        conversion_factors=[{'electricity_in': 0.65, 'h2_out': 1}],  # 65% efficiency
    )
    ```

    **Variables:** Flow rates + `size` variable for investment

    **Constraints:** $\eqref{eq:Linear-Transformer-Ratio-simple}$ plus investment constraints from [InvestParameters](../features/InvestParameters.md)

---

## Specialized LinearConverter Classes

FlixOpt provides specialized classes that automatically configure the conversion factors based on physical relationships:

- **[`HeatPump`][flixopt.linear_converters.HeatPump]** - COP-based electric heating
- **[`Power2Heat`][flixopt.linear_converters.Power2Heat]** - Direct electric heating (efficiency ≤ 1)
- **[`CHP`][flixopt.linear_converters.CHP]** - Combined heat and power generation
- **[`Boiler`][flixopt.linear_converters.Boiler]** - Fuel to heat conversion
- **[`HeatPumpWithSource`][flixopt.linear_converters.HeatPumpWithSource]** - Heat pump with explicit ambient source

These provide more intuitive interfaces for common applications.

---

## Implementation

- **Component Class:** [`LinearConverter`][flixopt.components.LinearConverter]
- **Model Class:** [`LinearConverterModel`][flixopt.components.LinearConverterModel]

## See Also

- **Elements:** [Flow](Flow.md) · [Bus](Bus.md) · [Storage](Storage.md)
- **Features:** [Piecewise](../features/Piecewise.md) · [InvestParameters](../features/InvestParameters.md) · [OnOffParameters](../features/OnOffParameters.md)
- **Patterns:** [Modeling Patterns](../modeling-patterns/index.md)
