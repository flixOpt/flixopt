# LinearConverter

A LinearConverter defines linear relationships (ratios) between incoming and outgoing flows, representing energy/material conversion processes with fixed or variable efficiencies.

**Implementation:**

- **Component Class:** [`LinearConverter`][flixopt.components.LinearConverter]
- **Model Class:** [`LinearConverterModel`][flixopt.components.LinearConverterModel]

**Related:** [`Flow`](Flow.md) · [`Bus`](Bus.md) · [`Storage`](Storage.md)

---

=== "Core Formulation"

    ## General Linear Ratio Constraint

    The fundamental constraint relates all incoming and outgoing flows through linear conversion factors:

    $$ \label{eq:Linear-Transformer-Ratio}
        \sum_{f_{\text{in}} \in \mathcal F_{in}} \text a_{f_{\text{in}}}(\text{t}_i) \cdot p_{f_\text{in}}(\text{t}_i) = \sum_{f_{\text{out}} \in \mathcal F_{out}}  \text b_{f_\text{out}}(\text{t}_i) \cdot p_{f_\text{out}}(\text{t}_i)
    $$

    ??? info "Variables"
        | Symbol | Description | Domain |
        |--------|-------------|--------|
        | $p_{f_\text{in}}(\text{t}_i)$ | Flow rate of incoming flow $f_\text{in}$ at time $\text{t}_i$ | $\mathbb{R}$ |
        | $p_{f_\text{out}}(\text{t}_i)$ | Flow rate of outgoing flow $f_\text{out}$ at time $\text{t}_i$ | $\mathbb{R}$ |

    ??? info "Parameters"
        | Symbol | Description | Interpretation |
        |--------|-------------|----------------|
        | $\text a_{f_\text{in}}(\text{t}_i)$ | Conversion factor for incoming flow $f_\text{in}$ | Weight/efficiency coefficient |
        | $\text b_{f_\text{out}}(\text{t}_i)$ | Conversion factor for outgoing flow $f_\text{out}$ | Weight/efficiency coefficient |

    ??? info "Sets"
        | Symbol | Description |
        |--------|-------------|
        | $\mathcal F_{in}$ | Set of all incoming flows |
        | $\mathcal F_{out}$ | Set of all outgoing flows |

    ## Simplified Single-Input Single-Output Form

    For the common case of one incoming and one outgoing flow, equation $\eqref{eq:Linear-Transformer-Ratio}$ simplifies to:

    $$ \label{eq:Linear-Transformer-Ratio-simple}
        \text a(\text{t}_i) \cdot p_{f_\text{in}}(\text{t}_i) = p_{f_\text{out}}(\text{t}_i)
    $$

    **Physical Interpretation:**

    - $\text a$ represents the **conversion efficiency** or **conversion ratio**
    - For a heat pump: $\text a$ is the Coefficient of Performance (COP)
    - For a boiler: $\text a$ is the thermal efficiency (typically 0.85-0.95)
    - For a generator: $\text a$ is the electrical efficiency (typically 0.3-0.5)

=== "Advanced & Edge Cases"

    ## Piecewise Linear Conversion Factors

    Conversion efficiencies often vary with operating conditions (e.g., partial load efficiency, temperature-dependent COP). This is modeled using [Piecewise](../features/Piecewise.md) linear approximations.

    **Example:** Heat pump COP as a function of ambient temperature or load fraction.

    See [Piecewise](../features/Piecewise.md) for the mathematical formulation of piecewise linear relationships.

    ## Multiple Inputs/Outputs

    LinearConverters can model complex multi-flow processes:

    - **CHP (Combined Heat and Power):** 1 fuel input → 2 outputs (electricity + heat)
    - **Heat pump with auxiliary:** 2 inputs (electricity + ambient heat) → 1 output (useful heat)
    - **Multi-fuel boiler:** Multiple fuel inputs → 1 heat output

    Each flow has its own conversion factor $\text a_{f}$ or $\text b_{f}$ defining the ratio.

    ## Time-Varying Conversion Factors

    Conversion factors can be time-dependent $\text a(\text{t}_i)$ to model:

    - Seasonal efficiency variations
    - Temperature-dependent performance
    - Degradation over time
    - Scheduled maintenance periods

    ## Investment Sizing

    When using [InvestParameters](../features/InvestParameters.md), the converter size becomes an optimization variable. Flow sizes are then linked to the converter's optimized capacity.

    ## On/Off Operation

    Combining with [OnOffParameters](../features/OnOffParameters.md) allows modeling:

    - Minimum run times
    - Startup costs
    - Part-load restrictions (minimum load when operating)

=== "Mathematical Patterns"

    LinearConverter formulation relies on:

    - **Linear equality constraints** - Enforcing the ratio relationship
    - **[Scaled Bounds](../modeling-patterns/bounds-and-states.md#scaled-bounds)** - For flow rate bounds
    - **[Piecewise Linear Approximations](../features/Piecewise.md)** - For non-constant conversion factors

=== "Examples"

    ## Simple Boiler (Single Input/Output)

    ```python
    from flixopt import LinearConverter, Flow

    boiler = LinearConverter(
        label='gas_boiler',
        inputs=[Flow(label='gas_in', bus='natural_gas', size=100)],
        outputs=[Flow(label='heat_out', bus='heating', size=90)],
        conversion_factors={('gas_in', 'heat_out'): 0.9},  # 90% efficiency
    )
    ```

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
        conversion_factors={
            ('fuel_in', 'electricity_out'): 0.35,  # 35% electrical efficiency
            ('fuel_in', 'heat_out'): 0.55,  # 55% thermal efficiency
        },
    )
    ```

    ## Heat Pump with Temperature-Dependent COP

    ```python
    from flixopt import LinearConverter, Flow, Piecewise, Piece
    import numpy as np

    # COP varies from 2.5 (cold) to 4.0 (warm)
    cop_curve = Piecewise(
        [
            Piece((-10, 2.5), (0, 3.0)),   # -10°C to 0°C
            Piece((0, 3.0), (10, 3.5)),    # 0°C to 10°C
            Piece((10, 3.5), (20, 4.0)),   # 10°C to 20°C
        ]
    )

    heat_pump = LinearConverter(
        label='heat_pump',
        inputs=[Flow(label='electricity_in', bus='electricity', size=25)],
        outputs=[Flow(label='heat_out', bus='heating', size=100)],
        conversion_factors={('electricity_in', 'heat_out'): cop_curve},
    )
    ```

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
        conversion_factors={('electricity_in', 'h2_out'): 0.65},  # 65% efficiency
    )
    ```

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

## See Also

- **Elements:** [Flow](Flow.md) · [Bus](Bus.md) · [Storage](Storage.md)
- **Features:** [Piecewise](../features/Piecewise.md) · [InvestParameters](../features/InvestParameters.md) · [OnOffParameters](../features/OnOffParameters.md)
- **Patterns:** [Modeling Patterns](../modeling-patterns/index.md)
