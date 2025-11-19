# InvestParameters

InvestParameters enable investment decision modeling in optimization, supporting both binary (invest/don't invest) and continuous sizing choices with comprehensive cost modeling.

=== "Variables"

    | Symbol | Python Name | Description | Domain | Created When |
    |--------|-------------|-------------|--------|--------------|
    | $v_\text{invest}$ | `size` | Investment size (continuous or fixed) | $\mathbb{R}_+$ | Always |
    | $s_\text{invest}$ | `invested` | Binary investment decision | $\{0,1\}$ | `mandatory=False` (optional investment) |

=== "Constraints"

    **Binary investment decision** (when `fixed_size` specified):

    $$\label{eq:invest_binary}
    v_\text{invest} = s_\text{invest} \cdot \text{size}_\text{fixed}
    $$

    ---

    **Continuous sizing decision** (when `minimum_size` and `maximum_size` specified):

    $$\label{eq:invest_continuous}
    s_\text{invest} \cdot \text{size}_\text{min} \leq v_\text{invest} \leq s_\text{invest} \cdot \text{size}_\text{max}
    $$

    When `mandatory=False`: $s_\text{invest} = 0$ means no investment ($v_\text{invest} = 0$), $s_\text{invest} = 1$ means invest with size in $[\text{size}_\text{min}, \text{size}_\text{max}]$

    ---

    **Mandatory investment** (when `mandatory=True`):

    $$\label{eq:invest_mandatory}
    s_\text{invest} = 1
    $$

    ---

    **Investment effects - Fixed** (when `effects_of_investment` specified):

    $$\label{eq:invest_fixed_effects}
    E_{e,\text{fix}} = s_\text{invest} \cdot \text{fix}_e
    $$

    One-time effects incurred if investment is made, independent of size (permits, grid connection, one-time environmental impacts).

    ---

    **Investment effects - Specific** (when `effects_of_investment_per_size` specified):

    $$\label{eq:invest_specific_effects}
    E_{e,\text{spec}} = v_\text{invest} \cdot \text{spec}_e
    $$

    Effects proportional to investment size (equipment costs €/kW, material requirements kg/kW, recurring maintenance €/kW/year).

    ---

    **Retirement effects** (when `effects_of_retirement` specified and `mandatory=False`):

    $$\label{eq:invest_retirement_effects}
    E_{e,\text{retirement}} = (1 - s_\text{invest}) \cdot \text{retirement}_e
    $$

    Effects incurred if investment is NOT made (demolition/disposal costs, decommissioning, penalties, opportunity costs).

    ---

    **Piecewise effects** (when `piecewise_effects_of_investment` specified):

    $$\label{eq:invest_piecewise_effects}
    E_{e,\text{pw}} = \sum_{k=1}^{K} \lambda_k \cdot r_{e,k}
    $$

    Subject to:

    $$
    v_\text{invest} = \sum_{k=1}^{K} \lambda_k \cdot v_k
    $$

    Non-linear effect relationships using piecewise linear approximations for economies of scale, technology learning curves, or threshold effects. See [Piecewise](Piecewise.md) for detailed formulation.

    ---

    **Total investment effects**:

    $$\label{eq:invest_total_effects}
    E_{e,\text{invest}} = E_{e,\text{fix}} + E_{e,\text{spec}} + E_{e,\text{pw}} + E_{e,\text{retirement}}
    $$

    **Mathematical Patterns:** [Bounds with State](../modeling-patterns/bounds-and-states.md#bounds-with-state), [Scaled Bounds](../modeling-patterns/bounds-and-states.md#scaled-bounds), [Piecewise Linear Approximations](Piecewise.md)

=== "Parameters"

    | Symbol | Python Parameter | Description | Default |
    |--------|------------------|-------------|---------|
    | $\text{fix}_e$ | `effects_of_investment` | Fixed effects if investment is made | None |
    | $\text{retirement}_e$ | `effects_of_retirement` | Effects if NOT investing | None |
    | $\text{size}_\text{fixed}$ | `fixed_size` | Predefined component size (binary decision) | None |
    | $\text{size}_\text{max}$ | `maximum_size` | Maximum investment size | CONFIG.Modeling.big |
    | $\text{size}_\text{min}$ | `minimum_size` | Minimum investment size (if investing) | CONFIG.Modeling.epsilon |
    | $\text{spec}_e$ | `effects_of_investment_per_size` | Per-unit effects proportional to size | None |
    | - | `linked_periods` | Describes which periods are linked for multi-period optimization | None |
    | - | `mandatory` | If True, investment must occur | False |
    | - | `piecewise_effects_of_investment` | PiecewiseEffects for non-linear cost structures | None |

=== "Use Cases"

    ## Binary Investment (Solar Panels)

    ```python
    from flixopt import Flow, InvestParameters

    solar_flow = Flow(
        label='solar_generation',
        bus='electricity',
        size=InvestParameters(
            fixed_size=100,  # 100 kW system (all or nothing)
            mandatory=False,  # Optional investment
            effects_of_investment={'cost': 25000},  # Fixed installation
            effects_of_investment_per_size={'cost': 1200},  # €1200/kW
        ),
    )
    ```

    **Variables:** `size`, `invested` (binary)

    **Constraints:** $\eqref{eq:invest_binary}$ with $v_\text{invest} = s_\text{invest} \cdot 100$ kW, $\eqref{eq:invest_fixed_effects}$ with fixed cost of €25,000, $\eqref{eq:invest_specific_effects}$ with €1,200/kW

    ---

    ## Continuous Sizing (Battery)

    ```python
    from flixopt import Storage, Flow, InvestParameters

    battery = Storage(
        label='battery_storage',
        charging=Flow(label='charge', bus='electricity', size=100),
        discharging=Flow(label='discharge', bus='electricity', size=100),
        capacity_in_flow_hours=InvestParameters(
            minimum_size=10,  # Minimum 10 kWh if investing
            maximum_size=1000,  # Maximum 1 MWh
            effects_of_investment={'cost': 5000},  # Grid connection
            effects_of_investment_per_size={'cost': 600},  # €600/kWh
        ),
    )
    ```

    **Variables:** `size` (continuous), `invested` (binary)

    **Constraints:** $\eqref{eq:invest_continuous}$ with $s_\text{invest} \cdot 10 \leq v_\text{invest} \leq s_\text{invest} \cdot 1000$ kWh, $\eqref{eq:invest_fixed_effects}$, $\eqref{eq:invest_specific_effects}$

    ---

    ## With Retirement Costs (Replacement Decision)

    ```python
    from flixopt import LinearConverter, Flow, InvestParameters

    boiler = LinearConverter(
        label='boiler_replacement',
        inputs=[Flow(
            label='gas_in',
            bus='natural_gas',
            size=InvestParameters(
                minimum_size=50,  # 50 kW minimum
                maximum_size=200,  # 200 kW maximum
                effects_of_investment={'cost': 15000},  # New installation
                effects_of_investment_per_size={'cost': 400},  # €400/kW
                effects_of_retirement={'cost': 8000},  # Demolition if not replaced
            ),
        )],
        outputs=[Flow(label='heat_out', bus='heating', size=1)],
        conversion_factors=[{'gas_in': 0.9, 'heat_out': 1}],
    )
    ```

    **Variables:** `size`, `invested`

    **Constraints:** $\eqref{eq:invest_continuous}$, $\eqref{eq:invest_fixed_effects}$, $\eqref{eq:invest_specific_effects}$, $\eqref{eq:invest_retirement_effects}$ with €8,000 cost if $s_\text{invest} = 0$

    ---

    ## Economies of Scale (Piecewise Costs)

    ```python
    from flixopt import Storage, Flow, InvestParameters, Piecewise, Piece, PiecewiseEffects

    battery_investment = InvestParameters(
        minimum_size=10,
        maximum_size=1000,
        piecewise_effects_of_investment=PiecewiseEffects(
            piecewise_origin=Piecewise([
                Piece((0, 0), (100, 100)),    # Small (0-100 kWh)
                Piece((100, 100), (500, 500)),  # Medium (100-500 kWh)
                Piece((500, 500), (1000, 1000)),  # Large (500-1000 kWh)
            ]),
            piecewise_shares={
                'cost': Piecewise([
                    Piece((0, 0), (100, 80000)),  # €800/kWh (small)
                    Piece((100, 80000), (500, 350000)),  # €750-€600/kWh (medium)
                    Piece((500, 350000), (1000, 850000)),  # €600-€500/kWh (large, bulk discount)
                ])
            },
        ),
    )
    ```

    **Variables:** `size`, `invested`, piecewise lambda variables $\lambda_k$

    **Constraints:** $\eqref{eq:invest_continuous}$, $\eqref{eq:invest_piecewise_effects}$ representing decreasing cost per kWh with scale (bulk discount from €800/kWh to €500/kWh)

    ---

    ## Mandatory Investment (Upgrade Required)

    ```python
    from flixopt import Flow, InvestParameters

    grid_upgrade = Flow(
        label='grid_connection',
        bus='electricity',
        size=InvestParameters(
            minimum_size=100,
            maximum_size=500,
            mandatory=True,  # Must upgrade
            effects_of_investment_per_size={'cost': 1000},  # €1000/kW
        ),
    )
    ```

    **Variables:** `size` (no binary variable since mandatory)

    **Constraints:** $\eqref{eq:invest_mandatory}$ forcing $s_\text{invest} = 1$, $\eqref{eq:invest_continuous}$ simplified to $100 \leq v_\text{invest} \leq 500$ kW

---

## Cost Annualization

**Important:** All investment cost values must be properly weighted to match the optimization model's time horizon.

For long-term investments, costs should be annualized:

$$\label{eq:annualization}
\text{cost}_\text{annual} = \frac{\text{cost}_\text{capital} \cdot r}{1 - (1 + r)^{-n}}
$$

Where $r$ is the discount rate and $n$ is the equipment lifetime (years).

**Example:** €1,000,000 equipment with 20-year life and 5% discount rate:

$$
\text{cost}_\text{annual} = \frac{1{,}000{,}000 \cdot 0.05}{1 - (1.05)^{-20}} \approx €80{,}243/\text{year}
$$

---

## Implementation

- **Feature Class:** [`InvestParameters`][flixopt.interface.InvestParameters]
- **Model Class:** [`InvestmentModel`][flixopt.features.InvestmentModel]
- **Used by:** [`Flow`](../elements/Flow.md) · [`Storage`](../elements/Storage.md) · [`LinearConverter`](../elements/LinearConverter.md)

## See Also

- **Elements:** [Flow](../elements/Flow.md) · [Storage](../elements/Storage.md) · [LinearConverter](../elements/LinearConverter.md)
- **Features:** [OnOffParameters](OnOffParameters.md) · [Piecewise](Piecewise.md)
- **Patterns:** [Bounds and States](../modeling-patterns/bounds-and-states.md)
- **System-Level:** [Effects, Penalty & Objective](../effects-penalty-objective.md)
