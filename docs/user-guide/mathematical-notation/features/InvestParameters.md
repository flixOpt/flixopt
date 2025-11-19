# InvestParameters

InvestParameters enable investment decision modeling in optimization, supporting both binary (invest/don't invest) and continuous sizing choices with comprehensive cost modeling.

**Implementation:**

- **Feature Class:** [`InvestParameters`][flixopt.interface.InvestParameters]
- **Used by:** [`Flow`][flixopt.elements.Flow] · [`Storage`][flixopt.components.Storage] · [`LinearConverter`][flixopt.components.LinearConverter]

**Related:** [`OnOffParameters`](OnOffParameters.md) · [`Piecewise`](Piecewise.md)

---

=== "Core Formulation"

    ## Binary Investment Decision

    Fixed-size investment creating a yes/no decision (e.g., install a 100 kW generator):

    $$\label{eq:invest_binary}
    v_\text{invest} = s_\text{invest} \cdot \text{size}_\text{fixed}
    $$

    ??? info "Variables"
        | Symbol | Description | Domain |
        |--------|-------------|--------|
        | $s_\text{invest}$ | Binary investment decision | $\{0, 1\}$ |
        | $v_\text{invest}$ | Resulting investment size | $\mathbb{R}_+$ |

    ??? info "Parameters"
        | Symbol | Description |
        |--------|-------------|
        | $\text{size}_\text{fixed}$ | Predefined component size |

    **Behavior:**

    - $s_\text{invest} = 0$: no investment ($v_\text{invest} = 0$)
    - $s_\text{invest} = 1$: invest at fixed size ($v_\text{invest} = \text{size}_\text{fixed}$)

    ## Continuous Sizing Decision

    Variable-size investment with bounds (e.g., battery capacity from 10-1000 kWh):

    $$\label{eq:invest_continuous}
    s_\text{invest} \cdot \text{size}_\text{min} \leq v_\text{invest} \leq s_\text{invest} \cdot \text{size}_\text{max}
    $$

    ??? info "Variables"
        | Symbol | Description | Domain |
        |--------|-------------|--------|
        | $s_\text{invest}$ | Binary investment decision | $\{0, 1\}$ |
        | $v_\text{invest}$ | Investment size (continuous) | $\mathbb{R}_+$ |

    ??? info "Parameters"
        | Symbol | Description |
        |--------|-------------|
        | $\text{size}_\text{min}$ | Minimum investment size (if investing) |
        | $\text{size}_\text{max}$ | Maximum investment size |

    **Behavior:**

    - $s_\text{invest} = 0$: no investment ($v_\text{invest} = 0$)
    - $s_\text{invest} = 1$: invest with size in $[\text{size}_\text{min}, \text{size}_\text{max}]$

    This uses the **[Bounds with State](../modeling-patterns/bounds-and-states.md#bounds-with-state)** pattern.

    ## Investment Effects (Costs)

    ### Fixed Effects

    One-time effects incurred if investment is made, independent of size:

    $$\label{eq:invest_fixed_effects}
    E_{e,\text{fix}} = s_\text{invest} \cdot \text{fix}_e
    $$

    **Examples:** Fixed installation costs (permits, grid connection), one-time environmental impacts.

    ### Specific Effects (Per-Unit Costs)

    Effects proportional to investment size:

    $$\label{eq:invest_specific_effects}
    E_{e,\text{spec}} = v_\text{invest} \cdot \text{spec}_e
    $$

    **Examples:** Equipment costs (€/kW), material requirements (kg steel/kW), recurring maintenance (€/kW/year).

    ### Total Investment Effects

    $$\label{eq:invest_total_effects}
    E_{e,\text{invest}} = E_{e,\text{fix}} + E_{e,\text{spec}} + E_{e,\text{pw}} + E_{e,\text{retirement}}
    $$

    Where $E_{e,\text{pw}}$ is the piecewise contribution (see Advanced tab) and $E_{e,\text{retirement}}$ are retirement effects (see Advanced tab).

=== "Advanced & Edge Cases"

    ## Optional vs. Mandatory Investment

    The `mandatory` parameter controls whether investment is required:

    **Optional Investment** (`mandatory=False`, default):
    $$\label{eq:invest_optional}
    s_\text{invest} \in \{0, 1\}
    $$

    The optimization can freely choose to invest or not.

    **Mandatory Investment** (`mandatory=True`):
    $$\label{eq:invest_mandatory}
    s_\text{invest} = 1
    $$

    The investment must occur (useful for mandatory upgrades or replacements).

    ## Retirement Effects

    Effects incurred if investment is NOT made (when retiring/not replacing existing equipment):

    $$\label{eq:invest_retirement_effects}
    E_{e,\text{retirement}} = (1 - s_\text{invest}) \cdot \text{retirement}_e
    $$

    **Behavior:**

    - $s_\text{invest} = 0$: retirement effects are incurred
    - $s_\text{invest} = 1$: no retirement effects

    **Examples:** Demolition/disposal costs, decommissioning expenses, contractual penalties, opportunity costs.

    ## Piecewise Effects (Economies of Scale)

    Non-linear effect relationships using piecewise linear approximations:

    $$\label{eq:invest_piecewise_effects}
    E_{e,\text{pw}} = \sum_{k=1}^{K} \lambda_k \cdot r_{e,k}
    $$

    Subject to:
    $$
    v_\text{invest} = \sum_{k=1}^{K} \lambda_k \cdot v_k
    $$

    ??? info "Piecewise Variables"
        | Symbol | Description |
        |--------|-------------|
        | $\lambda_k$ | Piecewise lambda variables (see [Piecewise](Piecewise.md)) |
        | $r_{e,k}$ | Effect rate at piece $k$ |
        | $v_k$ | Size points defining the pieces |

    **Use cases:** Economies of scale (bulk discounts), technology learning curves, threshold effects.

    See [Piecewise](Piecewise.md) for detailed mathematical formulation.

    ## Integration with Component Sizing

    Investment parameters modify component sizing:

    **Without Investment:**
    $$
    \text{size} = \text{size}_\text{nominal}
    $$

    **With Investment:**
    $$
    \text{size} = v_\text{invest}
    $$

    This size variable then appears in component constraints. For example, flow rate bounds become:

    $$
    v_\text{invest} \cdot \text{rel}_\text{lower} \leq p(t) \leq v_\text{invest} \cdot \text{rel}_\text{upper}
    $$

    Using the **[Scaled Bounds](../modeling-patterns/bounds-and-states.md#scaled-bounds)** pattern.

    ## Cost Annualization

    **Important:** All investment cost values must be properly weighted to match the optimization model's time horizon.

    For long-term investments, costs should be annualized:

    $$\label{eq:annualization}
    \text{cost}_\text{annual} = \frac{\text{cost}_\text{capital} \cdot r}{1 - (1 + r)^{-n}}
    $$

    ??? info "Annualization Parameters"
        | Symbol | Description |
        |--------|-------------|
        | $\text{cost}_\text{capital}$ | Upfront investment cost |
        | $r$ | Discount rate |
        | $n$ | Equipment lifetime (years) |

    **Example:** €1,000,000 equipment with 20-year life and 5% discount rate
    $$
    \text{cost}_\text{annual} = \frac{1{,}000{,}000 \cdot 0.05}{1 - (1.05)^{-20}} \approx €80{,}243/\text{year}
    $$

=== "Mathematical Patterns"

    InvestParameters relies on:

    - **[Bounds with State](../modeling-patterns/bounds-and-states.md#bounds-with-state)** - For continuous sizing with binary investment decision
    - **[Scaled Bounds](../modeling-patterns/bounds-and-states.md#scaled-bounds)** - For linking investment size to component constraints
    - **[Piecewise Linear Approximations](Piecewise.md)** - For non-linear cost structures

=== "Examples"

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

    ## Continuous Sizing (Battery)

    ```python
    from flixopt import Storage, Flow, InvestParameters

    battery = Storage(
        label='battery_storage',
        inputs=[Flow(label='charge', bus='electricity', size=100)],
        outputs=[Flow(label='discharge', bus='electricity', size=100)],
        capacity_in_flow_hours=InvestParameters(
            minimum_size=10,  # Minimum 10 kWh if investing
            maximum_size=1000,  # Maximum 1 MWh
            effects_of_investment={'cost': 5000},  # Grid connection
            effects_of_investment_per_size={'cost': 600},  # €600/kWh
        ),
    )
    ```

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
        conversion_factors={('gas_in', 'heat_out'): 0.9},
    )
    ```

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
                    Piece((0, 800), (100, 750)),  # €800-750/kWh (small)
                    Piece((100, 750), (500, 600)),  # €750-600/kWh (medium)
                    Piece((500, 600), (1000, 500)),  # €600-500/kWh (large, bulk discount)
                ])
            },
        ),
    )
    ```

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

---

## See Also

- **Elements:** [Flow](../elements/Flow.md) · [Storage](../elements/Storage.md) · [LinearConverter](../elements/LinearConverter.md)
- **Features:** [OnOffParameters](OnOffParameters.md) · [Piecewise](Piecewise.md)
- **Patterns:** [Bounds and States](../modeling-patterns/bounds-and-states.md)
- **System-Level:** [Effects, Penalty & Objective](../effects-penalty-objective.md)
