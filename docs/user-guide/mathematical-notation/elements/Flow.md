# Flow

A Flow represents the transfer of energy or material between a Bus and a Component, with the flow rate as the primary optimization variable.

**Implementation:**

- **Element Class:** [`Flow`][flixopt.elements.Flow]
- **Model Class:** [`FlowModel`][flixopt.elements.FlowModel]

**Related:** [`Bus`](Bus.md) · [`Storage`](Storage.md) · [`LinearConverter`](LinearConverter.md)

---

=== "Core Formulation"

    ## Flow Rate Bounds

    The flow rate $p(\text{t}_{i})$ is constrained by the flow size and relative bounds:

    $$ \label{eq:flow_rate}
        \text P \cdot \text p^{\text{L}}_{\text{rel}}(\text{t}_{i})
        \leq p(\text{t}_{i}) \leq
        \text P \cdot \text p^{\text{U}}_{\text{rel}}(\text{t}_{i})
    $$

    ??? info "Variables"
        | Symbol | Description | Domain |
        |--------|-------------|--------|
        | $p(\text{t}_{i})$ | Flow rate at time $\text{t}_{i}$ | $\mathbb{R}$ |
        | $\text P$ | Flow size (capacity) | $\mathbb{R}_+$ or decision variable (with [InvestParameters](../features/InvestParameters.md)) |

    ??? info "Parameters"
        | Symbol | Description | Typical Value |
        |--------|-------------|---------------|
        | $\text p^{\text{L}}_{\text{rel}}(\text{t}_{i})$ | Relative lower bound at time $\text{t}_{i}$ | 0 |
        | $\text p^{\text{U}}_{\text{rel}}(\text{t}_{i})$ | Relative upper bound at time $\text{t}_{i}$ | 1 |

    ## Simplified Form

    With standard bounds $\text p^{\text{L}}_{\text{rel}}(\text{t}_{i}) = 0$ and $\text p^{\text{U}}_{\text{rel}}(\text{t}_{i}) = 1$, equation \eqref{eq:flow_rate} simplifies to:

    $$
        0 \leq p(\text{t}_{i}) \leq \text P
    $$

=== "Advanced & Edge Cases"

    ## Extensions with Optional Features

    The basic flow formulation can be extended with additional constraints:

    ### On/Off Operation

    When combined with [OnOffParameters](../features/OnOffParameters.md), the flow gains binary on/off states with startup costs, minimum run times, and switching constraints. The formulation becomes:

    - **[Scaled Bounds with State](../modeling-patterns/bounds-and-states.md#scaled-bounds-with-state)**

    ### Investment Sizing

    When using [InvestParameters](../features/InvestParameters.md), the flow size $\text P$ becomes an optimization variable rather than a fixed parameter:

    - **[Bounds with State](../modeling-patterns/bounds-and-states.md#bounds-with-state)**

    ### Load Factor Constraints

    Minimum and maximum average utilization over time periods:

    $$
        \text{load\_factor\_min} \cdot \text P \cdot N_t \leq \sum_{i=1}^{N_t} p(\text{t}_{i}) \leq \text{load\_factor\_max} \cdot \text P \cdot N_t
    $$

    Where $N_t$ is the number of timesteps in the period.

    ### Flow Hours Limits

    Direct constraints on cumulative flow-hours (energy/material throughput):

    $$
        \text{flow\_hours\_min} \leq \sum_{i=1}^{N_t} p(\text{t}_{i}) \cdot \Delta t_i \leq \text{flow\_hours\_max}
    $$

    ### Fixed Relative Profile

    When a predetermined pattern is specified, the flow rate becomes:

    $$
        p(\text{t}_{i}) = \text P \cdot \text{profile}(\text{t}_{i})
    $$

    This is commonly used for renewable generation (solar, wind) or fixed demand patterns.

    ### Initial Flow State

    For flows with on/off parameters, the previous flow rate can be specified to properly model startup/shutdown transitions at the beginning of the optimization horizon:

    - `previous_flow_rate`: Initial condition for on/off dynamics (default: None, interpreted as off/zero flow)

=== "Mathematical Patterns"

    Flow formulation builds on the following modeling patterns:

    - **[Scaled Bounds](../modeling-patterns/bounds-and-states.md#scaled-bounds)** - Basic flow rate constraint (equation $\eqref{eq:flow_rate}$)
    - **[Scaled Bounds with State](../modeling-patterns/bounds-and-states.md#scaled-bounds-with-state)** - When combined with [OnOffParameters](../features/OnOffParameters.md)
    - **[Bounds with State](../modeling-patterns/bounds-and-states.md#bounds-with-state)** - Investment decisions with [InvestParameters](../features/InvestParameters.md)

    See [Modeling Patterns](../modeling-patterns/index.md) for detailed explanations of these building blocks.

=== "Examples"

    ## Basic Fixed Capacity Flow

    ```python
    from flixopt import Flow

    generator_output = Flow(
        label='electricity_out',
        bus='electricity_grid',
        size=100,  # 100 MW capacity
        relative_minimum=0.4,  # Cannot operate below 40 MW
        effects_per_flow_hour={'fuel_cost': 45, 'co2_emissions': 0.8},
    )
    ```

    ## Investment Decision

    ```python
    from flixopt import Flow, InvestParameters

    battery_flow = Flow(
        label='electricity_storage',
        bus='electricity_grid',
        size=InvestParameters(
            minimum_size=10,  # Minimum 10 MWh
            maximum_size=100,  # Maximum 100 MWh
            specific_effects={'cost': 150_000},  # €150k/MWh annualized
        ),
    )
    ```

    ## On/Off Operation with Constraints

    ```python
    from flixopt import Flow, OnOffParameters

    heat_pump_flow = Flow(
        label='heat_output',
        bus='heating_network',
        size=50,  # 50 kW thermal
        relative_minimum=0.3,  # Minimum 15 kW when on
        effects_per_flow_hour={'electricity_cost': 25},
        on_off_parameters=OnOffParameters(
            effects_per_switch_on={'startup_cost': 100},
            consecutive_on_hours_min=2,  # Min run time
            consecutive_off_hours_min=1,  # Min off time
        ),
    )
    ```

    ## Fixed Profile (Renewable Generation)

    ```python
    import numpy as np
    from flixopt import Flow

    solar_generation = Flow(
        label='solar_power',
        bus='electricity_grid',
        size=25,  # 25 MW installed
        fixed_relative_profile=np.array([0, 0.1, 0.4, 0.8, 0.9, 0.7, 0.3, 0.1, 0]),
    )
    ```

---

## See Also

- **Features:** [OnOffParameters](../features/OnOffParameters.md) · [InvestParameters](../features/InvestParameters.md)
- **Elements:** [Bus](Bus.md) · [Storage](Storage.md) · [LinearConverter](LinearConverter.md)
- **Patterns:** [Modeling Patterns](../modeling-patterns/index.md)
- **Effects:** [Effects, Penalty & Objective](../effects-penalty-objective.md)
