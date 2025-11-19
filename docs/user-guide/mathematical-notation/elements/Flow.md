# Flow

A Flow represents the transfer of energy or material between a Bus and a Component, with the flow rate as the primary optimization variable.

=== "Variables"

    | Variable Name | Symbol | Description | Domain | Created When |
    |---------------|--------|-------------|--------|--------------|
    | **flow_rate** | $p(\text{t}_{i})$ | Flow rate at each timestep | $\mathbb{R}$ | Always |
    | **size** | $\text P$ | Flow capacity (decision variable) | $\mathbb{R}_+$ | `size` is `InvestParameters` |
    | **invest_binary** | $s_\text{invest}$ | Binary investment decision | $\{0,1\}$ | `size` is `InvestParameters` |
    | **total_flow_hours** | - | Cumulative flow-hours per period | $\mathbb{R}_+$ | `flow_hours_min/max` or `load_factor_min/max` specified |
    | **on_off_state** | $s(\text{t}_i)$ | Binary on/off state | $\{0,1\}$ | `on_off_parameters` specified |
    | **switch_on** | - | Startup indicator | $\{0,1\}$ | `on_off_parameters` specified |
    | **switch_off** | - | Shutdown indicator | $\{0,1\}$ | `on_off_parameters` specified |

=== "Constraints"

    | Constraint | Equation | Parameters | Active When |
    |------------|----------|------------|-------------|
    | **Flow rate bounds** | $\text P \cdot \text p^{\text{L}}_{\text{rel}}(\text{t}_{i}) \leq p(\text{t}_{i}) \leq \text P \cdot \text p^{\text{U}}_{\text{rel}}(\text{t}_{i})$ | `size` ($\text P$), `relative_minimum` ($\text p^{\text{L}}_{\text{rel}}$, default: 0), `relative_maximum` ($\text p^{\text{U}}_{\text{rel}}$, default: 1) | Always |
    | **Load factor** | $\text{LF}_\text{min} \cdot \text P \cdot N_t \leq \sum_{i=1}^{N_t} p(\text{t}_{i}) \leq \text{LF}_\text{max} \cdot \text P \cdot N_t$ | `load_factor_min` ($\text{LF}_\text{min}$, default: 0), `load_factor_max` ($\text{LF}_\text{max}$, default: 1) | When `load_factor_min` or `load_factor_max` specified |
    | **Flow hours limits** | $\text{FH}_\text{min} \leq \sum_{i=1}^{N_t} p(\text{t}_{i}) \cdot \Delta t_i \leq \text{FH}_\text{max}$ | `flow_hours_min`, `flow_hours_max`, `flow_hours_min_over_periods`, `flow_hours_max_over_periods` | When any flow hours parameter specified |
    | **Fixed profile** | $p(\text{t}_{i}) = \text P \cdot \text{profile}(\text{t}_{i})$ | `fixed_relative_profile` (array) | When `fixed_relative_profile` specified |
    | **On/off operation** | See [OnOffParameters](../features/OnOffParameters.md) | `on_off_parameters` | When `on_off_parameters` specified |
    | **Initial conditions** | - | `previous_flow_rate` (default: None/0) | When `on_off_parameters` specified |

    **Mathematical Patterns:** [Scaled Bounds](../modeling-patterns/bounds-and-states.md#scaled-bounds), [Scaled Bounds with State](../modeling-patterns/bounds-and-states.md#scaled-bounds-with-state)

=== "Use Cases"

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

    **Variables:** `flow_rate[t]`
    **Constraints:** $40 \leq p(t) \leq 100$ MW

    ---

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

    **Variables:** `flow_rate[t]`, `size`, `invest_binary`
    **Constraints:** If investing: $10 \leq \text{size} \leq 100$; Flow bounds: $0 \leq p(t) \leq \text{size}$

    ---

    ## On/Off Operation

    ```python
    from flixopt import Flow, OnOffParameters

    heat_pump_flow = Flow(
        label='heat_output',
        bus='heating_network',
        size=50,  # 50 kW thermal
        relative_minimum=0.3,  # Minimum 15 kW when on
        on_off_parameters=OnOffParameters(
            effects_per_switch_on={'startup_cost': 100},
            consecutive_on_hours_min=2,  # Min run time
            consecutive_off_hours_min=1,  # Min off time
        ),
    )
    ```

    **Variables:** `flow_rate[t]`, `on_off_state[t]`, `switch_on[t]`, `switch_off[t]`
    **Constraints:** When on: $15 \leq p(t) \leq 50$ kW; When off: $p(t) = 0$; Min 2h on, min 1h off

    ---

    ## Fixed Profile (Renewable)

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

    **Variables:** `flow_rate[t]` (fixed by profile)
    **Constraints:** $p(t) = 25 \cdot \text{profile}(t)$ MW

    ---

    ## Load Factor Constraint

    ```python
    from flixopt import Flow

    baseload_plant = Flow(
        label='baseload_output',
        bus='electricity',
        size=200,  # 200 MW
        load_factor_min=0.7,  # Must run at least 70% average
        effects_per_flow_hour={'cost': 30},
    )
    ```

    **Variables:** `flow_rate[t]`, `total_flow_hours`
    **Constraints:** $0 \leq p(t) \leq 200$ MW; $\sum_t p(t) \geq 0.7 \times 200 \times N_t$

---

## Implementation

- **Element Class:** [`Flow`][flixopt.elements.Flow]
- **Model Class:** [`FlowModel`][flixopt.elements.FlowModel]

## See Also

- **Features:** [OnOffParameters](../features/OnOffParameters.md) · [InvestParameters](../features/InvestParameters.md)
- **Elements:** [Bus](Bus.md) · [Storage](Storage.md) · [LinearConverter](LinearConverter.md)
- **Patterns:** [Modeling Patterns](../modeling-patterns/index.md) · [Bounds and States](../modeling-patterns/bounds-and-states.md)
- **Effects:** [Effects, Penalty & Objective](../effects-penalty-objective.md)
