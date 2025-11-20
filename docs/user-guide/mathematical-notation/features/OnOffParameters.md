# OnOffParameters

OnOffParameters define operational constraints and effects for binary on/off equipment behavior, capturing realistic operational limits and associated costs.

=== "Variables"

    | Symbol | Python Name | Description | Domain | Created When |
    |--------|-------------|-------------|--------|--------------|
    | $s(\text{t}_i)$ | `on` | Binary on/off state at time $\text{t}_i$ | $\{0,1\}$ | Always |
    | - | `off` | Binary off state (complement of `on`) | $\{0,1\}$ | Any consecutive off constraints specified |
    | - | `switch_on` | Startup indicator at time $\text{t}_i$ | $\{0,1\}$ | `switch_on_max` or `effects_per_switch_on` or consecutive constraints specified |
    | - | `switch_off` | Shutdown indicator at time $\text{t}_i$ | $\{0,1\}$ | `switch_on_max` or `effects_per_switch_on` or consecutive constraints specified |
    | - | `on_hours_total` | Total operating hours in period | $\mathbb{R}_+$ | `on_hours_min` or `on_hours_max` specified |
    | - | `switch_count` | Count of startups in period | $\mathbb{Z}_+$ | `switch_on_max` specified |
    | - | `consecutive_on_hours` | Duration tracking for minimum/maximum run times | $\mathbb{R}_+$ | `consecutive_on_hours_min` or `consecutive_on_hours_max` specified |
    | - | `consecutive_off_hours` | Duration tracking for minimum/maximum off times | $\mathbb{R}_+$ | `consecutive_off_hours_min` or `consecutive_off_hours_max` specified |

=== "Constraints"

    **Complementary on/off states** (when off state needed):

    $$\label{eq:onoff_complementary}
    s(\text{t}_i) + s_\text{off}(\text{t}_i) = 1
    $$

    ---

    **Total operating hours** (when `on_hours_min` or `on_hours_max` specified):

    $$\label{eq:onoff_total_hours}
    H_\text{min} \leq \sum_i s(\text{t}_i) \cdot \Delta \text{t}_i \leq H_\text{max}
    $$

    ---

    **State transitions** (when switch tracking enabled):

    $$\label{eq:onoff_transitions}
    \text{switch\_on}(\text{t}_i) - \text{switch\_off}(\text{t}_i) = s(\text{t}_i) - s(\text{t}_{i-1})
    $$

    ---

    **Maximum startup count** (when `switch_on_max` specified):

    $$\label{eq:onoff_switch_count}
    \sum_i \text{switch\_on}(\text{t}_i) \leq N_\text{switch,max}
    $$

    ---

    **Minimum consecutive on time** (when `consecutive_on_hours_min` specified):

    $$\label{eq:onoff_min_on}
    \text{If } s(\text{t}_{i-1}) = 0 \text{ and } s(\text{t}_i) = 1, \text{ then } \sum_{j=i}^{i+k} s(\text{t}_j) \cdot \Delta \text{t}_j \geq T_\text{on,min}
    $$

    ---

    **Maximum consecutive on time** (when `consecutive_on_hours_max` specified):

    $$\label{eq:onoff_max_on}
    \sum_{j=i}^{i+k} s(\text{t}_j) \cdot \Delta \text{t}_j \leq T_\text{on,max}
    $$

    ---

    **Minimum consecutive off time** (when `consecutive_off_hours_min` specified):

    $$\label{eq:onoff_min_off}
    \text{If } s(\text{t}_{i-1}) = 1 \text{ and } s(\text{t}_i) = 0, \text{ then } \sum_{j=i}^{i+k} (1-s(\text{t}_j)) \cdot \Delta \text{t}_j \geq T_\text{off,min}
    $$

    ---

    **Effects per switch on** (when `effects_per_switch_on` specified):

    $$\label{eq:onoff_switch_effects}
    E_{e,\text{switch}} = \sum_i \text{switch\_on}(\text{t}_i) \cdot c_{e,\text{switch}}
    $$

    ---

    **Effects per running hour** (when `effects_per_running_hour` specified):

    $$\label{eq:onoff_running_effects}
    E_{e,\text{run}} = \sum_i s(\text{t}_i) \cdot \Delta \text{t}_i \cdot c_{e,\text{run}}
    $$

    **Mathematical Patterns:** [State Transitions](../modeling-patterns/bounds-and-states.md), [Duration Tracking](../modeling-patterns/bounds-and-states.md)

=== "Parameters"

    | Symbol | Python Parameter | Description | Default |
    |--------|------------------|-------------|---------|
    | $c_{e,\text{run}}$ | `effects_per_running_hour` | Ongoing effects while equipment operates | None |
    | $c_{e,\text{switch}}$ | `effects_per_switch_on` | Effects for each startup | None |
    | $\Delta \text{t}_i$ | - | Timestep duration (hours) | From system |
    | $H_\text{max}$ | `on_hours_max` | Maximum total operating hours per period | None |
    | $H_\text{min}$ | `on_hours_min` | Minimum total operating hours per period | None |
    | $N_\text{switch,max}$ | `switch_on_max` | Maximum number of startups per period | None |
    | $T_\text{off,min}$ | `consecutive_off_hours_min` | Minimum continuous shutdown duration | None |
    | $T_\text{on,max}$ | `consecutive_on_hours_max` | Maximum continuous operating duration | None |
    | $T_\text{on,min}$ | `consecutive_on_hours_min` | Minimum continuous operating duration | None |
    | - | `consecutive_off_hours_max` | Maximum continuous shutdown duration | None |
    | - | `force_switch_on` | Create switch variables even without max constraint | False |

=== "Use Cases"

    ## Power Plant with Startup Costs

    ```python
    from flixopt import Flow, OnOffParameters

    generator = Flow(
        label='power_output',
        bus='electricity',
        size=100,  # 100 MW
        on_off_parameters=OnOffParameters(
            effects_per_switch_on={'cost': 25000},  # €25k startup cost
            consecutive_on_hours_min=8,  # Must run 8+ hours once started
            consecutive_off_hours_min=4,  # Must stay off 4+ hours
        ),
    )
    ```

    **Variables:** `on[t]`, `switch_on[t]`, `switch_off[t]`, `consecutive_on_hours`, `consecutive_off_hours`

    **Constraints:** $\eqref{eq:onoff_transitions}$, $\eqref{eq:onoff_min_on}$ with 8h minimum, $\eqref{eq:onoff_min_off}$ with 4h minimum, $\eqref{eq:onoff_switch_effects}$ with €25k per startup

    ---

    ## Industrial Process with Cycling Limits

    ```python
    from flixopt import Flow, OnOffParameters

    batch_reactor = Flow(
        label='process_output',
        bus='production',
        size=50,
        on_off_parameters=OnOffParameters(
            effects_per_switch_on={'setup_cost': 1500},
            effects_per_running_hour={'utilities': 200},
            consecutive_on_hours_min=12,  # 12h minimum batch
            switch_on_max=20,  # Max 20 batches per period
        ),
    )
    ```

    **Variables:** `on[t]`, `switch_on[t]`, `switch_off[t]`, `switch_count`, `consecutive_on_hours`

    **Constraints:** $\eqref{eq:onoff_transitions}$, $\eqref{eq:onoff_switch_count}$ with max 20 startups, $\eqref{eq:onoff_min_on}$ with 12h minimum, $\eqref{eq:onoff_switch_effects}$, $\eqref{eq:onoff_running_effects}$

    ---

    ## HVAC with Operating Hour Limits

    ```python
    from flixopt import Flow, OnOffParameters

    chiller = Flow(
        label='cooling_output',
        bus='chilled_water',
        size=500,  # 500 kW cooling
        on_off_parameters=OnOffParameters(
            on_hours_min=2000,  # Minimum 2000h/year utilization
            on_hours_max=5000,  # Maximum 5000h/year (maintenance limit)
            consecutive_on_hours_max=18,  # Max 18h continuous operation
        ),
    )
    ```

    **Variables:** `on[t]`, `on_hours_total`, `consecutive_on_hours`

    **Constraints:** $\eqref{eq:onoff_total_hours}$ with 2000-5000h bounds, $\eqref{eq:onoff_max_on}$ with 18h maximum

---

## Implementation

- **Feature Class:** [`OnOffParameters`][flixopt.interface.OnOffParameters]
- **Model Class:** [`OnOffModel`][flixopt.features.OnOffModel]
- **Used by:** [`Flow`](../elements/Flow.md) · [`LinearConverter`](../elements/LinearConverter.md)

## See Also

- **Elements:** [Flow](../elements/Flow.md) · [LinearConverter](../elements/LinearConverter.md)
- **Features:** [InvestParameters](InvestParameters.md)
- **Patterns:** [Bounds and States](../modeling-patterns/bounds-and-states.md)
- **System-Level:** [Effects, Penalty & Objective](../effects-penalty-objective.md)
