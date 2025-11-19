# Bus

A Bus represents a node in the energy/material flow network where flow balance constraints ensure conservation (inflows equal outflows).

=== "Variables"

    | Symbol | Python Name | Description | Domain | Created When |
    |--------|-------------|-------------|--------|--------------|
    | $p_{f_\text{in}}(\text{t}_i)$ | - | Flow rate of incoming flow $f_\text{in}$ at time $\text{t}_i$ | $\mathbb{R}$ | Always (from connected Flows) |
    | $p_{f_\text{out}}(\text{t}_i)$ | - | Flow rate of outgoing flow $f_\text{out}$ at time $\text{t}_i$ | $\mathbb{R}$ | Always (from connected Flows) |
    | $\phi_\text{in}(\text{t}_i)$ | `excess_input` | Missing inflow (shortage) at time $\text{t}_i$ | $\mathbb{R}_+$ | `excess_penalty_per_flow_hour` is specified |
    | $\phi_\text{out}(\text{t}_i)$ | `excess_output` | Excess outflow (surplus) at time $\text{t}_i$ | $\mathbb{R}_+$ | `excess_penalty_per_flow_hour` is specified |

=== "Constraints"

    **Nodal balance equation** (always active):

    $$\label{eq:bus_balance}
    \sum_{f_\text{in} \in \mathcal{F}_\text{in}} p_{f_\text{in}}(\text{t}_i) =
    \sum_{f_\text{out} \in \mathcal{F}_\text{out}} p_{f_\text{out}}(\text{t}_i)
    $$

    ---

    **Modified balance with excess** (when `excess_penalty_per_flow_hour` specified):

    $$\label{eq:bus_balance_excess}
    \sum_{f_\text{in} \in \mathcal{F}_\text{in}} p_{f_\text{in}}(\text{t}_i) + \phi_\text{in}(\text{t}_i) =
    \sum_{f_\text{out} \in \mathcal{F}_\text{out}} p_{f_\text{out}}(\text{t}_i) + \phi_\text{out}(\text{t}_i)
    $$

    ---

    **Penalty cost** (when `excess_penalty_per_flow_hour` specified):

    $$\label{eq:bus_penalty}
    \Phi(\text{t}_i) = \text a_{b \rightarrow \Phi}(\text{t}_i) \cdot \Delta \text{t}_i \cdot [ \phi_\text{in}(\text{t}_i) + \phi_\text{out}(\text{t}_i) ]
    $$

    **Mathematical Patterns:** [Basic Equality](../modeling-patterns/bounds-and-states.md)

=== "Parameters"

    | Symbol | Python Parameter | Description | Default |
    |--------|------------------|-------------|---------|
    | $\text a_{b \rightarrow \Phi}(\text{t}_i)$ | `excess_penalty_per_flow_hour` | Penalty coefficient for balance violations (cost per unit flow-hour) | 1e5 |
    | $\Delta \text{t}_i$ | - | Timestep duration (hours) | From system time index |
    | $\mathcal{F}_\text{in}$ | - | Set of all incoming flows to the bus | From connected Flows |
    | $\mathcal{F}_\text{out}$ | - | Set of all outgoing flows from the bus | From connected Flows |

=== "Use Cases"

    ## Basic Bus with Strict Balance

    ```python
    from flixopt import Bus

    electricity_grid = Bus(
        label='electricity_grid',
        excess_penalty_per_flow_hour=None,  # No imbalance allowed
    )
    ```

    **Variables:** Flow rates from connected flows (no excess variables)

    **Constraints:** $\eqref{eq:bus_balance}$ enforces strict equality: all electricity inflows must exactly equal all outflows at every timestep.

    ---

    ## Bus with Excess Penalty

    ```python
    from flixopt import Bus

    heat_network = Bus(
        label='heating_network',
        excess_penalty_per_flow_hour=1000,  # High penalty for unmet demand
    )
    ```

    **Variables:** Flow rates + `excess_input` + `excess_output`

    **Constraints:** $\eqref{eq:bus_balance_excess}$ allows violations with penalty $\eqref{eq:bus_penalty}$

    This allows the model to violate the heat balance if necessary, but applies a penalty of 1000 cost units per kWh of unbalanced flow. Useful for debugging infeasible models or modeling emergency scenarios.

    ---

    ## Time-Varying Penalty

    ```python
    from flixopt import Bus
    import numpy as np

    material_hub = Bus(
        label='material_processing_hub',
        excess_penalty_per_flow_hour=np.array([100, 200, 300, 500]),  # Higher penalty during peak hours
    )
    ```

    **Variables:** Flow rates + `excess_input` + `excess_output`

    **Constraints:** $\eqref{eq:bus_balance_excess}$ with time-varying penalty $\eqref{eq:bus_penalty}$ where $\text a_{b \rightarrow \Phi}(\text{t}_i)$ varies by timestep.

---

## Implementation

- **Element Class:** [`Bus`][flixopt.elements.Bus]
- **Model Class:** [`BusModel`][flixopt.elements.BusModel]

## See Also

- **Elements:** [Flow](Flow.md) · [Storage](Storage.md) · [LinearConverter](LinearConverter.md)
- **System-Level:** [Effects, Penalty & Objective](../effects-penalty-objective.md)
- **Patterns:** [Modeling Patterns](../modeling-patterns/index.md)
