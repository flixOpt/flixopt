# Bus

A Bus represents a node in the energy/material flow network where flow balance constraints ensure conservation (inflows equal outflows).

**Implementation:**

- **Element Class:** [`Bus`][flixopt.elements.Bus]
- **Model Class:** [`BusModel`][flixopt.elements.BusModel]

**Related:** [`Flow`](Flow.md) · [`Storage`](Storage.md) · [`LinearConverter`](LinearConverter.md)

---

=== "Core Formulation"

    ## Nodal Balance Equation

    The fundamental constraint of a Bus is that all incoming flow rates must equal all outgoing flow rates at every timestep:

    $$ \label{eq:bus_balance}
      \sum_{f_\text{in} \in \mathcal{F}_\text{in}} p_{f_\text{in}}(\text{t}_i) =
      \sum_{f_\text{out} \in \mathcal{F}_\text{out}} p_{f_\text{out}}(\text{t}_i)
    $$

    ??? info "Variables"
        | Symbol | Description | Domain |
        |--------|-------------|--------|
        | $p_{f_\text{in}}(\text{t}_i)$ | Flow rate of incoming flow $f_\text{in}$ at time $\text{t}_i$ | $\mathbb{R}$ |
        | $p_{f_\text{out}}(\text{t}_i)$ | Flow rate of outgoing flow $f_\text{out}$ at time $\text{t}_i$ | $\mathbb{R}$ |

    ??? info "Sets"
        | Symbol | Description |
        |--------|-------------|
        | $\mathcal{F}_\text{in}$ | Set of all incoming flows to the bus |
        | $\mathcal{F}_\text{out}$ | Set of all outgoing flows from the bus |

    This strict equality ensures energy/material conservation at each node.

=== "Advanced & Edge Cases"

    ## Excess Penalty (Soft Constraints)

    When `excess_penalty_per_flow_hour` is specified, the Bus allows balance violations with a penalty cost. This creates a "soft" constraint useful for handling potential infeasibilities gracefully.

    ### Modified Balance Equation

    The balance equation becomes:

    $$ \label{eq:bus_balance_excess}
      \sum_{f_\text{in} \in \mathcal{F}_\text{in}} p_{f_ \text{in}}(\text{t}_i) + \phi_\text{in}(\text{t}_i) =
      \sum_{f_\text{out} \in \mathcal{F}_\text{out}} p_{f_\text{out}}(\text{t}_i) + \phi_\text{out}(\text{t}_i)
    $$

    ??? info "Additional Variables"
        | Symbol | Description | Domain |
        |--------|-------------|--------|
        | $\phi_\text{in}(\text{t}_i)$ | Missing inflow (shortage) at time $\text{t}_i$ | $\mathbb{R}_+ $ |
        | $\phi_\text{out}(\text{t}_i)$ | Excess outflow (surplus) at time $\text{t}_i$ | $\mathbb{R}_+$ |

    ### Penalty Cost

    The penalty term added to the objective function:

    $$ \label{eq:bus_penalty}
      s_{b \rightarrow \Phi}(\text{t}_i) =
          \text a_{b \rightarrow \Phi}(\text{t}_i) \cdot \Delta \text{t}_i
          \cdot [ \phi_\text{in}(\text{t}_i) + \phi_\text{out}(\text{t}_i) ]
    $$

    ??? info "Parameters"
        | Symbol | Description | Units |
        |--------|-------------|-------|
        | $\text a_{b \rightarrow \Phi}(\text{t}_i)$ | Penalty coefficient (`excess_penalty_per_flow_hour`) | Cost per unit flow-hour |
        | $\Delta \text{t}_i$ | Timestep duration | hours |
        | $s_{b \rightarrow \Phi}(\text{t}_i)$ | Total penalty cost at time $\text{t}_i$ | Cost units |

    **Use Case:** This soft constraint approach prevents model infeasibility when supply and demand cannot be perfectly balanced, making it easier to identify and diagnose problem areas in the energy system model.

=== "Examples"

    ## Basic Bus with Strict Balance

    ```python
    from flixopt import Bus

    electricity_grid = Bus(
        label='electricity_grid',
    )
    ```

    This creates a strict nodal balance: all electricity inflows must exactly equal all outflows at every timestep.

    ## Bus with Excess Penalty

    ```python
    from flixopt import Bus

    heat_network = Bus(
        label='heating_network',
        excess_penalty_per_flow_hour=1000,  # High penalty for unmet demand
    )
    ```

    This allows the model to violate the heat balance if necessary, but applies a penalty of 1000 cost units per kWh of unbalanced flow. Useful for debugging infeasible models or modeling emergency scenarios.

---

## See Also

- **Elements:** [Flow](Flow.md) · [Storage](Storage.md) · [LinearConverter](LinearConverter.md)
- **System-Level:** [Effects, Penalty & Objective](../effects-penalty-objective.md)
- **Patterns:** [Modeling Patterns](../modeling-patterns/index.md)
