# Storage

A Storage component represents energy or material accumulation with charging/discharging flows, state of charge tracking, and efficiency losses.

**Implementation:**

- **Component Class:** [`Storage`][flixopt.components.Storage]
- **Model Class:** [`StorageModel`][flixopt.components.StorageModel]

**Related:** [`Flow`](Flow.md) · [`Bus`](Bus.md) · [`LinearConverter`](LinearConverter.md)

---

=== "Core Formulation"

    ## State of Charge Bounds

    The state of charge $c(\text{t}_i)$ is bounded by the storage size and relative limits:

    $$ \label{eq:Storage_Bounds}
        \text C \cdot \text c^{\text{L}}_{\text{rel}}(\text t_{i})
        \leq c(\text{t}_i) \leq
        \text C \cdot \text c^{\text{U}}_{\text{rel}}(\text t_{i})
    $$

    ??? info "Variables"
        | Symbol | Description | Domain |
        |--------|-------------|--------|
        | $c(\text{t}_i)$ | State of charge at time $\text{t}_i$ | $\mathbb{R}_+$ |
        | $\text C$ | Storage capacity | $\mathbb{R}_+$ or decision variable (with [InvestParameters](../features/InvestParameters.md)) |

    ??? info "Parameters"
        | Symbol | Description | Typical Value |
        |--------|-------------|---------------|
        | $\text c^{\text{L}}_{\text{rel}}(\text t_{i})$ | Relative lower bound at time $\text{t}_i$ | 0 |
        | $\text c^{\text{U}}_{\text{rel}}(\text t_{i})$ | Relative upper bound at time $\text{t}_i$ | 1 |

    ### Simplified Form

    With standard bounds $\text c^{\text{L}}_{\text{rel}} = 0$ and $\text c^{\text{U}}_{\text{rel}} = 1$, equation $\eqref{eq:Storage_Bounds}$ simplifies to:

    $$ 0 \leq c(\text t_{i}) \leq \text C $$

    ## Storage Balance Equation

    The state of charge evolves according to charging/discharging flows and self-discharge losses:

    $$
    \begin{align}
        c(\text{t}_{i+1}) &= c(\text{t}_{i}) \cdot (1-\dot{\text{c}}_\text{rel,loss}(\text{t}_i))^{\Delta \text{t}_{i}} \nonumber \\
        &\quad + p_{f_\text{in}}(\text{t}_i) \cdot \Delta \text{t}_i \cdot \eta_\text{in}(\text{t}_i) \nonumber \\
        &\quad - p_{f_\text{out}}(\text{t}_i) \cdot \Delta \text{t}_i / \eta_\text{out}(\text{t}_i)
        \label{eq:storage_balance}
    \end{align}
    $$

    ??? info "Flow Variables"
        | Symbol | Description | Domain |
        |--------|-------------|--------|
        | $p_{f_\text{in}}(\text{t}_i)$ | Input flow rate (charging power) at time $\text{t}_i$ | $\mathbb{R}_+$ |
        | $p_{f_\text{out}}(\text{t}_i)$ | Output flow rate (discharging power) at time $\text{t}_i$ | $\mathbb{R}_+$ |

    ??? info "Efficiency Parameters"
        | Symbol | Description | Typical Range | Units |
        |--------|-------------|---------------|-------|
        | $\eta_\text{in}(\text{t}_i)$ | Charging efficiency | 0.85-0.98 | dimensionless |
        | $\eta_\text{out}(\text{t}_i)$ | Discharging efficiency | 0.85-0.98 | dimensionless |
        | $\dot{\text{c}}_\text{rel,loss}(\text{t}_i)$ | Relative self-discharge rate | 0-0.05 | per hour |
        | $\Delta \text{t}_{i}$ | Timestep duration | - | hours |

    **Physical Interpretation:**

    - The first term represents self-discharge (exponential decay)
    - The second term adds energy from charging (accounting for charging losses)
    - The third term subtracts energy from discharging (accounting for discharging losses)

=== "Advanced & Edge Cases"

    ## Initial State of Charge

    The storage must be initialized with a starting charge state:

    - **Parameter:** `initial_charge_state` sets $c(\text{t}_0)$
    - **Default:** Often set to some fraction of capacity (e.g., 0.5 × $\text C$)

    ## Final State of Charge Constraints

    Optional bounds on the state of charge at the end of the optimization horizon:

    $$
        \text c^{\text{L}}_{\text{final}} \leq c(\text{t}_\text{end}) \leq \text c^{\text{U}}_{\text{final}}
    $$

    **Use Case:** Ensure the storage ends with sufficient charge for the next optimization period, or enforce cyclic conditions.

    **Parameters:**
    - `minimal_final_charge_state`: Lower bound $\text c^{\text{L}}_{\text{final}}$
    - `maximal_final_charge_state`: Upper bound $\text c^{\text{U}}_{\text{final}}$

    ## Investment Sizing

    When using [InvestParameters](../features/InvestParameters.md), the storage capacity $\text C$ becomes an optimization variable:

    - Storage size is determined by the optimizer to minimize total system cost
    - Flow capacities (charging/discharging rates) can be independently sized or linked to storage capacity

    ## Prevent Simultaneous Charging/Discharging

    Some storage types (e.g., batteries with shared converters) cannot charge and discharge simultaneously. This is modeled using [OnOffParameters](../features/OnOffParameters.md) on the input/output flows with appropriate constraints.

    ## Variable Timesteps

    The formulation naturally handles variable timestep durations $\Delta \text{t}_i$ through the balance equation $\eqref{eq:storage_balance}$.

=== "Mathematical Patterns"

    Storage formulation builds on the following modeling patterns:

    - **[Basic Bounds](../modeling-patterns/bounds-and-states.md#basic-bounds)** - Charge state bounds (equation $\eqref{eq:Storage_Bounds}$)
    - **[Scaled Bounds](../modeling-patterns/bounds-and-states.md#scaled-bounds)** - Flow rate bounds relative to storage size
    - **[Bounds with State](../modeling-patterns/bounds-and-states.md#bounds-with-state)** - When using [InvestParameters](../features/InvestParameters.md) for capacity investment

=== "Examples"

    ## Basic Battery Storage

    ```python
    from flixopt import Storage, Flow

    battery = Storage(
        label='battery',
        inputs=[Flow(label='charge', bus='electricity', size=50)],  # 50 kW charging
        outputs=[Flow(label='discharge', bus='electricity', size=50)],  # 50 kW discharging
        capacity_in_flow_hours=200,  # 200 kWh capacity
        initial_charge_state=100,  # Start at 100 kWh (50% SOC)
        eta_charge=0.95,  # 95% charging efficiency
        eta_discharge=0.95,  # 95% discharging efficiency
        relative_loss_per_hour=0.001,  # 0.1% self-discharge per hour
    )
    ```

    ## Thermal Storage with Final State Constraint

    ```python
    from flixopt import Storage, Flow

    thermal_storage = Storage(
        label='heat_tank',
        inputs=[Flow(label='heat_in', bus='heating', size=100)],
        outputs=[Flow(label='heat_out', bus='heating', size=100)],
        capacity_in_flow_hours=500,  # 500 kWh thermal capacity
        initial_charge_state=250,  # Start half-full
        minimal_final_charge_state=250,  # End at least half-full
        eta_charge=0.98,  # Minimal losses charging
        eta_discharge=0.98,
        relative_loss_per_hour=0.02,  # 2% heat loss per hour
    )
    ```

    ## Storage with Investment Decision

    ```python
    from flixopt import Storage, Flow, InvestParameters

    optimized_battery = Storage(
        label='battery_investment',
        inputs=[Flow(label='charge', bus='electricity', size=100)],
        outputs=[Flow(label='discharge', bus='electricity', size=100)],
        capacity_in_flow_hours=InvestParameters(
            minimum_size=0,  # Can choose not to build
            maximum_size=1000,  # Up to 1 MWh
            specific_effects={'cost': 200},  # €200 per kWh annualized
        ),
        initial_charge_state=0,
        eta_charge=0.92,
        eta_discharge=0.92,
    )
    ```

---

## See Also

- **Elements:** [Flow](Flow.md) · [Bus](Bus.md)
- **Features:** [InvestParameters](../features/InvestParameters.md) · [OnOffParameters](../features/OnOffParameters.md)
- **Patterns:** [Modeling Patterns](../modeling-patterns/index.md)
- **Components:** [LinearConverter](LinearConverter.md)
