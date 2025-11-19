# Storage

A Storage component represents energy or material accumulation with charging/discharging flows, state of charge tracking, and efficiency losses.

=== "Variables"

    | Symbol | Python Name | Description | Domain | Created When |
    |--------|-------------|-------------|--------|--------------|
    | $c(\text{t}_i)$ | `charge_state` | State of charge at time $\text{t}_i$ | $\mathbb{R}_+$ | Always |
    | $p_{f_\text{in}}(\text{t}_i)$ | - | Input flow rate (charging power) at time $\text{t}_i$ | $\mathbb{R}_+$ | Always (from `charging` Flow) |
    | $p_{f_\text{out}}(\text{t}_i)$ | - | Output flow rate (discharging power) at time $\text{t}_i$ | $\mathbb{R}_+$ | Always (from `discharging` Flow) |
    | - | `netto_discharge` | Net discharge rate (discharge - charge) | $\mathbb{R}$ | Always |
    | $\text C$ | `size` | Storage capacity (decision variable) | $\mathbb{R}_+$ | `capacity_in_flow_hours` is `InvestParameters` |

=== "Constraints"

    **State of charge bounds** (always active):

    $$\label{eq:Storage_Bounds}
    \text C \cdot \text c^{\text{L}}_{\text{rel}}(\text t_{i})
    \leq c(\text{t}_i) \leq
    \text C \cdot \text c^{\text{U}}_{\text{rel}}(\text t_{i})
    $$

    ---

    **Storage balance equation** (always active):

    $$\label{eq:storage_balance}
    \begin{align}
    c(\text{t}_{i+1}) &= c(\text{t}_{i}) \cdot (1-\dot{\text{c}}_\text{rel,loss}(\text{t}_i))^{\Delta \text{t}_{i}} \nonumber \\
    &\quad + p_{f_\text{in}}(\text{t}_i) \cdot \Delta \text{t}_i \cdot \eta_\text{in}(\text{t}_i) \nonumber \\
    &\quad - p_{f_\text{out}}(\text{t}_i) \cdot \Delta \text{t}_i / \eta_\text{out}(\text{t}_i)
    \end{align}
    $$

    ---

    **Initial charge state** (when `initial_charge_state` specified):

    $$\label{eq:storage_initial}
    c(\text{t}_0) = \text{c}_\text{initial}
    $$

    Or for cyclic condition (`initial_charge_state='equals_final'`):

    $$\label{eq:storage_cyclic}
    c(\text{t}_0) = c(\text{t}_\text{end})
    $$

    ---

    **Final charge state bounds** (when `minimal_final_charge_state` or `maximal_final_charge_state` specified):

    $$\label{eq:storage_final}
    \text c^{\text{L}}_{\text{final}} \leq c(\text{t}_\text{end}) \leq \text c^{\text{U}}_{\text{final}}
    $$

    **Mathematical Patterns:** [Basic Bounds](../modeling-patterns/bounds-and-states.md#basic-bounds), [Scaled Bounds](../modeling-patterns/bounds-and-states.md#scaled-bounds)

=== "Parameters"

    | Symbol | Python Parameter | Description | Default |
    |--------|------------------|-------------|---------|
    | $\text C$ | `capacity_in_flow_hours` | Storage capacity | Required |
    | $\text c^{\text{L}}_{\text{rel}}(\text t_{i})$ | `relative_minimum_charge_state` | Relative lower bound (fraction of capacity) | 0 |
    | $\text c^{\text{U}}_{\text{rel}}(\text t_{i})$ | `relative_maximum_charge_state` | Relative upper bound (fraction of capacity) | 1 |
    | $\text{c}_\text{initial}$ | `initial_charge_state` | Charge at start | 0 |
    | $\text c^{\text{L}}_{\text{final}}$ | `minimal_final_charge_state` | Minimum absolute charge required at end | None |
    | $\text c^{\text{U}}_{\text{final}}$ | `maximal_final_charge_state` | Maximum absolute charge allowed at end | None |
    | $\Delta \text{t}_{i}$ | - | Timestep duration | hours |
    | $\dot{\text{c}}_\text{rel,loss}(\text{t}_i)$ | `relative_loss_per_hour` | Relative self-discharge rate per hour | 0 |
    | $\eta_\text{in}(\text{t}_i)$ | `eta_charge` | Charging efficiency (0-1) | 1 |
    | $\eta_\text{out}(\text{t}_i)$ | `eta_discharge` | Discharging efficiency (0-1) | 1 |
    | - | `balanced` | If True, forces charging and discharging flows to have equal sizes | False |
    | - | `prevent_simultaneous_charge_and_discharge` | Prevents charging and discharging simultaneously | True |
    | - | `relative_minimum_final_charge_state` | Minimum relative charge at end | None |
    | - | `relative_maximum_final_charge_state` | Maximum relative charge at end | None |

=== "Use Cases"

    ## Basic Battery Storage

    ```python
    from flixopt import Storage, Flow

    battery = Storage(
        label='battery',
        charging=Flow(label='charge', bus='electricity', size=50),  # 50 kW charging
        discharging=Flow(label='discharge', bus='electricity', size=50),  # 50 kW discharging
        capacity_in_flow_hours=200,  # 200 kWh capacity
        initial_charge_state=100,  # Start at 100 kWh (50% SOC)
        eta_charge=0.95,  # 95% charging efficiency
        eta_discharge=0.95,  # 95% discharging efficiency
        relative_loss_per_hour=0.001,  # 0.1% self-discharge per hour
    )
    ```

    **Variables:** `charge_state[t]`, flow rates, `netto_discharge[t]`

    **Constraints:** $\eqref{eq:Storage_Bounds}$ with $0 \leq c(t) \leq 200$ kWh, $\eqref{eq:storage_balance}$, $\eqref{eq:storage_initial}$ with $c(t_0) = 100$ kWh

    ---

    ## Thermal Storage with Final State Constraint

    ```python
    from flixopt import Storage, Flow

    thermal_storage = Storage(
        label='heat_tank',
        charging=Flow(label='heat_in', bus='heating', size=100),
        discharging=Flow(label='heat_out', bus='heating', size=100),
        capacity_in_flow_hours=500,  # 500 kWh thermal capacity
        initial_charge_state=250,  # Start half-full
        minimal_final_charge_state=250,  # End at least half-full
        eta_charge=0.98,  # Minimal losses charging
        eta_discharge=0.98,
        relative_loss_per_hour=0.02,  # 2% heat loss per hour
    )
    ```

    **Variables:** `charge_state[t]`, flow rates, `netto_discharge[t]`

    **Constraints:** $\eqref{eq:Storage_Bounds}$, $\eqref{eq:storage_balance}$, $\eqref{eq:storage_initial}$, $\eqref{eq:storage_final}$ with $c(t_\text{end}) \geq 250$ kWh

    ---

    ## Storage with Investment Decision

    ```python
    from flixopt import Storage, Flow, InvestParameters

    optimized_battery = Storage(
        label='battery_investment',
        charging=Flow(label='charge', bus='electricity', size=100),
        discharging=Flow(label='discharge', bus='electricity', size=100),
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

    **Variables:** `charge_state[t]`, flow rates, `netto_discharge[t]`, `size` (decision variable)

    **Constraints:** $\eqref{eq:Storage_Bounds}$ with variable $\text C$, $\eqref{eq:storage_balance}$, plus investment constraints from [InvestParameters](../features/InvestParameters.md)

    ---

    ## Cyclic Storage Condition

    ```python
    from flixopt import Storage, Flow

    pumped_hydro = Storage(
        label='pumped_hydro',
        charging=Flow(label='pump', bus='electricity', size=100),
        discharging=Flow(label='turbine', bus='electricity', size=120),
        capacity_in_flow_hours=10000,  # 10 MWh
        initial_charge_state='equals_final',  # End with same charge as start
        eta_charge=0.85,  # Pumping efficiency
        eta_discharge=0.90,  # Turbine efficiency
        relative_loss_per_hour=0.0001,  # Minimal evaporation
    )
    ```

    **Variables:** `charge_state[t]`, flow rates, `netto_discharge[t]`

    **Constraints:** $\eqref{eq:Storage_Bounds}$, $\eqref{eq:storage_balance}$, $\eqref{eq:storage_cyclic}$ enforcing $c(t_0) = c(t_\text{end})$

---

## Implementation

- **Component Class:** [`Storage`][flixopt.components.Storage]
- **Model Class:** [`StorageModel`][flixopt.components.StorageModel]

## See Also

- **Elements:** [Flow](Flow.md) · [Bus](Bus.md) · [LinearConverter](LinearConverter.md)
- **Features:** [InvestParameters](../features/InvestParameters.md) · [OnOffParameters](../features/OnOffParameters.md)
- **Patterns:** [Modeling Patterns](../modeling-patterns/index.md) · [Bounds and States](../modeling-patterns/bounds-and-states.md)
