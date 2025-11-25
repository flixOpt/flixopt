# Storage

A Storage component accumulates energy or material over time, allowing you to decouple when you produce from when you consume.

!!! example "Real-world examples"
    - **Battery** — store electricity when cheap, discharge when expensive
    - **Thermal tank** — buffer heat production from demand
    - **Pumped hydro** — large-scale electricity storage

## Core Concept: Charge State Over Time

A storage has a **charge state** $c(t)$ that evolves based on charging and discharging:

$$
c(t_{i+1}) = c(t_i) \cdot (1 - \dot{c}_{loss})^{\Delta t} + p_{in}(t_i) \cdot \Delta t \cdot \eta_{in} - p_{out}(t_i) \cdot \Delta t / \eta_{out}
$$

Where:

| Symbol | Meaning |
|--------|---------|
| $c(t)$ | Charge state (energy stored) |
| $p_{in}(t)$ | Charging power |
| $p_{out}(t)$ | Discharging power |
| $\eta_{in}$ | Charging efficiency |
| $\eta_{out}$ | Discharging efficiency |
| $\dot{c}_{loss}$ | Self-discharge rate per hour |

## Charge State Bounds

$$
C \cdot c_{rel}^{min}(t) \leq c(t) \leq C \cdot c_{rel}^{max}(t)
$$

Where $C$ is the storage capacity.

!!! example "Battery with 20-80% SOC range"
    $c_{rel}^{min} = 0.2$, $c_{rel}^{max} = 0.8$ → Effective range: 20-80% of capacity

## Initial and Final Conditions

=== "Fixed Initial"

    $$
    c(t_0) = c_0
    $$

    Storage starts with a specified charge.

=== "Cyclic"

    $$
    c(t_0) = c(t_{end})
    $$

    Storage must end where it started. Use `initial_charge_state='equals_final'`.

    Prevents the optimizer from "cheating" by draining storage without replenishing.

=== "Final Bounds"

    $$
    c_{final}^{min} \leq c(t_{end}) \leq c_{final}^{max}
    $$

    Constrain the final charge state.

## Variables

| Symbol | Python Name | Description | When Created |
|--------|-------------|-------------|--------------|
| $c(t)$ | `charge_state` | Charge at timestep $t$ | Always |
| $p_{in}(t)$ | (from `charging` Flow) | Charging power | Always |
| $p_{out}(t)$ | (from `discharging` Flow) | Discharging power | Always |
| $C$ | `size` | Capacity (variable) | `capacity_in_flow_hours` is `InvestParameters` |

## Parameters

| Symbol | Python Name | Description | Default |
|--------|-------------|-------------|---------|
| $C$ | `capacity_in_flow_hours` | Storage capacity | Required |
| $\eta_{in}$ | `eta_charge` | Charging efficiency | 1.0 |
| $\eta_{out}$ | `eta_discharge` | Discharging efficiency | 1.0 |
| $\dot{c}_{loss}$ | `relative_loss_per_hour` | Self-discharge rate | 0 |
| $c_0$ | `initial_charge_state` | Starting charge | 0 |
| $c_{rel}^{min}(t)$ | `relative_minimum_charge_state` | Min SOC fraction | 0 |
| $c_{rel}^{max}(t)$ | `relative_maximum_charge_state` | Max SOC fraction | 1 |
| $c_{final}^{min}$ | `minimal_final_charge_state` | Min final charge | None |
| $c_{final}^{max}$ | `maximal_final_charge_state` | Max final charge | None |

## Usage Examples

### Basic Battery

```python
battery = fx.Storage(
    label='battery',
    charging=fx.Flow(label='charge', bus=elec_bus, size=50),
    discharging=fx.Flow(label='discharge', bus=elec_bus, size=50),
    capacity_in_flow_hours=200,  # 200 kWh
    initial_charge_state=100,    # Start at 50% SOC
    eta_charge=0.95,
    eta_discharge=0.95,
)
```

Round-trip efficiency: $0.95 \times 0.95 = 90.25\%$

### Thermal Storage with Losses

```python
tank = fx.Storage(
    label='tank',
    charging=fx.Flow(label='in', bus=heat_bus, size=100),
    discharging=fx.Flow(label='out', bus=heat_bus, size=100),
    capacity_in_flow_hours=500,
    relative_loss_per_hour=0.02,  # 2%/hour heat loss
    eta_charge=0.98,
    eta_discharge=0.98,
)
```

### Cyclic (Typical Day)

```python
buffer = fx.Storage(
    label='buffer',
    charging=fx.Flow(label='in', bus=bus, size=100),
    discharging=fx.Flow(label='out', bus=bus, size=100),
    capacity_in_flow_hours=400,
    initial_charge_state='equals_final',  # Must end where started
)
```

### Investment Decision

```python
battery = fx.Storage(
    label='battery',
    charging=fx.Flow(label='charge', bus=elec_bus, size=100),
    discharging=fx.Flow(label='discharge', bus=elec_bus, size=100),
    capacity_in_flow_hours=fx.InvestParameters(
        minimum_size=0,
        maximum_size=1000,
        specific_effects={'costs': 200},  # €200/kWh
    ),
    eta_charge=0.92,
    eta_discharge=0.92,
)
```

### Different Charge/Discharge Power

```python
pumped_hydro = fx.Storage(
    label='hydro',
    charging=fx.Flow(label='pump', bus=elec_bus, size=100),     # 100 MW
    discharging=fx.Flow(label='turbine', bus=elec_bus, size=120),  # 120 MW
    capacity_in_flow_hours=10000,
    eta_charge=0.85,
    eta_discharge=0.90,
)
```

## Simultaneous Charge/Discharge

By default, storage cannot charge and discharge simultaneously:

```python
storage = fx.Storage(
    ...,
    prevent_simultaneous_charge_and_discharge=True,  # Default
)
```

## Implementation Details

- **Component Class:** [`Storage`][flixopt.components.Storage]
- **Model Class:** [`StorageModel`][flixopt.components.StorageModel]

## See Also

- [Flow](Flow.md) — Charging and discharging flows
- [Bus](Bus.md) — Where storage connects
- [InvestParameters](../features/InvestParameters.md) — Capacity optimization
