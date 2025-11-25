# Storage

A Storage component accumulates energy or material over time, allowing you to decouple when you produce from when you consume. This is essential for integrating renewables, managing peak demand, and arbitraging price differences.

!!! example "Real-world examples"
    - **Battery** — store electricity when cheap, discharge when expensive
    - **Thermal tank** — buffer heat production from demand
    - **Warehouse** — inventory buffer in supply chains
    - **Pumped hydro** — large-scale electricity storage

## Core Concept: Charge State Over Time

A storage has a **charge state** $c(t)$ that evolves over time based on charging and discharging:

$$
c(t+1) = c(t) \cdot (1 - loss)^{\Delta t} + charge(t) \cdot \eta_{in} - \frac{discharge(t)}{\eta_{out}}
$$

Where:

- $c(t)$ — Energy/material stored at time $t$
- $charge(t)$ — Charging flow rate (power in)
- $discharge(t)$ — Discharging flow rate (power out)
- $\eta_{in}$ — Charging efficiency (losses when storing)
- $\eta_{out}$ — Discharging efficiency (losses when retrieving)
- $loss$ — Self-discharge rate per hour

!!! note "Units"
    - **Capacity** is in energy units (kWh, MWh)
    - **Charging/discharging flows** are in power units (kW, MW)
    - The timestep duration $\Delta t$ converts between them

## The Storage Balance Equation

At each timestep, the charge state updates:

$$
c(t_{i+1}) = c(t_i) \cdot (1 - loss)^{\Delta t_i} + p_{in}(t_i) \cdot \Delta t_i \cdot \eta_{in} - p_{out}(t_i) \cdot \Delta t_i / \eta_{out}
$$

This equation captures:

1. **Starting state** — what was stored at the beginning
2. **Self-discharge** — losses over time (exponential decay)
3. **Charging** — energy added (with efficiency loss)
4. **Discharging** — energy removed (with efficiency loss)

## Charge State Bounds

The charge state is bounded by the storage capacity:

$$
C \cdot relative\_min(t) \leq c(t) \leq C \cdot relative\_max(t)
$$

!!! example "Battery with 20-80% operating range"
    - Capacity: 1000 kWh
    - `relative_minimum_charge_state`: 0.2
    - `relative_maximum_charge_state`: 0.8
    - Effective range: 200-800 kWh (protects battery life)

## Initial and Final Conditions

### Initial State

What's in the storage at the start?

$$
c(t_0) = initial\_charge\_state
$$

### Final State

Optionally constrain where the storage ends up:

$$
c_{final,min} \leq c(t_{end}) \leq c_{final,max}
$$

### Cyclic Condition

For periodic optimization (e.g., typical day), ensure the storage ends where it started:

$$
c(t_0) = c(t_{end})
$$

This prevents "gaming" where the optimizer drains storage without replenishing.

## Variables

| Variable | Python Name | Description | When Created |
|----------|-------------|-------------|--------------|
| $c(t)$ | `charge_state` | Energy stored at each timestep | Always |
| $p_{in}(t)$ | (from charging Flow) | Charging power | Always |
| $p_{out}(t)$ | (from discharging Flow) | Discharging power | Always |
| $C$ | `size` | Storage capacity (decision variable) | When `capacity_in_flow_hours` is `InvestParameters` |

## Parameters

| Parameter | Python Name | Description | Default |
|-----------|-------------|-------------|---------|
| $C$ | `capacity_in_flow_hours` | Storage capacity (kWh, MWh) | Required |
| $\eta_{in}$ | `eta_charge` | Charging efficiency (0-1) | 1.0 |
| $\eta_{out}$ | `eta_discharge` | Discharging efficiency (0-1) | 1.0 |
| $loss$ | `relative_loss_per_hour` | Self-discharge rate per hour | 0 |
| $c_0$ | `initial_charge_state` | Starting charge (absolute or `'equals_final'`) | 0 |
| $c_{min}(t)$ | `relative_minimum_charge_state` | Min charge as fraction of capacity | 0 |
| $c_{max}(t)$ | `relative_maximum_charge_state` | Max charge as fraction of capacity | 1 |
| $c_{final,min}$ | `minimal_final_charge_state` | Minimum charge at end | None |
| $c_{final,max}$ | `maximal_final_charge_state` | Maximum charge at end | None |

## Usage Examples

### Basic Battery

```python
battery = fx.Storage(
    label='battery',
    charging=fx.Flow(label='charge', bus=electricity_bus, size=50),   # 50 kW charging
    discharging=fx.Flow(label='discharge', bus=electricity_bus, size=50),  # 50 kW discharging
    capacity_in_flow_hours=200,  # 200 kWh capacity
    initial_charge_state=100,    # Start at 100 kWh (50% SOC)
    eta_charge=0.95,
    eta_discharge=0.95,
)
```

Round-trip efficiency: $0.95 \times 0.95 = 90.25\%$

### Thermal Storage with Losses

```python
thermal_tank = fx.Storage(
    label='heat_tank',
    charging=fx.Flow(label='heat_in', bus=heat_bus, size=100),
    discharging=fx.Flow(label='heat_out', bus=heat_bus, size=100),
    capacity_in_flow_hours=500,  # 500 kWh thermal
    initial_charge_state=250,
    eta_charge=0.98,
    eta_discharge=0.98,
    relative_loss_per_hour=0.02,  # 2% heat loss per hour
)
```

### Storage with Cyclic Condition

For typical-day optimization, ensure the storage doesn't cheat:

```python
daily_storage = fx.Storage(
    label='daily_buffer',
    charging=fx.Flow(label='in', bus=some_bus, size=100),
    discharging=fx.Flow(label='out', bus=some_bus, size=100),
    capacity_in_flow_hours=400,
    initial_charge_state='equals_final',  # Must end where it started
)
```

### Storage with Investment Decision

```python
from flixopt import InvestParameters

battery_investment = fx.Storage(
    label='battery',
    charging=fx.Flow(label='charge', bus=electricity_bus, size=100),
    discharging=fx.Flow(label='discharge', bus=electricity_bus, size=100),
    capacity_in_flow_hours=InvestParameters(
        minimum_size=0,
        maximum_size=1000,  # Up to 1 MWh
        specific_effects={'costs': 200},  # €200/kWh annualized
    ),
    eta_charge=0.92,
    eta_discharge=0.92,
)
```

### Different Charge/Discharge Power

```python
pumped_hydro = fx.Storage(
    label='pumped_hydro',
    charging=fx.Flow(label='pump', bus=electricity_bus, size=100),      # 100 MW pumping
    discharging=fx.Flow(label='turbine', bus=electricity_bus, size=120),  # 120 MW generating
    capacity_in_flow_hours=10000,  # 10 GWh reservoir
    eta_charge=0.85,   # Pumping losses
    eta_discharge=0.90,  # Turbine losses
)
```

## Preventing Simultaneous Charging and Discharging

By default, storage can't charge and discharge simultaneously (which would waste energy through round-trip losses). This is controlled by:

```python
storage = fx.Storage(
    ...,
    prevent_simultaneous_charge_and_discharge=True,  # Default
)
```

This adds a constraint linking the charging and discharging flows.

## Implementation Details

- **Component Class:** [`Storage`][flixopt.components.Storage]
- **Model Class:** [`StorageModel`][flixopt.components.StorageModel]

## See Also

- [Flow](Flow.md) — Charging and discharging flows
- [Bus](Bus.md) — Where storage connects
- [InvestParameters](../features/InvestParameters.md) — Optimizing storage capacity
- [Core Concepts: Storages](../../core-concepts.md#storages-save-for-later) — High-level overview
