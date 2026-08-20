# Storage

A Storage accumulates energy over time — charge now, discharge later.

## Basic: Charge Dynamics

$$
c(t+1) = c(t) \cdot (1 - \dot{c}_{loss})^{\Delta t} + p_{in}(t) \cdot \Delta t \cdot \eta_{in} - p_{out}(t) \cdot \Delta t / \eta_{out}
$$

```python
battery = fx.Storage(
    label='battery',
    charging=fx.Flow(label='charge', bus=elec_bus, size=50),
    discharging=fx.Flow(label='discharge', bus=elec_bus, size=50),
    capacity_in_flow_hours=200,  # 200 kWh
    eta_charge=0.95,
    eta_discharge=0.95,
)
# Round-trip efficiency: 95% × 95% = 90.25%
```

---

## Charge State Bounds

$$
C \cdot c_{rel}^{min} \leq c(t) \leq C \cdot c_{rel}^{max}
$$

```python
fx.Storage(...,
    relative_minimum_charge_state=0.2,  # Min 20% SOC
    relative_maximum_charge_state=0.8,  # Max 80% SOC
)
```

---

## Initial & Final Conditions

=== "Fixed Start"

    ```python
    fx.Storage(..., initial_charge_state=100)  # Start at 100 kWh
    ```

=== "Cyclic"

    Must end where it started (prevents "cheating"):

    ```python
    fx.Storage(..., initial_charge_state='equals_final')
    ```

=== "Final Bounds"

    ```python
    fx.Storage(...,
        minimal_final_charge_state=50,
        maximal_final_charge_state=150,
    )
    ```

---

## Adding Features

=== "Self-Discharge"

    ```python
    tank = fx.Storage(...,
        relative_loss_per_hour=0.02,  # 2%/hour loss
    )
    ```

=== "Variable Capacity"

    Optimize storage size:

    ```python
    battery = fx.Storage(...,
        capacity_in_flow_hours=fx.InvestParameters(
            minimum_size=0,
            maximum_size=1000,
            specific_effects={'costs': 200},  # €/kWh
        ),
    )
    ```

=== "Asymmetric Power"

    Different charge/discharge rates:

    ```python
    fx.Storage(
        charging=fx.Flow(..., size=100),     # 100 MW pump
        discharging=fx.Flow(..., size=120),  # 120 MW turbine
        ...
    )
    ```

---

## Reference

| Symbol | Type | Description |
|--------|------|-------------|
| $c(t)$ | $\mathbb{R}_{\geq 0}$ | Charge state at timestep $t$ |
| $C$ | $\mathbb{R}_{\geq 0}$ | Capacity (`capacity_in_flow_hours`) |
| $p_{in}(t)$ | $\mathbb{R}_{\geq 0}$ | Charging power (from `charging` flow) |
| $p_{out}(t)$ | $\mathbb{R}_{\geq 0}$ | Discharging power (from `discharging` flow) |
| $\eta_{in}$ | $\mathbb{R}_{\geq 0}$ | Charge efficiency (`eta_charge`) |
| $\eta_{out}$ | $\mathbb{R}_{\geq 0}$ | Discharge efficiency (`eta_discharge`) |
| $\dot{c}_{loss}$ | $\mathbb{R}_{\geq 0}$ | Self-discharge rate (`relative_loss_per_hour`) |
| $c_{rel}^{min}$ | $\mathbb{R}_{\geq 0}$ | Min charge state (`relative_minimum_charge_state`) |
| $c_{rel}^{max}$ | $\mathbb{R}_{\geq 0}$ | Max charge state (`relative_maximum_charge_state`) |
| $\Delta t$ | $\mathbb{R}_{> 0}$ | Timestep duration (hours) |

**Classes:** [`Storage`][flixopt.components.Storage], [`StorageModel`][flixopt.components.StorageModel]
