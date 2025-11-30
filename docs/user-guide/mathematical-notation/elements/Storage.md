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

| Variable | Description |
|----------|-------------|
| $c(t)$ | Charge state |
| $p_{in}(t)$ | Charging power (from `charging` flow) |
| $p_{out}(t)$ | Discharging power (from `discharging` flow) |

| Parameter | Python | Default |
|-----------|--------|---------|
| Capacity | `capacity_in_flow_hours` | required |
| Charge efficiency | `eta_charge` | 1.0 |
| Discharge efficiency | `eta_discharge` | 1.0 |
| Self-discharge | `relative_loss_per_hour` | 0 |
| Initial charge | `initial_charge_state` | 0 |

**Classes:** [`Storage`][flixopt.components.Storage], [`StorageModel`][flixopt.components.StorageModel]
