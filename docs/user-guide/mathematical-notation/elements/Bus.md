# Bus

A Bus is where flows meet and must balance — inputs equal outputs at every timestep.

## Carriers

Buses can optionally be assigned a **carrier** — a type of energy or material (e.g., electricity, heat, gas). Carriers enable:

- **Automatic coloring** in plots based on energy type
- **Unit tracking** for better result visualization
- **Semantic grouping** of buses by type

```python
# Assign a carrier by name (uses CONFIG.Carriers defaults)
heat_bus = fx.Bus('HeatNetwork', carrier='heat')
elec_bus = fx.Bus('Grid', carrier='electricity')

# Or register custom carriers on the FlowSystem
biogas = fx.Carrier('biogas', color='#228B22', unit='kW', description='Biogas fuel')
flow_system.add_carrier(biogas)
gas_bus = fx.Bus('BiogasNetwork', carrier='biogas')
```

See [Color Management](../../../user-guide/results-plotting.md#color-management) for more on how carriers affect visualization.

---

## Basic: Balance Equation

$$
\sum_{in} p(t) = \sum_{out} p(t)
$$

```python
heat_bus = fx.Bus(label='heat')
# All flows connected to this bus must balance
```

If balance can't be achieved → model is **infeasible**.

---

## With Imbalance Penalty

Allow imbalance for debugging or soft constraints:

$$
\sum_{in} p(t) + \phi_{in}(t) = \sum_{out} p(t) + \phi_{out}(t)
$$

The slack variables $\phi$ are penalized: $(\phi_{in} + \phi_{out}) \cdot \Delta t \cdot c_\phi$

```python
heat_bus = fx.Bus(
    label='heat',
    imbalance_penalty_per_flow_hour=1e5  # High penalty for imbalance
)
```

!!! tip "Debugging"
    If you see a `virtual_demand` or `virtual_supply` and its non zero in results → your system couldn't meet demand. Check capacities and connections.

---

## Reference

| Symbol | Type | Description |
|--------|------|-------------|
| $p(t)$ | $\mathbb{R}_{\geq 0}$ | Flow rate of connected flows |
| $\phi_{in}(t)$ | $\mathbb{R}_{\geq 0}$ | Slack: virtual supply (covers shortages) |
| $\phi_{out}(t)$ | $\mathbb{R}_{\geq 0}$ | Slack: virtual demand (absorbs surplus) |
| $c_\phi$ | $\mathbb{R}_{\geq 0}$ | Penalty factor (`imbalance_penalty_per_flow_hour`) |
| $\Delta t$ | $\mathbb{R}_{> 0}$ | Timestep duration (hours) |

**Classes:** [`Bus`][flixopt.elements.Bus], [`BusModel`][flixopt.elements.BusModel]
