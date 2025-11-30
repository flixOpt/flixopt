# Bus

A Bus is where flows meet and must balance — inputs equal outputs at every timestep.

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

## With Excess Penalty

Allow imbalance for debugging or soft constraints:

$$
\sum_{in} p(t) + \phi_{in}(t) = \sum_{out} p(t) + \phi_{out}(t)
$$

The slack variables $\phi$ are penalized: $(\phi_{in} + \phi_{out}) \cdot \Delta t \cdot c_\phi$

```python
heat_bus = fx.Bus(
    label='heat',
    excess_penalty_per_flow_hour=1e5  # High penalty for imbalance
)
```

!!! tip "Debugging"
    If excess is non-zero in results → your system couldn't meet demand. Check capacities and connections.

---

## Reference

| Symbol | Type | Description |
|--------|------|-------------|
| $p(t)$ | $\mathbb{R}_{\geq 0}$ | Flow rate of connected flows |
| $\phi_{in}(t)$ | $\mathbb{R}_{\geq 0}$ | Slack: virtual supply (covers shortages) |
| $\phi_{out}(t)$ | $\mathbb{R}_{\geq 0}$ | Slack: virtual demand (absorbs surplus) |
| $c_\phi$ | $\mathbb{R}_{\geq 0}$ | Penalty factor (`excess_penalty_per_flow_hour`) |
| $\Delta t$ | $\mathbb{R}_{> 0}$ | Timestep duration (hours) |

**Classes:** [`Bus`][flixopt.elements.Bus], [`BusModel`][flixopt.elements.BusModel]
