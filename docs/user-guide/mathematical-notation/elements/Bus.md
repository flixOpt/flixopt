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

| Variable | Description |
|----------|-------------|
| $\phi_{in}(t)$ | Virtual supply (covers shortages) |
| $\phi_{out}(t)$ | Virtual demand (absorbs surplus) |

| Parameter | Python | Default |
|-----------|--------|---------|
| Penalty | `excess_penalty_per_flow_hour` | `1e5` |

**Classes:** [`Bus`][flixopt.elements.Bus], [`BusModel`][flixopt.elements.BusModel]
