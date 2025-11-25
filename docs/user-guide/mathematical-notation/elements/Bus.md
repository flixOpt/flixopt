# Bus

A Bus is a connection point where flows meet and must balance. Think of it as a junction in your system where energy or material from multiple sources combines and is distributed to multiple consumers.

!!! example "Real-world examples"
    - **Heat Bus** — where boiler output, heat pump output, and storage discharge meet building demand
    - **Electricity Bus** — where generators, grid imports, and battery discharge meet electrical loads
    - **Gas Bus** — connection point to the gas grid

## The Balance Equation

The fundamental rule: **what goes in must equal what goes out**.

=== "Without Excess (Strict)"

    $$
    \sum_{f \in \mathcal{F}_{in}} p_f(t) = \sum_{f \in \mathcal{F}_{out}} p_f(t)
    $$

    At every timestep $t$, the sum of all incoming flow rates must equal the sum of all outgoing flow rates. If this can't be satisfied, the model is infeasible.

=== "With Excess (Soft)"

    When `excess_penalty_per_flow_hour` is set, slack variables allow imbalance:

    $$
    \sum_{f \in \mathcal{F}_{in}} p_f(t) + \phi_{in}(t) = \sum_{f \in \mathcal{F}_{out}} p_f(t) + \phi_{out}(t)
    $$

    Where:

    - $\phi_{in}(t)$ — "virtual supply" to cover shortages
    - $\phi_{out}(t)$ — "virtual demand" to absorb surplus

    Both are penalized in the objective:

    $$
    \Phi_b(t) = (\phi_{in}(t) + \phi_{out}(t)) \cdot \Delta t \cdot c_\phi
    $$

!!! note "Direction matters"
    Flows have a defined direction. An *input* to the bus means energy/material flowing **into** the bus. An *output* means flowing **out of** the bus.

!!! tip "Debugging with excess"
    If excess variables are non-zero in your solution, it means your system couldn't meet all constraints. Check:

    - Is demand too high for available capacity?
    - Are there timesteps where no supply is available?
    - Did you forget to connect a component?

## Variables

| Symbol | Python Name | Description | When Created |
|--------|-------------|-------------|--------------|
| $\phi_{in}(t)$ | `excess_input` | Virtual supply to cover shortages | `excess_penalty_per_flow_hour` is set |
| $\phi_{out}(t)$ | `excess_output` | Virtual demand to absorb surplus | `excess_penalty_per_flow_hour` is set |

## Parameters

| Symbol | Python Name | Description | Default |
|--------|-------------|-------------|---------|
| $c_\phi$ | `excess_penalty_per_flow_hour` | Penalty cost per unit imbalance | `1e5` |
| $\mathcal{F}_{in}$ | - | Set of input flows | From connected flows |
| $\mathcal{F}_{out}$ | - | Set of output flows | From connected flows |

## Usage Examples

### Strict Balance (No Imbalance Allowed)

```python
electricity_bus = fx.Bus(
    label='electricity',
    excess_penalty_per_flow_hour=None  # No slack allowed
)
```

If balance can't be achieved, the solver returns infeasible.

### Balance with Penalty (Debugging/Soft Constraints)

```python
heat_bus = fx.Bus(
    label='heat',
    excess_penalty_per_flow_hour=1e5  # High penalty for imbalance
)
```

Imbalance is allowed but heavily penalized. Use this to:

- Debug infeasible models
- Model emergency scenarios
- Allow small numerical tolerances

### Time-Varying Penalty

```python
# Higher penalty during peak hours
penalty_profile = [100, 100, 500, 500, 500, 100, 100, ...]

material_bus = fx.Bus(
    label='material',
    excess_penalty_per_flow_hour=penalty_profile
)
```

## How Buses Connect Components

Buses don't exist in isolation — they connect components through flows:

```
                    ┌─────────┐
     gas_in ───────►│  Gas    │
                    │   Bus   │
                    └────┬────┘
                         │ gas_out
                         ▼
                    ┌─────────┐
                    │ Boiler  │
                    └────┬────┘
                         │ heat_out
                         ▼
                    ┌─────────┐
     storage_out ──►│  Heat   │◄─── heat_pump_out
                    │   Bus   │
                    └────┬────┘
                         │ demand_in
                         ▼
                    ┌─────────┐
                    │ Demand  │
                    │ (Sink)  │
                    └─────────┘
```

Each arrow is a Flow. The Bus ensures that at every timestep:

$$
heat\_pump\_out + boiler\_out + storage\_out = demand\_in
$$

## Implementation Details

- **Element Class:** [`Bus`][flixopt.elements.Bus]
- **Model Class:** [`BusModel`][flixopt.elements.BusModel]

## See Also

- [Flow](Flow.md) — The flows that connect to buses
- [Effects & Objective](../effects-penalty-objective.md) — How penalties affect the objective
- [Core Concepts: Buses](../../core-concepts.md#buses-where-things-connect) — High-level overview
