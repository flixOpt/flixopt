# Example Models & Use Cases

flixOpt can model a wide variety of energy and material flow systems. This page provides an overview of common model types and applications.

## Energy System Models

### Simple Dispatch

Optimize operation of existing capacity with given power plants, storage, and demand.

**Typical components:**

- Generator flows with capacity and cost parameters
- Storage with charge/discharge dynamics
- Fixed or time-varying demand
- Grid connection with import/export limits

**Applications:** Day-ahead dispatch, operational planning, market participation

### Capacity Expansion

Determine optimal investment in new capacity alongside operational decisions.

**Typical components:**

- Components with `InvestParameters` for sizing decisions
- Multiple technology options (solar, wind, storage, etc.)
- Long-term time series (full year or representative periods)
- Investment cost parameters

**Applications:** Generation expansion planning, storage sizing, grid reinforcement

### Multi-Period Planning

Sequential investment decisions across multiple time periods with changing conditions.

**Typical components:**

- Two-stage optimization with investment and operational models
- Evolving demand and technology costs
- Existing capacity degradation
- Long-term scenarios

**Applications:** Long-term energy system pathways, infrastructure planning

## Sector Coupling Models

### Power-to-Heat

Integration of electric heat pumps, thermal storage, and power systems.

**Typical components:**

- Heat pumps as `LinearConverter` with COP
- Thermal storage with temperature layers
- Combined electricity and heat demand
- District heating networks

### Power-to-Gas

Hydrogen production via electrolysis with storage and reconversion.

**Typical components:**

- Electrolyzer as `LinearConverter`
- Hydrogen storage
- Fuel cells or gas turbines for reconversion
- Multiple energy carriers (electricity, hydrogen, heat)

### Combined Heat and Power (CHP)

Cogeneration systems with heat and power outputs.

**Typical components:**

- CHP unit with fixed heat/power ratio
- Heat and electricity buses
- Thermal and electrical storage
- Multiple demand profiles

## Industrial Applications

### Process Optimization

Material and energy flows in industrial processes.

**Typical components:**

- Multiple material flows with conversion
- Energy inputs (electricity, gas, heat)
- Process constraints and sequencing
- Quality or composition requirements

### Multi-Commodity Networks

Systems with multiple interacting energy carriers.

**Typical components:**

- Multiple bus types (electricity, gas, heat, hydrogen)
- Conversion technologies between carriers
- Storage for different commodities
- Network flow constraints

## Modeling Patterns

### Time Series Handling

- **Full resolution** - Hourly data for entire year (8760 hours)
- **Representative periods** - Typical days/weeks
- **Aggregated periods** - Multi-hour timesteps
- **Rolling horizon** - Sequential optimization windows

### Uncertainty Modeling

- **Scenario analysis** - Multiple demand/price scenarios
- **Stochastic optimization** - Probabilistic parameters
- **Robust optimization** - Worst-case scenarios
- **Sensitivity analysis** - Parameter variation studies

### Operational Constraints

- **Ramping limits** - Maximum change between timesteps
- **Minimum run time** - On/off parameters with duration tracking
- **Load-following** - Constraints on partial load operation
- **Reserve requirements** - Capacity held for contingencies

## Getting Started

To build these types of models:

1. **Start with examples** - See [Examples](../examples/index.md) for working code
2. **Learn core concepts** - Read [Core Concepts](../user-guide/core-concepts.md)
3. **Use recipes** - Follow [Recipes](../user-guide/recipes/index.md) for common patterns
4. **Study formulations** - Review [Mathematical Notation](../user-guide/mathematical-notation/index.md)

## Model Library

*Coming soon: A library of reusable model templates and building blocks*
