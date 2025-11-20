# Model Types

flixOpt can model various energy and material flow systems.

## Common Model Types

### Dispatch Optimization

Optimize operation with fixed capacities:

- Generator and storage operation
- Fixed or time-varying demand
- Grid connection constraints
- Typically short term time horizons

### Capacity Expansion

Investment and operational decisions combined:

- Sizing with `InvestParameters`
- Technology comparison (solar, wind, storage)
- Annual time series or representative periods
- Investment costs and constraints

### Multi-Period Planning

Sequential decisions across multiple periods:

- Two-stage optimization
- Evolving costs and demand
- Long-term transformation pathways

## Sector Coupling

### Power-to-Heat
- Heat pumps (`LinearConverter` with COP)
- Thermal storage
- District heating networks

### Power-to-Gas
- Electrolyzers
- Hydrogen storage
- Fuel cells

### Combined Heat and Power
- CHP units with heat/power ratios
- Multiple demand profiles

## Time Resolution

- **Full resolution** - 8760 hours/year
- **Representative periods** - Typical days/weeks
- **Multi-hour timesteps** - Aggregated resolution

## Getting Started

1. Explore [Examples](../examples/index.md)
2. Read [Core Concepts](../user-guide/core-concepts.md)
3. Review [Mathematical Notation](../user-guide/mathematical-notation/index.md)
