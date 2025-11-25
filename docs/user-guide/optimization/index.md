# Optimization Overview

flixOpt formulates energy and material flow problems as Mixed-Integer Linear Programming (MILP) models that can be solved with various optimization solvers.

## What Gets Optimized?

### Decision Variables

flixOpt creates decision variables for:

- **Flow rates** - Energy or material transfer at each timestep
- **Storage states** - Charge levels over time
- **Investment sizes** - Capacity decisions (when using `InvestParameters`)
- **On/off states** - Binary operational decisions (when using `OnOffParameters`)
- **Effect totals** - Aggregated costs, emissions, etc.

### Objective Function

The optimization minimizes or maximizes one `Effect`:

```python
costs = fx.Effect('costs', 'EUR', 'Minimize total system costs', is_objective=True)
```

Effects aggregate contributions from:

- Flow-based costs (€/kWh)
- Capacity-based costs (€/kW)
- Investment costs (€)
- Fixed costs (€)
- Cross-effect relationships (e.g., carbon pricing)

## Constraint Types

### Balance Constraints

**Nodal balance** at each bus ensures supply equals demand:

$$\sum \text{inputs} = \sum \text{outputs}$$

See [Bus](../mathematical-notation/elements/Bus.md) for details.

### Capacity Constraints

**Flow bounds** limit transfer rates:

$$\text{flow}_\text{min} \leq \text{flow}(t) \leq \text{flow}_\text{max}$$

See [Flow](../mathematical-notation/elements/Flow.md) for details.

### Storage Dynamics

**Charge state evolution** tracks energy levels:

$$\text{charge}(t+1) = \text{charge}(t) + \eta_\text{charge} \cdot \text{charge\_flow}(t) - \frac{\text{discharge\_flow}(t)}{\eta_\text{discharge}}$$

See [Storage](../mathematical-notation/elements/Storage.md) for details.

### Conversion Relationships

**Linear conversions** between flows:

$$\text{output}(t) = \eta \cdot \text{input}(t)$$

See [LinearConverter](../mathematical-notation/elements/LinearConverter.md) for details.

## Optimization Types

### Operational Optimization (Dispatch)

Optimize operation with **fixed capacities**:

- All component sizes are parameters
- Only operational decisions (flow rates, storage states)
- Typically shorter time horizons (days to weeks)
- Fast solve times

**Example:** Day-ahead power plant dispatch

### Investment Optimization (Capacity Expansion)

Optimize **capacity and operation together**:

- Component sizes are decision variables
- Uses `InvestParameters` for sizing
- Longer time horizons (months to years)
- Slower solve times due to binary/integer variables

**Example:** Renewable energy system planning

### Multi-Period Planning

Sequential investment decisions across **multiple time periods**:

- Two-stage optimization (investment + operation)
- Evolving conditions and technology costs
- Path-dependent decisions
- Most complex formulation

**Example:** Long-term energy transition pathways

## Mathematical Formulation

For complete mathematical details, see:

- **[Mathematical Notation Overview](../mathematical-notation/index.md)**
- **[Elements](../mathematical-notation/elements/Flow.md)** - Flow, Bus, Storage, LinearConverter
- **[Features](../mathematical-notation/features/InvestParameters.md)** - Investment, On/Off, Piecewise
- **[Effects & Objective](../mathematical-notation/effects-penalty-objective.md)**
- **[Modeling Patterns](../mathematical-notation/modeling-patterns/index.md)**

## Solver Options

### Choosing a Solver

flixOpt supports multiple solvers:

| Solver | Type | Speed | License |
|--------|------|-------|---------|
| **HiGHS** | Open-source | Fast | Free |
| **Gurobi** | Commercial | Fastest | Academic/Commercial |
| **CPLEX** | Commercial | Fastest | Academic/Commercial |
| **GLPK** | Open-source | Slower | Free |

**Recommendation:** Start with HiGHS (default). Use Gurobi/CPLEX for large models or when speed matters.

### Solver Configuration

Specify solver when solving:

```python
calc = fx.Optimization('my_model', flow_system)

calc.solve(
    solver=fx.solvers.GurobiSolver(
        time_limit_seconds=3600,
        mip_gap=0.01,
        extra_options={  # Add solver-specific options we didn't map
            'Threads': 4,           # Parallel threads
            'Presolve': 2           # Aggressive presolve
        }
    )
)
```

## Performance Optimization

### Model Size Reduction

- Use longer timesteps where acceptable
- Aggregate time periods (representative days/weeks)
- Remove unnecessary components
- Simplify constraint formulations

### Solver Tuning

- Enable presolve and cuts
- Adjust optimality tolerances
- Use heuristics for quick feasible solutions
- Enable warm starting from previous solutions

### Problem Formulation

- Avoid unnecessary binary variables
- Use continuous relaxations where possible
- Tighten variable bounds
- Remove redundant constraints

## Debugging Optimization

### Infeasibility

Model has no feasible solution:

1. Enable penalty variables to identify conflicts
2. Check balance constraints
3. Verify capacity limits
4. Review storage state requirements
5. Simplify to isolate issue

See [Troubleshooting](../troubleshooting.md) for details.

### Poor Performance

Optimization takes too long:

1. Reduce problem size
2. Try different solver
3. Adjust solver options
4. Simplify model formulation
5. Use longer timesteps

### Unexpected Results

Solution doesn't match expectations:

1. Verify input data
2. Check units and scales
3. Visualize intermediate results
4. Start with simpler model
5. Review constraint formulations

## Next Steps

- Study the [Mathematical Notation](../mathematical-notation/index.md)
- Learn about [Investment Parameters](../mathematical-notation/features/InvestParameters.md)
- Explore [Modeling Patterns](../mathematical-notation/modeling-patterns/index.md)
- Review [Examples](../../examples/index.md)
