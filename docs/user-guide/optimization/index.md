# Running Optimizations

This section covers how to run optimizations in flixOpt, including different optimization modes and solver configuration.

## Optimization Modes

flixOpt provides three optimization modes to handle different problem sizes and requirements:

### Optimization (Full)

[`Optimization`][flixopt.optimization.Optimization] solves the entire problem at once.

```python
import flixopt as fx

optimization = fx.Optimization('my_model', flow_system)
optimization.solve(fx.solvers.HighsSolver())
```

**Best for:**

- Small to medium problems
- When you need the globally optimal solution
- Problems without time-coupling simplifications

### SegmentedOptimization

[`SegmentedOptimization`][flixopt.optimization.SegmentedOptimization] splits the time horizon into segments and solves them sequentially.

```python
optimization = fx.SegmentedOptimization(
    'segmented_model',
    flow_system,
    segment_length=24,  # Hours per segment
    overlap_length=4    # Hours of overlap between segments
)
optimization.solve(fx.solvers.HighsSolver())
```

**Best for:**

- Large problems that don't fit in memory
- Long time horizons (weeks, months)
- Problems where decisions are mostly local in time

**Trade-offs:**

- Faster solve times
- May miss globally optimal solutions
- Overlap helps maintain solution quality at segment boundaries

### ClusteredOptimization

[`ClusteredOptimization`][flixopt.optimization.ClusteredOptimization] uses time series aggregation to reduce problem size by identifying representative periods.

```python
clustering_params = fx.ClusteringParameters(
    n_periods=8,           # Number of typical periods
    hours_per_period=24    # Hours per typical period
)

optimization = fx.ClusteredOptimization(
    'clustered_model',
    flow_system,
    clustering_params
)
optimization.solve(fx.solvers.HighsSolver())
```

**Best for:**

- Investment planning problems
- Year-long optimizations
- When computational speed is critical

**Trade-offs:**

- Much faster solve times
- Approximates the full problem
- Best when patterns repeat (e.g., typical days)

## Choosing an Optimization Mode

| Mode | Problem Size | Solve Time | Solution Quality |
|------|-------------|------------|------------------|
| `Optimization` | Small-Medium | Slow | Optimal |
| `SegmentedOptimization` | Large | Medium | Near-optimal |
| `ClusteredOptimization` | Very Large | Fast | Approximate |

## Solver Configuration

### Available Solvers

| Solver | Type | Speed | License |
|--------|------|-------|---------|
| **HiGHS** | Open-source | Fast | Free |
| **Gurobi** | Commercial | Fastest | Academic/Commercial |
| **CPLEX** | Commercial | Fastest | Academic/Commercial |
| **GLPK** | Open-source | Slower | Free |

**Recommendation:** Start with HiGHS (included by default). Use Gurobi/CPLEX for large models or when speed matters.

### Solver Options

```python
# Basic usage with defaults
optimization.solve(fx.solvers.HighsSolver())

# With custom options
optimization.solve(
    fx.solvers.GurobiSolver(
        time_limit_seconds=3600,
        mip_gap=0.01,
        extra_options={
            'Threads': 4,
            'Presolve': 2
        }
    )
)
```

Common solver parameters:

- `time_limit_seconds` - Maximum solve time
- `mip_gap` - Acceptable optimality gap (0.01 = 1%)
- `log_to_console` - Show solver output

## Performance Tips

### Model Size Reduction

- Use longer timesteps where acceptable
- Use `ClusteredOptimization` for long horizons
- Remove unnecessary components
- Simplify constraint formulations

### Solver Tuning

- Enable presolve and cuts
- Adjust optimality tolerances for faster (approximate) solutions
- Use parallel threads when available

### Problem Formulation

- Avoid unnecessary binary variables
- Use continuous investment sizes when possible
- Tighten variable bounds
- Remove redundant constraints

## Debugging

### Infeasibility

If your model has no feasible solution:

1. **Enable excess penalties on buses** to allow balance violations:
   ```python
   # Allow imbalance with high penalty cost (default is 1e5)
   heat_bus = fx.Bus('Heat', excess_penalty_per_flow_hour=1e5)

   # Or disable penalty to enforce strict balance
   electricity_bus = fx.Bus('Electricity', excess_penalty_per_flow_hour=None)
   ```
   When `excess_penalty_per_flow_hour` is set, the optimization can violate bus balance constraints by paying a penalty, helping identify which constraints cause infeasibility.

2. **Use Gurobi for infeasibility analysis** - When using GurobiSolver and the model is infeasible, flixOpt automatically extracts and logs the Irreducible Inconsistent Subsystem (IIS):
   ```python
   # Gurobi provides detailed infeasibility analysis
   optimization.solve(fx.solvers.GurobiSolver())
   # If infeasible, check the model documentation file for IIS details
   ```
   The infeasible constraints are saved to the model documentation file in the results folder.

3. Check balance constraints - can supply meet demand?
4. Verify capacity limits are consistent
5. Review storage state requirements
6. Simplify model to isolate the issue

See [Troubleshooting](../troubleshooting.md) for more details.

### Unexpected Results

If solutions don't match expectations:

1. Verify input data (units, scales)
2. Enable logging: `fx.CONFIG.exploring()`
3. Visualize intermediate results
4. Start with a simpler model
5. Check constraint formulations

## Next Steps

- See [Examples](../../examples/03-Optimization Modes.md) for working code
- Learn about [Mathematical Notation](../mathematical-notation/index.md)
- Explore [Recipes](../recipes/index.md) for common patterns
