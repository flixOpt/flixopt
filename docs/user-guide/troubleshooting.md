# Troubleshooting

Common issues and their solutions when working with flixOpt.

## Installation Issues

### Import Error: No module named 'flixopt'

**Problem:** Python can't find the flixOpt package.

**Solutions:**

1. Verify installation:
   ```bash
   pip list | grep flixopt
   ```

2. Install if missing:
   ```bash
   pip install flixopt
   ```

3. Check you're using the correct Python environment:
   ```bash
   which python
   python --version
   ```

### Solver Not Found

**Problem:** Error message about missing solver.

**Solutions:**

1. For HiGHS (default), reinstall flixOpt:
   ```bash
   pip install --upgrade --force-reinstall flixopt
   ```

2. For Gurobi/CPLEX, ensure the solver is installed and licensed
3. Specify solver explicitly:
   ```python
   calc = fx.Optimization('model', flow_system, solver=fx.solvers.HighsSolver())
   ```

## Modeling Issues

### Infeasible Model

**Problem:** Solver reports the model is infeasible.

**Diagnostic Steps:**

1. **Enable penalty variables** to identify problematic constraints:
   ```python
   system.use_penalty_variables = True
   ```

2. **Check balance constraints:**
   - Can supply meet demand at all timesteps?
   - Are there isolated buses with no input or output?

3. **Verify capacity limits:**
   - Do components have sufficient size?
   - Are upper/lower bounds consistent?

4. **Review storage constraints:**
   - Is initial charge state feasible?
   - Can storage charge/discharge meet requirements?

5. **Check temporal constraints:**
   - Are minimum on/off times achievable?
   - Do ramp rate limits allow necessary changes?

**Common Causes:**

- Demand exceeds total available capacity
- Storage initial/final states incompatible
- Over-constrained on/off requirements
- Inconsistent flow bounds

### Unbounded Model

**Problem:** Solver reports the model is unbounded.

**Solutions:**

1. Add upper bounds to all decision variables
2. Check that all flows have maximum limits
3. Ensure investment parameters have maximum sizes
4. Verify effect coefficients have correct signs

### Unexpected Results

**Problem:** Model solves but results don't make sense.

**Debugging Steps:**

1. **Enable logging:**
   ```python
   from flixopt import CONFIG
   CONFIG.exploring()
   ```

2. **Start simple:**
   - Build a minimal model first
   - Add complexity incrementally
   - Verify each addition

3. **Check units:**
   - Are all units consistent?
   - Do time series align with timesteps?

4. **Visualize results:**
   - Plot flows over time
   - Check energy balances
   - Verify storage states

5. **Validate input data:**
   - Check for NaN or infinite values
   - Ensure arrays have correct length
   - Verify parameter signs (costs should be positive)

## Performance Issues

### Slow Solve Times

**Problem:** Optimization takes too long.

**Solutions:**

1. **Reduce model size:**
   - Use longer timesteps
   - Aggregate time periods
   - Remove unnecessary components

2. **Simplify constraints:**
   - Relax tight bounds where possible
   - Remove redundant constraints
   - Use continuous instead of binary variables when appropriate

3. **Use a better solver:**
   ```python
   calc = fx.Optimization('model', flow_system, solver=fx.solvers.GurobiSolver())
   ```

4. **Set solver options:**
   ```python
   calc = fx.Optimization(
       'model',
       flow_system,
       solver=fx.solvers.GurobiSolver(
           time_limit_seconds=3600,
           mip_gap=0.01,  # 1% optimality gap
           extra_options={'Threads': 4}
       )
   )
   ```

5. **Enable presolve and cuts:**
   Most solvers have aggressive presolve options

### Memory Issues

**Problem:** Running out of memory.

**Solutions:**

1. Reduce time resolution
2. Use sparse matrices (automatic in flixOpt)
3. Process results in chunks
4. Increase system RAM or use high-memory machine

## Result Issues

### Can't Access Results

**Problem:** Error when accessing result attributes.

**Solutions:**

1. **Check solve status:**
   ```python
   print(calc.results.status)
   ```

2. **Verify optimization completed:**
   ```python
   if calc.results.status == 'optimal':
       # Access results
   else:
       print(f"Solver status: {calc.results.status}")
   ```

3. **Check component/flow names:**
   ```python
   print(calc.results.component_names)
   print(calc.results.get_component('name').flow_names)
   ```

### Missing Time Series Data

**Problem:** Some time series results are None or empty.

**Solutions:**

1. Verify the variable was created (check conditions in documentation)
2. Check if the component/feature is active
3. Ensure results object is from a completed solve

## Numerical Issues

### Scaling Problems

**Problem:** Warning about poor numerical conditioning.

**Solutions:**

1. **Normalize units:**
   - Use MW instead of W
   - Use MWh instead of Wh
   - Keep coefficients between 1e-3 and 1e3

2. **Check parameter magnitudes:**
   - Avoid very large or very small numbers
   - Scale costs and capacities appropriately

3. **Set solver tolerances:**
   ```python
   solver_options={'FeasibilityTol': 1e-6, 'OptimalityTol': 1e-6}
   ```

### Numerical Precision

**Problem:** Results have unexpected precision errors.

**Solutions:**

1. Use appropriate tolerance when comparing values
2. Round results for display: `round(value, 2)`
3. Check solver numerical tolerances

## Common Error Messages

### `KeyError: 'component_name'`

**Cause:** Trying to access a component that doesn't exist.

**Solution:** Check spelling and verify component was added to system.

### `ValueError: Array length mismatch`

**Cause:** Time series length doesn't match model timesteps.

**Solution:** Ensure all time series have length equal to `time_series.number_of_time_steps`.

### `AttributeError: 'NoneType' object has no attribute`

**Cause:** Accessing results before solving or on failed solve.

**Solution:** Check solve status before accessing results.

## Getting Help

If you've tried these solutions and still have issues:

1. **Search existing issues:** [GitHub Issues](https://github.com/flixOpt/flixopt/issues)
2. **Ask the community:** [GitHub Discussions](https://github.com/flixOpt/flixopt/discussions)
3. **Report a bug:** Open a new [issue](https://github.com/flixOpt/flixopt/issues/new) with:
   - Minimal reproducible example
   - flixOpt version
   - Python version
   - Operating system
   - Full error message

## Additional Resources

- [FAQ](faq.md) - Frequently asked questions
- [Support](support.md) - How to get help
- [Examples](../examples/index.md) - Working code examples
- [API Reference](../api-reference/) - Detailed documentation
