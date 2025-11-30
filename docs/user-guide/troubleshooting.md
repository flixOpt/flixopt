# Troubleshooting

## Infeasible Model

**Problem:** Solver reports the model is infeasible.

**Solutions:**

1. Check that supply can meet demand at all timesteps
2. Verify capacity limits are sufficient
3. Review storage initial/final states

## Unbounded Model

**Problem:** Solver reports the model is unbounded.

**Solutions:**

1. Add upper bounds to all flows
2. Ensure investment parameters have maximum sizes
3. Verify effect coefficients have correct signs

## Unexpected Results

**Debugging Steps:**

1. Enable logging:
   ```python
   from flixopt import CONFIG
   CONFIG.exploring()
   ```

2. Start with a minimal model and add complexity incrementally

3. Check units are consistent

4. Visualize results to verify energy balances

## Slow Solve Times

**Solutions:**

1. Use longer timesteps or aggregate time periods
2. Use Gurobi instead of HiGHS for large models
3. Set solver options:
   ```python
   solver = fx.solvers.GurobiSolver(
       time_limit_seconds=3600,
       mip_gap=0.01
   )
   ```

## Getting Help

If you're stuck:

1. Search [GitHub Issues](https://github.com/flixOpt/flixopt/issues)
2. Open a new issue with:
   - Minimal reproducible example
   - flixopt and Python version
   - Full error message
