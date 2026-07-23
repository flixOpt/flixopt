# Migration Guide: v7.x → v8.0.0

!!! tip "Quick Start"
    ```bash
    pip install --upgrade flixopt
    ```
    v8.0.0 removes the long-deprecated v4-era APIs. If your code runs on v7.x
    **without `DeprecationWarning`s**, it will run unchanged on v8.0.0 — this
    guide is only relevant if you still use the old entry points below.

---

## Overview

Every API removed in v8.0.0 has been deprecated since v5/v6 and has emitted a
`DeprecationWarning` pointing to its replacement ever since. The replacements
are unchanged — nothing new to learn, only old spellings to drop:

| Removed | Replacement |
|---------|-------------|
| `fx.Optimization`, `fx.SegmentedOptimization` | `flow_system.optimize(solver)` or `build_model()` + `solve(solver)` |
| `fx.results` / `Results` classes | `flow_system.solution`, `flow_system.stats` |
| `FlowSystem.sel()` / `isel()` / `resample()` | `flow_system.transform.sel()` / `isel()` / `resample()` |
| `FlowSystem.coords` | `flow_system.indexes` |
| `FlowSystem.plot_network()`, `network_infos()` | `flow_system.topology.plot()`, `topology.infos()` |
| `start_network_app()` / `stop_network_app()` | `topology.start_app()` / `topology.stop_app()` |
| `topology.plot_legacy()` (PyVis) | `topology.plot()` (Plotly) |
| `FlowSystem.from_old_results()` | re-run the optimization with the current API |
| `Bus(excess_penalty_per_flow_hour=...)` | `Bus(imbalance_penalty_per_flow_hour=...)` |
| `optimize(..., normalize_weights=...)` | remove the argument — weights are always normalized |

---

## 💥 Breaking Changes in v8.0.0

### Optimization & Results classes removed

The pre-v5 workflow objects are gone. Solve directly on the `FlowSystem` and
read results from it:

=== "v7.x and earlier (removed)"
    ```python
    import flixopt as fx

    optimization = fx.Optimization('my_run', flow_system)
    optimization.do_modeling()
    optimization.solve(fx.solvers.HighsSolver())

    results = fx.results.Results.from_optimization(optimization)
    results.flow_rates()
    ```

=== "v8.0.0"
    ```python
    import flixopt as fx

    flow_system.optimize(fx.solvers.HighsSolver())

    flow_system.solution['Boiler(Q_th)|flow_rate']
    flow_system.stats.flow_rates
    ```

`SegmentedOptimization` is removed without a direct replacement yet; a new
segmented (rolling-horizon) API is planned.

### Data methods moved to the `transform` accessor

=== "v7.x and earlier (removed)"
    ```python
    fs_jan = flow_system.sel(time='2020-01')
    fs_2h = flow_system.resample('2h', method='mean')
    ```

=== "v8.0.0"
    ```python
    fs_jan = flow_system.transform.sel(time='2020-01')
    fs_2h = flow_system.transform.resample('2h', method='mean')
    ```

### Network visualization is Plotly-only

`plot_network()`, the PyVis-based `topology.plot_legacy()`, and the network
app wrappers on `FlowSystem` are removed. **PyVis is no longer a dependency.**

=== "v7.x and earlier (removed)"
    ```python
    flow_system.plot_network(show=True)
    flow_system.start_network_app()
    ```

=== "v8.0.0"
    ```python
    flow_system.topology.plot()
    flow_system.topology.start_app()
    ```

### Renamed `Bus` argument is no longer bridged

Passing the old name now raises a `TypeError` instead of warning:

=== "v7.x and earlier (removed)"
    ```python
    fx.Bus('Heat', excess_penalty_per_flow_hour=1e5)
    ```

=== "v8.0.0"
    ```python
    fx.Bus('Heat', imbalance_penalty_per_flow_hour=1e5)
    ```

### Old result files can no longer be loaded

`FlowSystem.from_old_results()` (the loader for pre-v5 `*--flow_system.nc4` +
`*--solution.nc4` result pairs) is removed — re-run those optimizations with
the current API.

!!! info "Old *configuration* files still load"
    `FlowSystem.from_old_dataset()` remains fully supported (and no longer warns): it loads a pre-v5
    `*--flow_system.nc4` configuration, renames old parameters, and returns a
    `FlowSystem` ready to re-optimize. Files saved with v5+
    (`to_netcdf`/`from_netcdf`) are unaffected by any of this.

    `from_old_dataset()` is planned for removal in **v9** — migrate pre-v5
    files to the current single-file format before then.

---

## Checklist

1. Run your project on v7.x with warnings visible:
   `python -W once::DeprecationWarning your_script.py`
2. Fix every `DeprecationWarning` using the table above.
3. Upgrade: `pip install --upgrade flixopt`.

If step 1 is silent, v8.0.0 is a drop-in upgrade.
