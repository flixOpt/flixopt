# Changelog

This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Formatting is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) & [Gitmoji](https://gitmoji.dev).
For more details regarding the individual PRs and contributors, please refer to our [GitHub releases](https://github.com/flixOpt/flixopt/releases).

!!! tip

    If upgrading from v2.x, see the [v3.0.0 release notes](https://github.com/flixOpt/flixOpt/releases/tag/v3.0.0) and [Migration Guide](https://flixopt.github.io/flixopt/latest/user-guide/migration-guide-v3/).

---

<!-- This text won't be rendered
Note: The CI will automatically append a "What's Changed" section to the changelog for github releases.
This contains all commits, PRs, and contributors.
Therefore, the Changelog should focus on the user-facing changes.

Please remove all irrelevant sections before releasing.
Please keep the format of the changelog consistent: ## [VERSION] - YYYY-MM-DD
---

## [Template] - ????-??-??

**Summary**:

If upgrading from v2.x, see the [v3.0.0 release notes](https://github.com/flixOpt/flixOpt/releases/tag/v3.0.0) and [Migration Guide](https://flixopt.github.io/flixopt/latest/user-guide/migration-guide-v3/).

### ✨ Added

### 💥 Breaking Changes

### ♻️ Changed

### 🗑️ Deprecated

### 🔥 Removed

### 🐛 Fixed

### 🔒 Security

### 📦 Dependencies

### 📝 Docs

### 👷 Development

### 🚧 Known Issues

---

Until here -->

## [6.1.0] - Upcoming

**Summary**: Adds inner-period segmentation support to time-series clustering, enabling further reduction of problem size by grouping adjacent timesteps within each typical period into variable-length segments.

### ✨ Added

#### Inner-Period Segmentation for Clustering

Segmentation divides each typical period (cluster) into variable-length segments, dramatically reducing problem size while preserving key features of the time series.

```python
# Without segmentation: 8760h → 8 clusters × 24h = 192 timesteps
# With segmentation: 8760h → 8 clusters × 6 segments = 48 timesteps

fs_segmented = flow_system.transform.cluster(
    n_clusters=8,
    cluster_duration='1D',
    segmentation=True,           # Enable inner-period segmentation
    n_segments=6,                # Segments per cluster
)
fs_segmented.optimize(solver)
fs_expanded = fs_segmented.transform.expand_solution()
```

**New Parameters**:

| Parameter | Description |
|-----------|-------------|
| `segmentation` | Enable inner-period segmentation (default: `False`) |
| `n_segments` | Number of segments per cluster (required when `segmentation=True`) |
| `segment_representation_method` | How to represent segment values: `'meanRepresentation'` (default), `'medoidRepresentation'`, etc. |

**Key Features**:

- **Variable segment durations**: Each segment can have different duration (in hours), automatically determined by tsam based on time series characteristics
- **Full storage integration**: Works with all storage `cluster_mode` options including `'intercluster_cyclic'`
- **Solution expansion**: `expand_solution()` correctly maps segmented results back to original timesteps
- **RangeIndex timesteps**: Segmented FlowSystems use `RangeIndex` instead of `DatetimeIndex` for the time dimension
- **`is_segmented` property**: Check if a FlowSystem uses segmentation via `flow_system.is_segmented`

**Example with Storage**:

```python
storage = fx.Storage(
    'Battery',
    capacity_in_flow_hours=100,
    cluster_mode='intercluster_cyclic',
    ...
)

# Cluster with segmentation - extreme reduction
fs_segmented = flow_system.transform.cluster(
    n_clusters=12,
    cluster_duration='1D',
    segmentation=True,
    n_segments=4,  # 12 clusters × 4 segments = 48 timesteps (vs 12 × 24 = 288)
)
fs_segmented.optimize(solver)

# Expand back to full resolution
fs_expanded = fs_segmented.transform.expand_solution()
```

!!! tip "When to Use Segmentation"
    Segmentation is most beneficial for large-scale optimization problems where the additional reduction from 24 timesteps per cluster to ~4-8 segments significantly improves solve time. For problems that already solve quickly, standard clustering without segmentation may be sufficient.

---

## [6.0.0] - Upcoming

**Summary**: Major release featuring a complete rewrite of the clustering/aggregation system with tsam integration, new `fxplot` plotting accessor, FlowSystem comparison tools, and removal of deprecated v5.0 classes.

!!! warning "Breaking Changes"
    This release removes `ClusteredOptimization` and `ClusteringParameters` which were deprecated in v5.0.0. Use `flow_system.transform.cluster()` instead. See [Migration](#migration-from-clusteredoptimization) below.

### Key Features

- **Clustering/Aggregation Rework** (#549, #552) - Complete rewrite with tsam integration, inter-cluster storage linking, and 4 storage modes
- **fxplot Plotting Accessor** (#548) - Universal xarray plotting with automatic faceting
- **Comparison Module** (#550) - Compare multiple FlowSystems side-by-side
- **Improved Notebooks** (#542, #551) - Better tutorial data and faster CI execution

### ✨ Added

#### Time-Series Clustering (#549, #552)

Reduce large time series to representative typical periods for faster investment optimization, then expand results back to full resolution.

```python
# Stage 1: Cluster and optimize (fast sizing)
fs_clustered = flow_system.transform.cluster(
    n_clusters=12,                    # 12 typical days from a year
    cluster_duration='1D',            # Each cluster represents one day
    time_series_for_high_peaks=['HeatDemand(Q)|fixed_relative_profile'],
)
fs_clustered.optimize(solver)

# Stage 2: Expand back to full resolution
fs_expanded = fs_clustered.transform.expand_solution()
```

**Storage Modes for Clustering**: Control how storage behaves across clustered periods via `Storage(cluster_mode=...)`:

| Mode | Description | Use Case |
|------|-------------|----------|
| `'intercluster_cyclic'` | Links storage across clusters + yearly cyclic (default) | Seasonal storage with yearly optimization |
| `'intercluster'` | Links storage across clusters, free start/end | Multi-year optimization without cyclic constraint |
| `'cyclic'` | Each cluster independent, but cyclic (start = end) | Daily storage only, ignores seasonal patterns |
| `'independent'` | Each cluster fully independent, free start/end | Fastest solve, no long-term storage value |

**Clustering Parameters**:

| Parameter | Description |
|-----------|-------------|
| `n_clusters` | Number of representative periods to create |
| `cluster_duration` | Duration of each cluster (e.g., `'1D'`, `'24h'`, or hours as float) |
| `time_series_for_high_peaks` | Time series labels whose peaks should be preserved |
| `time_series_for_low_peaks` | Time series labels whose minima should be preserved |
| `cluster_method` | Algorithm: `'hierarchical'` (default), `'k_means'`, `'k_medoids'`, `'k_maxoids'`, `'averaging'` |
| `representation_method` | How to represent clusters: `'medoidRepresentation'` (default), `'meanRepresentation'`, `'distributionAndMinMaxRepresentation'` |
| `extreme_period_method` | How to handle extreme periods: `'append'`, `'new_cluster_center'`, `'replace_cluster_center'` |
| `rescale_cluster_periods` | Whether to rescale cluster periods to match original statistics (default: `True`) |
| `predef_cluster_order` | Predefined cluster assignment for reproducibility |
| `**tsam_kwargs` | Additional arguments passed to tsam |

**Key Features**:

- **Inter-cluster storage linking**: For `'intercluster'` and `'intercluster_cyclic'` modes, a `SOC_boundary` variable tracks absolute state-of-charge at period boundaries, enabling accurate seasonal storage modeling
- **Self-discharge decay**: Storage losses are correctly applied during solution expansion using the formula: `actual_SOC(t) = SOC_boundary × (1 - loss)^t + ΔE(t)`
- **Multi-dimensional support**: Works with periods, scenarios, and clusters dimensions simultaneously
- **Solution expansion**: `transform.expand_solution()` maps clustered results back to original timesteps with proper storage state reconstruction
- **Clustering IO**: Save and load clustered FlowSystems with full state preservation via `to_netcdf()` / `from_netcdf()`

**Example: Seasonal Storage with Clustering**:

```python
# Configure storage for seasonal behavior
storage = fx.Storage(
    'SeasonalPit',
    capacity_in_flow_hours=5000,
    cluster_mode='intercluster_cyclic',  # Enable seasonal storage in clustering
    relative_loss_per_hour=0.0001,       # Small self-discharge
    ...
)

# Cluster, optimize, and expand
fs_clustered = flow_system.transform.cluster(n_clusters=12, cluster_duration='1D')
fs_clustered.optimize(solver)
fs_expanded = fs_clustered.transform.expand_solution()

# Full-resolution charge state now available
charge_state = fs_expanded.solution['SeasonalPit|charge_state']
```

!!! tip "Choosing the Right Storage Mode"
    Use `'intercluster_cyclic'` (default) for seasonal storage like pit storage or underground thermal storage.
    Use `'cyclic'` for short-term storage like batteries or hot water tanks where only daily patterns matter.
    Use `'independent'` for quick estimates when storage behavior isn't critical.

#### FXPlot Accessor (#548)

New global xarray accessors for universal plotting with automatic faceting and smart dimension handling. Works on any xarray Dataset, not just flixopt results.

```python
import flixopt as fx  # Registers accessors automatically

# Plot any xarray Dataset with automatic faceting
dataset.fxplot.bar(x='component')
dataset.fxplot.area(x='time')
dataset.fxplot.heatmap(x='time', y='component')
dataset.fxplot.line(x='time', facet_col='scenario')

# DataArray support
data_array.fxplot.line()

# Statistics transformations
dataset.fxstats.to_duration_curve()
```

**Available Plot Methods**:

| Method | Description |
|--------|-------------|
| `.fxplot.bar()` | Grouped bar charts |
| `.fxplot.stacked_bar()` | Stacked bar charts |
| `.fxplot.line()` | Line charts with faceting |
| `.fxplot.area()` | Stacked area charts |
| `.fxplot.heatmap()` | Heatmap visualizations |
| `.fxplot.scatter()` | Scatter plots |
| `.fxplot.pie()` | Pie charts with faceting |
| `.fxstats.to_duration_curve()` | Transform to duration curve format |

**Key Features**:

- **Auto-faceting**: Automatically assigns extra dimensions (period, scenario, cluster) to `facet_col`, `facet_row`, or `animation_frame`
- **Smart x-axis**: Intelligently selects x dimension based on priority (time > duration > period > scenario)
- **Universal**: Works on any xarray Dataset/DataArray, not limited to flixopt
- **Configurable**: Customize via `CONFIG.Plotting` (colorscales, facet columns, line shapes)

#### FlowSystem Comparison (#550)

New `Comparison` class for comparing multiple FlowSystems side-by-side:

```python
# Compare systems (uses FlowSystem.name by default)
comp = fx.Comparison([fs_base, fs_modified])

# Or with custom names
comp = fx.Comparison([fs1, fs2, fs3], names=['baseline', 'low_cost', 'high_eff'])

# Side-by-side plots (auto-facets by 'case' dimension)
comp.statistics.plot.balance('Heat')
comp.statistics.flow_rates.fxplot.line()

# Access combined data with 'case' dimension
comp.solution  # xr.Dataset
comp.statistics.flow_rates  # xr.Dataset

# Compute differences relative to a reference case
comp.diff()  # vs first case
comp.diff('baseline')  # vs named case
```

- Concatenates solutions and statistics from multiple FlowSystems with a `'case'` dimension
- Mirrors all `StatisticsAccessor` properties (`flow_rates`, `flow_hours`, `sizes`, `charge_states`, `temporal_effects`, `periodic_effects`, `total_effects`)
- Mirrors all `StatisticsPlotAccessor` methods (`balance`, `carrier_balance`, `flows`, `sizes`, `duration_curve`, `effects`, `charge_states`, `heatmap`, `storage`)
- Existing plotting infrastructure automatically handles faceting by `'case'`

### 💥 Breaking Changes

- `FlowSystem.scenario_weights` are now always normalized to sum to 1 when set (including after `.sel()` subsetting)

### ♻️ Changed

- `FlowSystem.weights` returns `dict[str, xr.DataArray]` (unit weights instead of `1.0` float fallback)
- `FlowSystemDimensions` type now includes `'cluster'`

### 🗑️ Deprecated

The following items are deprecated and will be removed in **v7.0.0**:

**Classes** (use FlowSystem methods instead):

- `Optimization` class → Use `flow_system.optimize(solver)`
- `SegmentedOptimization` class → Use `flow_system.optimize.rolling_horizon()`
- `Results` class → Use `flow_system.solution` and `flow_system.statistics`
- `SegmentedResults` class → Use segment FlowSystems directly

**FlowSystem methods** (use `transform` or `topology` accessor instead):

- `flow_system.sel()` → Use `flow_system.transform.sel()`
- `flow_system.isel()` → Use `flow_system.transform.isel()`
- `flow_system.resample()` → Use `flow_system.transform.resample()`
- `flow_system.plot_network()` → Use `flow_system.topology.plot()`
- `flow_system.start_network_app()` → Use `flow_system.topology.start_app()`
- `flow_system.stop_network_app()` → Use `flow_system.topology.stop_app()`
- `flow_system.network_infos()` → Use `flow_system.topology.infos()`

**Parameters:**

- `normalize_weights` parameter in `create_model()`, `build_model()`, `optimize()`

**Topology method name simplifications** (old names still work with deprecation warnings, removal in v7.0.0):

| Old (v5.x) | New (v6.0.0) |
|------------|--------------|
| `topology.plot_network()` | `topology.plot()` |
| `topology.start_network_app()` | `topology.start_app()` |
| `topology.stop_network_app()` | `topology.stop_app()` |
| `topology.network_infos()` | `topology.infos()` |

Note: `topology.plot()` now renders a Sankey diagram. The old PyVis visualization is available via `topology.plot_legacy()`.

### 🔥 Removed

**Clustering classes removed** (deprecated in v5.0.0):

- `ClusteredOptimization` class - Use `flow_system.transform.cluster()` then `optimize()`
- `ClusteringParameters` class - Parameters are now passed directly to `transform.cluster()`
- `flixopt/clustering.py` module - Restructured to `flixopt/clustering/` package with new classes

#### Migration from ClusteredOptimization

=== "v5.x (Old - No longer works)"
    ```python
    from flixopt import ClusteredOptimization, ClusteringParameters

    params = ClusteringParameters(hours_per_period=24, nr_of_periods=8)
    calc = ClusteredOptimization('model', flow_system, params)
    calc.do_modeling_and_solve(solver)
    results = calc.results
    ```

=== "v6.0.0 (New)"
    ```python
    # Cluster using transform accessor
    fs_clustered = flow_system.transform.cluster(
        n_clusters=8,           # was: nr_of_periods
        cluster_duration='1D',  # was: hours_per_period=24
    )
    fs_clustered.optimize(solver)

    # Results on the clustered FlowSystem
    costs = fs_clustered.solution['costs'].item()

    # Expand back to full resolution if needed
    fs_expanded = fs_clustered.transform.expand_solution()
    ```

### 🐛 Fixed

- `temporal_weight` and `sum_temporal()` now use consistent implementation
- `FlowSystem.from_old_results()` now sets `previous_flow_rate=0` for flows of components with `status_parameters`, fixing startup cost calculation mismatch when re-optimizing migrated v4 results

### 📝 Docs

**New Documentation Pages:**

- [Time-Series Clustering Guide](https://flixopt.github.io/flixopt/latest/user-guide/optimization/clustering/) - Comprehensive guide to clustering workflows
- Cluster architecture design documentation (`docs/design/cluster_architecture.md`)

**New Jupyter Notebooks** (#542):

- **08c-clustering.ipynb** - Introduction to time-series clustering
- **08c2-clustering-storage-modes.ipynb** - Comparison of all 4 storage cluster modes
- **08d-clustering-multiperiod.ipynb** - Clustering with periods and scenarios
- **08e-clustering-internals.ipynb** - Understanding clustering internals
- **fxplot_accessor_demo.ipynb** - Demo of the new fxplot accessor

**Improved Tutorials:**

- Added `tutorial_data.py` helper module for cleaner notebook examples
- Updated all existing notebooks to use new clustering and plotting APIs

### 👷 Development

**CI Improvements** (#551):

- Speedup notebook execution in documentation builds

**New Test Suites for Clustering**:

- `TestStorageClusterModes`: Tests for all 4 storage `cluster_mode` options
- `TestInterclusterStorageLinking`: Tests for `SOC_boundary` variable and expansion logic
- `TestMultiPeriodClustering`: Tests for clustering with periods and scenarios dimensions
- `TestPeakSelection`: Tests for `time_series_for_high_peaks` and `time_series_for_low_peaks` parameters

**New Test Suites for Other Features**:

- `test_clustering_io.py` - Tests for clustering serialization roundtrip
- `test_sel_isel_single_selection.py` - Tests for transform selection methods

---

## [5.0.4] - 2026-01-05

**Summary**: Dependency updates.

### 🐛 Fixed

- Fixed netcdf dependency

### 📦 Dependencies

- Updated `mkdocs-material` to v9.7.1
- Updated `mkdocstrings-python` to v1.19.0
- Updated `ruff` to v0.14.10
- Updated `pymdown-extensions` to v10.19.1
- Updated `werkzeug` to v3.1.4

### 👷 Development

- Updated `actions/checkout` action to v6

---

## [5.0.3] - 2025-12-18

**Summary**: Cleaner notebook outputs and improved `CONFIG.notebook()` preset.

### ♻️ Changed

- `CONFIG.notebook()` now suppresses linopy progress bars via `progress=False` in solve calls
- Downgraded "FlowSystem not connected" message from WARNING to INFO (auto-connects anyway)

### 🐛 Fixed

- Fixed notebooks triggering unnecessary warnings (removed `relative_minimum` without `status_parameters`)

### 📝 Docs

- Consolidated verbose print statements into concise single-line summaries across all tutorial notebooks
- Added output suppression (`;`) to `optimize()` calls for cleaner cell output
- Fixed notebook 07 parameters so CHP investment produces interesting results (was 0 kW, now 100 kW)

---

## [5.0.2] - 2025-12-17

### ♻️ Changed

- `statistics.plot.effects()` now defaults to `by=None` for aggregated totals; use `by='component'` for the previous behavior

---

## [5.0.1] - 2025-12-17

### 👷 Development

- Fixed docs deployment in CI workflow

---

## [5.0.0] - 2025-12-16

**Summary**: This is a major release that introduces the new **FlowSystem-centric API**, dramatically simplifying workflows by integrating optimization, results access, and visualization directly into the FlowSystem object. This release also completes the terminology standardization (OnOff → Status) and **deprecates the old Optimization/Results workflow** (to be removed in v6.0.0).

!!! tip "Migration Guide"

    See the [Migration Guide v5](https://flixopt.github.io/flixopt/latest/user-guide/migration-guide-v5/) for step-by-step upgrade instructions.

### ✨ Added

**FlowSystem-Centric Architecture**: The FlowSystem is now the central hub for all operations:

```python
import flixopt as fx

# Create and configure your system
flow_system = fx.FlowSystem(timesteps)
flow_system.add_elements(boiler, heat_bus, costs)

# Optimize directly on FlowSystem
flow_system.optimize(fx.solvers.HighsSolver())

# Access results via solution Dataset
total_costs = flow_system.solution['costs'].item()
flow_rate = flow_system.solution['Boiler(Q_th)|flow_rate'].values

# Plot with new accessor API
flow_system.statistics.plot.balance('HeatBus')
flow_system.statistics.plot.sankey.flows()
```

**New Accessor-Based API**: Four accessor patterns provide organized, discoverable interfaces:

| Accessor | Purpose | Example |
|----------|---------|---------|
| `flow_system.statistics` | Data access (flow rates, sizes, effects) | `flow_system.statistics.flow_rates` |
| `flow_system.statistics.plot` | Visualization methods | `flow_system.statistics.plot.balance('Bus')` |
| `flow_system.transform` | FlowSystem transformations | `flow_system.transform.cluster(params)` |
| `flow_system.topology` | Network structure & visualization | `flow_system.topology.plot_network()` |

**Statistics Accessor**: Access aggregated results data with clean, consistent naming:

```python
stats = flow_system.statistics

# Flow data (clean labels, no |flow_rate suffix needed)
stats.flow_rates['Boiler(Q_th)']
stats.flow_hours['Boiler(Q_th)']
stats.sizes['Boiler(Q_th)']
stats.charge_states['Battery']

# Effect breakdown by contributor
stats.temporal_effects['costs']   # Per timestep, per contributor
stats.periodic_effects['costs']   # Investment costs per contributor
stats.total_effects['costs']      # Total per contributor
```

**Comprehensive Plotting API**: All plots return `PlotResult` objects with chainable methods:

```python
# Balance plots for buses and components
flow_system.statistics.plot.balance('ElectricityBus')
flow_system.statistics.plot.balance('Boiler', mode='area')

# Storage visualization with charge state
flow_system.statistics.plot.storage('Battery')

# Heatmaps with automatic time reshaping
flow_system.statistics.plot.heatmap('Boiler(Q_th)|flow_rate', reshape=('D', 'h'))

# Flow-based Sankey diagrams
flow_system.statistics.plot.sankey.flows()
flow_system.statistics.plot.sankey.flows(select={'bus': 'ElectricityBus'})

# Effect contribution Sankey
flow_system.statistics.plot.sankey.effects('costs')

# Method chaining for customization and export
flow_system.statistics.plot.balance('Bus') \
    .update(title='Custom Title', height=600) \
    .to_html('plot.html') \
    .to_csv('data.csv') \
    .show()
```

**Carrier Management**: New `Carrier` class for consistent styling across visualizations:

```python
# Define custom carriers
electricity = fx.Carrier('electricity', '#FFD700', 'kW', 'Electrical power')
district_heat = fx.Carrier('district_heat', '#FF6B6B', 'kW_th')

# Register with FlowSystem
flow_system.add_carrier(electricity)

# Use with buses (reference by name)
elec_bus = fx.Bus('MainGrid', carrier='electricity')

# Or use predefined carriers from CONFIG
fx.CONFIG.Carriers.electricity
fx.CONFIG.Carriers.heat
```

**Transform Accessor**: Transformations that create new FlowSystem instances:

```python
# Time selection and resampling
fs_subset = flow_system.transform.sel(time=slice('2023-01-01', '2023-06-30'))
fs_resampled = flow_system.transform.resample(time='4h', method='mean')

# Clustered optimization
params = fx.ClusteringParameters(hours_per_period=24, nr_of_periods=8)
clustered_fs = flow_system.transform.cluster(params)
clustered_fs.optimize(solver)
```

**Rolling Horizon Optimization**: Decompose large operational problems into sequential segments:

```python
# Solve with rolling horizon
segments = flow_system.optimize.rolling_horizon(
    solver,
    horizon=192,    # Timesteps per segment
    overlap=48,     # Lookahead for storage optimization
)

# Combined solution available on original FlowSystem
total_cost = flow_system.solution['costs'].item()

# Individual segments also available
for seg in segments:
    print(seg.solution['costs'].item())
```

**Solution Persistence**: FlowSystem now stores and persists solutions:

```python
# Optimize and save with solution
flow_system.optimize(solver)
flow_system.to_netcdf('results/my_model.nc4')

# Load FlowSystem with solution intact
loaded_fs = fx.FlowSystem.from_netcdf('results/my_model.nc4')
print(loaded_fs.solution['costs'].item())  # Solution is available!
```

**Migration Helper for Old Results** (deprecated, temporary):

```python
# Migrate old result files to new FlowSystem format
fs = fx.FlowSystem.from_old_results('results_folder', 'my_model')
# Or convert Results object directly
fs = results.convert_to_flow_system()
```

**FlowSystem Locking**: FlowSystem automatically locks after optimization to prevent accidental modifications:

```python
flow_system.optimize(solver)

# This would raise an error:
# flow_system.add_elements(new_component)  # Locked!

# Call reset() to unlock for modifications
flow_system.reset()
flow_system.add_elements(new_component)  # Now works
```

**NetCDF Improvements**:
- Default compression level 5 for smaller files
- `overwrite=False` parameter to prevent accidental overwrites
- Solution data included in FlowSystem NetCDF files
- Automatic name assignment from filename

**PlotResult Class**: All plotting methods return a `PlotResult` object containing both:
- `data`: An xarray Dataset with the prepared data
- `figure`: A Plotly Figure object

**Component color parameter**: Components now accept a `color` parameter for consistent visualization styling.

### 💥 Breaking Changes

**Renamed `OnOffParameters` → `StatusParameters`**: Complete terminology update to align with industry standards (PyPSA, unit commitment). Old NetCDF files with `OnOffParameters` are automatically converted on load.

| Old Term | New Term |
|----------|----------|
| `OnOffParameters` | `StatusParameters` |
| `on_off_parameters` | `status_parameters` |
| `on` variable | `status` |
| `switch_on` | `startup` |
| `switch_off` | `shutdown` |
| `switch_on_nr` | `startup_count` |
| `on_hours_total` | `active_hours` |
| `consecutive_on_hours` | `uptime` |
| `consecutive_off_hours` | `downtime` |
| `effects_per_switch_on` | `effects_per_startup` |
| `effects_per_running_hour` | `effects_per_active_hour` |
| `consecutive_on_hours_min` | `min_uptime` |
| `consecutive_on_hours_max` | `max_uptime` |
| `consecutive_off_hours_min` | `min_downtime` |
| `consecutive_off_hours_max` | `max_downtime` |
| `switch_on_total_max` | `startup_limit` |
| `force_switch_on` | `force_startup_tracking` |
| `on_hours_min` | `active_hours_min` |
| `on_hours_max` | `active_hours_max` |

**Bus imbalance terminology and default changed**:
- `excess_penalty_per_flow_hour` → `imbalance_penalty_per_flow_hour`
- Default changed from `1e5` to `None` (strict balance)
- `with_excess` → `allows_imbalance`
- `excess_input` → `virtual_supply`
- `excess_output` → `virtual_demand`

**Storage charge_state changes**:
- `charge_state` no longer has an extra timestep
- Final charge state is now a separate variable: `charge_state|final`

**Effect.description** now defaults to `''` (empty string) instead of `None`.

**Stricter I/O**: NetCDF loading is stricter to prevent silent errors. Missing or corrupted data now raises explicit errors.

**Validation**: Component with `status_parameters` now validates that all flows have sizes (required for big-M constraints).

### ♻️ Changed

- Renamed `BusModel.excess_input` → `virtual_supply` and `BusModel.excess_output` → `virtual_demand` for clearer semantics
- Renamed `Bus.excess_penalty_per_flow_hour` → `imbalance_penalty_per_flow_hour`
- Renamed `Bus.with_excess` → `allows_imbalance`

### 🗑️ Deprecated

All deprecated items will be removed in v6.0.0.

**Old Optimization Workflow** - Use FlowSystem methods instead:

```python
# Old (deprecated, still works with warning)
optimization = fx.Optimization('model', flow_system)
optimization.do_modeling()
optimization.solve(solver)
results = optimization.results
costs = results.model['costs'].solution.item()

# New (recommended)
flow_system.optimize(solver)
costs = flow_system.solution['costs'].item()
```

**Classes deprecated:**
- `Optimization` → Use `flow_system.optimize(solver)`
- `ClusteredOptimization` → Use `flow_system.transform.cluster()` then `optimize()`
- `SegmentedOptimization` → Use `flow_system.optimize.rolling_horizon()`
- `Results` → Use `flow_system.solution` and `flow_system.statistics`
- `SegmentedResults` → Use segment FlowSystems directly

**FlowSystem methods deprecated:**
- `flow_system.sel()` → Use `flow_system.transform.sel()`
- `flow_system.isel()` → Use `flow_system.transform.isel()`
- `flow_system.resample()` → Use `flow_system.transform.resample()`
- `flow_system.plot_network()` → Use `flow_system.topology.plot_network()`
- `flow_system.start_network_app()` → Use `flow_system.topology.start_network_app()`
- `flow_system.stop_network_app()` → Use `flow_system.topology.stop_network_app()`
- `flow_system.network_infos()` → Use `flow_system.topology.network_infos()`

**Results methods deprecated:**
- `results.flow_rates()` → Use `flow_system.statistics.flow_rates`
- `results.flow_hours()` → Use `flow_system.statistics.flow_hours`

**Migration helpers (temporary, also deprecated):**
- `FlowSystem.from_old_results()` → For migrating old result files
- `Results.convert_to_flow_system()` → For converting Results objects

**Plotting parameters deprecated:**
- `indexer` parameter → Use `select` instead
- `heatmap_timeframes` parameter → Use `reshape_time=(timeframes, timesteps_per_frame)` instead
- `heatmap_timesteps_per_frame` parameter → Use `reshape_time=(timeframes, timesteps_per_frame)` instead
- `color_map` parameter → Use `colors` instead

### 🔥 Removed

**Python version changes:**
- Dropped Python 3.10 support
- Added Python 3.14 support

**Classes removed** (already renamed/deprecated in v4.x):
- `OnOffParameters` → Use `StatusParameters`
- `Calculation` → Use `Optimization` (deprecated) or `flow_system.optimize()` (recommended)
- `FullCalculation` → Use `Optimization` (deprecated) or `flow_system.optimize()` (recommended)
- `AggregatedCalculation` → Use `ClusteredOptimization` (deprecated) or `flow_system.transform.cluster()` (recommended)
- `SegmentedCalculation` → Use `SegmentedOptimization` (deprecated) or `flow_system.optimize.rolling_horizon()` (recommended)
- `Aggregation` → Use `Clustering`
- `AggregationParameters` → Use `ClusteringParameters`
- `AggregationModel` → Use `ClusteringModel`
- `CalculationResults` → Use `Results` (deprecated) or `flow_system.solution` (recommended)
- `SegmentedCalculationResults` → Use `SegmentedResults` (deprecated)

**Modules removed:**
- `calculation.py` module → Use `optimization.py` (deprecated) or FlowSystem methods (recommended)

**Functions removed:**
- `change_logging_level()` → Use `CONFIG.Logging.enable_console()`

**Properties removed:**
- `FlowSystem.all_elements` → Use dict-like interface (`flow_system['label']`, `.keys()`, `.values()`, `.items()`)
- `FlowSystem.weights` → Use `scenario_weights`

### 📝 Docs

**Comprehensive Tutorial Notebooks** - 12 new Jupyter notebooks covering all major use cases:

1. **01-Quickstart** - Minimal working example
2. **02-Heat System** - District heating with storage
3. **03-Investment Optimization** - Optimal equipment sizing
4. **04-Operational Constraints** - Startup costs, uptime/downtime
5. **05-Multi-Carrier System** - CHP producing electricity and heat
6. **06a-Time-Varying Parameters** - Temperature-dependent COP
7. **06b-Piecewise Conversion** - Load-dependent efficiency
8. **06c-Piecewise Effects** - Economies of scale
9. **07-Scenarios and Periods** - Multi-year planning
10. **08-Large-Scale Optimization** - Resampling and two-stage
11. **09-Plotting and Data Access** - Visualization guide
12. **10-Transmission** - Connecting sites with pipelines/cables

**New Documentation Pages:**
- Migration Guide v5 - Step-by-step upgrade instructions
- Results & Plotting Guide - Comprehensive plotting documentation
- Building Models Guide - Component selection and modeling patterns
- FAQ - Common questions and answers
- Troubleshooting - Problem diagnosis and solutions

### 👷 Development

**New Test Suites:**
- `test_flow_system_locking.py` - FlowSystem locking behavior
- `test_solution_and_plotting.py` - Statistics accessor and plotting
- `test_solution_persistence.py` - Solution save/load
- `test_io_conversion.py` - Old file format conversion
- `test_topology_accessor.py` - Network visualization

**CI Improvements:**
- Separate docs build and deploy workflow
- Improved test organization with deprecated tests in separate folder

### Migration Checklist

The old `Optimization`/`Results` workflow still works with deprecation warnings. Migrate at your own pace before v6.0.0.

| Task | Action |
|------|--------|
| Update `OnOffParameters` | Rename to `StatusParameters` with new parameter names (breaking) |
| Update `on_off_parameters` | Rename to `status_parameters` (breaking) |
| Update Bus excess parameters | Use `imbalance_penalty_per_flow_hour` (breaking) |
| Replace `Optimization` class | Use `flow_system.optimize(solver)` (deprecated) |
| Replace `SegmentedOptimization` | Use `flow_system.optimize.rolling_horizon(solver, ...)` (deprecated) |
| Replace `Results` access | Use `flow_system.solution['var_name']` (deprecated) |
| Update transform methods | Use `flow_system.transform.sel/isel/resample()` (deprecated) |
| Update I/O code | Use `to_netcdf()` / `from_netcdf()` on FlowSystem |
| Migrate old result files | Use `FlowSystem.from_old_results(folder, name)` (temporary helper) |

---

## [4.3.5] - 2025-11-29

**Summary**: Fix zenodo again

If upgrading from v2.x, see the [v3.0.0 release notes](https://github.com/flixOpt/flixOpt/releases/tag/v3.0.0) and [Migration Guide](https://flixopt.github.io/flixopt/latest/user-guide/migration-guide-v3/).

---

## [4.3.4] - 2025-11-27

**Summary**: Fix zenodo again

If upgrading from v2.x, see the [v3.0.0 release notes](https://github.com/flixOpt/flixOpt/releases/tag/v3.0.0) and [Migration Guide](https://flixopt.github.io/flixopt/latest/user-guide/migration-guide-v3/).

---

## [4.3.3] - 2025-11-27

**Summary**: Fix zenodo again

If upgrading from v2.x, see the [v3.0.0 release notes](https://github.com/flixOpt/flixOpt/releases/tag/v3.0.0) and [Migration Guide](https://flixopt.github.io/flixopt/latest/user-guide/migration-guide-v3/).

---

## [4.3.2] - 2025-11-27

**Summary**: Fix zenodo

If upgrading from v2.x, see the [v3.0.0 release notes](https://github.com/flixOpt/flixOpt/releases/tag/v3.0.0) and [Migration Guide](https://flixopt.github.io/flixopt/latest/user-guide/migration-guide-v3/).

---

## [4.3.1] - 2025-11-26

**Summary**: Add zenodo for better citations and archiving.

If upgrading from v2.x, see the [v3.0.0 release notes](https://github.com/flixOpt/flixOpt/releases/tag/v3.0.0) and [Migration Guide](https://flixopt.github.io/flixopt/latest/user-guide/migration-guide-v3/).

### 📝 Docs
- Added Zenodo DOI badge to README.md (placeholder, to be updated after first Zenodo release)

### 👷 Development
- Added Zenodo integration for automatic archival and citation
  - Created `.zenodo.json` file for Zenodo metadata configuration
  - Repository now ready for DOI assignment upon next release

---

## [4.3.0] - 2025-11-25

**Summary**: Penalty is now a first-class Effect - add penalty contributions anywhere (e.g., `effects_per_flow_hour={'Penalty': 2.5}`) and optionally define bounds as with any other effect.

If upgrading from v2.x, see the [v3.0.0 release notes](https://github.com/flixOpt/flixOpt/releases/tag/v3.0.0) and [Migration Guide](https://flixopt.github.io/flixopt/latest/user-guide/migration-guide-v3/).

### ✨ Added

- **Penalty as first-class Effect**: Users can now add Penalty contributions anywhere effects are used:
  ```python
  fx.Flow('Q', 'Bus', effects_per_flow_hour={'Penalty': 2.5})
  fx.InvestParameters(..., effects_of_investment={'Penalty': 100})
  ```
- **User-definable Penalty**: Optionally define custom Penalty with constraints (auto-created if not defined):
  ```python
  penalty = fx.Effect(fx.PENALTY_EFFECT_LABEL, unit='€', maximum_total=1e6)
  flow_system.add_elements(penalty)
  ```

### ♻️ Changed

- Penalty is now a standard Effect with temporal/periodic dimensions, and periodic weights in the objective
- **Results structure**: Penalty now has same structure as other effects in solution Dataset
  - Use `results.solution['Penalty']` for total penalty value (same as before, but now it's an effect variable)
  - Access components via `results.solution['Penalty(temporal)']` and `results.solution['Penalty(periodic)']` if needed

### 📝 Docs

- Updated mathematical notation for Penalty as Effect

### 👷 Development

- Unified interface: Penalty uses same `add_share_to_effects()` as other effects (internal only)

---

## [4.2.0] - 2025-11-25

**Summary**: Renamed classes and parameters related to `Calculation`, `Aggregation` and `Results`. Fully backwards compatible

If upgrading from v2.x, see the [v3.0.0 release notes](https://github.com/flixOpt/flixOpt/releases/tag/v3.0.0) and [Migration Guide](https://flixopt.github.io/flixopt/latest/user-guide/migration-guide-v3/).

### ✨ Added
- `overwrite` parameter when saving results to file. If True, overwrite existing files.

### ♻️ Changed

- Now creates the Results folder even if parents didnt exist

### 🗑️ Deprecated

**Class and module renaming:**
- `FullCalculation` → `Optimization`
- `AggregatedCalculation` → `ClusteredOptimization`
- `SegmentedCalculation` → `SegmentedOptimization`
- `CalculationResults` → `Results`
- `SegmentedCalculationResults` → `SegmentedResults`
- `Aggregation` → `Clustering`
- `AggregationParameters` → `ClusteringParameters`
- `AggregationModel` → `ClusteringModel`
- Module: `calculation.py` → `optimization.py`
- Module: `aggregation.py` → `clustering.py`

Old names remain available with deprecation warnings (removed in v5.0.0).

### 🐛 Fixed

- Fixed `fix_sizes()` docstring/implementation inconsistency for optional `ds` parameter

### 👷 Development
- Fixed `active_timesteps` type annotation to include `None`
- Fixed xarray truth-value ambiguity in `main_results` buses with excess filter
- Added validation for `nr_of_previous_values` in `SegmentedOptimization` to prevent silent indexing bugs

---

## [4.1.4] - 2025-11-25

**Summary**: Added file logging encoding to prevent issues

If upgrading from v2.x, see the [v3.0.0 release notes](https://github.com/flixOpt/flixOpt/releases/tag/v3.0.0) and [Migration Guide](https://flixopt.github.io/flixopt/latest/user-guide/migration-guide-v3/).

### 🐛 Fixed
- Issues with windows file system when logging to file due to non ASCII characters

---

## [4.1.3] - 2025-11-25

**Summary**: Re-add mistakenly removed method for loading a config from file

If upgrading from v2.x, see the [v3.0.0 release notes](https://github.com/flixOpt/flixOpt/releases/tag/v3.0.0) and [Migration Guide](https://flixopt.github.io/flixopt/latest/user-guide/migration-guide-v3/).

### 🐛 Fixed
- Re-added `CONFIG.load_from_file()` method that was accidentally removed

---

## [4.1.2] - 2025-11-24

**Summary**:

If upgrading from v2.x, see the [v3.0.0 release notes](https://github.com/flixOpt/flixOpt/releases/tag/v3.0.0) and [Migration Guide](https://flixopt.github.io/flixopt/latest/user-guide/migration-guide-v3/).

### ✨ Added
- Exported SUCCESS log level (`SUCCESS_LEVEL`) for use with `logger.log(SUCCESS_LEVEL, ...)`
- Added proper deprecation tests

### ♻️ Changed
- logger coloring improved

### 👷 Development
- Fixed Deprecation warnings in tests

---

## [4.1.1] - 2025-11-23

**Summary**: Finalize preparations for removal of all deprecated parameters in v5.0.0

### ✨ Added

- Added missing infos about removal to remaining deprecated parameters and methods

### 🐛 Fixed

- Missing release notes of v4.1.0

---

## [4.1.0] - 2025-11-21

**Summary**: Logging migrated from loguru to standard Python logging for stability and security. Simpler API with convenient presets.

!!! info "Migration Required?"
    **Most users**: No action needed (silent by default). Methods like `CONFIG.exploring()`, `CONFIG.debug()`, etc. continue to work exactly as before.
    **If you customized logging**: Simple API update (see migration below).
    **If you used loguru directly**: Breaking change (loguru only in v3.6.0-v4.0.0, ~4 days).

If upgrading from v2.x, see the [v3.0.0 release notes](https://github.com/flixOpt/flixOpt/releases/tag/v3.0.0) and [Migration Guide](https://flixopt.github.io/flixopt/latest/user-guide/migration-guide-v3/).

### ✨ Added

**New logging presets**:
```python
CONFIG.production('app.log')    # File-only, no console/plots
```

**New logging methods**:
- `CONFIG.Logging.enable_console(level, colored, stream)` - Console output with colors
- `CONFIG.Logging.enable_file(level, path, max_bytes, backup_count)` - File logging with rotation
- `CONFIG.Logging.disable()` - Disable all logging
- `CONFIG.Logging.set_colors(log_colors)` - Customize colors

**Enhanced formatting**:
- Multi-line messages with box borders (┌─, │, └─)
- Exception tracebacks with proper indentation
- Timestamps: `2025-11-21 14:30:45.123`

### 💥 Breaking Changes

**Logging migration** (edge cases only):

| Old (v3.6.0-v4.0.0) | New (v4.1.0+) |
|---------------------|---------------|
| `CONFIG.Logging.level = 'INFO'`<br>`CONFIG.Logging.console = True`<br>`CONFIG.apply()` | `CONFIG.Logging.enable_console('INFO')`<br>or `CONFIG.exploring()` |
| `CONFIG.Logging.file = 'app.log'` | `CONFIG.Logging.enable_file('INFO', 'app.log')` |
| `logger.opt(lazy=True)` | Built-in (automatic) |

**Migration**:
```python
# Before (v3.6.0-v4.0.0)
CONFIG.Logging.level = 'INFO'
CONFIG.Logging.console = True
CONFIG.apply()

# After (v4.1.0+)
CONFIG.Logging.enable_console('INFO')  # or CONFIG.exploring()
```

### ♻️ Changed

- Replaced loguru with Python `logging` + optional `colorlog` for colors
- Configuration immediate (no `CONFIG.apply()` needed)
- Log format: `[dimmed timestamp] [colored level] │ message`
- Logs to `stdout` by default (configurable)
- SUCCESS level preserved (green, level 25)
- Performance: Expensive operations guarded with `logger.isEnabledFor()` checks

### 🗑️ Deprecated

- `change_logging_level(level)` → Use `CONFIG.Logging.enable_console(level)`. Removal in v5.0.0.

### 🔥 Removed

**CONFIG methods/attributes**:
- `CONFIG.apply()` → Use helper methods directly
- `CONFIG.Logging.level`, `.console`, `.file` → Use `enable_console()`/`enable_file()`
- `CONFIG.Logging.verbose_tracebacks`, `.rich`, `.Colors`, `.date_format`, `.format`, `.console_width`, `.show_path`, `.show_logger_name` → Use standard logging
- loguru features (`logger.opt()`, etc.)

### 🐛 Fixed

- `TypeError` in `check_bounds()` with loguru-style formatting
- Exception tracebacks not appearing in custom formatters
- Inconsistent formatting between console and file logs

### 🔒 Security

- Removed loguru dependency for reduced supply chain risk

### 📦 Dependencies

- **Removed:** `loguru >= 0.7.0`
- **Added:** `colorlog >= 6.8.0, < 7` (optional)

### 📝 Docs

- Preset comparison table in `CONFIG.Logging` docstring
- Color customization examples
- Migration guide with before/after code

---

## [4.0.0] - 2025-11-19

**Summary**: This release introduces clearer parameter naming for linear converters and constraints, enhanced period handling with automatic weight computation, and new sum-over-all-periods constraints for multi-period optimization. All deprecated parameter names continue to work with warnings.

If upgrading from v2.x, see the [v3.0.0 release notes](https://github.com/flixOpt/flixOpt/releases/tag/v3.0.0) and [Migration Guide](https://flixopt.github.io/flixopt/latest/user-guide/migration-guide-v3/).

### ✨ Key Features

**Sum-over-all-periods constraints:**
New constraint parameters enable limiting weighted totals across all periods:
- `Effect`: `minimum_over_periods` and `maximum_over_periods`
- `Flow`: `flow_hours_max_over_periods` and `flow_hours_min_over_periods`

```python
# Per-period: limits apply to EACH period individually
effect = fx.Effect('costs', maximum_total=1000)  # ≤1000 per period

# Over-periods: limits apply to WEIGHTED SUM across ALL periods
# With periods=[2020, 2030, 2040] (weights: [10, 10, 10] from 10-year intervals)
effect = fx.Effect('costs', maximum_over_periods=25000)  # 10×costs₂₀₂₀ + 10×costs₂₀₃₀ + 10×costs₂₀₄₀ ≤ 25000
```

**Improved period weight handling:**
- Period weights now computed automatically from period index (like `hours_per_timestep` for time)
- Weights correctly recalculate when using `.sel()` or `.isel()` on periods
- Separate tracking of `period_weights`, `scenario_weights`, and combined `weights`

**Simplified workflow:**
- `Calculation.solve()` now automatically calls `do_modeling()` if needed

### 💥 Breaking Changes

**FlowSystem weights parameter renamed:**
```python
# Old (v3.x)
fs = FlowSystem(..., weights=np.array([0.3, 0.5, 0.2]))

# New (v4.0)
fs = FlowSystem(..., scenario_weights=np.array([0.3, 0.5, 0.2]))
```
Period weights are now always computed from the period index.

  **Note**: If you were previously passing period × scenario weights to `weights`, you now need to:
  1. Pass only scenario weights to `scenario_weights`
  2. Period weights will be computed automatically from your `periods` index

### 🗑️ Deprecated Parameters

**Linear converters** (`Boiler`, `CHP`, `HeatPump`, etc.) - descriptive names replace abbreviations:
- Flow: `Q_fu` → `fuel_flow`, `P_el` → `electrical_flow`, `Q_th` → `thermal_flow`, `Q_ab` → `heat_source_flow`
- Efficiency: `eta` → `thermal_efficiency`, `eta_th` → `thermal_efficiency`, `eta_el` → `electrical_efficiency`, `COP` → `cop` (lowercase)

**Constraint parameters** - removed redundant `_total` suffix:
- `Flow`: `flow_hours_total_max` → `flow_hours_max`, `flow_hours_total_min` → `flow_hours_min`
- `OnOffParameters`: `on_hours_total_max` → `on_hours_max`, `on_hours_total_min` → `on_hours_min`, `switch_on_total_max` → `switch_on_max`

**Storage**:
- `initial_charge_state="lastValueOfSim"` → `initial_charge_state="equals_final"`

All deprecated names continue working with warnings. **They will be removed in v5.0.0.**

**Additional property deprecations now include removal version:**
- `InvestParameters`: `fix_effects`, `specific_effects`, `divest_effects`, `piecewise_effects`
- `OnOffParameters`: `on_hours_total_min`, `on_hours_total_max`, `switch_on_total_max`
- `Flow`: `flow_hours_total_min`, `flow_hours_total_max`

### 🐛 Fixed
- Fixed inconsistent boundary checks in linear converters with array-like inputs

### 👷 Development
- Eliminated circular dependencies with two-phase modeling pattern
- Enhanced validation for cross-element references and FlowSystem assignment
- Added helper methods for cleaner data transformation code
- Improved logging and cache invalidation
- Improved argument consistency in internal effect coordinate fitting

---

## [3.6.1] - 2025-11-17

**Summary**: Documentation improvements and dependency updates.

If upgrading from v2.x, see the [v3.0.0 release notes](https://github.com/flixOpt/flixOpt/releases/tag/v3.0.0) and [Migration Guide](https://flixopt.github.io/flixopt/latest/user-guide/migration-guide-v3/).

### 📦 Dependencies
- Updated `astral-sh/uv` to v0.9.8
- Updated `mkdocs-git-revision-date-localized-plugin` to v1.5.0

### 📝 Docs
- Improved type specifications in `flixopt/types.py` for better documentation generation
- Fixed minor mkdocs warnings in `flixopt/io.py` and `mkdocs.yml`

---

## [3.6.0] - 2025-11-15

**Summary**: Type system overhaul and migration to loguru for logging. If you are heavily using our logs, this might be breaking!

If upgrading from v2.x, see the [v3.0.0 release notes](https://github.com/flixOpt/flixOpt/releases/tag/v3.0.0) and [Migration Guide](https://flixopt.github.io/flixopt/latest/user-guide/migration-guide-v3/).

### ✨ Added
- **New type system** (`flixopt/types.py`):
    - Introduced dimension-aware type aliases using suffix notation (`_TPS`, `_PS`, `_S`) to clearly indicate which dimensions data can have
    - Added `Numeric_TPS`, `Numeric_PS`, `Numeric_S` for numeric data with Time/Period/Scenario dimensions
    - Added `Bool_TPS`, `Bool_PS`, `Bool_S` for boolean data with dimension support
    - Added `Effect_TPS`, `Effect_PS`, `Effect_S` for effect dictionaries with dimension support
    - Added `Scalar` type for scalar-only numeric values
    - Added `NumericOrBool` utility type for internal use
    - Type system supports scalars, numpy arrays, pandas Series/DataFrames, and xarray DataArrays
- Lazy logging evaluation - expensive log operations only execute when log level is active
- `CONFIG.Logging.verbose_tracebacks` option for detailed debugging with variable values

### 💥 Breaking Changes
- **Logging framework**: Migrated to [loguru](https://loguru.readthedocs.io/)
    - Removed `CONFIG.Logging` parameters: `rich`, `Colors`, `date_format`, `format`, `console_width`, `show_path`, `show_logger_name`
    - For advanced formatting, use loguru's API directly after `CONFIG.apply()`

### ♻️ Changed
- **Code structure**: Removed `commons.py` module and moved all imports directly to `__init__.py` for cleaner code organization (no public API changes)
- **Type handling improvements**: Updated internal data handling to work seamlessly with the new type system

### 🐛 Fixed
- Fixed `ShareAllocationModel` inconsistency where None/inf conversion happened in `__init__` instead of during modeling, which could cause issues with parameter validation
- Fixed numerous type hint inconsistencies across the codebase

### 📦 Dependencies
- Updated `mkdocs-material` to v9.6.23
- Replaced `rich >= 13.0.0` with `loguru >= 0.7.0` for logging

### 📝 Docs
- Enhanced documentation in `flixopt/types.py` with comprehensive examples and dimension explanation table
- Clarified Effect type docstrings - Effect types are dicts, but single numeric values work through union types
- Added clarifying comments in `effects.py` explaining parameter handling and transformation
- Improved OnOffParameters attribute documentation
- Updated getting-started guide with loguru examples
- Updated `config.py` docstrings for loguru integration

### 👷 Development
- Added test for FlowSystem resampling

---

## [3.5.0] - 2025-11-06

**Summary**: Improve representations and improve resampling

If upgrading from v2.x, see the [v3.0.0 release notes](https://github.com/flixOpt/flixOpt/releases/tag/v3.0.0) and [Migration Guide](https://flixopt.github.io/flixopt/latest/user-guide/migration-guide-v3/).

### ✨ Added
- Added options to resample and select subsets of flowsystems without converting to and from Dataset each time. Use the new methods `FlowSystem.__dataset_resample()`, `FlowSystem.__dataset_sel()` and `FlowSystem.__dataset_isel()`. All of them expect and return a dataset.

### 💥 Breaking Changes

### ♻️ Changed
- Truncate repr of FlowSystem and CalculationResults to only show the first 10 items of each category
- Greatly sped up the resampling of a FlowSystem again

---

## [3.4.1] - 2025-11-04

**Summary**: Speed up resampling by 20-40 times.

If upgrading from v2.x, see the [v3.0.0 release notes](https://github.com/flixOpt/flixOpt/releases/tag/v3.0.0) and [Migration Guide](https://flixopt.github.io/flixopt/latest/user-guide/migration-guide-v3/).

### ♻️ Changed
- Greatly sped up the resampling of a FlowSystem (x20 - x40) by converting to dataarray internally

---

## [3.4.0] - 2025-11-01

**Summary**: Enhanced solver configuration with new CONFIG.Solving section for centralized solver parameter management.

If upgrading from v2.x, see the [v3.0.0 release notes](https://github.com/flixOpt/flixOpt/releases/tag/v3.0.0) and [Migration Guide](https://flixopt.github.io/flixopt/latest/user-guide/migration-guide-v3/).

### ✨ Added

**Solver configuration:**
- **New `CONFIG.Solving` configuration section** for centralized solver parameter management:
    - `mip_gap`: Default MIP gap tolerance for solver convergence (default: 0.01)
    - `time_limit_seconds`: Default time limit in seconds for solver runs (default: 300)
    - `log_to_console`: Whether solver should output to console (default: True)
    - `log_main_results`: Whether to log main results after solving (default: True)
- Solvers (`HighsSolver`, `GurobiSolver`) now use `CONFIG.Solving` defaults for parameters, allowing global configuration
- Solver parameters can still be explicitly overridden when creating solver instances
- New `log_to_console` parameter in all Solver classes

### ♻️ Changed
- Individual solver output is now hidden in **SegmentedCalculation**. To return to the prior behaviour, set `show_individual_solves=True` in `do_modeling_and_solve()`.

### 🐛 Fixed
-  New compacted list representation for periods and scenarios also in results log and console print

### 📝 Docs
- Unified contributing guides in docs and on github

### 👷 Development
- Added type hints for submodel in all Interface classes

---

## [3.3.1] - 2025-10-30

**Summary**: Small Bugfix and improving readability

If upgrading from v2.x, see the [v3.0.0 release notes](https://github.com/flixOpt/flixOpt/releases/tag/v3.0.0) and [Migration Guide](https://flixopt.github.io/flixopt/latest/user-guide/migration-guide-v3/).

### ♻️ Changed
- Improved `summary.yaml` to use a compacted list representation for periods and scenarios

### 🐛 Fixed
- Using `switch_on_total_max` with periods or scenarios failed

### 📝 Docs
- Add more comprehensive `CONTRIBUTE.md`
- Improve logical structure in User Guide

---

## [3.3.0] - 2025-10-30

**Summary**: Better access to Elements stored in the FLowSystem and better representations (repr)

If upgrading from v2.x, see the [v3.0.0 release notes](https://github.com/flixOpt/flixOpt/releases/tag/v3.0.0) and [Migration Guide](https://flixopt.github.io/flixopt/latest/user-guide/migration-guide-v3/).

### ♻️ Changed
**Improved repr methods:**
- **Results classes** (`ComponentResults`, `BusResults`, `FlowResults`, `EffectResults`) now show concise header with key metadata followed by xarray Dataset repr
- **Element classes** (`Component`, `Bus`, `Flow`, `Effect`, `Storage`) now show one-line summaries with essential information (connections, sizes, capacities, constraints)

**Container-based access:**
- **FlowSystem** now provides dict-like access patterns for all elements
- Use `flow_system['element_label']`, `flow_system.keys()`, `flow_system.values()`, and `flow_system.items()` for unified element access
- Specialized containers (`components`, `buses`, `effects`, `flows`) offer type-specific access with helpful error messages

### 🗑️ Deprecated
- **`FlowSystem.all_elements`** property is deprecated in favor of dict-like interface (`flow_system['label']`, `.keys()`, `.values()`, `.items()`). Will be removed in v4.0.0.

---

## [3.2.1] - 2025-10-29

**Summary**:

If upgrading from v2.x, see the [v3.0.0 release notes](https://github.com/flixOpt/flixOpt/releases/tag/v3.0.0) and [Migration Guide](https://flixopt.github.io/flixopt/latest/user-guide/migration-guide-v3/).

### 🐛 Fixed
- Fixed resampling of FlowSystem to reset `hours_of_last_timestep` and `hours_of_previous_timesteps` properly

### 👷 Development
- Improved issue templates

---

## [3.2.0] - 2025-10-26

**Summary**: Enhanced plotting capabilities with consistent color management, custom plotting kwargs support, and centralized I/O handling.

If upgrading from v2.x, see the [v3.0.0 release notes](https://github.com/flixOpt/flixOpt/releases/tag/v3.0.0) and [Migration Guide](https://flixopt.github.io/flixopt/latest/user-guide/migration-guide-v3/).

### ✨ Added

**Color management:**
- **`setup_colors()` method** for `CalculationResults` and `SegmentedCalculationResults` to configure consistent colors across all plots
    - Group components by colorscales: `results.setup_colors({'CHP': 'reds', 'Storage': 'blues', 'Greys': ['Grid', 'Demand']})`
    - Automatically propagates to all segments in segmented calculations
    - Colors persist across all plot calls unless explicitly overridden
- **Flexible color inputs**: Supports colorscale names (e.g., 'turbo', 'plasma'), color lists, or label-to-color dictionaries
- **Cross-backend compatibility**: Seamless color handling for both Plotly and Matplotlib

**Plotting customization:**
- **Plotting kwargs support**: Pass additional arguments to plotting backends via `px_kwargs`, `plot_kwargs`, and `backend_kwargs` parameters
- **New `CONFIG.Plotting` configuration section**:
    - `default_show`: Control default plot visibility
    - `default_engine`: Choose 'plotly' or 'matplotlib'
    - `default_dpi`: Set resolution for saved plots
    - `default_facet_cols`: Configure default faceting columns
    - `default_sequential_colorscale`: Default for heatmaps (now 'turbo')
    - `default_qualitative_colorscale`: Default for categorical plots (now 'plotly')

**I/O improvements:**
- Centralized JSON/YAML I/O with auto-format detection
- Enhanced NetCDF handling with consistent engine usage
- Better numeric formatting in YAML exports

### ♻️ Changed
- **Default colorscale**: Changed from 'viridis' to 'turbo' for better perceptual uniformity
- **Color terminology**: Standardized from "colormap" to "colorscale" throughout for Plotly consistency
- **Plotting internals**: Now use `xr.Dataset` as primary data type (DataFrames automatically converted)
- **NetCDF engine**: Switched back to netcdf4 engine following xarray updates and performance benchmarks

### 🔥 Removed
- Removed unused `plotting.pie_with_plotly()` method

### 🐛 Fixed
- Improved error messages when using `engine='matplotlib'` with multidimensional data
- Better dimension validation in `results.plot_heatmap()`

### 📝 Docs
- Enhanced examples demonstrating `setup_colors()` usage
- Updated terminology from "colormap" to "colorscale" in docstrings

### 👷 Development
- Fixed concurrency issue in CI
- Centralized color processing logic into dedicated module
- Refactored to function-based color handling for simpler API

---

## [3.1.1] - 2025-10-20
**Summary**: Fixed a bug when accessing the `effects_per_component` dataset in results without periodic effects.

If upgrading from v2.x, see the [v3.0.0 release notes](https://github.com/flixOpt/flixOpt/releases/tag/v3.0.0) and [Migration Guide](https://flixopt.github.io/flixopt/latest/user-guide/migration-guide-v3/).

### 🐛 Fixed
- Fixed ValueError in effects_per_component when all periodic effects are scalars/NaN by explicitly creating mode-specific templates (via _create_template_for_mode) with correct dimensions

### 👷 Development
- Converted all remaining numpy style docstrings to google style

---

## [3.1.0] - 2025-10-19

**Summary**: This release adds faceting and animation support for multidimensional plots and redesigns the documentation website. Plotting results across scenarios or periods is now significantly simpler (Plotly only).

If upgrading from v2.x, see the [Migration Guide](https://flixopt.github.io/flixopt/latest/user-guide/migration-guide-v3/) and [v3.0.0 release notes](https://github.com/flixOpt/flixOpt/releases/tag/v3.0.0).

### ✨ Added
- **Faceting and animation for multidimensional plots**: All plotting methods now support `facet_by` and `animate_by` parameters to create subplot grids and animations from multidimensional data (scenarios, periods, etc.). *Plotly only.*
- **Flexible data selection with `select` parameter**: Select data using single values, lists, slices, or index arrays for precise control over what gets plotted
- **Heatmap fill control**: New `fill` parameter in heatmap methods controls how missing values are filled after reshaping (`'ffill'` or `'bfill'`)
- **Smart line styling for mixed variables**: Area plots now automatically style variables containing both positive and negative values with dashed lines, while stacking purely positive or negative variables

### ♻️ Changed
- **Breaking: Selection behavior**: Plotting methods no longer automatically select the first value for non-time dimensions. Use the `select` parameter for explicit selection of scenarios, periods, or other dimensions
- **Better error messages**: Enhanced error messages when using Matplotlib with multidimensional data, with clearer guidance on dimension requirements and suggestions to use Plotly
- **Improved examples**: Enhanced `scenario_example.py` with better demonstration of new features
- **Robust validation**: Improved dimension validation in `plot_heatmap()` with clearer error messages

### 🗑️ Deprecated
- **`indexer` parameter**: Use the new `select` parameter instead. The `indexer` parameter will be removed in v4.0.0
- **`heatmap_timeframes` and `heatmap_timesteps_per_frame` parameters**: Use the new `reshape_time=(timeframes, timesteps_per_frame)` parameter instead in heatmap plotting methods
- **`color_map` parameter**: Use the new `colors` parameter instead in heatmap plotting methods

### 🐛 Fixed
- Fixed cryptic errors when working with empty buses by adding proper validation
- Added early validation for non-existent periods when using linked periods with tuples

### 📝 Documentation
- **Redesigned documentation website** with custom css

### 👷 Development
- Renamed internal `_apply_indexer_to_data()` to `_apply_selection_to_data()` for consistency with new API naming

---

## [3.0.3] - 2025-10-16
**Summary**: Hotfixing new plotting parameter `style`. Continue to use `mode`.

**Note**: If upgrading from v2.x, see the [Migration Guide](https://flixopt.github.io/flixopt/latest/user-guide/migration-guide-v3/) and [v3.0.0 release notes](https://github.com/flixOpt/flixOpt/releases/tag/v3.0.0).

### 🐛 Fixed
- Reverted breaking change from v3.0.0: continue to use `mode parameter in plotting instead of new `style`
- Renamed new `mode` parameter in plotting methods to `unit_type`

### 📝 Docs
- Updated Migration Guide and added missing entries.
- Improved Changelog of v3.0.0

---

## [3.0.2] - 2025-10-15
**Summary**: This is a follow-up release to **[v3.0.0](https://github.com/flixOpt/flixOpt/releases/tag/v3.0.0)**, improving the documentation.

**Note**: If upgrading from v2.x, see the [Migration Guide](https://flixopt.github.io/flixopt/latest/user-guide/migration-guide-v3/) and [v3.0.0 release notes](https://github.com/flixOpt/flixOpt/releases/tag/v3.0.0).

### 📝 Docs
- Update the Readme
- Add a project roadmap to the docs
- Change Development status to "Production/Stable"
- Regroup parts in docs

---

## [3.0.1] - 2025-10-14
**Summary**: This is a follow-up release to **[v3.0.0](https://github.com/flixOpt/flixOpt/releases/tag/v3.0.0)**, adding a Migration Guide and bugfixing the docs.

**Note**: If upgrading from v2.x, see the [Migration Guide](https://flixopt.github.io/flixopt/latest/user-guide/migration-guide-v3/) and [v3.0.0 release notes](https://github.com/flixOpt/flixOpt/releases/tag/v3.0.0).

### 📝 Docs
- Fixed deployed docs
- Added Migration Guide for flixopt 3

### 👷 Development
- Added missing type hints

---

## [3.0.0] - 2025-10-13
**Summary**: This release introduces new model dimensions (periods and scenarios) for multi-period investments and stochastic modeling, along with a redesigned effect sharing system and enhanced I/O capabilities.

**Note**: If upgrading from v2.x, see the [Migration Guide](https://flixopt.github.io/flixopt/latest/user-guide/migration-guide-v3/) and [v3.0.0 release notes](https://github.com/flixOpt/flixOpt/releases/tag/v3.0.0).

### ✨ Added

**New model dimensions:**

- **Period dimension**: Enables multi-period investment modeling with distinct decisions in each period for transformation pathway optimization
- **Scenario dimension**: Supports stochastic modeling with weighted scenarios for robust decision-making under uncertainty (demand, prices, weather)
    - Control variable independence across scenarios via `scenario_independent_sizes` and `scenario_independent_flow_rates` parameters
    - By default, investment sizes are shared across scenarios while flow rates vary per scenario

**Redesigned effect sharing system:**

Effects now use intuitive `share_from_*` syntax that clearly shows contribution sources:

```python
costs = fx.Effect('costs', '€', 'Total costs',
    share_from_temporal={'CO2': 0.2},      # From temporal effects
    share_from_periodic={'land': 100})     # From periodic effects
```

This replaces `specific_share_to_other_effects_*` parameters and inverts the direction for clearer relationships.

**Enhanced I/O and data handling:**

- NetCDF/JSON serialization for all Interface objects and FlowSystem with round-trip support
- FlowSystem manipulation: `sel()`, `isel()`, `resample()`, `copy()`, `__eq__()` methods
- Direct access to FlowSystem from results without manual restoring (lazily loaded)
- New `FlowResults` class and precomputed DataArrays for sizes/flow_rates/flow_hours
- `effects_per_component` dataset for component impact evaluation, including all indirect effects through effect shares

**Other additions:**

- Balanced storage - charging and discharging sizes can be forced equal via `balanced` parameter
- New Storage parameters: `relative_minimum_final_charge_state` and `relative_maximum_final_charge_state` for final state control
- Improved filter methods in results
- Example for 2-stage investment decisions leveraging FlowSystem resampling

### 💥 Breaking Changes

**API and Behavior Changes:**

- **Effect system redesigned** (no deprecation):
    - **Terminology changes**: Effect domains renamed for clarity: `operation` → `temporal`, `invest`/`investment` → `periodic`
    - **Sharing system**: The old `specific_share_to_other_effects_*` parameters were completely replaced with the new `share_from_temporal` and `share_from_periodic` syntax (see 🔥 Removed section)
- **FlowSystem independence**: FlowSystems cannot be shared across multiple Calculations anymore. A copy of the FlowSystem is created instead, making every Calculation independent. Each Subcalculation in `SegmentedCalculation` now has its own distinct `FlowSystem` object
- **Bus and Effect object assignment**: Direct assignment of Bus/Effect objects is no longer supported. Use labels (strings) instead:
    - `Flow.bus` must receive a string label, not a Bus object
    - Effect shares must use effect labels (strings) in dictionaries, not Effect objects
- **Logging defaults** (from v2.2.0): Console and file logging are now disabled by default. Enable explicitly with `CONFIG.Logging.console = True` and `CONFIG.apply()`

**Class and Method Renaming:**

- Renamed class `SystemModel` to `FlowSystemModel`
- Renamed class `Model` to `Submodel`
- Renamed `mode` parameter in plotting methods to `style`
- `Calculation.do_modeling()` now returns the `Calculation` object instead of its `linopy.Model`. Callers that previously accessed the linopy model directly should now use `calculation.do_modeling().model` instead of `calculation.do_modeling()`

**Variable Renaming in Results:**

- Investment binary variable: `is_invested` → `invested` in `InvestmentModel`
- Switch tracking variables in `OnOffModel`:
    - `switch_on` → `switch|on`
    - `switch_off` → `switch|off`
    - `switch_on_nr` → `switch|count`
- Effect submodel variables (following terminology changes):
    - `Effect(invest)|total` → `Effect(periodic)`
    - `Effect(operation)|total` → `Effect(temporal)`
    - `Effect(operation)|total_per_timestep` → `Effect(temporal)|per_timestep`
    - `Effect|total` → `Effect`

**Data Structure Changes:**

- `relative_minimum_charge_state` and `relative_maximum_charge_state` don't have an extra timestep anymore. Use the new `relative_minimum_final_charge_state` and `relative_maximum_final_charge_state` parameters for final state control

### ♻️ Changed

- Type system overhaul - added clear separation between temporal and non-temporal data throughout codebase for better clarity
- Enhanced FlowSystem interface with improved `__repr__()` and `__str__()` methods
- Improved Model Structure - Views and organisation is now divided into:
    - Model: The main Model (linopy.Model) that is used to create and store the variables and constraints for the FlowSystem.
    - Submodel: The base class for all submodels. Each is a subset of the Model, for simpler access and clearer code.
- Made docstrings in `config.py` more compact and easier to read
- Improved format handling in configuration module
- Enhanced console output to support both `stdout` and `stderr` stream selection
- Added `show_logger_name` parameter to `CONFIG.Logging` for displaying logger names in messages

### 🗑️ Deprecated

- The `agg_group` and `agg_weight` parameters of `TimeSeriesData` are deprecated and will be removed in a future version. Use `aggregation_group` and `aggregation_weight` instead.
- The `active_timesteps` parameter of `Calculation` is deprecated and will be removed in a future version. Use the new `sel(time=...)` method on the FlowSystem instead.
- **InvestParameters** parameters renamed for improved clarity around investment and retirement effects:
    - `fix_effects` → `effects_of_investment`
    - `specific_effects` → `effects_of_investment_per_size`
    - `divest_effects` → `effects_of_retirement`
    - `piecewise_effects` → `piecewise_effects_of_investment`
- **Effect** parameters renamed:
    - `minimum_investment` → `minimum_periodic`
    - `maximum_investment` → `maximum_periodic`
    - `minimum_operation` → `minimum_temporal`
    - `maximum_operation` → `maximum_temporal`
    - `minimum_operation_per_hour` → `minimum_per_hour`
    - `maximum_operation_per_hour` → `maximum_per_hour`
- **Component** parameters renamed:
    - `Source.source` → `Source.outputs`
    - `Sink.sink` → `Sink.inputs`
    - `SourceAndSink.source` → `SourceAndSink.outputs`
    - `SourceAndSink.sink` → `SourceAndSink.inputs`
    - `SourceAndSink.prevent_simultaneous_sink_and_source` → `SourceAndSink.prevent_simultaneous_flow_rates`

### 🔥 Removed

- **Effect share parameters**: The old `specific_share_to_other_effects_*` parameters were replaced WITHOUT DEPRECATION
    - `specific_share_to_other_effects_operation` → `share_from_temporal` (with inverted direction)
    - `specific_share_to_other_effects_invest` → `share_from_periodic` (with inverted direction)

### 🐛 Fixed

- Enhanced NetCDF I/O with proper attribute preservation for DataArrays
- Improved error handling and validation in serialization processes
- Better type consistency across all framework components
- Added extra validation in `config.py` to improve error handling

### 📝 Docs

- Reorganized mathematical notation docs: moved to lowercase `mathematical-notation/` with subdirectories (`elements/`, `features/`, `modeling-patterns/`)
- Added comprehensive documentation pages: `dimensions.md` (time/period/scenario), `effects-penalty-objective.md`, modeling patterns
- Enhanced all element pages with implementation details, cross-references, and "See Also" sections
- Rewrote README and landing page with clearer vision, roadmap, and universal applicability emphasis
- Removed deprecated `docs/SUMMARY.md`, updated `mkdocs.yml` for new structure
- Tightened docstrings in core modules with better cross-referencing
- Added recipes section to docs

### 🚧 Known Issues

- IO for single Interfaces/Elements to Datasets might not work properly if the Interface/Element is not part of a fully transformed and connected FlowSystem. This arises from Numeric Data not being stored as xr.DataArray by the user. To avoid this, always use the `to_dataset()` on Elements inside a FlowSystem that's connected and transformed.

### 👷 Development

- **Centralized deprecation pattern**: Added `_handle_deprecated_kwarg()` helper method to `Interface` base class that provides reusable deprecation handling with consistent warnings, conflict detection, and optional value transformation. Applied across 5 classes (InvestParameters, Source, Sink, SourceAndSink, Effect) reducing deprecation boilerplate by 72%.
- FlowSystem data management simplified - removed `time_series_collection` pattern in favor of direct timestep properties
- Change modeling hierarchy to allow for more flexibility in future development. This leads to minimal changes in the access and creation of Submodels and their variables.
- Added new module `.modeling` that contains modeling primitives and utilities
- Clearer separation between the main Model and "Submodels"
- Improved access to the Submodels and their variables, constraints and submodels
- Added `__repr__()` for Submodels to easily inspect its content
- Enhanced data handling methods
    - `fit_to_model_coords()` method for data alignment
    - `fit_effects_to_model_coords()` method for effect data processing
    - `connect_and_transform()` method replacing several operations
- **Testing improvements**: Eliminated warnings during test execution
    - Updated deprecated code patterns in tests and examples (e.g., `sink`/`source` → `inputs`/`outputs`, `'H'` → `'h'` frequency)
    - Refactored plotting logic to handle test environments explicitly with non-interactive backends
    - Added comprehensive warning filters in `__init__.py` and `pyproject.toml` to suppress third-party library warnings
    - Improved test fixtures with proper figure cleanup to prevent memory leaks
    - Enhanced backend detection and handling in `plotting.py` for both Matplotlib and Plotly
    - Always run dependent tests in order

---

## [2.2.0] - 2025-10-11
**Summary:** This release is a Configuration and Logging management release.

### ✨ Added
- Added `CONFIG.reset()` method to restore configuration to default values
- Added configurable log file rotation settings: `CONFIG.Logging.max_file_size` and `CONFIG.Logging.backup_count`
- Added configurable log format settings: `CONFIG.Logging.date_format` and `CONFIG.Logging.format`
- Added configurable console settings: `CONFIG.Logging.console_width` and `CONFIG.Logging.show_path`
- Added `CONFIG.Logging.Colors` nested class for customizable log level colors using ANSI escape codes (works with both standard and Rich handlers)
- All examples now enable console logging to demonstrate proper logging usage
- Console logging now outputs to `sys.stdout` instead of `sys.stderr` for better compatibility with output redirection

### 💥 Breaking Changes
- Console logging is now disabled by default (`CONFIG.Logging.console = False`). Enable it explicitly in your scripts with `CONFIG.Logging.console = True` and `CONFIG.apply()`
- File logging is now disabled by default (`CONFIG.Logging.file = None`). Set a file path to enable file logging

### ♻️ Changed
- Logging and Configuration management changed
- Improved default logging colors: DEBUG is now gray (`\033[90m`) for de-emphasized messages, INFO uses terminal default color (`\033[0m`) for clean output

### 🗑️ Deprecated
- `change_logging_level()` function is now deprecated in favor of `CONFIG.Logging.level` and `CONFIG.apply()`. Will be removed in version 3.0.0.

### 🔥 Removed
- Removed unused `config.merge_configs` function from configuration module

### 👷 Development
- Greatly expanded test coverage for `config.py` module
- Added `@pytest.mark.xdist_group` to `TestConfigModule` tests to prevent global config interference

---

## [2.1.11] - 2025-10-05
**Summary:** Important bugfix in `Storage` leading to wrong results due to incorrect discharge losses.

### ♻️ Changed
- Using `h5netcdf` instead of `netCDF4` for dataset I/O operations. This follows the update in `xarray==2025.09.01`

### 🐛 Fixed
- Fix `charge_state` Constraint in `Storage` leading to incorrect losses in discharge and therefore incorrect charge states and discharge values.

### 📦 Dependencies
- Updated `renovate.config` to treat CalVer packages (xarray and dask) with more care
- Updated packaging configuration

---

## [2.1.10] - 2025-09-29
**Summary:** This release is a Documentation and Development release.

### 📝 Docs
- Improved CHANGELOG.md formatting by adding better categories and formating by Gitmoji.
- Added a script to extract the release notes from the CHANGELOG.md file for better organized documentation.

### 👷 Development
- Improved `renovate.config`
- Sped up CI by not running examples in every run and using `pytest-xdist`

---

## [2.1.9] - 2025-09-23

**Summary:** Small bugfix release addressing network visualization error handling.

### 🐛 Fixed
- Fix error handling in network visualization if `networkx` is not installed

---

## [2.1.8] - 2025-09-22

**Summary:** Code quality improvements, enhanced documentation, and bug fixes for heat pump components and visualization features.

### ✨ Added
- Extra Check for HeatPumpWithSource.COP to be strictly > 1 to avoid division by zero
- Apply deterministic color assignment by using sorted() in `plotting.py`
- Add missing args in docstrings in `plotting.py`, `solvers.py`, and `core.py`.

### ♻️ Changed
- Greatly improved docstrings and documentation of all public classes
- Make path handling to be gentle about missing .html suffix in `plotting.py`
- Default for `relative_losses` in `Transmission` is now 0 instead of None
- Setter of COP in `HeatPumpWithSource` now completely overwrites the conversion factors, which is safer.
- Fix some docstrings in plotting.py
- Change assertions to raise Exceptions in `plotting.py`

### 🐛 Fixed

**Core Components:**
- Fix COP getter and setter of `HeatPumpWithSource` returning and setting wrong conversion factors
- Fix custom compression levels in `io.save_dataset_to_netcdf`
- Fix `total_max` did not work when total min was not used

**Visualization:**
- Fix color scheme selection in network_app; color pickers now update when a scheme is selected

### 📝 Docs
- Fix broken links in docs
- Fix some docstrings in plotting.py

### 👷 Development
- Pin dev dependencies to specific versions
- Improve CI workflows to run faster and smarter

---

## [2.1.7] - 2025-09-13

**Summary:** Maintenance release to improve Code Quality, CI and update the dependencies. There are no changes or new features.

### ✨ Added
- Added `__version__` to flixopt

### 👷 Development
- ruff format the whole Codebase
- Added renovate config
- Added pre-commit
- lint and format in CI
- improved CI
- Updated Dependencies
- Updated Issue Templates

---

## [2.1.6] - 2025-09-02

**Summary:** Enhanced Sink/Source components with multi-flow support and new interactive network visualization.

### ✨ Added
- **Network Visualization**: Added `FlowSystem.start_network_app()` and `FlowSystem.stop_network_app()` to easily visualize the network structure of a flow system in an interactive Dash web app
    - *Note: This is still experimental and might change in the future*

### ♻️ Changed
- **Multi-Flow Support**: `Sink`, `Source`, and `SourceAndSink` now accept multiple `flows` as `inputs` and `outputs` instead of just one. This enables modeling more use cases with these classes
- **Flow Control**: Both `Sink` and `Source` now have a `prevent_simultaneous_flow_rates` argument to prevent simultaneous flow rates of more than one of their flows

### 🗑️ Deprecated
- For the classes `Sink`, `Source` and `SourceAndSink`: `.sink`, `.source` and `.prevent_simultaneous_sink_and_source` are deprecated in favor of the new arguments `inputs`, `outputs` and `prevent_simultaneous_flow_rates`

### 🐛 Fixed
- Fixed testing issue with new `linopy` version 0.5.6

### 👷 Development
- Added dependency "nbformat>=4.2.0" to dev dependencies to resolve issue with plotly CI

---

## [2.1.5] - 2025-07-08

### 🐛 Fixed
- Fixed Docs deployment

---

## [2.1.4] - 2025-07-08

### 🐛 Fixed
- Fixing release notes of 2.1.3, as well as documentation build.

---

## [2.1.3] - 2025-07-08

### 🐛 Fixed
- Using `Effect.maximum_operation_per_hour` raised an error, needing an extra timestep. This has been fixed thanks to @PRse4.

---

## [2.1.2] - 2025-06-14

### 🐛 Fixed
- Storage losses per hour were not calculated correctly, as mentioned by @brokenwings01. This might have led to issues when modeling large losses and long timesteps.
    - Old implementation:     $c(\text{t}_{i}) \cdot (1-\dot{\text{c}}_\text{rel,loss}(\text{t}_i)) \cdot \Delta \text{t}_{i}$
    - Correct implementation: $c(\text{t}_{i}) \cdot (1-\dot{\text{c}}_\text{rel,loss}(\text{t}_i)) ^{\Delta \text{t}_{i}}$

### 🚧 Known Issues
- Just to mention: Plotly >= 6 may raise errors if "nbformat" is not installed. We pinned plotly to <6, but this may be fixed in the future.

---

## [2.1.1] - 2025-05-08

### ♻️ Changed
- Improved docstring and tests

### 🐛 Fixed
- Fixed bug in the `_ElementResults.constraints` not returning the constraints but rather the variables

---
## [2.1.0] - 2025-04-11

### ✨ Added
- Python 3.13 support added
- Logger warning if relative_minimum is used without on_off_parameters in Flow
- Greatly improved internal testing infrastructure by leveraging linopy's testing framework

### 💥 Breaking Changes
- Restructured the modeling of the On/Off state of Flows or Components
    - Variable renaming: `...|consecutive_on_hours` → `...|ConsecutiveOn|hours`
    - Variable renaming: `...|consecutive_off_hours` → `...|ConsecutiveOff|hours`
    - Constraint renaming: `...|consecutive_on_hours_con1` → `...|ConsecutiveOn|con1`
    - Similar pattern for all consecutive on/off constraints

### 🐛 Fixed
- Fixed the lower bound of `flow_rate` when using optional investments without OnOffParameters
- Fixed bug that prevented divest effects from working
- Added lower bounds of 0 to two unbounded vars (numerical improvement)

---

## [2.0.1] - 2025-04-10

### ✨ Added
- Logger warning if relative_minimum is used without on_off_parameters in Flow

### 🐛 Fixed
- Replace "|" with "__" in filenames when saving figures (Windows compatibility)
- Fixed bug that prevented the load factor from working without InvestmentParameters

## [2.0.0] - 2025-03-29

**Summary:** 💥 **MAJOR RELEASE** - Complete framework migration from Pyomo to Linopy with redesigned architecture.

### ✨ Added

**Model Capabilities:**
- Full model serialization support - save and restore unsolved Models
- Enhanced model documentation with YAML export containing human-readable mathematical formulations
- Extend flixopt models with native linopy language support
- Full Model Export/Import capabilities via linopy.Model

**Results & Data:**
- Unified solution exploration through `Calculation.results` attribute
- Compression support for result files
- `to_netcdf/from_netcdf` methods for FlowSystem and core components
- xarray integration for TimeSeries with improved datatypes support

### 💥 Breaking Changes

**Framework Migration:**
- **Optimization Engine**: Complete migration from Pyomo to Linopy optimization framework
- **Package Import**: Framework renamed from flixOpt to flixopt (`import flixopt as fx`)
- **Data Architecture**: Redesigned data handling to rely on xarray.Dataset throughout the package
- **Results System**: Results handling completely redesigned with new `CalculationResults` class

**Variable Structure:**
- Restructured the modeling of the On/Off state of Flows or Components
    - Variable renaming: `...|consecutive_on_hours` → `...|ConsecutiveOn|hours`
    - Variable renaming: `...|consecutive_off_hours` → `...|ConsecutiveOff|hours`
    - Constraint renaming: `...|consecutive_on_hours_con1` → `...|ConsecutiveOn|con1`
    - Similar pattern for all consecutive on/off constraints

### 🔥 Removed
- **Pyomo dependency** (replaced by linopy)
- **Period concepts** in time management (simplified to timesteps)

### 🐛 Fixed
- Improved infeasible model detection and reporting
- Enhanced time series management and serialization
- Reduced file size through improved compression

### 📝 Docs
- Google Style Docstrings throughout the codebase
