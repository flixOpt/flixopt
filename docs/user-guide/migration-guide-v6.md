# Migration Guide: v5.x → v6.0.0

!!! tip "Quick Start"
    ```bash
    pip install --upgrade flixopt
    ```
    v6.0.0 brings tsam v3 integration, faster I/O, and new clustering features. Review this guide to update your code.

---

## Overview

v6.0.0 introduces major improvements to clustering and I/O performance. The key changes are:

| Aspect | Old API (v5.x) | New API (v6.0.0) |
|--------|----------------|------------------|
| **Clustering config** | Individual parameters | `ClusterConfig`, `ExtremeConfig` objects |
| **Peak forcing** | `time_series_for_high_peaks` | `extremes=ExtremeConfig(max_value=[...])` |
| **Clustering class** | `ClusteredOptimization` (deprecated) | `flow_system.transform.cluster()` |

---

## 💥 Breaking Changes in v6.0.0

### tsam v3 API Migration

The clustering API now uses tsam v3's configuration objects instead of individual parameters.

=== "v5.x (Old)"
    ```python
    import flixopt as fx

    fs_clustered = flow_system.transform.cluster(
        n_clusters=8,
        cluster_duration='1D',
        cluster_method='hierarchical',
        representation_method='medoid',
        time_series_for_high_peaks=['HeatDemand(Q)|fixed_relative_profile'],
        time_series_for_low_peaks=['SolarThermal(Q)|fixed_relative_profile'],
        extreme_period_method='new_cluster',
    )
    ```

=== "v6.0.0 (New)"
    ```python
    import flixopt as fx
    from tsam import ClusterConfig, ExtremeConfig

    fs_clustered = flow_system.transform.cluster(
        n_clusters=8,
        cluster_duration='1D',
        cluster=ClusterConfig(
            method='hierarchical',
            representation='medoid',
        ),
        extremes=ExtremeConfig(
            method='new_cluster',
            max_value=['HeatDemand(Q)|fixed_relative_profile'],
            min_value=['SolarThermal(Q)|fixed_relative_profile'],
        ),
    )
    ```

#### Parameter Mapping

| Old Parameter (v5.x) | New Parameter (v6.0.0) |
|---------------------|------------------------|
| `cluster_method` | `cluster=ClusterConfig(method=...)` |
| `representation_method` | `cluster=ClusterConfig(representation=...)` |
| `time_series_for_high_peaks` | `extremes=ExtremeConfig(max_value=[...])` |
| `time_series_for_low_peaks` | `extremes=ExtremeConfig(min_value=[...])` |
| `extreme_period_method` | `extremes=ExtremeConfig(method=...)` |
| `predef_cluster_order` | `predef_cluster_assignments` |

!!! note "tsam Installation"
    v6.0.0 requires tsam with `SegmentConfig` and `ExtremeConfig` support. Install with:
    ```bash
    pip install "flixopt[full]"
    ```
    This installs the compatible tsam version from the VCS dependency.

---

### Removed: ClusteredOptimization

`ClusteredOptimization` and `ClusteringParameters` were deprecated in v5.0.0 and are now **removed**.

=== "v4.x/v5.x (Old)"
    ```python
    from flixopt import ClusteredOptimization, ClusteringParameters

    params = ClusteringParameters(
        n_clusters=8,
        hours_per_cluster=24,
        cluster_method='hierarchical',
    )
    optimization = ClusteredOptimization('clustered', flow_system, params)
    optimization.do_modeling_and_solve(solver)
    ```

=== "v6.0.0 (New)"
    ```python
    import flixopt as fx
    from tsam import ClusterConfig, ExtremeConfig

    fs_clustered = flow_system.transform.cluster(
        n_clusters=8,
        cluster_duration='1D',
        cluster=ClusterConfig(method='hierarchical'),
        extremes=ExtremeConfig(method='new_cluster', max_value=['Demand|profile']),
    )
    fs_clustered.optimize(solver)

    # Expand back to full resolution
    fs_expanded = fs_clustered.transform.expand()
    ```

---

### Scenario Weights Normalization

`FlowSystem.scenario_weights` are now always normalized to sum to 1 when set, including after `.sel()` subsetting.

=== "v5.x (Old)"
    ```python
    # Weights could be any values
    flow_system.scenario_weights = {'low': 0.3, 'high': 0.7}

    # After subsetting, weights were unchanged
    fs_subset = flow_system.sel(scenario='low')
    # fs_subset.scenario_weights might be {'low': 0.3}
    ```

=== "v6.0.0 (New)"
    ```python
    # Weights are normalized to sum to 1
    flow_system.scenario_weights = {'low': 0.3, 'high': 0.7}

    # After subsetting, weights are renormalized
    fs_subset = flow_system.sel(scenario='low')
    # fs_subset.scenario_weights = {'low': 1.0}
    ```

---

## ✨ New Features in v6.0.0

### Time-Series Segmentation

New intra-period segmentation reduces timesteps within each cluster:

```python
from tsam import SegmentConfig, ExtremeConfig

fs_segmented = flow_system.transform.cluster(
    n_clusters=8,
    cluster_duration='1D',
    segments=SegmentConfig(n_segments=6),  # 6 segments per day instead of 24 hours
    extremes=ExtremeConfig(method='new_cluster', max_value=['Demand|profile']),
)

# Variable timestep durations
print(fs_segmented.timestep_duration)  # Different duration per segment

# Expand back to original resolution
fs_expanded = fs_segmented.transform.expand()
```

See [08f-Segmentation notebook](../notebooks/08f-clustering-segmentation.ipynb) for details.

---

### I/O Performance

2-3x faster NetCDF I/O for large systems:

```python
# Save - now faster with variable stacking
flow_system.to_netcdf('system.nc')

# Load - faster DataArray construction
fs_loaded = fx.FlowSystem.from_netcdf('system.nc')

# Version tracking
ds = flow_system.to_dataset()
print(ds.attrs['flixopt_version'])  # e.g., '6.0.0'
```

---

### Clustering Inspection

New methods to inspect clustering data before and after:

```python
# Before clustering: see what data will be used
clustering_data = flow_system.transform.clustering_data()
print(list(clustering_data.data_vars))

# After clustering: access metadata
fs_clustered.clustering.n_clusters
fs_clustered.clustering.cluster_assignments
fs_clustered.clustering.cluster_occurrences
fs_clustered.clustering.metrics.to_dataframe()

# Visualize
fs_clustered.clustering.plot.compare()
fs_clustered.clustering.plot.heatmap()
```

---

### Apply Existing Clustering

Reuse clustering from one FlowSystem on another:

```python
# Create reference clustering
fs_reference = flow_system.transform.cluster(n_clusters=8, cluster_duration='1D')

# Apply same clustering to modified system
flow_system_modified = flow_system.copy()
flow_system_modified.components['Storage'].capacity_in_flow_hours.maximum_size = 2000

fs_modified = flow_system_modified.transform.apply_clustering(fs_reference.clustering)
```

---

## Migration Checklist

- [ ] Update `transform.cluster()` calls to use `ClusterConfig` and `ExtremeConfig`
- [ ] Replace `ClusteredOptimization` with `transform.cluster()` + `optimize()`
- [ ] Replace `time_series_for_high_peaks` with `extremes=ExtremeConfig(max_value=[...])`
- [ ] Replace `cluster_method` with `cluster=ClusterConfig(method=...)`
- [ ] Review code that depends on `scenario_weights` not being normalized
- [ ] Test clustering workflows with new API

---

## Need Help?

- [Clustering User Guide](optimization/clustering.md)
- [Clustering Notebooks](../notebooks/08c-clustering.ipynb)
- [CHANGELOG](https://github.com/flixOpt/flixopt/blob/main/CHANGELOG.md)
- [GitHub Issues](https://github.com/flixOpt/flixopt/issues)
