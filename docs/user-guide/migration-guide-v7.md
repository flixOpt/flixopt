# Migration Guide: v6.x → v7.0.0

!!! tip "Quick Start"
    ```bash
    pip install --upgrade flixopt
    ```
    v7.0.0 has a single breaking change: clustering now runs on
    [tsam_xarray](https://github.com/FZJ-IEK3-VSA/tsam_xarray) instead of tsam.
    The `transform.cluster()` call signature is unchanged — only the `Clustering`
    result object and a few helpers changed.

---

## Overview

| Aspect | v6.x | v7.0.0 |
|--------|------|--------|
| **Clustering backend** | per-slice `tsam.aggregate()` loop | single `tsam_xarray.aggregate()` call |
| **Result object** | `ClusteringResults` (flixopt) | delegates to `tsam_xarray.ClusteringResult` |
| **Expansion** | `expand_data()` / `timestep_mapping` | `disaggregate()` |
| **Clustering on a subset** | `data_vars=[...]` | `cluster_on=[...]` |

Config objects (`ClusterConfig`, `ExtremeConfig`, `SegmentConfig`) and the
`cluster()` / `apply_clustering()` / `expand()` methods are **unchanged**.

### Dependencies

- `tsam_xarray >= 0.6.0, < 1` (new; replaces the tsam backend)
- `tsam >= 3.4.0, < 4` (still required directly for the config objects)

---

## 💥 Breaking Changes

### Removed: `data_vars` parameter → use `cluster_on`

The v5/v6 `data_vars` argument ("cluster on these variables only") is replaced by
`cluster_on`, which keeps the same semantics: the clustering is computed on
the listed subset and the assignments are applied to the full dataset. Excluded
variables are aggregated but have **no** influence on the assignments.

=== "v6.x (Old)"
    ```python
    fs_clustered = flow_system.transform.cluster(
        n_clusters=8, cluster_duration='1D',
        data_vars=['HeatDemand(Q)|fixed_relative_profile'],  # cluster on this only
    )
    ```

=== "v7.0.0 (New)"
    ```python
    fs_clustered = flow_system.transform.cluster(
        n_clusters=8, cluster_duration='1D',
        cluster_on=['HeatDemand(Q)|fixed_relative_profile'],  # cluster on this only
    )
    ```

`cluster_on` combines with `ClusterConfig(weights=...)` to set relative importance
among the kept variables (weights may not reference an excluded variable).

!!! note "Why not `weights={var: 0}`?"
    You *can* express exclusion through weights, but a `0` weight is **not** true
    exclusion — tsam clamps it up to a minimal tolerable value, so the variable
    still nudges the assignment (and you must enumerate every column via
    [`cluster_inputs()`](#cluster-inputs) to zero the rest). Prefer
    `cluster_on` for genuine exclusion. Note this is a correctness feature, not a
    speed one: subset-then-apply is a second pass, so it is not faster than a full
    clustering. Variables omitted from `weights` (without `cluster_on`) keep the
    default weight of **1.0** and still influence assignments.

### Removed: `TimeSeriesData(clustering_group=..., clustering_weight=...)`

Auto-weighting from these attributes is gone. Pass weights explicitly via
`ClusterConfig(weights={...})`.

### Removed: `flow_system.transform.clustering_data()`

Not a rename. v7 passes **all** time-varying inputs (including constants) to
tsam_xarray, so the old "non-constant inputs" preview is no longer meaningful.
See [`cluster_inputs()`](#cluster-inputs) for the v7 equivalent
(different semantics — it includes constants).

### Removed: metrics, plotting, and original-data serialization

`Clustering.metrics` (RMSE/MAE), `clustering.plot.*`, and the
`include_original_data=...` flag on `to_netcdf()` / `to_dataset()` are gone. For
accuracy analysis or plotting, use `clustering.aggregation_result` (a tsam_xarray
`AggregationResult`) **before** serialization, or rebuild via
`transform.apply_clustering(...)` after loading.

### Expansion: `disaggregate()` replaces `expand_data()` / `timestep_mapping`

```python
# v6.x
full = fs.clustering.expand_data(reduced_da)
# v7.0.0
full = fs.clustering.disaggregate(reduced_da)
```

`transform.expand()` (whole-FlowSystem expansion) is unchanged.

### Removed / renamed `Clustering` properties

| Removed (v6.x) | Replacement (v7.0.0) |
|---|---|
| `Clustering.results` | `Clustering.clustering_result` |
| `Clustering.dims`, `Clustering.coords` | `clustering_result.slice_dims` + `.coords` on returned DataArrays |
| `Clustering.sel(...)`, `Clustering.get_result(...)` | `clustering.aggregation_result` (pre-IO only) |
| `Clustering.n_representatives` | `n_clusters * (n_segments or timesteps_per_cluster)` |
| `Clustering.timestep_mapping`, `Clustering.expand_data(da)` | `clustering.disaggregate(da)` |
| `Clustering.cluster_start_positions` | `np.arange(0, n_clusters * step, step)` |
| `Clustering.representative_weights` | `Clustering.cluster_occurrences` |
| `AggregationResults` alias | `Clustering` (use directly) |

**Still available:** `n_clusters`, `timesteps_per_cluster`, `n_original_clusters`,
`n_segments`, `is_segmented`, `dim_names`, `cluster_assignments`,
`cluster_occurrences`, `segment_assignments`, `segment_durations`,
`disaggregate()`, `apply()`, `to_json()` / `from_json()`, plus
`clustering_result` (tsam_xarray) and `aggregation_result` (pre-IO only).

### Serialization / NetCDF compatibility

The embedded clustering is now serialized via `tsam_xarray.ClusteringResult`
(`to_dict()` / `from_dict()`). **NetCDF files written by v6 cannot be loaded in
v7.** Re-run `transform.cluster()` after upgrading, or re-save from v6.

### Removed notebooks

`08d-clustering-multiperiod`, `08e-clustering-internals`, and
`08f-clustering-segmentation` were removed; their content lives in
`08c-clustering` and `08c2-clustering-storage-modes`.

---

## ✨ New: `transform.cluster_inputs()` { #cluster-inputs }

Returns an `xr.Dataset` of **every** variable with a `time` dim — exactly what
`cluster()` feeds to tsam_xarray, **constants included**. Use it to build a
complete `weights` map:

```python
cols = list(flow_system.transform.cluster_inputs())
target = 'HeatDemand(Q)|fixed_relative_profile'
weights = {target: 1, **{v: 0 for v in cols if v != target}}

fs_clustered = flow_system.transform.cluster(
    n_clusters=8, cluster_duration='1D',
    cluster=ClusterConfig(weights=weights),
)
```

---

## Migration Checklist

- [ ] Replace `data_vars=[...]` with `cluster_on=[...]`
- [ ] Remove `clustering_group` / `clustering_weight` from `TimeSeriesData`; pass weights explicitly
- [ ] Replace `expand_data()` / `timestep_mapping` with `disaggregate()`
- [ ] Update removed `Clustering` properties (see table above)
- [ ] Move any metrics/plotting to `clustering.aggregation_result` before serialization
- [ ] Re-run `transform.cluster()` for any NetCDF saved with v6

---

## Need Help?

- [Clustering User Guide](optimization/clustering.md)
- [Clustering Notebooks](../notebooks/08c-clustering.ipynb)
- [tsam_xarray](https://github.com/FZJ-IEK3-VSA/tsam_xarray)
- [CHANGELOG](https://github.com/flixOpt/flixopt/blob/main/CHANGELOG.md)
- [GitHub Issues](https://github.com/flixOpt/flixopt/issues)
