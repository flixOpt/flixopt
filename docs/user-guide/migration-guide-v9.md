# Migration Guide: v8.x → v9.0.0

!!! tip "Quick Start"
    ```bash
    pip install --upgrade flixopt
    ```
    v9.0.0 has a single breaking change: clustering weights move from
    `ClusterConfig(weights={...})` to a top-level `cluster(weights={...})`
    argument, following [tsam 4.0](https://tsam.readthedocs.io/en/latest/migration/v3-to-v4/).
    Everything else in `transform.cluster()` is unchanged.

---

## Overview

| Aspect | v8.x | v9.0.0 |
|--------|------|--------|
| **Clustering weights** | `cluster=ClusterConfig(weights={...})` | `weights={...}` |
| **tsam requirement** | `>= 3.4.0, < 4` | `>= 4.0.0, < 5` |
| **`cluster()` arguments** | positional after `cluster_duration` allowed | keyword-only after `cluster_duration` |

### Dependencies

- `tsam >= 4.0.0, < 5` (was `>= 3.4.0, < 4`) — tsam 4 removed the class-based
  `TimeSeriesAggregation` API and moved per-column `weights` out of `ClusterConfig`.
- `tsam_xarray >= 0.6.5, < 1` (unchanged)

---

## 💥 Breaking Changes

### Moved: `ClusterConfig(weights=...)` → `cluster(weights=...)`

tsam 4 made per-column weights a top-level argument of `aggregate()`;
`ClusterConfig(weights=...)` now raises `TypeError`. flixopt follows suit.

=== "v8.x (Old)"
    ```python
    from tsam import ClusterConfig

    fs_clustered = flow_system.transform.cluster(
        n_clusters=8,
        cluster_duration='1D',
        cluster=ClusterConfig(
            method='hierarchical',
            weights={'HeatDemand(Q)|fixed_relative_profile': 2},
        ),
    )
    ```

=== "v9.0.0 (New)"
    ```python
    from tsam import ClusterConfig

    fs_clustered = flow_system.transform.cluster(
        n_clusters=8,
        cluster_duration='1D',
        cluster=ClusterConfig(method='hierarchical'),
        weights={'HeatDemand(Q)|fixed_relative_profile': 2},
    )
    ```

Semantics are unchanged: variables omitted from `weights` keep the default weight
of **1.0** and still influence cluster assignments. A `0` weight is still not true
exclusion — tsam clamps it up to a minimal tolerable value — so use
[`cluster_on`](migration-guide-v7.md#cluster-inputs) when a variable should have no
influence at all. `weights` may not reference a variable that `cluster_on` excludes.

Call `flow_system.transform.cluster_inputs()` to list the valid variable names.

### `cluster()` arguments after `cluster_duration` are keyword-only

`n_clusters` and `cluster_duration` may still be positional; everything after them
must be passed by keyword. This prevents a positional argument from silently
binding to the wrong parameter now that `weights` sits in the signature, and matches
tsam 4's own `aggregate()` signature.

```python
# Still fine
flow_system.transform.cluster(8, '1D', extremes=...)

# Now raises TypeError
flow_system.transform.cluster(8, '1D', None, None, extremes_config)
```

### Renamed in tsam: `normalize_column_means` → `scale_by_column_means`

If you passed `normalize_column_means` through to tsam, it is now
`ClusterConfig(scale_by_column_means=...)`. See the
[tsam v3→v4 migration guide](https://tsam.readthedocs.io/en/latest/migration/v3-to-v4/)
for the full list of tsam-side renames; every internal tsam identifier moved from
camelCase to snake_case.

---

## Migration Checklist

- [ ] Move `weights` out of `ClusterConfig(...)` into the `cluster(weights=...)` argument
- [ ] Pass all `cluster()` arguments after `cluster_duration` by keyword
- [ ] Rename `normalize_column_means` to `scale_by_column_means` (on `ClusterConfig`)
- [ ] Upgrade `tsam` to `>= 4.0.0`
