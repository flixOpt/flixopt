"""
Aggregation helpers for building reduced datasets from clustering results.

This module provides functions for constructing reduced FlowSystem datasets
from tsam aggregation results, including building typical DataArrays,
segment durations, cluster weights, and metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

from .iteration import DimInfo, iter_dim_slices

if TYPE_CHECKING:
    from tsam import AggregationResult


def combine_slices_to_dataarray(
    slices: dict[tuple, xr.DataArray],
    dim_info: DimInfo,
    base_dims: list[str] | None = None,
    name: str | None = None,
    attrs: dict | None = None,
) -> xr.DataArray:
    """Combine per-(period, scenario) slices into a multi-dimensional DataArray.

    This is the unified slice combination function that handles both generic
    base dimensions and the common (cluster, time) case.

    Args:
        slices: Dict mapping (period, scenario) tuples to DataArrays.
        dim_info: DimInfo with period/scenario information.
        base_dims: Base dimensions of each slice. If None, inferred from first slice.
            Common examples: ['cluster', 'time'], ['original_cluster'], ['time_series'].
        name: Name for the resulting DataArray.
        attrs: Attributes to assign to the result.

    Returns:
        DataArray with dimensions [base_dims..., period?, scenario?].
    """
    first_key = (dim_info.periods[0], dim_info.scenarios[0])

    # Infer base_dims if not provided
    if base_dims is None:
        base_dims = list(slices[first_key].dims)

    # Simple case: no period/scenario dimensions
    if not dim_info.has_periods and not dim_info.has_scenarios:
        result = slices[first_key]
        if name is not None:
            result = result.rename(name)
        if attrs is not None:
            result = result.assign_attrs(attrs)
        return result

    # Multi-dimensional: use xr.concat to stack along period/scenario dims
    # Use join='outer' to handle cases where different periods/scenarios have
    # different coordinate values (e.g., different time_series after drop_constant_arrays)
    if dim_info.has_periods and dim_info.has_scenarios:
        # Stack scenarios first, then periods
        period_arrays = []
        for p in dim_info.periods:
            scenario_arrays = [slices[(p, s)] for s in dim_info.scenarios]
            period_arrays.append(
                xr.concat(
                    scenario_arrays,
                    dim=pd.Index(dim_info.scenarios, name='scenario'),
                    join='outer',
                    fill_value=np.nan,
                )
            )
        result = xr.concat(
            period_arrays,
            dim=pd.Index(dim_info.periods, name='period'),
            join='outer',
            fill_value=np.nan,
        )
    elif dim_info.has_periods:
        result = xr.concat(
            [slices[(p, None)] for p in dim_info.periods],
            dim=pd.Index(dim_info.periods, name='period'),
            join='outer',
            fill_value=np.nan,
        )
    else:
        result = xr.concat(
            [slices[(None, s)] for s in dim_info.scenarios],
            dim=pd.Index(dim_info.scenarios, name='scenario'),
            join='outer',
            fill_value=np.nan,
        )

    # Put all base dimensions first in order (standard order)
    result = result.transpose(*base_dims, ...)

    if name is not None:
        result = result.rename(name)
    if attrs is not None:
        result = result.assign_attrs(attrs)

    return result


def build_typical_dataarrays(
    tsam_aggregation_results: dict[tuple, AggregationResult],
    n_clusters: int,
    n_time_points: int,
    cluster_coords: np.ndarray,
    time_coords: pd.DatetimeIndex | pd.RangeIndex,
    is_segmented: bool = False,
) -> dict[str, dict[tuple, xr.DataArray]]:
    """Build typical periods DataArrays with (cluster, time) shape.

    Args:
        tsam_aggregation_results: Dict mapping (period, scenario) to tsam AggregationResult.
        n_clusters: Number of clusters.
        n_time_points: Number of time points per cluster (timesteps or segments).
        cluster_coords: Cluster coordinate values.
        time_coords: Time coordinate values.
        is_segmented: Whether segmentation was used.

    Returns:
        Nested dict: {column_name: {(period, scenario): DataArray}}.
    """
    typical_das: dict[str, dict[tuple, xr.DataArray]] = {}

    for key, tsam_result in tsam_aggregation_results.items():
        typical_df = tsam_result.cluster_representatives

        if is_segmented:
            # Segmented data: MultiIndex with cluster as first level
            # Each cluster has exactly n_time_points rows (segments)
            # Extract all data at once using numpy reshape, avoiding slow .loc calls
            columns = typical_df.columns.tolist()

            # Get all values as numpy array: (n_clusters * n_time_points, n_columns)
            all_values = typical_df.values

            # Reshape to (n_clusters, n_time_points, n_columns)
            reshaped = all_values.reshape(n_clusters, n_time_points, -1)

            for col_idx, col in enumerate(columns):
                # reshaped[:, :, col_idx] selects all clusters, all time points, single column
                # Result shape: (n_clusters, n_time_points)
                typical_das.setdefault(col, {})[key] = xr.DataArray(
                    reshaped[:, :, col_idx],
                    dims=['cluster', 'time'],
                    coords={'cluster': cluster_coords, 'time': time_coords},
                )
        else:
            # Non-segmented: flat data that can be reshaped
            for col in typical_df.columns:
                flat_data = typical_df[col].values
                reshaped = flat_data.reshape(n_clusters, n_time_points)
                typical_das.setdefault(col, {})[key] = xr.DataArray(
                    reshaped,
                    dims=['cluster', 'time'],
                    coords={'cluster': cluster_coords, 'time': time_coords},
                )

    return typical_das


def build_segment_durations(
    tsam_aggregation_results: dict[tuple, AggregationResult],
    n_clusters: int,
    n_segments: int,
    cluster_coords: np.ndarray,
    time_coords: pd.RangeIndex,
    dt: float,
    dim_info: DimInfo,
) -> xr.DataArray:
    """Build timestep_duration DataArray from segment durations.

    For segmented systems, each segment represents multiple original timesteps.
    The duration is segment_duration_in_original_timesteps * dt.

    Args:
        tsam_aggregation_results: Dict mapping (period, scenario) to tsam AggregationResult.
        n_clusters: Number of clusters.
        n_segments: Number of segments per cluster.
        cluster_coords: Cluster coordinate values.
        time_coords: Time coordinate values (RangeIndex for segments).
        dt: Hours per original timestep.
        dim_info: DimInfo with period/scenario information.

    Returns:
        DataArray with dims [cluster, time] or [cluster, time, period?, scenario?]
        containing duration in hours for each segment.
    """
    segment_duration_slices: dict[tuple, xr.DataArray] = {}

    for key, tsam_result in tsam_aggregation_results.items():
        # segment_durations is tuple of tuples: ((dur1, dur2, ...), (dur1, dur2, ...), ...)
        seg_durs = tsam_result.segment_durations

        # Build 2D array (cluster, segment) of durations in hours
        data = np.zeros((n_clusters, n_segments))
        for cluster_id in range(n_clusters):
            cluster_seg_durs = seg_durs[cluster_id]
            for seg_id in range(n_segments):
                data[cluster_id, seg_id] = cluster_seg_durs[seg_id] * dt

        segment_duration_slices[key] = xr.DataArray(
            data,
            dims=['cluster', 'time'],
            coords={'cluster': cluster_coords, 'time': time_coords},
        )

    return combine_slices_to_dataarray(
        segment_duration_slices,
        dim_info,
        base_dims=['cluster', 'time'],
        name='timestep_duration',
    )


def build_cluster_weights(
    cluster_occurrences: dict[tuple, dict[int, int]],
    n_clusters: int,
    cluster_coords: np.ndarray,
    dim_info: DimInfo,
) -> xr.DataArray:
    """Build cluster_weight DataArray from occurrence counts.

    Args:
        cluster_occurrences: Dict mapping (period, scenario) tuples to
            dicts of {cluster_id: occurrence_count}.
        n_clusters: Number of clusters.
        cluster_coords: Cluster coordinate values.
        dim_info: DimInfo with period/scenario information.

    Returns:
        DataArray with dims [cluster] or [cluster, period?, scenario?].
    """

    def _weight_for_key(key: tuple) -> xr.DataArray:
        occurrences = cluster_occurrences[key]
        # Missing clusters contribute zero weight (not 1)
        weights = np.array([occurrences.get(c, 0) for c in range(n_clusters)])
        return xr.DataArray(weights, dims=['cluster'], coords={'cluster': cluster_coords})

    weight_slices = {key: _weight_for_key(key) for key in cluster_occurrences}
    return combine_slices_to_dataarray(
        weight_slices,
        dim_info,
        base_dims=['cluster'],
        name='cluster_weight',
    )


def build_clustering_metrics(
    clustering_metrics: dict[tuple, pd.DataFrame],
    dim_info: DimInfo,
) -> xr.Dataset:
    """Build clustering metrics Dataset from per-slice DataFrames.

    Args:
        clustering_metrics: Dict mapping (period, scenario) to metric DataFrames.
        dim_info: DimInfo with period/scenario information.

    Returns:
        Dataset with RMSE, MAE, RMSE_duration metrics.
    """
    non_empty_metrics = {k: v for k, v in clustering_metrics.items() if not v.empty}

    if not non_empty_metrics:
        return xr.Dataset()

    first_key = (dim_info.periods[0], dim_info.scenarios[0])

    if len(clustering_metrics) == 1 and len(non_empty_metrics) == 1:
        metrics_df = non_empty_metrics.get(first_key)
        if metrics_df is None:
            metrics_df = next(iter(non_empty_metrics.values()))
        return xr.Dataset(
            {
                col: xr.DataArray(
                    metrics_df[col].values,
                    dims=['time_series'],
                    coords={'time_series': metrics_df.index},
                )
                for col in metrics_df.columns
            }
        )

    # Multi-dim case
    sample_df = next(iter(non_empty_metrics.values()))
    metric_names = list(sample_df.columns)
    data_vars = {}

    for metric in metric_names:
        slices = {}
        for (p, s), df in clustering_metrics.items():
            if df.empty:
                slices[(p, s)] = xr.DataArray(
                    np.full(len(sample_df.index), np.nan),
                    dims=['time_series'],
                    coords={'time_series': list(sample_df.index)},
                )
            else:
                slices[(p, s)] = xr.DataArray(
                    df[metric].values,
                    dims=['time_series'],
                    coords={'time_series': list(df.index)},
                )
        data_vars[metric] = combine_slices_to_dataarray(
            slices,
            dim_info,
            base_dims=['time_series'],
            name=metric,
        )

    return xr.Dataset(data_vars)


def build_cluster_assignments_dataarray(
    cluster_assignments: dict[tuple, np.ndarray],
    dim_info: DimInfo,
) -> xr.DataArray:
    """Build cluster_assignments DataArray from cluster assignments.

    Args:
        cluster_assignments: Dict mapping (period, scenario) to cluster assignment arrays.
        dim_info: DimInfo with period/scenario information.

    Returns:
        DataArray with dims [original_cluster] or [original_cluster, period?, scenario?].
    """
    if dim_info.has_periods or dim_info.has_scenarios:
        # Multi-dimensional case
        slices = {}
        for ctx in iter_dim_slices(dim_info):
            slices[ctx.key] = xr.DataArray(
                cluster_assignments[ctx.key],
                dims=['original_cluster'],
                name='cluster_assignments',
            )
        return combine_slices_to_dataarray(
            slices,
            dim_info,
            base_dims=['original_cluster'],
            name='cluster_assignments',
        )
    else:
        # Simple case
        first_key = (dim_info.periods[0], dim_info.scenarios[0])
        return xr.DataArray(
            cluster_assignments[first_key],
            dims=['original_cluster'],
            name='cluster_assignments',
        )


def calculate_clustering_weights(ds: xr.Dataset) -> dict[str, float]:
    """Calculate weights for clustering based on dataset attributes.

    Variables in the same clustering_group share weight equally (1/count).
    Variables without a group use their clustering_weight attribute or default to 1.

    Args:
        ds: Dataset with variables that may have clustering_group or
            clustering_weight attributes.

    Returns:
        Dict mapping variable names to their clustering weights.
    """
    from collections import Counter

    groups = [da.attrs.get('clustering_group') for da in ds.data_vars.values() if 'clustering_group' in da.attrs]
    group_counts = Counter(groups)

    # Calculate weight for each group (1/count)
    group_weights = {group: 1 / count for group, count in group_counts.items()}

    weights = {}
    variables = ds.variables
    for name in ds.data_vars:
        var_attrs = variables[name].attrs
        clustering_group = var_attrs.get('clustering_group')
        group_weight = group_weights.get(clustering_group)
        if group_weight is not None:
            weights[name] = group_weight
        else:
            weights[name] = var_attrs.get('clustering_weight', 1)

    return weights


def build_cluster_config_with_weights(
    cluster: Any | None,
    auto_weights: dict[str, float],
) -> Any:
    """Merge auto-calculated weights into ClusterConfig.

    Args:
        cluster: Optional user-provided ClusterConfig from tsam.
        auto_weights: Automatically calculated weights based on data variance.

    Returns:
        ClusterConfig with weights set (either user-provided or auto-calculated).
    """
    from tsam import ClusterConfig

    # User provided ClusterConfig with weights - use as-is
    if cluster is not None and cluster.weights is not None:
        return cluster

    # No ClusterConfig provided - use defaults with auto-calculated weights
    if cluster is None:
        return ClusterConfig(weights=auto_weights)

    # ClusterConfig provided without weights - add auto-calculated weights
    return ClusterConfig(
        method=cluster.method,
        representation=cluster.representation,
        weights=auto_weights,
        normalize_column_means=cluster.normalize_column_means,
        use_duration_curves=cluster.use_duration_curves,
        include_period_sums=cluster.include_period_sums,
        solver=cluster.solver,
    )


def accuracy_to_dataframe(accuracy: Any) -> pd.DataFrame:
    """Convert tsam AccuracyMetrics to DataFrame.

    Args:
        accuracy: tsam AccuracyMetrics object.

    Returns:
        DataFrame with RMSE, MAE, and RMSE_duration columns.
    """
    return pd.DataFrame(
        {
            'RMSE': accuracy.rmse,
            'MAE': accuracy.mae,
            'RMSE_duration': accuracy.rmse_duration,
        }
    )
