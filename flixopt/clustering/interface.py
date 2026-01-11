"""
Clustering class that inherits from Interface for IO serialization.

This module is kept separate to avoid circular imports - it's lazily imported
after the core structure module is fully loaded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

from ..structure import Interface, register_class_for_io

if TYPE_CHECKING:
    from ..flow_system import FlowSystem
    from ..plot_result import PlotResult
    from ..statistics_accessor import SelectType


@register_class_for_io
class Clustering(Interface):
    """Minimal clustering info for expansion, inter-cluster storage, and IO.

    Inherits from Interface for automatic IO serialization.
    All derived values (timestep_mapping, n_original_clusters) are computed on demand.

    Attributes:
        cluster_assignments: Cluster ID for each original period. Shape [original_cluster, period?, scenario?].
        cluster_weights: Count of original periods per cluster. Shape [cluster, period?, scenario?].
        n_clusters: Number of distinct clusters (e.g., 8 typical days).
        timesteps_per_cluster: Timesteps within each cluster (e.g., 24 for daily).
        original_timesteps_iso: Original time coordinates as ISO strings.
        predefined: tsam PredefinedConfig dict for transferring clustering.
        metrics_rmse, metrics_mae, metrics_rmse_duration: Optional metric DataArrays.

    Example:
        >>> fs_clustered = fs.transform.cluster(n_clusters=8)
        >>> fs_clustered.clustering.n_clusters
        8
        >>> fs_clustered.clustering.expand_data(some_data)  # Expand to original timesteps
    """

    def __init__(
        self,
        cluster_assignments: xr.DataArray,
        cluster_weights: xr.DataArray,
        n_clusters: int,
        timesteps_per_cluster: int,
        original_timesteps_iso: list[str],
        predefined: dict | Any | None = None,
        metrics_rmse: xr.DataArray | None = None,
        metrics_mae: xr.DataArray | None = None,
        metrics_rmse_duration: xr.DataArray | None = None,
    ):
        # Ensure DataArrays have names for IO
        if cluster_assignments.name is None:
            cluster_assignments = cluster_assignments.rename('cluster_assignments')
        if cluster_weights.name is None:
            cluster_weights = cluster_weights.rename('cluster_weights')

        self.cluster_assignments = cluster_assignments
        self.cluster_weights = cluster_weights
        self.n_clusters = n_clusters
        self.timesteps_per_cluster = timesteps_per_cluster
        self.original_timesteps_iso = original_timesteps_iso

        # Convert predefined dict to PredefinedConfig if tsam available
        if isinstance(predefined, dict):
            try:
                import tsam

                predefined = tsam.PredefinedConfig.from_dict(predefined)
            except (ImportError, Exception):
                pass
        self.predefined = predefined

        # Store metrics (named for IO)
        if metrics_rmse is not None and metrics_rmse.name is None:
            metrics_rmse = metrics_rmse.rename('metrics_rmse')
        if metrics_mae is not None and metrics_mae.name is None:
            metrics_mae = metrics_mae.rename('metrics_mae')
        if metrics_rmse_duration is not None and metrics_rmse_duration.name is None:
            metrics_rmse_duration = metrics_rmse_duration.rename('metrics_rmse_duration')

        self.metrics_rmse = metrics_rmse
        self.metrics_mae = metrics_mae
        self.metrics_rmse_duration = metrics_rmse_duration

        # Not serialized
        self._original_data: xr.Dataset | None = None

    def transform_data(self) -> None:
        """No-op (required by Interface)."""
        pass

    def _create_reference_structure(self) -> tuple[dict, dict[str, xr.DataArray]]:
        """Override to serialize predefined via to_dict()."""
        ref, arrays = super()._create_reference_structure()

        # Override predefined with proper to_dict() serialization
        if self.predefined is not None and hasattr(self.predefined, 'to_dict'):
            ref['predefined'] = self.predefined.to_dict()

        return ref, arrays

    # ═══════════════════════════════════════════════════════════════════════════
    # Convenience constructor
    # ═══════════════════════════════════════════════════════════════════════════

    @classmethod
    def create(
        cls,
        cluster_assignments: xr.DataArray,
        cluster_weights: xr.DataArray,
        n_clusters: int,
        timesteps_per_cluster: int,
        original_timesteps: pd.DatetimeIndex,
        predefined: Any | None = None,
        metrics: xr.Dataset | None = None,
    ) -> Clustering:
        """Create Clustering with DatetimeIndex and Dataset (convenience method).

        Converts DatetimeIndex to ISO strings and Dataset to individual DataArrays.
        """
        original_timesteps_iso = [t.isoformat() for t in original_timesteps]

        # Convert predefined to dict for storage (will be converted back in __init__)
        predefined_dict = None
        if predefined is not None:
            predefined_dict = predefined.to_dict() if hasattr(predefined, 'to_dict') else predefined

        # Extract metrics DataArrays
        metrics_rmse = metrics_mae = metrics_rmse_duration = None
        if metrics is not None and len(metrics.data_vars) > 0:
            if 'RMSE' in metrics:
                metrics_rmse = metrics['RMSE'].rename('metrics_rmse')
            if 'MAE' in metrics:
                metrics_mae = metrics['MAE'].rename('metrics_mae')
            if 'RMSE_DURATION' in metrics:
                metrics_rmse_duration = metrics['RMSE_DURATION'].rename('metrics_rmse_duration')

        return cls(
            cluster_assignments=cluster_assignments,
            cluster_weights=cluster_weights,
            n_clusters=n_clusters,
            timesteps_per_cluster=timesteps_per_cluster,
            original_timesteps_iso=original_timesteps_iso,
            predefined=predefined_dict,
            metrics_rmse=metrics_rmse,
            metrics_mae=metrics_mae,
            metrics_rmse_duration=metrics_rmse_duration,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # Computed Properties
    # ═══════════════════════════════════════════════════════════════════════════

    @property
    def original_timesteps(self) -> pd.DatetimeIndex:
        """Original time coordinates before clustering."""
        return pd.DatetimeIndex(self.original_timesteps_iso)

    @property
    def n_original_clusters(self) -> int:
        """Number of original periods before clustering."""
        return self.cluster_assignments.sizes.get('original_cluster', len(self.cluster_assignments))

    @property
    def cluster_order(self) -> xr.DataArray:
        """Alias for cluster_assignments."""
        return self.cluster_assignments

    @property
    def cluster_occurrences(self) -> xr.DataArray:
        """Alias for cluster_weights."""
        return self.cluster_weights

    @property
    def metrics(self) -> xr.Dataset:
        """Clustering quality metrics as Dataset."""
        data_vars = {}
        if self.metrics_rmse is not None:
            data_vars['RMSE'] = self.metrics_rmse
        if self.metrics_mae is not None:
            data_vars['MAE'] = self.metrics_mae
        if self.metrics_rmse_duration is not None:
            data_vars['RMSE_DURATION'] = self.metrics_rmse_duration
        return xr.Dataset(data_vars)

    # ═══════════════════════════════════════════════════════════════════════════
    # Core Methods
    # ═══════════════════════════════════════════════════════════════════════════

    def get_cluster_assignments_for_slice(self, period: str | None = None, scenario: str | None = None) -> np.ndarray:
        """Get cluster assignments for a specific period/scenario slice."""
        return _select_dims(self.cluster_assignments, period, scenario).values.astype(int)

    def get_timestep_mapping(self, period: str | None = None, scenario: str | None = None) -> np.ndarray:
        """Compute timestep mapping from cluster_assignments."""
        assignments = self.get_cluster_assignments_for_slice(period, scenario)
        n_original = len(self.original_timesteps)
        mapping = np.zeros(n_original, dtype=np.int32)

        for t in range(n_original):
            orig_cluster = t // self.timesteps_per_cluster
            if orig_cluster < len(assignments):
                cluster_id = assignments[orig_cluster]
                time_within = t % self.timesteps_per_cluster
                mapping[t] = cluster_id * self.timesteps_per_cluster + time_within

        return mapping

    def expand_data(
        self,
        data: xr.DataArray,
        period: str | None = None,
        scenario: str | None = None,
    ) -> xr.DataArray:
        """Expand clustered data back to original timesteps."""
        extra_dims = [d for d in data.dims if d not in ('cluster', 'time')]

        if not extra_dims:
            return self._expand_slice(data, period, scenario)

        # Multi-dimensional: expand each slice
        dim_coords = {d: list(data.coords[d].values) for d in extra_dims}
        expanded_slices = {}

        for combo in np.ndindex(*[len(v) for v in dim_coords.values()]):
            selector = {d: dim_coords[d][i] for d, i in zip(extra_dims, combo, strict=True)}
            data_slice = data.sel(**selector)
            p = selector.get('period', period)
            s = selector.get('scenario', scenario)
            expanded_slices[tuple(selector.values())] = self._expand_slice(data_slice, p, s)

        # Concatenate back
        result_arrays = expanded_slices
        for dim in reversed(extra_dims):
            dim_vals = dim_coords[dim]
            grouped = {}
            for key, arr in result_arrays.items():
                rest_key = key[:-1] if len(key) > 1 else ()
                grouped.setdefault(rest_key, []).append(arr)
            result_arrays = {k: xr.concat(v, dim=pd.Index(dim_vals, name=dim)) for k, v in grouped.items()}

        return list(result_arrays.values())[0].transpose('time', ...).assign_attrs(data.attrs)

    def _expand_slice(self, data: xr.DataArray, period: str | None, scenario: str | None) -> xr.DataArray:
        """Expand a single (cluster, time) or (time,) slice."""
        mapping = self.get_timestep_mapping(period, scenario)
        has_cluster_dim = 'cluster' in data.dims

        if has_cluster_dim:
            cluster_ids = mapping // self.timesteps_per_cluster
            time_within = mapping % self.timesteps_per_cluster
            expanded_values = data.values[cluster_ids, time_within]
        else:
            expanded_values = data.values[mapping]

        return xr.DataArray(
            expanded_values,
            coords={'time': self.original_timesteps},
            dims=['time'],
            attrs=data.attrs,
        )

    def __repr__(self) -> str:
        return (
            f'Clustering(n_clusters={self.n_clusters}, '
            f'timesteps_per_cluster={self.timesteps_per_cluster}, '
            f'n_original_clusters={self.n_original_clusters})'
        )

    @property
    def plot(self) -> ClusteringPlotAccessor:
        """Access plotting methods."""
        return ClusteringPlotAccessor(self)


def _select_dims(da: xr.DataArray, period: str | None = None, scenario: str | None = None) -> xr.DataArray:
    """Select from DataArray by period/scenario if those dimensions exist."""
    result = da
    if period is not None and 'period' in da.dims:
        result = result.sel(period=period)
    if scenario is not None and 'scenario' in da.dims:
        result = result.sel(scenario=scenario)
    return result


class ClusteringPlotAccessor:
    """Accessor for clustering visualization methods."""

    def __init__(self, clustering: Clustering, flow_system: FlowSystem | None = None):
        self._clustering = clustering
        self._flow_system = flow_system

    def heatmap(
        self,
        *,
        select: SelectType | None = None,
        colors: str | list[str] | None = None,
        facet_col: str | None = 'auto',
        animation_frame: str | None = 'auto',
        show: bool | None = None,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot cluster assignments as heatmap timeline."""
        from ..config import CONFIG
        from ..plot_result import PlotResult
        from ..statistics_accessor import _apply_selection

        cluster_order_da = self._clustering.cluster_assignments
        timesteps_per_cluster = self._clustering.timesteps_per_cluster
        original_time = self._clustering.original_timesteps

        if select:
            cluster_order_da = _apply_selection(cluster_order_da.to_dataset(name='cluster'), select)['cluster']

        extra_dims = [d for d in cluster_order_da.dims if d != 'original_cluster']
        expanded_values = np.repeat(cluster_order_da.values, timesteps_per_cluster, axis=0)

        if len(original_time) < expanded_values.shape[0]:
            expanded_values = expanded_values[: len(original_time)]

        coords = {'time': original_time}
        coords.update({d: cluster_order_da.coords[d].values for d in extra_dims})
        cluster_da = xr.DataArray(expanded_values, dims=['time'] + extra_dims, coords=coords)

        heatmap_da = cluster_da.expand_dims('y', axis=-1).assign_coords(y=['Cluster'])
        heatmap_da.name = 'cluster_assignment'
        heatmap_da = heatmap_da.transpose('time', 'y', ...)

        fig = heatmap_da.fxplot.heatmap(
            colors=colors,
            title='Cluster Assignments',
            facet_col=facet_col,
            animation_frame=animation_frame,
            aspect='auto',
            **plotly_kwargs,
        )

        fig.update_yaxes(showticklabels=False)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))

        plot_result = PlotResult(data=cluster_da.to_dataset(name='cluster'), figure=fig)

        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            plot_result.show()

        return plot_result
