"""
Clustering dataclass for storing clustering results and metadata.

Simple dataclass with explicit IO methods - no Interface inheritance needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    import tsam.api

    from ..flow_system import FlowSystem
    from ..plot_result import PlotResult
    from ..statistics_accessor import SelectType


def _select_dims(da: xr.DataArray, period: str | None = None, scenario: str | None = None) -> xr.DataArray:
    """Select from DataArray by period/scenario if those dimensions exist."""
    result = da
    if period is not None and 'period' in da.dims:
        result = result.sel(period=period)
    if scenario is not None and 'scenario' in da.dims:
        result = result.sel(scenario=scenario)
    return result


@dataclass
class Clustering:
    """Clustering results and metadata for expansion, inter-cluster storage, and IO.

    Attributes:
        cluster_assignments: Cluster ID for each original period. Shape [original_cluster, period?, scenario?].
        cluster_weights: Count of original periods per cluster. Shape [cluster, period?, scenario?].
        original_timesteps: Original time coordinates before clustering.
        predefined: tsam PredefinedConfig for transferring clustering to other systems.
        metrics: Clustering quality metrics (RMSE, MAE, etc.) as Dataset.

    Example:
        >>> fs_clustered = fs.transform.cluster(n_clusters=8)
        >>> fs_clustered.clustering.n_clusters
        8
        >>> fs_clustered.clustering.expand_data(some_data)
    """

    cluster_assignments: xr.DataArray
    cluster_weights: xr.DataArray
    original_timesteps: pd.DatetimeIndex
    predefined: tsam.api.PredefinedConfig | None = None
    metrics: xr.Dataset | None = None

    # Not serialized
    _original_data: xr.Dataset | None = field(default=None, repr=False)

    def __post_init__(self):
        """Ensure DataArrays have names for IO."""
        if self.cluster_assignments.name is None:
            object.__setattr__(self, 'cluster_assignments', self.cluster_assignments.rename('cluster_assignments'))
        if self.cluster_weights.name is None:
            object.__setattr__(self, 'cluster_weights', self.cluster_weights.rename('cluster_weights'))

        # Convert predefined dict to PredefinedConfig if tsam available
        if isinstance(self.predefined, dict):
            try:
                import tsam

                object.__setattr__(self, 'predefined', tsam.PredefinedConfig.from_dict(self.predefined))
            except (ImportError, Exception):
                pass

    # ═══════════════════════════════════════════════════════════════════════════
    # Derived Properties
    # ═══════════════════════════════════════════════════════════════════════════

    @property
    def n_clusters(self) -> int:
        """Number of distinct clusters."""
        return int(self.cluster_weights.sizes.get('cluster', len(self.cluster_weights)))

    @property
    def timesteps_per_cluster(self) -> int:
        """Timesteps within each cluster (e.g., 24 for daily clustering)."""
        return len(self.original_timesteps) // self.n_original_clusters

    @property
    def n_original_clusters(self) -> int:
        """Number of original periods before clustering."""
        return int(self.cluster_assignments.sizes.get('original_cluster', len(self.cluster_assignments)))

    @property
    def cluster_order(self) -> xr.DataArray:
        """Alias for cluster_assignments."""
        return self.cluster_assignments

    @property
    def cluster_occurrences(self) -> xr.DataArray:
        """Alias for cluster_weights."""
        return self.cluster_weights

    # ═══════════════════════════════════════════════════════════════════════════
    # IO Methods
    # ═══════════════════════════════════════════════════════════════════════════

    def to_reference(self) -> tuple[dict, dict[str, xr.DataArray]]:
        """Serialize to reference dict + DataArrays for IO."""
        # Store original_timesteps as a DataArray (preserve index name in attrs)
        original_timesteps_da = xr.DataArray(
            self.original_timesteps,
            dims=['original_time'],
            name='original_timesteps',
            attrs={'index_name': self.original_timesteps.name},
        )

        arrays: dict[str, xr.DataArray] = {
            'cluster_assignments': self.cluster_assignments,
            'cluster_weights': self.cluster_weights,
            'original_timesteps': original_timesteps_da,
        }

        ref: dict[str, Any] = {'__class__': 'Clustering'}

        # Serialize predefined via to_dict() if available
        if self.predefined is not None:
            if hasattr(self.predefined, 'to_dict'):
                ref['predefined'] = self.predefined.to_dict()
            else:
                ref['predefined'] = self.predefined

        # Add metrics DataArrays with prefix
        if self.metrics is not None:
            for name, arr in self.metrics.data_vars.items():
                arrays[f'metrics_{name}'] = arr.rename(f'metrics_{name}')

        return ref, arrays

    @classmethod
    def from_reference(cls, ref: dict, arrays: dict[str, xr.DataArray]) -> Clustering:
        """Reconstruct from reference dict + DataArrays."""
        original_timesteps_da = arrays['original_timesteps']
        index_name = original_timesteps_da.attrs.get('index_name')
        original_timesteps = pd.DatetimeIndex(original_timesteps_da.values, name=index_name)

        # Reconstruct metrics Dataset from metrics_ prefixed arrays
        metrics_vars = {k[8:]: v for k, v in arrays.items() if k.startswith('metrics_')}
        metrics = xr.Dataset(metrics_vars) if metrics_vars else None

        # Get predefined (will be converted to PredefinedConfig in __post_init__)
        predefined = ref.get('predefined')

        return cls(
            cluster_assignments=arrays['cluster_assignments'],
            cluster_weights=arrays['cluster_weights'],
            original_timesteps=original_timesteps,
            predefined=predefined,
            metrics=metrics,
        )

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
