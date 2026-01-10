"""
Minimal clustering data structure for time series aggregation.

This module provides a single `Clustering` dataclass that stores only the essential
data needed for:
- Expanding solutions back to original timesteps
- Inter-cluster storage linking constraints
- IO serialization

All other values (timestep_mapping, etc.) are computed on demand from the core data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from ..color_processing import ColorType
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
    """Minimal clustering info for expansion, inter-cluster storage, and IO.

    All derived values (timestep_mapping, n_original_clusters) are computed on demand.

    Attributes:
        cluster_assignments: Cluster ID for each original period. Shape [original_cluster, period?, scenario?].
            For daily clustering of 365 days into 8 clusters, this has 365 values (0-7).
        cluster_weights: Count of original periods per cluster. Shape [cluster, period?, scenario?].
            For 365 days into 8 clusters, values sum to 365.
        n_clusters: Number of distinct clusters (e.g., 8 typical days).
        timesteps_per_cluster: Timesteps within each cluster (e.g., 24 for daily).
        original_timesteps: Original time coordinates before clustering.
        predefined: tsam PredefinedConfig for transferring clustering to another system.
        metrics: Optional clustering quality metrics (RMSE, MAE, etc.).

    Example:
        >>> fs_clustered = fs.transform.cluster(n_clusters=8)
        >>> fs_clustered.clustering.n_clusters
        8
        >>> fs_clustered.clustering.n_original_clusters
        365
        >>> fs_clustered.clustering.expand_data(some_data)  # Expand to original timesteps
    """

    cluster_assignments: xr.DataArray
    cluster_weights: xr.DataArray
    n_clusters: int
    timesteps_per_cluster: int
    original_timesteps: pd.DatetimeIndex
    predefined: Any = None  # tsam.PredefinedConfig
    metrics: xr.Dataset = field(default_factory=lambda: xr.Dataset())

    # Optional reference to original data for comparison plots
    _original_data: xr.Dataset | None = field(default=None, repr=False)

    # ═══════════════════════════════════════════════════════════════════════════
    # Computed Properties
    # ═══════════════════════════════════════════════════════════════════════════

    @property
    def n_original_clusters(self) -> int:
        """Number of original periods before clustering (e.g., 365 days)."""
        return self.cluster_assignments.sizes.get('original_cluster', len(self.cluster_assignments))

    @property
    def cluster_order(self) -> xr.DataArray:
        """Alias for cluster_assignments (legacy name)."""
        return self.cluster_assignments

    @property
    def cluster_occurrences(self) -> xr.DataArray:
        """Alias for cluster_weights (legacy name)."""
        return self.cluster_weights

    # ═══════════════════════════════════════════════════════════════════════════
    # Core Methods
    # ═══════════════════════════════════════════════════════════════════════════

    def get_cluster_assignments_for_slice(self, period: str | None = None, scenario: str | None = None) -> np.ndarray:
        """Get cluster assignments for a specific period/scenario slice.

        Args:
            period: Period label to select (if multi-period).
            scenario: Scenario label to select (if multi-scenario).

        Returns:
            1D numpy array of cluster IDs for each original period.
        """
        return _select_dims(self.cluster_assignments, period, scenario).values.astype(int)

    def get_timestep_mapping(self, period: str | None = None, scenario: str | None = None) -> np.ndarray:
        """Compute timestep mapping on demand from cluster_assignments.

        Maps each original timestep to its representative index in the (cluster, time) structure.

        Args:
            period: Period label to select (if multi-period).
            scenario: Scenario label to select (if multi-scenario).

        Returns:
            1D numpy array where mapping[t] = cluster_id * timesteps_per_cluster + time_within.
        """
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
        """Expand clustered data back to original timesteps.

        Args:
            data: DataArray with (cluster, time) or (time,) dimensions.
            period: Period label for multi-period data.
            scenario: Scenario label for multi-scenario data.

        Returns:
            DataArray expanded to original timesteps.
        """
        # Handle multi-dimensional data by selecting slice or iterating
        extra_dims = [d for d in data.dims if d not in ('cluster', 'time')]

        if not extra_dims:
            # Simple case: just (cluster, time) or (time,)
            return self._expand_slice(data, period, scenario)

        # Multi-dimensional: expand each slice and recombine
        # Check if we need to iterate over period/scenario in data
        dim_coords = {d: list(data.coords[d].values) for d in extra_dims}
        expanded_slices = {}

        for combo in np.ndindex(*[len(v) for v in dim_coords.values()]):
            selector = {d: dim_coords[d][i] for d, i in zip(extra_dims, combo, strict=True)}
            data_slice = data.sel(**selector)

            # Determine which period/scenario to use for mapping
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

        result = list(result_arrays.values())[0]
        return result.transpose('time', ...).assign_attrs(data.attrs)

    def _expand_slice(
        self,
        data: xr.DataArray,
        period: str | None,
        scenario: str | None,
    ) -> xr.DataArray:
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

    # ═══════════════════════════════════════════════════════════════════════════
    # IO Serialization
    # ═══════════════════════════════════════════════════════════════════════════

    def _create_reference_structure(self) -> tuple[dict, dict[str, xr.DataArray]]:
        """Create reference structure for netCDF serialization."""
        ref = {'__class__': self.__class__.__name__}

        data_arrays = {
            'clustering|cluster_assignments': self.cluster_assignments,
            'clustering|cluster_weights': self.cluster_weights,
        }

        # Store scalars as attrs
        ref['n_clusters'] = int(self.n_clusters)
        ref['timesteps_per_cluster'] = int(self.timesteps_per_cluster)
        ref['original_timesteps'] = self.original_timesteps.tolist()

        # Store metrics if present
        if self.metrics is not None and len(self.metrics.data_vars) > 0:
            for name, da in self.metrics.items():
                data_arrays[f'clustering|metrics|{name}'] = da

        return ref, data_arrays

    @classmethod
    def from_dataset(cls, ds: xr.Dataset, ref: dict) -> Clustering:
        """Reconstruct Clustering from netCDF dataset."""
        cluster_assignments = ds['clustering|cluster_assignments']
        cluster_weights = ds['clustering|cluster_weights']

        # Reconstruct metrics
        metrics_vars = {
            name.replace('clustering|metrics|', ''): ds[name]
            for name in ds.data_vars
            if name.startswith('clustering|metrics|')
        }
        metrics = xr.Dataset(metrics_vars) if metrics_vars else xr.Dataset()

        return cls(
            cluster_assignments=cluster_assignments,
            cluster_weights=cluster_weights,
            n_clusters=ref['n_clusters'],
            timesteps_per_cluster=ref['timesteps_per_cluster'],
            original_timesteps=pd.DatetimeIndex(ref['original_timesteps']),
            metrics=metrics,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # Convenience
    # ═══════════════════════════════════════════════════════════════════════════

    def __repr__(self) -> str:
        return (
            f'Clustering(\n'
            f'  n_clusters={self.n_clusters}\n'
            f'  timesteps_per_cluster={self.timesteps_per_cluster}\n'
            f'  n_original_clusters={self.n_original_clusters}\n'
            f'  original_timesteps={len(self.original_timesteps)}\n'
            f')'
        )

    @property
    def plot(self) -> ClusteringPlotAccessor:
        """Access plotting methods for clustering visualization."""
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
        """Plot cluster assignments over time as a heatmap timeline.

        Shows which cluster each timestep belongs to as a horizontal color bar.

        Args:
            select: xarray-style selection dict, e.g. {'scenario': 'Base Case'}.
            colors: Colorscale name or list of colors.
            facet_col: Dimension to facet on columns. 'auto' uses CONFIG priority.
            animation_frame: Dimension for animation slider.
            show: Whether to display the figure.
            **plotly_kwargs: Additional arguments passed to plotly.

        Returns:
            PlotResult containing the heatmap figure.
        """
        from ..config import CONFIG
        from ..plot_result import PlotResult
        from ..statistics_accessor import _apply_selection

        clustering = self._clustering
        cluster_order_da = clustering.cluster_assignments
        timesteps_per_cluster = clustering.timesteps_per_cluster
        original_time = clustering.original_timesteps

        # Apply selection if provided
        if select:
            cluster_order_da = _apply_selection(cluster_order_da.to_dataset(name='cluster'), select)['cluster']

        # Expand cluster_order to per-timestep
        extra_dims = [d for d in cluster_order_da.dims if d != 'original_cluster']
        expanded_values = np.repeat(cluster_order_da.values, timesteps_per_cluster, axis=0)

        # Trim to match original_time length if needed
        if len(original_time) < expanded_values.shape[0]:
            expanded_values = expanded_values[: len(original_time)]

        coords = {'time': original_time}
        coords.update({d: cluster_order_da.coords[d].values for d in extra_dims})
        cluster_da = xr.DataArray(expanded_values, dims=['time'] + extra_dims, coords=coords)

        # Add dummy y dimension for heatmap visualization
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

    def compare(
        self,
        variables: str | list[str] | None = None,
        *,
        kind: str = 'timeseries',
        select: SelectType | None = None,
        colors: ColorType | None = None,
        color: str | None = 'representation',
        line_dash: str | None = None,
        facet_col: str | None = 'auto',
        facet_row: str | None = 'auto',
        show: bool | None = None,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Compare original vs clustered time series data.

        Note: Requires original_data to be stored (set during clustering).

        Args:
            variables: Variable(s) to plot.
            kind: 'timeseries' or 'duration_curve'.
            select: xarray-style selection dict.
            colors: Color specification.
            color: Dimension for line colors.
            line_dash: Dimension for line dash styles.
            facet_col: Dimension for subplot columns.
            facet_row: Dimension for subplot rows.
            show: Whether to display the figure.
            **plotly_kwargs: Additional arguments passed to plotly.

        Returns:
            PlotResult containing the comparison figure.
        """
        from ..config import CONFIG
        from ..plot_result import PlotResult
        from ..statistics_accessor import _apply_selection

        if self._clustering._original_data is None:
            raise ValueError(
                'No original data available for comparison. Original data is not stored by default to save memory.'
            )

        if kind not in ('timeseries', 'duration_curve'):
            raise ValueError(f"Unknown kind '{kind}'. Use 'timeseries' or 'duration_curve'.")

        original_data = self._clustering._original_data
        resolved_variables = self._resolve_variables(variables, original_data)

        # Build Dataset with original and clustered (expanded) data
        data_vars = {}
        for var in resolved_variables:
            original = original_data[var]
            # Get clustered data from FlowSystem if available, else skip
            if self._flow_system is not None:
                fs_ds = self._flow_system.to_dataset(include_solution=False)
                if var in fs_ds:
                    clustered = self._clustering.expand_data(fs_ds[var])
                    combined = xr.concat(
                        [original, clustered], dim=pd.Index(['Original', 'Clustered'], name='representation')
                    )
                    data_vars[var] = combined

        if not data_vars:
            raise ValueError('No matching variables found for comparison')

        ds = xr.Dataset(data_vars)
        ds = _apply_selection(ds, select)

        # For duration curve: flatten and sort
        if kind == 'duration_curve':
            sorted_vars = {}
            for var in ds.data_vars:
                for rep in ds.coords['representation'].values:
                    values = np.sort(ds[var].sel(representation=rep).values.flatten())[::-1]
                    sorted_vars[(var, rep)] = values
            n = len(values)
            ds = xr.Dataset(
                {
                    var: xr.DataArray(
                        [sorted_vars[(var, r)] for r in ['Original', 'Clustered']],
                        dims=['representation', 'duration'],
                        coords={'representation': ['Original', 'Clustered'], 'duration': range(n)},
                    )
                    for var in resolved_variables
                    if var in data_vars
                }
            )

        title = 'Original vs Clustered' if len(data_vars) > 1 else f'Original vs Clustered: {list(data_vars)[0]}'
        if kind == 'duration_curve':
            title = 'Duration Curve' if len(data_vars) > 1 else f'Duration Curve: {list(data_vars)[0]}'

        line_kwargs = {}
        if line_dash is not None:
            line_kwargs['line_dash'] = line_dash
            if line_dash == 'representation':
                line_kwargs['line_dash_map'] = {'Original': 'dot', 'Clustered': 'solid'}

        fig = ds.fxplot.line(
            colors=colors,
            color=color,
            title=title,
            facet_col=facet_col,
            facet_row=facet_row,
            **line_kwargs,
            **plotly_kwargs,
        )
        fig.update_yaxes(matches=None)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))

        plot_result = PlotResult(data=ds, figure=fig)

        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            plot_result.show()

        return plot_result

    def _resolve_variables(self, variables: str | list[str] | None, data: xr.Dataset) -> list[str]:
        """Resolve variables parameter to a list of valid variable names."""
        time_vars = [
            name
            for name in data.data_vars
            if 'time' in data[name].dims and not np.isclose(data[name].min(), data[name].max())
        ]
        if not time_vars:
            raise ValueError('No time-varying variables found')

        if variables is None:
            return time_vars
        elif isinstance(variables, str):
            if variables not in time_vars:
                raise ValueError(f"Variable '{variables}' not found. Available: {time_vars}")
            return [variables]
        else:
            invalid = [v for v in variables if v not in time_vars]
            if invalid:
                raise ValueError(f'Variables {invalid} not found. Available: {time_vars}')
            return list(variables)


def _register_clustering_classes():
    """Register clustering classes for IO deserialization.

    This is called from flow_system.py to defer the import and avoid circular imports.
    """
    from ..structure import CLASS_REGISTRY

    CLASS_REGISTRY['Clustering'] = Clustering
