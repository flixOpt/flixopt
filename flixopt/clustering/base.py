"""
Base classes and data structures for time series aggregation (clustering).

This module provides an abstraction layer for time series aggregation that
supports multiple backends (TSAM, manual/external, etc.).

Terminology:
- "cluster" = a group of similar time chunks (e.g., similar days grouped together)
- "typical period" = a representative time chunk for a cluster (TSAM terminology)
- "cluster duration" = the length of each time chunk (e.g., 24h for daily clustering)

Note: This is separate from the model's "period" dimension (years/months) and
"scenario" dimension. The aggregation operates on the 'time' dimension.

All data structures use xarray for consistent handling of coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from ..color_processing import ColorType
    from ..plot_result import PlotResult
    from ..statistics_accessor import SelectType


@dataclass
class ClusterStructure:
    """Structure information for inter-cluster storage linking.

    This class captures the hierarchical structure of time series clustering,
    which is needed for proper storage state-of-charge tracking across
    typical periods when using cluster().

    Note: The "original_cluster" dimension indexes the original cluster-sized
    time segments (e.g., 0..364 for 365 days), NOT the model's "period" dimension
    (years). Each original segment gets assigned to a representative cluster.

    Attributes:
        cluster_order: Maps original cluster index → representative cluster ID.
            dims: [original_cluster] for simple case, or
            [original_cluster, period, scenario] for multi-period/scenario systems.
            Values are cluster IDs (0 to n_clusters-1).
        cluster_occurrences: Count of how many original time chunks each cluster represents.
            dims: [cluster] for simple case, or [cluster, period, scenario] for multi-dim.
        n_clusters: Number of distinct clusters (typical periods).
        timesteps_per_cluster: Number of timesteps in each cluster (e.g., 24 for daily).

    Example:
        For 365 days clustered into 8 typical days:
        - cluster_order: shape (365,), values 0-7 indicating which cluster each day belongs to
        - cluster_occurrences: shape (8,), e.g., [45, 46, 46, 46, 46, 45, 45, 46]
        - n_clusters: 8
        - timesteps_per_cluster: 24 (for hourly data)

        For multi-scenario (e.g., 2 scenarios):
        - cluster_order: shape (365, 2) with dims [original_cluster, scenario]
        - cluster_occurrences: shape (8, 2) with dims [cluster, scenario]
    """

    cluster_order: xr.DataArray
    cluster_occurrences: xr.DataArray
    n_clusters: int | xr.DataArray
    timesteps_per_cluster: int

    def __post_init__(self):
        """Validate and ensure proper DataArray formatting."""
        # Ensure cluster_order is a DataArray with proper dims
        if not isinstance(self.cluster_order, xr.DataArray):
            self.cluster_order = xr.DataArray(self.cluster_order, dims=['original_cluster'], name='cluster_order')
        elif self.cluster_order.name is None:
            self.cluster_order = self.cluster_order.rename('cluster_order')

        # Ensure cluster_occurrences is a DataArray with proper dims
        if not isinstance(self.cluster_occurrences, xr.DataArray):
            self.cluster_occurrences = xr.DataArray(
                self.cluster_occurrences, dims=['cluster'], name='cluster_occurrences'
            )
        elif self.cluster_occurrences.name is None:
            self.cluster_occurrences = self.cluster_occurrences.rename('cluster_occurrences')

    def __repr__(self) -> str:
        n_clusters = (
            int(self.n_clusters) if isinstance(self.n_clusters, (int, np.integer)) else int(self.n_clusters.values)
        )
        occ = [int(self.cluster_occurrences.sel(cluster=c).values) for c in range(n_clusters)]
        return (
            f'ClusterStructure(\n'
            f'  {self.n_original_clusters} original periods → {n_clusters} clusters\n'
            f'  timesteps_per_cluster={self.timesteps_per_cluster}\n'
            f'  occurrences={occ}\n'
            f')'
        )

    def _create_reference_structure(self) -> tuple[dict, dict[str, xr.DataArray]]:
        """Create reference structure for serialization."""
        ref = {'__class__': self.__class__.__name__}
        arrays = {}

        # Store DataArrays with references
        arrays[str(self.cluster_order.name)] = self.cluster_order
        ref['cluster_order'] = f':::{self.cluster_order.name}'

        arrays[str(self.cluster_occurrences.name)] = self.cluster_occurrences
        ref['cluster_occurrences'] = f':::{self.cluster_occurrences.name}'

        # Store scalar values
        if isinstance(self.n_clusters, xr.DataArray):
            n_clusters_name = self.n_clusters.name or 'n_clusters'
            self.n_clusters = self.n_clusters.rename(n_clusters_name)
            arrays[n_clusters_name] = self.n_clusters
            ref['n_clusters'] = f':::{n_clusters_name}'
        else:
            ref['n_clusters'] = int(self.n_clusters)

        ref['timesteps_per_cluster'] = self.timesteps_per_cluster

        return ref, arrays

    @property
    def n_original_clusters(self) -> int:
        """Number of original periods (before clustering)."""
        return len(self.cluster_order.coords['original_cluster'])

    @property
    def has_multi_dims(self) -> bool:
        """Check if cluster_order has period/scenario dimensions."""
        return 'period' in self.cluster_order.dims or 'scenario' in self.cluster_order.dims

    def get_cluster_order_for_slice(self, period: str | None = None, scenario: str | None = None) -> np.ndarray:
        """Get cluster_order for a specific (period, scenario) combination.

        Args:
            period: Period label (None if no period dimension).
            scenario: Scenario label (None if no scenario dimension).

        Returns:
            1D numpy array of cluster indices for the specified slice.
        """
        order = self.cluster_order
        if 'period' in order.dims and period is not None:
            order = order.sel(period=period)
        if 'scenario' in order.dims and scenario is not None:
            order = order.sel(scenario=scenario)
        return order.values.astype(int)

    def get_cluster_occurrences_for_slice(
        self, period: str | None = None, scenario: str | None = None
    ) -> dict[int, int]:
        """Get cluster occurrence counts for a specific (period, scenario) combination.

        Args:
            period: Period label (None if no period dimension).
            scenario: Scenario label (None if no scenario dimension).

        Returns:
            Dict mapping cluster ID to occurrence count.
        """
        occurrences = self.cluster_occurrences
        if 'period' in occurrences.dims and period is not None:
            occurrences = occurrences.sel(period=period)
        if 'scenario' in occurrences.dims and scenario is not None:
            occurrences = occurrences.sel(scenario=scenario)
        return {int(c): int(occurrences.sel(cluster=c).values) for c in occurrences.coords['cluster'].values}

    def get_cluster_weight_per_timestep(self) -> xr.DataArray:
        """Get weight for each representative timestep.

        Returns an array where each timestep's weight equals the number of
        original periods its cluster represents.

        Returns:
            DataArray with dims [time] or [time, period, scenario].
        """
        # Expand cluster_occurrences to timesteps
        n_clusters = (
            int(self.n_clusters) if isinstance(self.n_clusters, (int, np.integer)) else int(self.n_clusters.values)
        )

        # Get occurrence for each cluster, then repeat for timesteps
        weights_list = []
        for c in range(n_clusters):
            occ = self.cluster_occurrences.sel(cluster=c)
            weights_list.append(np.repeat(float(occ.values), self.timesteps_per_cluster))

        weights = np.concatenate(weights_list)
        return xr.DataArray(
            weights,
            dims=['time'],
            coords={'time': np.arange(len(weights))},
            name='cluster_weight',
        )

    def plot(self, colors: str | list[str] | None = None, show: bool | None = None) -> PlotResult:
        """Plot cluster assignment visualization.

        Shows which cluster each original period belongs to, and the
        number of occurrences per cluster.

        Args:
            colors: Colorscale name (str) or list of colors.
                Defaults to CONFIG.Plotting.default_sequential_colorscale.
            show: Whether to display the figure. Defaults to CONFIG.Plotting.default_show.

        Returns:
            PlotResult containing the figure and underlying data.
        """
        from ..config import CONFIG
        from ..plot_result import PlotResult

        n_clusters = (
            int(self.n_clusters) if isinstance(self.n_clusters, (int, np.integer)) else int(self.n_clusters.values)
        )

        cluster_order = self.get_cluster_order_for_slice()

        # Build DataArray for fxplot heatmap
        cluster_da = xr.DataArray(
            cluster_order.reshape(1, -1),
            dims=['y', 'original_cluster'],
            coords={'y': ['Cluster'], 'original_cluster': range(1, len(cluster_order) + 1)},
            name='cluster_assignment',
        )

        # Use fxplot.heatmap for smart defaults
        colorscale = colors or CONFIG.Plotting.default_sequential_colorscale
        fig = cluster_da.fxplot.heatmap(
            colors=colorscale,
            title=f'Cluster Assignment ({self.n_original_clusters} periods → {n_clusters} clusters)',
        )
        fig.update_yaxes(showticklabels=False)
        fig.update_coloraxes(colorbar_title='Cluster')

        # Build data for PlotResult
        data = xr.Dataset(
            {
                'cluster_order': self.cluster_order,
                'cluster_occurrences': self.cluster_occurrences,
            }
        )
        plot_result = PlotResult(data=data, figure=fig)

        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            plot_result.show()

        return plot_result


@dataclass
class ClusterResult:
    """Universal result from any time series aggregation method.

    This dataclass captures all information needed to:
    1. Transform a FlowSystem to use aggregated (clustered) timesteps
    2. Expand a solution back to original resolution
    3. Properly weight results for statistics

    Attributes:
        timestep_mapping: Maps each original timestep to its representative index.
            dims: [original_time] for simple case, or
            [original_time, period, scenario] for multi-period/scenario systems.
            Values are indices into the representative timesteps (0 to n_representatives-1).
        n_representatives: Number of representative timesteps after aggregation.
        representative_weights: Weight for each representative timestep.
            dims: [time] or [time, period, scenario]
            Typically equals the number of original timesteps each representative covers.
            Used as cluster_weight in the FlowSystem.
        aggregated_data: Time series data aggregated to representative timesteps.
            Optional - some backends may not aggregate data.
        cluster_structure: Hierarchical clustering structure for storage linking.
            Optional - only needed when using cluster() mode.
        original_data: Reference to original data before aggregation.
            Optional - useful for expand_solution().

    Example:
        For 8760 hourly timesteps clustered into 192 representative timesteps (8 clusters x 24h):
        - timestep_mapping: shape (8760,), values 0-191
        - n_representatives: 192
        - representative_weights: shape (192,), summing to 8760
    """

    timestep_mapping: xr.DataArray
    n_representatives: int | xr.DataArray
    representative_weights: xr.DataArray
    aggregated_data: xr.Dataset | None = None
    cluster_structure: ClusterStructure | None = None
    original_data: xr.Dataset | None = None

    def __post_init__(self):
        """Validate and ensure proper DataArray formatting."""
        # Ensure timestep_mapping is a DataArray
        if not isinstance(self.timestep_mapping, xr.DataArray):
            self.timestep_mapping = xr.DataArray(self.timestep_mapping, dims=['original_time'], name='timestep_mapping')
        elif self.timestep_mapping.name is None:
            self.timestep_mapping = self.timestep_mapping.rename('timestep_mapping')

        # Ensure representative_weights is a DataArray
        # Can be (cluster, time) for 2D structure or (time,) for flat structure
        if not isinstance(self.representative_weights, xr.DataArray):
            self.representative_weights = xr.DataArray(self.representative_weights, name='representative_weights')
        elif self.representative_weights.name is None:
            self.representative_weights = self.representative_weights.rename('representative_weights')

    def __repr__(self) -> str:
        n_rep = (
            int(self.n_representatives)
            if isinstance(self.n_representatives, (int, np.integer))
            else int(self.n_representatives.values)
        )
        has_structure = self.cluster_structure is not None
        has_data = self.original_data is not None and self.aggregated_data is not None
        return (
            f'ClusterResult(\n'
            f'  {self.n_original_timesteps} original → {n_rep} representative timesteps\n'
            f'  weights sum={float(self.representative_weights.sum().values):.0f}\n'
            f'  cluster_structure={has_structure}, data={has_data}\n'
            f')'
        )

    def _create_reference_structure(self) -> tuple[dict, dict[str, xr.DataArray]]:
        """Create reference structure for serialization."""
        ref = {'__class__': self.__class__.__name__}
        arrays = {}

        # Store DataArrays with references
        arrays[str(self.timestep_mapping.name)] = self.timestep_mapping
        ref['timestep_mapping'] = f':::{self.timestep_mapping.name}'

        arrays[str(self.representative_weights.name)] = self.representative_weights
        ref['representative_weights'] = f':::{self.representative_weights.name}'

        # Store scalar values
        if isinstance(self.n_representatives, xr.DataArray):
            n_rep_name = self.n_representatives.name or 'n_representatives'
            self.n_representatives = self.n_representatives.rename(n_rep_name)
            arrays[n_rep_name] = self.n_representatives
            ref['n_representatives'] = f':::{n_rep_name}'
        else:
            ref['n_representatives'] = int(self.n_representatives)

        # Store nested ClusterStructure if present
        if self.cluster_structure is not None:
            cs_ref, cs_arrays = self.cluster_structure._create_reference_structure()
            ref['cluster_structure'] = cs_ref
            arrays.update(cs_arrays)

        # Skip aggregated_data and original_data - not needed for serialization

        return ref, arrays

    @property
    def n_original_timesteps(self) -> int:
        """Number of original timesteps (before aggregation)."""
        return len(self.timestep_mapping.coords['original_time'])

    def get_expansion_mapping(self) -> xr.DataArray:
        """Get mapping from original timesteps to representative indices.

        This is the same as timestep_mapping but ensures proper naming
        for use in expand_solution().

        Returns:
            DataArray mapping original timesteps to representative indices.
        """
        return self.timestep_mapping.rename('expansion_mapping')

    def get_timestep_mapping_for_slice(self, period: str | None = None, scenario: str | None = None) -> np.ndarray:
        """Get timestep_mapping for a specific (period, scenario) combination.

        Args:
            period: Period label (None if no period dimension).
            scenario: Scenario label (None if no scenario dimension).

        Returns:
            1D numpy array of representative timestep indices for the specified slice.
        """
        mapping = self.timestep_mapping
        if 'period' in mapping.dims and period is not None:
            mapping = mapping.sel(period=period)
        if 'scenario' in mapping.dims and scenario is not None:
            mapping = mapping.sel(scenario=scenario)
        return mapping.values.astype(int)

    def expand_data(self, aggregated: xr.DataArray, original_time: xr.DataArray | None = None) -> xr.DataArray:
        """Expand aggregated data back to original timesteps.

        Uses the stored timestep_mapping to map each original timestep to its
        representative value from the aggregated data. Handles multi-dimensional
        data with period/scenario dimensions.

        Args:
            aggregated: DataArray with aggregated (reduced) time dimension.
            original_time: Original time coordinates. If None, uses coords from
                original_data if available.

        Returns:
            DataArray expanded to original timesteps.

        Example:
            >>> result = fs_clustered.clustering.result
            >>> aggregated_values = result.aggregated_data['Demand|profile']
            >>> expanded = result.expand_data(aggregated_values)
            >>> len(expanded.time) == len(original_timesteps)  # True
        """
        import pandas as pd

        if original_time is None:
            if self.original_data is None:
                raise ValueError('original_time required when original_data is not available')
            original_time = self.original_data.coords['time']

        timestep_mapping = self.timestep_mapping
        has_periods = 'period' in timestep_mapping.dims
        has_scenarios = 'scenario' in timestep_mapping.dims
        has_cluster_dim = 'cluster' in aggregated.dims

        # Simple case: no period/scenario dimensions
        if not has_periods and not has_scenarios:
            mapping = timestep_mapping.values
            if has_cluster_dim:
                # 2D cluster structure: convert flat indices to (cluster, time_within)
                # Use cluster_structure's timesteps_per_cluster, not aggregated.sizes['time']
                # because the solution may include extra timesteps (timesteps_extra)
                timesteps_per_cluster = self.cluster_structure.timesteps_per_cluster
                cluster_ids = mapping // timesteps_per_cluster
                time_within = mapping % timesteps_per_cluster
                expanded_values = aggregated.values[cluster_ids, time_within]
            else:
                expanded_values = aggregated.values[mapping]
            return xr.DataArray(
                expanded_values,
                coords={'time': original_time},
                dims=['time'],
                attrs=aggregated.attrs,
            )

        # Multi-dimensional: expand each (period, scenario) slice and recombine
        periods = list(timestep_mapping.coords['period'].values) if has_periods else [None]
        scenarios = list(timestep_mapping.coords['scenario'].values) if has_scenarios else [None]

        expanded_slices: dict[tuple, xr.DataArray] = {}
        for p in periods:
            for s in scenarios:
                # Get mapping for this slice
                mapping_slice = timestep_mapping
                if p is not None:
                    mapping_slice = mapping_slice.sel(period=p)
                if s is not None:
                    mapping_slice = mapping_slice.sel(scenario=s)
                mapping = mapping_slice.values

                # Select the data slice
                selector = {}
                if p is not None and 'period' in aggregated.dims:
                    selector['period'] = p
                if s is not None and 'scenario' in aggregated.dims:
                    selector['scenario'] = s

                slice_da = aggregated.sel(**selector, drop=True) if selector else aggregated

                if has_cluster_dim:
                    # 2D cluster structure: convert flat indices to (cluster, time_within)
                    # Use cluster_structure's timesteps_per_cluster, not slice_da.sizes['time']
                    # because the solution may include extra timesteps (timesteps_extra)
                    timesteps_per_cluster = self.cluster_structure.timesteps_per_cluster
                    cluster_ids = mapping // timesteps_per_cluster
                    time_within = mapping % timesteps_per_cluster
                    expanded_values = slice_da.values[cluster_ids, time_within]
                    expanded = xr.DataArray(expanded_values, dims=['time'])
                else:
                    expanded = slice_da.isel(time=xr.DataArray(mapping, dims=['time']))
                expanded_slices[(p, s)] = expanded.assign_coords(time=original_time)

        # Recombine slices using xr.concat
        if has_periods and has_scenarios:
            period_arrays = []
            for p in periods:
                scenario_arrays = [expanded_slices[(p, s)] for s in scenarios]
                period_arrays.append(xr.concat(scenario_arrays, dim=pd.Index(scenarios, name='scenario')))
            result = xr.concat(period_arrays, dim=pd.Index(periods, name='period'))
        elif has_periods:
            result = xr.concat([expanded_slices[(p, None)] for p in periods], dim=pd.Index(periods, name='period'))
        else:
            result = xr.concat(
                [expanded_slices[(None, s)] for s in scenarios], dim=pd.Index(scenarios, name='scenario')
            )

        return result.transpose('time', ...).assign_attrs(aggregated.attrs)

    def validate(self) -> None:
        """Validate that all fields are consistent.

        Raises:
            ValueError: If validation fails.
        """
        n_rep = (
            int(self.n_representatives)
            if isinstance(self.n_representatives, (int, np.integer))
            else int(self.n_representatives.max().values)
        )

        # Check mapping values are within range
        max_idx = int(self.timestep_mapping.max().values)
        if max_idx >= n_rep:
            raise ValueError(f'timestep_mapping contains index {max_idx} but n_representatives is {n_rep}')

        # Check weights dimensions
        # representative_weights should have (cluster,) dimension with n_clusters elements
        # (plus optional period/scenario dimensions)
        if self.cluster_structure is not None:
            n_clusters = self.cluster_structure.n_clusters
            if 'cluster' in self.representative_weights.dims:
                weights_n_clusters = self.representative_weights.sizes['cluster']
                if weights_n_clusters != n_clusters:
                    raise ValueError(
                        f'representative_weights has {weights_n_clusters} clusters '
                        f'but cluster_structure has {n_clusters}'
                    )

        # Check weights sum roughly equals number of original periods
        # (each weight is how many original periods that cluster represents)
        # Sum should be checked per period/scenario slice, not across all dimensions
        if self.cluster_structure is not None:
            n_original_clusters = self.cluster_structure.n_original_clusters
            # Sum over cluster dimension only (keep period/scenario if present)
            weight_sum_per_slice = self.representative_weights.sum(dim='cluster')
            # Check each slice
            if weight_sum_per_slice.size == 1:
                # Simple case: no period/scenario
                weight_sum = float(weight_sum_per_slice.values)
                if abs(weight_sum - n_original_clusters) > 1e-6:
                    import warnings

                    warnings.warn(
                        f'representative_weights sum ({weight_sum}) does not match '
                        f'n_original_clusters ({n_original_clusters})',
                        stacklevel=2,
                    )
            else:
                # Multi-dimensional: check each slice
                for val in weight_sum_per_slice.values.flat:
                    if abs(float(val) - n_original_clusters) > 1e-6:
                        import warnings

                        warnings.warn(
                            f'representative_weights sum per slice ({float(val)}) does not match '
                            f'n_original_clusters ({n_original_clusters})',
                            stacklevel=2,
                        )
                        break  # Only warn once


class ClusteringPlotAccessor:
    """Plot accessor for Clustering objects.

    Provides visualization methods for comparing original vs aggregated data
    and understanding the clustering structure.

    Example:
        >>> fs_clustered = flow_system.transform.cluster(n_clusters=8, cluster_duration='1D')
        >>> fs_clustered.clustering.plot.compare()  # timeseries comparison
        >>> fs_clustered.clustering.plot.compare(kind='duration_curve')  # duration curve
        >>> fs_clustered.clustering.plot.heatmap()  # structure visualization
        >>> fs_clustered.clustering.plot.clusters()  # cluster profiles
    """

    def __init__(self, clustering: Clustering):
        self._clustering = clustering

    def compare(
        self,
        kind: str = 'timeseries',
        variables: str | list[str] | None = None,
        *,
        select: SelectType | None = None,
        colors: ColorType | None = None,
        facet_col: str | None = 'auto',
        facet_row: str | None = 'auto',
        show: bool | None = None,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Compare original vs aggregated data.

        Args:
            kind: Type of comparison plot.
                - 'timeseries': Time series comparison (default)
                - 'duration_curve': Sorted duration curve comparison
            variables: Variable(s) to plot. Can be a string, list of strings,
                or None to plot all time-varying variables.
            select: xarray-style selection dict, e.g. {'scenario': 'Base Case'}.
            colors: Color specification (colorscale name, color list, or label-to-color dict).
            facet_col: Dimension for subplot columns. 'auto' uses CONFIG priority.
                Use 'variable' to create separate columns per variable.
            facet_row: Dimension for subplot rows. 'auto' uses CONFIG priority.
                Use 'variable' to create separate rows per variable.
            show: Whether to display the figure.
                Defaults to CONFIG.Plotting.default_show.
            **plotly_kwargs: Additional arguments passed to plotly.

        Returns:
            PlotResult containing the comparison figure and underlying data.
        """
        import pandas as pd

        from ..config import CONFIG
        from ..plot_result import PlotResult
        from ..statistics_accessor import _apply_selection

        if kind not in ('timeseries', 'duration_curve'):
            raise ValueError(f"Unknown kind '{kind}'. Use 'timeseries' or 'duration_curve'.")

        result = self._clustering.result
        if result.original_data is None or result.aggregated_data is None:
            raise ValueError('No original/aggregated data available for comparison')

        resolved_variables = self._resolve_variables(variables)

        # Build Dataset with variables as data_vars
        data_vars = {}
        for var in resolved_variables:
            original = result.original_data[var]
            clustered = result.expand_data(result.aggregated_data[var])
            combined = xr.concat([original, clustered], dim=pd.Index(['Original', 'Clustered'], name='representation'))
            data_vars[var] = combined
        ds = xr.Dataset(data_vars)

        # Apply selection
        ds = _apply_selection(ds, select)

        # For duration curve: flatten and sort values
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
                }
            )

        # Set title based on kind
        if kind == 'timeseries':
            title = (
                'Original vs Clustered'
                if len(resolved_variables) > 1
                else f'Original vs Clustered: {resolved_variables[0]}'
            )
        else:
            title = 'Duration Curve' if len(resolved_variables) > 1 else f'Duration Curve: {resolved_variables[0]}'

        # Use fxplot for smart defaults with line_dash for representation
        fig = ds.fxplot.line(
            colors=colors,
            title=title,
            facet_col=facet_col,
            facet_row=facet_row,
            line_dash='representation',
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

    def _get_time_varying_variables(self) -> list[str]:
        """Get list of time-varying variables from original data."""
        result = self._clustering.result
        if result.original_data is None:
            return []
        return [
            name
            for name in result.original_data.data_vars
            if 'time' in result.original_data[name].dims
            and not np.isclose(result.original_data[name].min(), result.original_data[name].max())
        ]

    def _resolve_variables(self, variables: str | list[str] | None) -> list[str]:
        """Resolve variables parameter to a list of valid variable names."""
        time_vars = self._get_time_varying_variables()
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
        The x-axis is time, color indicates cluster assignment. This visualization
        aligns with time series data, making it easy to correlate cluster
        assignments with other plots.

        For multi-period/scenario data, uses faceting and/or animation.

        Args:
            select: xarray-style selection dict, e.g. {'scenario': 'Base Case'}.
            colors: Colorscale name (str) or list of colors for heatmap coloring.
                Dicts are not supported for heatmaps.
                Defaults to CONFIG.Plotting.default_sequential_colorscale.
            facet_col: Dimension to facet on columns. 'auto' uses CONFIG priority.
            animation_frame: Dimension for animation slider. 'auto' uses CONFIG priority.
            show: Whether to display the figure.
                Defaults to CONFIG.Plotting.default_show.
            **plotly_kwargs: Additional arguments passed to plotly.

        Returns:
            PlotResult containing the heatmap figure and cluster assignment data.
            The data has 'cluster' variable with time dimension, matching original timesteps.
        """
        import pandas as pd

        from ..config import CONFIG
        from ..plot_result import PlotResult
        from ..statistics_accessor import _apply_selection

        result = self._clustering.result
        cs = result.cluster_structure
        if cs is None:
            raise ValueError('No cluster structure available')

        cluster_order_da = cs.cluster_order
        timesteps_per_period = cs.timesteps_per_cluster
        original_time = result.original_data.coords['time'] if result.original_data is not None else None

        # Apply selection if provided
        if select:
            cluster_order_da = _apply_selection(cluster_order_da.to_dataset(name='cluster'), select)['cluster']

        # Check for multi-dimensional data
        has_periods = 'period' in cluster_order_da.dims
        has_scenarios = 'scenario' in cluster_order_da.dims

        # Get dimension values
        periods = list(cluster_order_da.coords['period'].values) if has_periods else [None]
        scenarios = list(cluster_order_da.coords['scenario'].values) if has_scenarios else [None]

        # Build cluster assignment per timestep for each (period, scenario) slice
        cluster_slices: dict[tuple, xr.DataArray] = {}
        for p in periods:
            for s in scenarios:
                cluster_order = cs.get_cluster_order_for_slice(period=p, scenario=s)
                # Expand: each cluster repeated timesteps_per_period times
                cluster_per_timestep = np.repeat(cluster_order, timesteps_per_period)
                cluster_slices[(p, s)] = xr.DataArray(
                    cluster_per_timestep,
                    dims=['time'],
                    coords={'time': original_time} if original_time is not None else None,
                )

        # Combine slices into multi-dimensional DataArray
        if has_periods and has_scenarios:
            period_arrays = []
            for p in periods:
                scenario_arrays = [cluster_slices[(p, s)] for s in scenarios]
                period_arrays.append(xr.concat(scenario_arrays, dim=pd.Index(scenarios, name='scenario')))
            cluster_da = xr.concat(period_arrays, dim=pd.Index(periods, name='period'))
        elif has_periods:
            cluster_da = xr.concat(
                [cluster_slices[(p, None)] for p in periods],
                dim=pd.Index(periods, name='period'),
            )
        elif has_scenarios:
            cluster_da = xr.concat(
                [cluster_slices[(None, s)] for s in scenarios],
                dim=pd.Index(scenarios, name='scenario'),
            )
        else:
            cluster_da = cluster_slices[(None, None)]

        # Add dummy y dimension for heatmap visualization (single row)
        heatmap_da = cluster_da.expand_dims('y', axis=-1)
        heatmap_da = heatmap_da.assign_coords(y=['Cluster'])
        heatmap_da.name = 'cluster_assignment'

        # Use fxplot.heatmap for smart defaults
        fig = heatmap_da.fxplot.heatmap(
            colors=colors,
            title='Cluster Assignments',
            facet_col=facet_col,
            animation_frame=animation_frame,
            aspect='auto',
            **plotly_kwargs,
        )

        # Clean up: hide y-axis since it's just a single row
        fig.update_yaxes(showticklabels=False)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))

        # Data is exactly what we plotted (without dummy y dimension)
        cluster_da.name = 'cluster'
        data = xr.Dataset({'cluster': cluster_da})
        plot_result = PlotResult(data=data, figure=fig)

        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            plot_result.show()

        return plot_result

    def clusters(
        self,
        variables: str | list[str] | None = None,
        *,
        select: SelectType | None = None,
        colors: ColorType | None = None,
        facet_cols: int | None = None,
        show: bool | None = None,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot each cluster's typical period profile.

        Shows each cluster as a separate faceted subplot with all variables
        colored differently. Useful for understanding what each cluster represents.

        Args:
            variables: Variable(s) to plot. Can be a string, list of strings,
                or None to plot all time-varying variables.
            select: xarray-style selection dict, e.g. {'scenario': 'Base Case'}.
            colors: Color specification (colorscale name, color list, or label-to-color dict).
            facet_cols: Max columns before wrapping facets.
                Defaults to CONFIG.Plotting.default_facet_cols.
            show: Whether to display the figure.
                Defaults to CONFIG.Plotting.default_show.
            **plotly_kwargs: Additional arguments passed to plotly.

        Returns:
            PlotResult containing the figure and underlying data.
        """
        from ..config import CONFIG
        from ..plot_result import PlotResult
        from ..statistics_accessor import _apply_selection

        result = self._clustering.result
        cs = result.cluster_structure
        if result.aggregated_data is None or cs is None:
            raise ValueError('No aggregated data or cluster structure available')

        # Apply selection to aggregated data
        aggregated_data = _apply_selection(result.aggregated_data, select)

        time_vars = self._get_time_varying_variables()
        if not time_vars:
            raise ValueError('No time-varying variables found')

        # Resolve variables
        resolved_variables = self._resolve_variables(variables)

        n_clusters = int(cs.n_clusters) if isinstance(cs.n_clusters, (int, np.integer)) else int(cs.n_clusters.values)
        timesteps_per_cluster = cs.timesteps_per_cluster

        # Build Dataset with cluster dimension, using labels with occurrence counts
        cluster_labels = [
            f'Cluster {c} (×{int(cs.cluster_occurrences.sel(cluster=c).values)})' for c in range(n_clusters)
        ]

        data_vars = {}
        for var in resolved_variables:
            data = aggregated_data[var].values
            data_by_cluster = data.reshape(n_clusters, timesteps_per_cluster)
            data_vars[var] = xr.DataArray(
                data_by_cluster,
                dims=['cluster', 'time'],
                coords={'cluster': cluster_labels, 'time': range(timesteps_per_cluster)},
            )

        ds = xr.Dataset(data_vars)
        title = 'Clusters' if len(resolved_variables) > 1 else f'Clusters: {resolved_variables[0]}'

        # Use fxplot for smart defaults
        fig = ds.fxplot.line(
            colors=colors,
            title=title,
            facet_col='cluster',
            facet_cols=facet_cols,
            **plotly_kwargs,
        )
        fig.update_yaxes(matches=None)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))

        # Include occurrences in result data
        data_vars['occurrences'] = cs.cluster_occurrences
        result_data = xr.Dataset(data_vars)
        plot_result = PlotResult(data=result_data, figure=fig)

        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            plot_result.show()

        return plot_result


@dataclass
class Clustering:
    """Information about an aggregation stored on a FlowSystem.

    This is stored on the FlowSystem after aggregation to enable:
    - expand_solution() to map back to original timesteps
    - Statistics to properly weight results
    - Inter-cluster storage linking
    - Serialization/deserialization of aggregated models

    Attributes:
        result: The ClusterResult from the aggregation backend.
        backend_name: Name of the aggregation backend used (e.g., 'tsam', 'manual').
        metrics: Clustering quality metrics (RMSE, MAE, etc.) as xr.Dataset.
            Each metric (e.g., 'RMSE', 'MAE') is a DataArray with dims
            ``[time_series, period?, scenario?]``.

    Example:
        >>> fs_clustered = flow_system.transform.cluster(n_clusters=8, cluster_duration='1D')
        >>> fs_clustered.clustering.n_clusters
        8
        >>> fs_clustered.clustering.plot.compare()
        >>> fs_clustered.clustering.plot.heatmap()
    """

    result: ClusterResult
    backend_name: str = 'unknown'
    metrics: xr.Dataset | None = None

    def _create_reference_structure(self) -> tuple[dict, dict[str, xr.DataArray]]:
        """Create reference structure for serialization."""
        ref = {'__class__': self.__class__.__name__}
        arrays = {}

        # Store nested ClusterResult
        result_ref, result_arrays = self.result._create_reference_structure()
        ref['result'] = result_ref
        arrays.update(result_arrays)

        # Store scalar values
        ref['backend_name'] = self.backend_name

        return ref, arrays

    def __repr__(self) -> str:
        cs = self.result.cluster_structure
        if cs is not None:
            n_clusters = (
                int(cs.n_clusters) if isinstance(cs.n_clusters, (int, np.integer)) else int(cs.n_clusters.values)
            )
            structure_info = f'{cs.n_original_clusters} periods → {n_clusters} clusters'
        else:
            structure_info = 'no structure'
        return f'Clustering(\n  backend={self.backend_name!r}\n  {structure_info}\n)'

    @property
    def plot(self) -> ClusteringPlotAccessor:
        """Access plotting methods for clustering visualization.

        Returns:
            ClusteringPlotAccessor with compare(), heatmap(), and clusters() methods.

        Example:
            >>> fs.clustering.plot.compare()  # timeseries comparison
            >>> fs.clustering.plot.compare(kind='duration_curve')  # duration curve
            >>> fs.clustering.plot.heatmap()  # structure visualization
            >>> fs.clustering.plot.clusters()  # cluster profiles
        """
        return ClusteringPlotAccessor(self)

    # Convenience properties delegating to nested objects

    @property
    def cluster_order(self) -> xr.DataArray:
        """Which cluster each original period belongs to."""
        if self.result.cluster_structure is None:
            raise ValueError('No cluster_structure available')
        return self.result.cluster_structure.cluster_order

    @property
    def occurrences(self) -> xr.DataArray:
        """How many original periods each cluster represents."""
        if self.result.cluster_structure is None:
            raise ValueError('No cluster_structure available')
        return self.result.cluster_structure.cluster_occurrences

    @property
    def n_clusters(self) -> int:
        """Number of clusters."""
        if self.result.cluster_structure is None:
            raise ValueError('No cluster_structure available')
        n = self.result.cluster_structure.n_clusters
        return int(n) if isinstance(n, (int, np.integer)) else int(n.values)

    @property
    def n_original_clusters(self) -> int:
        """Number of original periods (before clustering)."""
        if self.result.cluster_structure is None:
            raise ValueError('No cluster_structure available')
        return self.result.cluster_structure.n_original_clusters

    @property
    def timesteps_per_period(self) -> int:
        """Number of timesteps in each period/cluster.

        Alias for :attr:`timesteps_per_cluster`.
        """
        return self.timesteps_per_cluster

    @property
    def timesteps_per_cluster(self) -> int:
        """Number of timesteps in each cluster."""
        if self.result.cluster_structure is None:
            raise ValueError('No cluster_structure available')
        return self.result.cluster_structure.timesteps_per_cluster

    @property
    def timestep_mapping(self) -> xr.DataArray:
        """Mapping from original timesteps to representative timestep indices."""
        return self.result.timestep_mapping

    @property
    def cluster_start_positions(self) -> np.ndarray:
        """Integer positions where clusters start.

        Returns the indices of the first timestep of each cluster.
        Use these positions to build masks for specific use cases.

        Returns:
            1D numpy array of positions: [0, T, 2T, ...] where T = timesteps_per_period.

        Example:
            For 2 clusters with 24 timesteps each:
            >>> clustering.cluster_start_positions
            array([0, 24])
        """
        if self.result.cluster_structure is None:
            raise ValueError('No cluster_structure available')

        n_timesteps = self.n_clusters * self.timesteps_per_period
        return np.arange(0, n_timesteps, self.timesteps_per_period)

    @property
    def original_timesteps(self) -> pd.DatetimeIndex:
        """Original timesteps before clustering.

        Derived from the 'original_time' coordinate of timestep_mapping.

        Raises:
            KeyError: If 'original_time' coordinate is missing from timestep_mapping.
        """
        if 'original_time' not in self.result.timestep_mapping.coords:
            raise KeyError(
                "timestep_mapping is missing 'original_time' coordinate. "
                'This may indicate corrupted or incompatible clustering results.'
            )
        return pd.DatetimeIndex(self.result.timestep_mapping.coords['original_time'].values)


def create_cluster_structure_from_mapping(
    timestep_mapping: xr.DataArray,
    timesteps_per_cluster: int,
) -> ClusterStructure:
    """Create ClusterStructure from a timestep mapping.

    This is a convenience function for creating ClusterStructure when you
    have the timestep mapping but not the full clustering metadata.

    Args:
        timestep_mapping: Mapping from original timesteps to representative indices.
        timesteps_per_cluster: Number of timesteps per cluster period.

    Returns:
        ClusterStructure derived from the mapping.
    """
    n_original = len(timestep_mapping)
    n_original_clusters = n_original // timesteps_per_cluster

    # Determine cluster order from the mapping
    # Each original period maps to the cluster of its first timestep
    cluster_order = []
    for p in range(n_original_clusters):
        start_idx = p * timesteps_per_cluster
        cluster_idx = int(timestep_mapping.isel(original_time=start_idx).values) // timesteps_per_cluster
        cluster_order.append(cluster_idx)

    cluster_order_da = xr.DataArray(cluster_order, dims=['original_cluster'], name='cluster_order')

    # Count occurrences of each cluster
    unique_clusters = np.unique(cluster_order)
    occurrences = {}
    for c in unique_clusters:
        occurrences[int(c)] = sum(1 for x in cluster_order if x == c)

    n_clusters = len(unique_clusters)
    cluster_occurrences_da = xr.DataArray(
        [occurrences.get(c, 0) for c in range(n_clusters)],
        dims=['cluster'],
        name='cluster_occurrences',
    )

    return ClusterStructure(
        cluster_order=cluster_order_da,
        cluster_occurrences=cluster_occurrences_da,
        n_clusters=n_clusters,
        timesteps_per_cluster=timesteps_per_cluster,
    )


def _register_clustering_classes():
    """Register clustering classes for IO.

    Called from flow_system.py after all imports are complete to avoid circular imports.
    """
    from ..structure import CLASS_REGISTRY

    CLASS_REGISTRY['ClusterStructure'] = ClusterStructure
    CLASS_REGISTRY['ClusterResult'] = ClusterResult
    CLASS_REGISTRY['Clustering'] = Clustering
