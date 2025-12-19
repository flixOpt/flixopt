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

import numpy as np
import xarray as xr


@dataclass
class ClusterStructure:
    """Structure information for inter-cluster storage linking.

    This class captures the hierarchical structure of time series clustering,
    which is needed for proper storage state-of-charge tracking across
    typical periods when using cluster().

    Note: "original_period" here refers to the original time chunks before
    clustering (e.g., 365 original days), NOT the model's "period" dimension
    (years/months). Each original time chunk gets assigned to a cluster.

    Attributes:
        cluster_order: Maps each original time chunk index to its cluster ID.
            dims: [original_period] for simple case, or
            [original_period, period, scenario] for multi-period/scenario systems.
            Values are cluster indices (0 to n_clusters-1).
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
        - cluster_order: shape (365, 2) with dims [original_period, scenario]
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
            self.cluster_order = xr.DataArray(self.cluster_order, dims=['original_period'], name='cluster_order')
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
            f'  {self.n_original_periods} original periods → {n_clusters} clusters\n'
            f'  timesteps_per_cluster={self.timesteps_per_cluster}\n'
            f'  occurrences={occ}\n'
            f')'
        )

    @property
    def n_original_periods(self) -> int:
        """Number of original periods (before clustering)."""
        return len(self.cluster_order.coords['original_period'])

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

    def plot(self, show: bool | None = None):
        """Plot cluster assignment visualization.

        Shows which cluster each original period belongs to, and the
        number of occurrences per cluster.

        Args:
            show: Whether to display the figure. Defaults to CONFIG.Plotting.default_show.

        Returns:
            PlotResult containing the figure and underlying data.
        """
        import plotly.express as px

        from ..config import CONFIG
        from ..plot_result import PlotResult

        n_clusters = (
            int(self.n_clusters) if isinstance(self.n_clusters, (int, np.integer)) else int(self.n_clusters.values)
        )

        # Create DataFrame for plotting
        import pandas as pd

        cluster_order = self.get_cluster_order_for_slice()
        df = pd.DataFrame(
            {
                'Original Period': range(1, len(cluster_order) + 1),
                'Cluster': cluster_order,
            }
        )

        # Bar chart showing cluster assignment
        fig = px.bar(
            df,
            x='Original Period',
            y=[1] * len(df),
            color='Cluster',
            color_continuous_scale='Viridis',
            title=f'Cluster Assignment ({self.n_original_periods} periods → {n_clusters} clusters)',
        )
        fig.update_layout(yaxis_visible=False, coloraxis_colorbar_title='Cluster')

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
        if not isinstance(self.representative_weights, xr.DataArray):
            self.representative_weights = xr.DataArray(
                self.representative_weights, dims=['time'], name='representative_weights'
            )
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

        # Check weights length matches n_representatives
        if len(self.representative_weights) != n_rep:
            raise ValueError(
                f'representative_weights has {len(self.representative_weights)} elements '
                f'but n_representatives is {n_rep}'
            )

        # Check weights sum roughly equals original timesteps
        weight_sum = float(self.representative_weights.sum().values)
        n_original = self.n_original_timesteps
        if abs(weight_sum - n_original) > 1e-6:
            # Warning only - some aggregation methods may not preserve this exactly
            import warnings

            warnings.warn(
                f'representative_weights sum ({weight_sum}) does not match n_original_timesteps ({n_original})',
                stacklevel=2,
            )

    def plot(self, colormap: str | None = None, show: bool | None = None):
        """Plot original vs aggregated data comparison.

        Visualizes the original time series (dashed lines) overlaid with
        the aggregated/clustered time series (solid lines) for comparison.
        Constants (time-invariant variables) are excluded from the plot.

        Args:
            colormap: Colorscale name for the time series colors.
                Defaults to CONFIG.Plotting.default_qualitative_colorscale.
            show: Whether to display the figure.
                Defaults to CONFIG.Plotting.default_show.

        Returns:
            PlotResult containing the comparison figure and underlying data.
        """
        import plotly.express as px

        from ..color_processing import process_colors
        from ..config import CONFIG
        from ..plot_result import PlotResult

        if self.original_data is None or self.aggregated_data is None:
            raise ValueError('ClusterResult must contain both original_data and aggregated_data for plotting')

        # Filter to only time-varying variables (exclude constants)
        time_vars = [
            name
            for name in self.original_data.data_vars
            if 'time' in self.original_data[name].dims
            and not np.isclose(self.original_data[name].min(), self.original_data[name].max())
        ]
        if not time_vars:
            raise ValueError('No time-varying variables found in original_data')

        original_filtered = self.original_data[time_vars]
        aggregated_filtered = self.aggregated_data[time_vars]

        # Convert xarray to DataFrames
        original_df = original_filtered.to_dataframe()
        aggregated_df = aggregated_filtered.to_dataframe()

        # Expand aggregated data to original length using mapping
        mapping = self.timestep_mapping.values
        expanded_agg = aggregated_df.iloc[mapping].reset_index(drop=True)

        # Rename for legend
        original_df = original_df.rename(columns={col: f'Original - {col}' for col in original_df.columns})
        expanded_agg = expanded_agg.rename(columns={col: f'Aggregated - {col}' for col in expanded_agg.columns})

        colors = list(
            process_colors(
                colormap or CONFIG.Plotting.default_qualitative_colorscale, list(original_df.columns)
            ).values()
        )

        # Create line plot for original data (dashed)
        original_df = original_df.reset_index()
        index_name = original_df.columns[0]
        df_org_long = original_df.melt(id_vars=index_name, var_name='variable', value_name='value')
        fig = px.line(df_org_long, x=index_name, y='value', color='variable', color_discrete_sequence=colors)
        for trace in fig.data:
            trace.update(line=dict(dash='dash'))

        # Add aggregated data (solid lines)
        expanded_agg[index_name] = original_df[index_name]
        df_agg_long = expanded_agg.melt(id_vars=index_name, var_name='variable', value_name='value')
        fig2 = px.line(df_agg_long, x=index_name, y='value', color='variable', color_discrete_sequence=colors)
        for trace in fig2.data:
            fig.add_trace(trace)

        fig.update_layout(
            title='Original vs Aggregated Data (original = ---)',
            xaxis_title='Time',
            yaxis_title='Value',
        )

        # Build xarray Dataset with both original and aggregated data
        data = xr.Dataset(
            {
                'original': original_filtered.to_array(dim='variable'),
                'aggregated': aggregated_filtered.to_array(dim='variable'),
            }
        )
        plot_result = PlotResult(data=data, figure=fig)

        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            plot_result.show()

        return plot_result

    def plot_typical_periods(self, variable: str | None = None, show: bool | None = None):
        """Plot each cluster's typical period profile.

        Shows each cluster as a separate subplot with its occurrence count
        in the title. Useful for understanding what each cluster represents.

        Args:
            variable: Variable to plot. If None, plots the first available variable.
            show: Whether to display the figure. Defaults to CONFIG.Plotting.default_show.

        Returns:
            PlotResult containing the figure and underlying data.
        """
        from plotly.subplots import make_subplots

        from ..config import CONFIG
        from ..plot_result import PlotResult

        if self.aggregated_data is None or self.cluster_structure is None:
            raise ValueError('ClusterResult must contain aggregated_data and cluster_structure for this plot')

        cs = self.cluster_structure
        n_clusters = int(cs.n_clusters) if isinstance(cs.n_clusters, (int, np.integer)) else int(cs.n_clusters.values)

        # Select variable
        variables = list(self.aggregated_data.data_vars)
        if variable is None:
            variable = variables[0]
        elif variable not in variables:
            raise ValueError(f'Variable {variable} not found. Available: {variables}')

        data = self.aggregated_data[variable].values

        # Reshape to [n_clusters, timesteps_per_cluster]
        data_by_cluster = data.reshape(n_clusters, cs.timesteps_per_cluster)

        # Create subplots
        n_cols = min(4, n_clusters)
        n_rows = (n_clusters + n_cols - 1) // n_cols
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=[
                f'Cluster {c} (×{int(cs.cluster_occurrences.sel(cluster=c).values)})' for c in range(n_clusters)
            ],
        )

        x = np.arange(cs.timesteps_per_cluster)
        for c in range(n_clusters):
            row = c // n_cols + 1
            col = c % n_cols + 1
            fig.add_trace(
                {'type': 'scatter', 'x': x, 'y': data_by_cluster[c], 'mode': 'lines', 'showlegend': False},
                row=row,
                col=col,
            )

        fig.update_layout(
            title=f'Typical Periods: {variable}',
            height=200 * n_rows,
        )

        # Build data for PlotResult
        result_data = xr.Dataset(
            {
                'typical_periods': xr.DataArray(
                    data_by_cluster,
                    dims=['cluster', 'timestep'],
                    coords={'cluster': range(n_clusters), 'timestep': range(cs.timesteps_per_cluster)},
                ),
                'occurrences': cs.cluster_occurrences,
            }
        )
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
        original_flow_system: Reference to the FlowSystem before aggregation.
        backend_name: Name of the aggregation backend used (e.g., 'tsam', 'manual').
        storage_inter_cluster_linking: Whether to add inter-cluster storage constraints.
        storage_cyclic: Whether to enforce cyclic storage (SOC[start] = SOC[end]).
    """

    result: ClusterResult
    original_flow_system: object  # FlowSystem - avoid circular import
    backend_name: str = 'unknown'
    storage_inter_cluster_linking: bool = True
    storage_cyclic: bool = True

    def __repr__(self) -> str:
        cs = self.result.cluster_structure
        if cs is not None:
            n_clusters = (
                int(cs.n_clusters) if isinstance(cs.n_clusters, (int, np.integer)) else int(cs.n_clusters.values)
            )
            structure_info = f'{cs.n_original_periods} periods → {n_clusters} clusters'
        else:
            structure_info = 'no structure'
        return (
            f'Clustering(\n'
            f'  backend={self.backend_name!r}\n'
            f'  {structure_info}\n'
            f'  storage_linking={self.storage_inter_cluster_linking}, cyclic={self.storage_cyclic}\n'
            f')'
        )

    def plot(self, colormap: str | None = None, show: bool | None = None):
        """Plot original vs aggregated data comparison.

        Convenience method that calls result.plot().

        Args:
            colormap: Colorscale name for the time series colors.
                Defaults to CONFIG.Plotting.default_qualitative_colorscale.
            show: Whether to display the figure.
                Defaults to CONFIG.Plotting.default_show.

        Returns:
            PlotResult containing the comparison figure and underlying data.

        Example:
            >>> fs_clustered = flow_system.transform.cluster(n_clusters=8, cluster_duration='1D')
            >>> fs_clustered.clustering.plot()
        """
        return self.result.plot(colormap=colormap, show=show)

    def plot_typical_periods(self, variable: str | None = None, show: bool | None = None):
        """Plot each cluster's typical period profile.

        Convenience method that calls result.plot_typical_periods().

        Args:
            variable: Variable to plot. If None, plots the first available variable.
            show: Whether to display the figure.
                Defaults to CONFIG.Plotting.default_show.

        Returns:
            PlotResult containing the figure and underlying data.

        Example:
            >>> fs_clustered = flow_system.transform.cluster(n_clusters=8, cluster_duration='1D')
            >>> fs_clustered.clustering.plot_typical_periods()
        """
        return self.result.plot_typical_periods(variable=variable, show=show)

    def plot_structure(self, show: bool | None = None):
        """Plot cluster assignment visualization.

        Shows which original period belongs to which cluster.

        Args:
            show: Whether to display the figure.
                Defaults to CONFIG.Plotting.default_show.

        Returns:
            PlotResult containing the figure and underlying data.

        Example:
            >>> fs_clustered = flow_system.transform.cluster(n_clusters=8, cluster_duration='1D')
            >>> fs_clustered.clustering.plot_structure()
        """
        if self.result.cluster_structure is None:
            raise ValueError('No cluster_structure available')
        return self.result.cluster_structure.plot(show=show)

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
    def n_original_periods(self) -> int:
        """Number of original periods (before clustering)."""
        if self.result.cluster_structure is None:
            raise ValueError('No cluster_structure available')
        return self.result.cluster_structure.n_original_periods

    @property
    def timesteps_per_period(self) -> int:
        """Number of timesteps in each period/cluster."""
        if self.result.cluster_structure is None:
            raise ValueError('No cluster_structure available')
        return self.result.cluster_structure.timesteps_per_cluster

    @property
    def timestep_mapping(self) -> xr.DataArray:
        """Mapping from original timesteps to representative timestep indices."""
        return self.result.timestep_mapping


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
    n_original_periods = n_original // timesteps_per_cluster

    # Determine cluster order from the mapping
    # Each original period maps to the cluster of its first timestep
    cluster_order = []
    for p in range(n_original_periods):
        start_idx = p * timesteps_per_cluster
        cluster_idx = int(timestep_mapping.isel(original_time=start_idx).values) // timesteps_per_cluster
        cluster_order.append(cluster_idx)

    cluster_order_da = xr.DataArray(cluster_order, dims=['original_period'], name='cluster_order')

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


def plot_aggregation(
    result: ClusterResult,
    colormap: str | None = None,
    show: bool | None = None,
):
    """Plot original vs aggregated data comparison.

    .. deprecated::
        Use ``result.plot()`` directly instead.

    Args:
        result: ClusterResult containing original and aggregated data.
        colormap: Colorscale name for the time series colors.
        show: Whether to display the figure.

    Returns:
        PlotResult containing the comparison figure and underlying data.
    """
    return result.plot(colormap=colormap, show=show)
