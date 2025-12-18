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
    typical periods when using cluster_reduce().

    Note: "original_period" here refers to the original time chunks before
    clustering (e.g., 365 original days), NOT the model's "period" dimension
    (years/months). Each original time chunk gets assigned to a cluster.

    Attributes:
        cluster_order: Maps each original time chunk index to its cluster ID.
            dims: [original_period] where original_period indexes the time chunks
            (e.g., days) before clustering. Values are cluster indices (0 to n_clusters-1).
        cluster_occurrences: Count of how many original time chunks each cluster represents.
            dims: [cluster]
        n_clusters: Number of distinct clusters (typical periods).
        timesteps_per_cluster: Number of timesteps in each cluster (e.g., 24 for daily).

    Example:
        For 365 days clustered into 8 typical days:
        - cluster_order: shape (365,), values 0-7 indicating which cluster each day belongs to
        - cluster_occurrences: shape (8,), e.g., [45, 46, 46, 46, 46, 45, 45, 46]
        - n_clusters: 8
        - timesteps_per_cluster: 24 (for hourly data)
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

    @property
    def n_original_periods(self) -> int:
        """Number of original periods (before clustering)."""
        return len(self.cluster_order.coords['original_period'])

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


@dataclass
class AggregationResult:
    """Universal result from any time series aggregation method.

    This dataclass captures all information needed to:
    1. Transform a FlowSystem to use aggregated (clustered) timesteps
    2. Expand a solution back to original resolution
    3. Properly weight results for statistics

    Attributes:
        timestep_mapping: Maps each original timestep to its representative index.
            dims: [original_time]
            Values are indices into the representative timesteps (0 to n_representatives-1).
        n_representatives: Number of representative timesteps after aggregation.
        representative_weights: Weight for each representative timestep.
            dims: [time]
            Typically equals the number of original timesteps each representative covers.
            Used as cluster_weight in the FlowSystem.
        aggregated_data: Time series data aggregated to representative timesteps.
            Optional - some backends may not aggregate data.
        cluster_structure: Hierarchical clustering structure for storage linking.
            Optional - only needed when using cluster_reduce() mode.
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


@dataclass
class AggregationInfo:
    """Information about an aggregation stored on a FlowSystem.

    This is stored on the FlowSystem after aggregation to enable:
    - expand_solution() to map back to original timesteps
    - Statistics to properly weight results
    - Inter-cluster storage linking
    - Serialization/deserialization of aggregated models

    Attributes:
        result: The AggregationResult from the aggregation backend.
        original_flow_system: Reference to the FlowSystem before aggregation.
        backend_name: Name of the aggregation backend used (e.g., 'tsam', 'manual').
        storage_inter_cluster_linking: Whether to add inter-cluster storage constraints.
        storage_cyclic: Whether to enforce cyclic storage (SOC[start] = SOC[end]).
    """

    result: AggregationResult
    original_flow_system: object  # FlowSystem - avoid circular import
    backend_name: str = 'unknown'
    storage_inter_cluster_linking: bool = True
    storage_cyclic: bool = True


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
    result: AggregationResult,
    colormap: str | None = None,
    show: bool | None = None,
):
    """Plot original vs aggregated data comparison.

    Visualizes the original time series (dashed lines) overlaid with
    the aggregated/clustered time series (solid lines) for comparison.

    Args:
        result: AggregationResult containing original and aggregated data.
        colormap: Colorscale name for the time series colors.
            Defaults to CONFIG.Plotting.default_qualitative_colorscale.
        show: Whether to display the figure.
            Defaults to CONFIG.Plotting.default_show.

    Returns:
        PlotResult containing the comparison figure and underlying data.

    Example:
        >>> fs_clustered = flow_system.transform.cluster(n_clusters=8, cluster_duration='1D')
        >>> plot_aggregation(fs_clustered._aggregation_info.result)
    """
    import plotly.express as px

    from ..color_processing import process_colors
    from ..config import CONFIG
    from ..plot_result import PlotResult

    if result.original_data is None or result.aggregated_data is None:
        raise ValueError('AggregationResult must contain both original_data and aggregated_data for plotting')

    # Convert xarray to DataFrames
    original_df = result.original_data.to_dataframe()
    aggregated_df = result.aggregated_data.to_dataframe()

    # Expand aggregated data to original length using mapping
    mapping = result.timestep_mapping.values
    expanded_agg = aggregated_df.iloc[mapping].reset_index(drop=True)

    # Rename for legend
    original_df = original_df.rename(columns={col: f'Original - {col}' for col in original_df.columns})
    expanded_agg = expanded_agg.rename(columns={col: f'Aggregated - {col}' for col in expanded_agg.columns})

    colors = list(
        process_colors(colormap or CONFIG.Plotting.default_qualitative_colorscale, list(original_df.columns)).values()
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
            'original': result.original_data.to_array(dim='variable'),
            'aggregated': result.aggregated_data.to_array(dim='variable'),
        }
    )
    plot_result = PlotResult(data=data, figure=fig)

    if show is None:
        show = CONFIG.Plotting.default_show
    if show:
        plot_result.show()

    return plot_result
