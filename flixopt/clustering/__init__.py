"""
Time Series Aggregation Module for flixopt.

This module provides wrapper classes around tsam's clustering functionality:
- Clustering: Top-level class stored on FlowSystem after clustering
- ClusteringResults: Manages collection of tsam ClusteringResult objects (for IO)

Example usage:

    # Cluster a FlowSystem to reduce timesteps
    from tsam import ExtremeConfig

    fs_clustered = flow_system.transform.cluster(
        n_clusters=8,
        cluster_duration='1D',
        extremes=ExtremeConfig(method='new_cluster', max_value=['Demand|fixed_relative_profile']),
    )

    # Access clustering structure (available before AND after IO)
    clustering = fs_clustered.clustering
    print(f'Number of clusters: {clustering.n_clusters}')
    print(f'Dims: {clustering.dims}')  # e.g., ('period', 'scenario')
    print(f'Coords: {clustering.coords}')  # e.g., {'period': [2024, 2025]}

    # Access tsam AggregationResult for detailed analysis
    # NOTE: Only available BEFORE saving/loading. Lost after IO.
    result = clustering.sel(period=2024, scenario='high')
    result.cluster_representatives  # DataFrame with aggregated time series
    result.accuracy  # AccuracyMetrics (rmse, mae)
    result.plot.compare()  # tsam's built-in comparison plot

    # Iterate over all results (only before IO)
    for key, result in clustering.items():
        print(f'{key}: {result.n_clusters} clusters')

    # Save and load - structure preserved, AggregationResult access lost
    fs_clustered.to_netcdf('system.nc')
    # Use include_original_data=False for smaller files (~38% reduction)
    fs_clustered.to_netcdf('system.nc', include_original_data=False)

    # Expand back to full resolution
    fs_expanded = fs_clustered.transform.expand()
"""

from .aggregation import (
    accuracy_to_dataframe,
    build_cluster_assignments_dataarray,
    build_cluster_config_with_weights,
    build_cluster_weights,
    build_clustering_metrics,
    build_segment_durations,
    build_typical_dataarrays,
    calculate_clustering_weights,
    combine_slices_to_dataarray,
)
from .base import AggregationResults, Clustering, ClusteringResults
from .expansion import (
    VariableExpansionHandler,
    append_final_state,
    build_segment_total_varnames,
    expand_first_timestep_only,
    interpolate_charge_state_segmented,
)
from .intercluster_helpers import (
    CapacityBounds,
    apply_soc_decay,
    build_boundary_coords,
    combine_intercluster_charge_states,
    extract_capacity_bounds,
)
from .iteration import DimInfo, DimSliceContext, iter_dim_slices, iter_dim_slices_simple

__all__ = [
    # Base classes
    'ClusteringResults',
    'AggregationResults',
    'Clustering',
    # Iteration utilities
    'DimSliceContext',
    'DimInfo',
    'iter_dim_slices',
    'iter_dim_slices_simple',
    # Aggregation helpers
    'combine_slices_to_dataarray',
    'build_typical_dataarrays',
    'build_segment_durations',
    'build_cluster_weights',
    'build_clustering_metrics',
    'build_cluster_assignments_dataarray',
    'calculate_clustering_weights',
    'build_cluster_config_with_weights',
    'accuracy_to_dataframe',
    # Expansion helpers
    'VariableExpansionHandler',
    'build_segment_total_varnames',
    'interpolate_charge_state_segmented',
    'expand_first_timestep_only',
    'append_final_state',
    # Intercluster helpers
    'CapacityBounds',
    'extract_capacity_bounds',
    'build_boundary_coords',
    'combine_intercluster_charge_states',
    'apply_soc_decay',
]
