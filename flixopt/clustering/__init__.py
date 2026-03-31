"""
Time Series Aggregation Module for flixopt.

This module provides the Clustering class stored on FlowSystem after clustering,
wrapping tsam_xarray's ClusteringInfo.

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

    # Access tsam_xarray AggregationResult for detailed analysis
    # NOTE: Only available BEFORE saving/loading. Lost after IO.
    result = clustering.aggregation_result
    result.cluster_representatives  # DataArray with aggregated time series
    result.accuracy  # AccuracyMetrics (rmse, mae)

    # Save and load - structure preserved, AggregationResult access lost
    fs_clustered.to_netcdf('system.nc')
    # Use include_original_data=False for smaller files (~38% reduction)
    fs_clustered.to_netcdf('system.nc', include_original_data=False)

    # Expand back to full resolution
    fs_expanded = fs_clustered.transform.expand()
"""

from .base import Clustering

__all__ = [
    'Clustering',
]
