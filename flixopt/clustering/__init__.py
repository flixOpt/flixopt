"""
Time Series Aggregation Module for flixopt.

This module provides wrapper classes around tsam's clustering functionality:
- Clustering: Top-level class stored on FlowSystem after clustering
- ClusteringResults: Manages collection of tsam ClusteringResult objects (for IO)

Example usage:

    # Cluster a FlowSystem to reduce timesteps
    from tsam.config import ExtremeConfig

    fs_clustered = flow_system.transform.cluster(
        n_clusters=8,
        cluster_duration='1D',
        extremes=ExtremeConfig(method='new_cluster', max_value=['Demand|fixed_relative_profile']),
    )

    # Access clustering structure
    clustering = fs_clustered.clustering
    print(f'Number of clusters: {clustering.n_clusters}')
    print(f'Dims: {clustering.dims}')  # e.g., ('period', 'scenario')
    print(f'Coords: {clustering.coords}')  # e.g., {'period': [2024, 2025]}

    # Access tsam AggregationResult for detailed analysis
    result = clustering.sel(period=2024, scenario='high')
    result.cluster_representatives  # DataFrame with aggregated time series
    result.accuracy  # AccuracyMetrics (rmse, mae)
    result.plot.compare()  # tsam's built-in comparison plot

    # Iterate over all results
    for key, result in clustering.items():
        print(f'{key}: {result.n_clusters} clusters')

    # Save clustering for reuse
    fs_clustered.clustering.to_json('clustering.json')

    # Expand back to full resolution
    fs_expanded = fs_clustered.transform.expand()
"""

from .base import AggregationResults, Clustering, ClusteringResults

__all__ = [
    'ClusteringResults',
    'AggregationResults',
    'Clustering',
]
