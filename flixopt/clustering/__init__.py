"""
Time Series Aggregation Module for flixopt.

This module provides wrapper classes around tsam's clustering functionality:
- ClusteringResults: Manages collection of tsam ClusteringResult objects for multi-dim data
- Clustering: Top-level class stored on FlowSystem after clustering

Example usage:

    # Cluster a FlowSystem to reduce timesteps
    from tsam.config import ExtremeConfig

    fs_clustered = flow_system.transform.cluster(
        n_clusters=8,
        cluster_duration='1D',
        extremes=ExtremeConfig(method='new_cluster', max_value=['Demand|fixed_relative_profile']),
    )

    # Access clustering metadata
    info = fs_clustered.clustering
    print(f'Number of clusters: {info.n_clusters}')

    # Access individual results
    result = fs_clustered.clustering.get_result(period=2024, scenario='high')

    # Save clustering for reuse
    fs_clustered.clustering.to_json('clustering.json')

    # Expand back to full resolution
    fs_expanded = fs_clustered.transform.expand()
"""

from .base import Clustering, ClusteringResultCollection, ClusteringResults

__all__ = [
    'ClusteringResults',
    'Clustering',
    'ClusteringResultCollection',  # Alias for backwards compat
]
