"""
Time Series Aggregation Module for flixopt.

This module provides the Clustering class stored on FlowSystem after clustering,
wrapping tsam_xarray's ClusteringResult.

Example usage:

    from tsam import ExtremeConfig

    fs_clustered = flow_system.transform.cluster(
        n_clusters=8,
        cluster_duration='1D',
        extremes=ExtremeConfig(method='new_cluster', max_value=['Demand|fixed_relative_profile']),
    )

    clustering = fs_clustered.clustering
    print(f'Number of clusters: {clustering.n_clusters}')
    print(f'Clustering result: {clustering.clustering_result}')

    # Access tsam_xarray AggregationResult (only before saving/loading)
    result = clustering.aggregation_result
    result.cluster_representatives  # DataArray
    result.accuracy  # AccuracyMetrics

    # Expand back to full resolution
    fs_expanded = fs_clustered.transform.expand()
"""

from .base import Clustering

__all__ = [
    'Clustering',
]
