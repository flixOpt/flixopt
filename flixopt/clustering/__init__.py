"""
Time Series Aggregation Module for flixopt.

This module provides data structures for time series clustering/aggregation.

Key classes:
- ClusterResult: Universal result container for clustering
- ClusterStructure: Hierarchical structure info for storage inter-cluster linking
- Clustering: Stored on FlowSystem after clustering
- ClusteringResultCollection: Wrapper for multi-dimensional tsam ClusteringResult objects

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

    # Save and reuse clustering
    fs_clustered.clustering.tsam_results.to_json('clustering.json')

    # Expand back to full resolution
    fs_expanded = fs_clustered.transform.expand()
"""

from .base import (
    Clustering,
    ClusteringResultCollection,
    ClusterResult,
    ClusterStructure,
    create_cluster_structure_from_mapping,
)

__all__ = [
    # Core classes
    'ClusterResult',
    'Clustering',
    'ClusteringResultCollection',
    'ClusterStructure',
    # Utilities
    'create_cluster_structure_from_mapping',
]
