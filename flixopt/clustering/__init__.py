"""
Time Series Aggregation Module for flixopt.

This module provides data structures for time series clustering/aggregation.

Key classes:
- ClusterResult: Universal result container for clustering
- ClusterStructure: Hierarchical structure info for storage inter-cluster linking
- Clustering: Stored on FlowSystem after clustering

Example usage:

    # Cluster a FlowSystem to reduce timesteps
    fs_clustered = flow_system.transform.cluster(
        n_clusters=8,
        cluster_duration='1D',
        time_series_for_high_peaks=['Demand|fixed_relative_profile'],
    )

    # Access clustering metadata
    info = fs_clustered.clustering
    print(f'Number of clusters: {info.result.cluster_structure.n_clusters}')

    # Expand back to full resolution
    fs_expanded = fs_clustered.transform.expand()
"""

from .base import (
    Clustering,
    ClusterResult,
    ClusterStructure,
    create_cluster_structure_from_mapping,
)

__all__ = [
    # Core classes
    'ClusterResult',
    'Clustering',
    'ClusterStructure',
    # Utilities
    'create_cluster_structure_from_mapping',
]
