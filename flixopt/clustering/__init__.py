"""
Time Series Aggregation Module for flixopt.

This module provides a minimal `Clustering` dataclass for time series aggregation.

Key class:
- Clustering: Stores essential clustering info for expansion, inter-cluster storage, and IO.

Example usage:

    # Cluster a FlowSystem to reduce timesteps (uses tsam v3 API)
    fs_clustered = fs.transform.cluster(n_clusters=8)

    # With peak preservation
    from tsam import ClusterConfig, ExtremeConfig
    fs_clustered = fs.transform.cluster(
        n_clusters=8,
        cluster=ClusterConfig(method='hierarchical'),
        extremes=ExtremeConfig(max_value=['Demand|fixed_relative_profile']),
    )

    # Access clustering metadata
    info = fs_clustered.clustering
    print(f'Number of clusters: {info.n_clusters}')
    print(f'Original periods: {info.n_original_clusters}')

    # Transfer clustering to another system
    fs2_clustered = fs2.transform.cluster(
        n_clusters=8,
        predefined=fs_clustered.clustering.predefined,
    )

    # Expand back to full resolution
    fs_expanded = fs_clustered.transform.expand()
"""

from . import tsam_adapter
from .interface import Clustering

__all__ = [
    'Clustering',
    'tsam_adapter',
]
