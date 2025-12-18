"""
Base classes and data structures for time series aggregation.

This module provides an abstraction layer for time series aggregation that
supports multiple backends (TSAM, manual/external, etc.) while maintaining
proper handling of multi-dimensional data (period, scenario dimensions).

All data structures use xarray for consistent multi-dimensional support.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import xarray as xr


@dataclass
class ClusterStructure:
    """Structure information for inter-period storage linking.

    This class captures the hierarchical structure of time series clustering,
    which is needed for proper storage state-of-charge tracking across
    typical periods when using cluster_reduce().

    All arrays use xarray DataArrays to properly handle multi-dimensional
    cases (period, scenario dimensions).

    Attributes:
        cluster_order: Maps original periods to cluster IDs.
            dims: [original_period] or [original_period, period, scenario]
            Each value indicates which typical period (cluster) the original
            period belongs to.
        cluster_occurrences: Count of how many original periods each cluster represents.
            dims: [cluster] or [cluster, period, scenario]
        n_clusters: Number of distinct clusters (typical periods).
            Can be int (same for all) or DataArray (varies by period/scenario).
        timesteps_per_cluster: Number of timesteps in each cluster period.

    Example:
        For 365 days clustered into 8 typical days:
        - cluster_order: shape (365,), values 0-7
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
    1. Transform a FlowSystem to use aggregated timesteps
    2. Expand a solution back to original resolution
    3. Properly weight results for statistics

    All arrays use xarray DataArrays to properly handle multi-dimensional
    cases (period, scenario dimensions).

    Attributes:
        timestep_mapping: Maps each original timestep to its representative index.
            dims: [original_time] or [original_time, period, scenario]
            Values are indices into the representative timesteps (0 to n_representatives-1).
        n_representatives: Number of representative timesteps after aggregation.
            Can be int (same for all) or DataArray (varies by period/scenario).
        representative_weights: Weight for each representative timestep.
            dims: [time] or [time, period, scenario]
            Typically equals the number of original timesteps each representative covers.
        aggregated_data: Time series data aggregated to representative timesteps.
            Optional - some backends may not aggregate data.
        cluster_structure: Hierarchical clustering structure for storage linking.
            Optional - only needed when using cluster_reduce() mode.
        original_data: Reference to original data before aggregation.
            Optional - useful for expand_solution().

    Example:
        For 8760 hourly timesteps -> 192 representative timesteps (8 days x 24h):
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


@runtime_checkable
class Aggregator(Protocol):
    """Protocol that any aggregation backend must implement.

    This protocol defines the interface for time series aggregation backends.
    Implementations can use any aggregation algorithm (TSAM, sklearn k-means,
    hierarchical clustering, etc.) as long as they return an AggregationResult.

    The input data is an xarray Dataset to properly handle multi-dimensional
    time series with period and scenario dimensions.

    Example implementation:
        class MyAggregator:
            def aggregate(
                self,
                data: xr.Dataset,
                n_representatives: int,
                **kwargs
            ) -> AggregationResult:
                # Custom aggregation logic
                ...
                return AggregationResult(
                    timestep_mapping=mapping,
                    n_representatives=n_representatives,
                    representative_weights=weights,
                )
    """

    def aggregate(
        self,
        data: xr.Dataset,
        n_representatives: int,
        **kwargs,
    ) -> AggregationResult:
        """Perform time series aggregation.

        Args:
            data: Input time series data as xarray Dataset.
                Must have 'time' dimension. May also have 'period' and/or
                'scenario' dimensions for multi-dimensional optimization.
            n_representatives: Target number of representative timesteps.
            **kwargs: Backend-specific options.

        Returns:
            AggregationResult containing mapping, weights, and optionally
            aggregated data and cluster structure.
        """
        ...


@dataclass
class AggregationInfo:
    """Information about an aggregation stored on a FlowSystem.

    This is stored on the FlowSystem after aggregation to enable:
    - expand_solution() to map back to original timesteps
    - Statistics to properly weight results
    - Serialization/deserialization of aggregated models

    Attributes:
        result: The AggregationResult from the aggregation backend.
        original_flow_system: Reference to the FlowSystem before aggregation.
        mode: Whether aggregation used 'reduce' (fewer timesteps) or
              'constrain' (same timesteps with equality constraints).
        backend_name: Name of the aggregation backend used (e.g., 'tsam', 'manual').
    """

    result: AggregationResult
    original_flow_system: object  # FlowSystem - avoid circular import
    mode: str  # 'reduce' or 'constrain'
    backend_name: str = 'unknown'


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
