"""
Manual aggregation backend for user-provided clustering results.

This backend enables PyPSA-style workflows where users perform aggregation
externally (using sklearn, custom algorithms, etc.) and then provide the
mapping and weights to flixopt.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from .base import AggregationResult, ClusterStructure, create_cluster_structure_from_mapping


class ManualBackend:
    """Backend for user-provided aggregation results.

    This backend accepts pre-computed aggregation mapping and weights,
    enabling users to use any external clustering tool (sklearn k-means,
    hierarchical clustering, etc.) with flixopt.

    This is similar to PyPSA's approach where aggregation is done externally
    and the framework just accepts the results.

    Args:
        timestep_mapping: Mapping from original timesteps to representative indices.
            DataArray with dims [original_time] or [original_time, period, scenario].
            Values should be integers in range [0, n_representatives).
        representative_weights: Weight for each representative timestep.
            DataArray with dims [time] or [time, period, scenario].
            Typically equals count of original timesteps each representative covers.
        cluster_structure: Optional cluster structure for storage inter-period linking.
            If not provided and timesteps_per_cluster is given, will be inferred from mapping.
        timesteps_per_cluster: Number of timesteps per cluster period.
            Required to infer cluster_structure if not explicitly provided.

    Example:
        >>> # External clustering with sklearn
        >>> from sklearn.cluster import KMeans
        >>> kmeans = KMeans(n_clusters=8)
        >>> labels = kmeans.fit_predict(my_data)
        >>>
        >>> # Create mapping (original timestep -> representative)
        >>> mapping = ...  # compute from labels
        >>> weights = ...  # count occurrences
        >>>
        >>> # Use with flixopt
        >>> backend = ManualBackend(
        ...     timestep_mapping=xr.DataArray(mapping, dims=['original_time']),
        ...     representative_weights=xr.DataArray(weights, dims=['time']),
        ... )
        >>> result = backend.aggregate(data, n_representatives=192)
    """

    def __init__(
        self,
        timestep_mapping: xr.DataArray,
        representative_weights: xr.DataArray,
        cluster_structure: ClusterStructure | None = None,
        timesteps_per_cluster: int | None = None,
    ):
        # Validate and store mapping
        if not isinstance(timestep_mapping, xr.DataArray):
            timestep_mapping = xr.DataArray(timestep_mapping, dims=['original_time'], name='timestep_mapping')
        self.timestep_mapping = timestep_mapping

        # Validate and store weights
        if not isinstance(representative_weights, xr.DataArray):
            representative_weights = xr.DataArray(representative_weights, dims=['time'], name='representative_weights')
        self.representative_weights = representative_weights

        # Store or infer cluster structure
        self.cluster_structure = cluster_structure
        self.timesteps_per_cluster = timesteps_per_cluster

        # Validate
        self._validate()

    def _validate(self) -> None:
        """Validate input arrays."""
        # Check mapping has required dimension
        if 'original_time' not in self.timestep_mapping.dims:
            if 'time' in self.timestep_mapping.dims:
                # Rename for clarity
                self.timestep_mapping = self.timestep_mapping.rename({'time': 'original_time'})
            else:
                raise ValueError("timestep_mapping must have 'original_time' or 'time' dimension")

        # Check weights has required dimension
        if 'time' not in self.representative_weights.dims:
            raise ValueError("representative_weights must have 'time' dimension")

        # Check mapping values are non-negative integers
        min_val = int(self.timestep_mapping.min().values)
        if min_val < 0:
            raise ValueError(f'timestep_mapping contains negative value: {min_val}')

        # Check mapping values are within bounds
        max_val = int(self.timestep_mapping.max().values)
        n_weights = len(self.representative_weights.coords['time'])
        if max_val >= n_weights:
            raise ValueError(
                f'timestep_mapping contains index {max_val} but representative_weights only has {n_weights} elements'
            )

    def aggregate(
        self,
        data: xr.Dataset,
        n_representatives: int | None = None,
        **kwargs,
    ) -> AggregationResult:
        """Create AggregationResult from stored mapping and weights.

        The data parameter is used to:
        1. Validate dimensions match the mapping
        2. Create aggregated data by indexing with the mapping

        Args:
            data: Input time series data as xarray Dataset.
                Used for validation and to create aggregated_data.
            n_representatives: Number of representatives. If None, inferred from weights.
            **kwargs: Ignored (for protocol compatibility).

        Returns:
            AggregationResult with the stored mapping and weights.
        """
        # Infer n_representatives if not provided
        if n_representatives is None:
            n_representatives = len(self.representative_weights.coords['time'])

        # Validate data dimensions match mapping
        self._validate_data_dimensions(data)

        # Create aggregated data by indexing original data
        aggregated_data = self._create_aggregated_data(data, n_representatives)

        # Infer cluster structure if needed
        cluster_structure = self.cluster_structure
        if cluster_structure is None and self.timesteps_per_cluster is not None:
            cluster_structure = create_cluster_structure_from_mapping(self.timestep_mapping, self.timesteps_per_cluster)

        return AggregationResult(
            timestep_mapping=self.timestep_mapping,
            n_representatives=n_representatives,
            representative_weights=self.representative_weights,
            aggregated_data=aggregated_data,
            cluster_structure=cluster_structure,
            original_data=data,
        )

    def _validate_data_dimensions(self, data: xr.Dataset) -> None:
        """Validate that data dimensions are compatible with mapping."""
        # Check time dimension length
        if 'time' not in data.dims:
            raise ValueError("Input data must have 'time' dimension")

        n_data_timesteps = len(data.coords['time'])
        n_mapping_timesteps = len(self.timestep_mapping.coords['original_time'])

        if n_data_timesteps != n_mapping_timesteps:
            raise ValueError(f'Data has {n_data_timesteps} timesteps but mapping expects {n_mapping_timesteps}')

        # Check period/scenario dimensions if present in mapping
        for dim in ['period', 'scenario']:
            if dim in self.timestep_mapping.dims:
                if dim not in data.dims:
                    raise ValueError(f"Mapping has '{dim}' dimension but data does not")
                mapping_coords = self.timestep_mapping.coords[dim].values
                data_coords = data.coords[dim].values
                if not np.array_equal(mapping_coords, data_coords):
                    raise ValueError(f"'{dim}' coordinates don't match between mapping and data")

    def _create_aggregated_data(
        self,
        data: xr.Dataset,
        n_representatives: int,
    ) -> xr.Dataset:
        """Create aggregated data by extracting representative timesteps.

        For each representative timestep, we take the value from the first
        original timestep that maps to it (simple selection, not averaging).
        """
        # Find first original timestep for each representative
        mapping_vals = self.timestep_mapping.values
        if mapping_vals.ndim > 1:
            # Multi-dimensional - use first slice
            mapping_vals = mapping_vals[:, 0] if mapping_vals.ndim == 2 else mapping_vals[:, 0, 0]

        # For each representative, find the first original that maps to it
        first_original = {}
        for orig_idx, rep_idx in enumerate(mapping_vals):
            if rep_idx not in first_original:
                first_original[int(rep_idx)] = orig_idx

        # Build index array for selecting representative values
        rep_indices = [first_original.get(i, 0) for i in range(n_representatives)]

        # Select from data
        aggregated_vars = {}
        for var_name, var_data in data.data_vars.items():
            if 'time' in var_data.dims:
                # Select representative timesteps
                selected = var_data.isel(time=rep_indices)
                # Reassign time coordinate
                selected = selected.assign_coords(time=np.arange(n_representatives))
                aggregated_vars[var_name] = selected
            else:
                # Non-time variable - keep as is
                aggregated_vars[var_name] = var_data

        return xr.Dataset(aggregated_vars)


def create_manual_backend_from_labels(
    labels: np.ndarray,
    timesteps_per_cluster: int,
    n_timesteps: int | None = None,
) -> ManualBackend:
    """Create ManualBackend from cluster labels (e.g., from sklearn KMeans).

    This is a convenience function for creating a ManualBackend when you have
    cluster labels from a standard clustering algorithm.

    Args:
        labels: Cluster label for each timestep (from KMeans.fit_predict, etc.).
            Shape: (n_timesteps,) with values in [0, n_clusters).
        timesteps_per_cluster: Number of timesteps per cluster period.
        n_timesteps: Total number of timesteps. If None, inferred from labels.

    Returns:
        ManualBackend configured with the label-derived mapping.

    Example:
        >>> from sklearn.cluster import KMeans
        >>> kmeans = KMeans(n_clusters=8).fit(daily_profiles)
        >>> labels = np.repeat(kmeans.labels_, 24)  # Expand to hourly
        >>> backend = create_manual_backend_from_labels(labels, timesteps_per_cluster=24)
    """
    if n_timesteps is None:
        n_timesteps = len(labels)

    # Get unique clusters and count occurrences
    unique_clusters = np.unique(labels)
    n_clusters = len(unique_clusters)

    # Remap labels to 0..n_clusters-1 if needed
    if not np.array_equal(unique_clusters, np.arange(n_clusters)):
        label_map = {old: new for new, old in enumerate(unique_clusters)}
        labels = np.array([label_map[label] for label in labels])

    # Build timestep mapping
    # Each original timestep maps to: cluster_id * timesteps_per_cluster + position_in_period
    n_original_periods = n_timesteps // timesteps_per_cluster
    timestep_mapping = np.zeros(n_timesteps, dtype=np.int32)

    for period_idx in range(n_original_periods):
        cluster_id = labels[period_idx * timesteps_per_cluster]  # Label of first timestep in period
        for pos in range(timesteps_per_cluster):
            orig_idx = period_idx * timesteps_per_cluster + pos
            if orig_idx < n_timesteps:
                timestep_mapping[orig_idx] = cluster_id * timesteps_per_cluster + pos

    # Build weights (count of originals per representative)
    n_representative_timesteps = n_clusters * timesteps_per_cluster
    representative_weights = np.zeros(n_representative_timesteps, dtype=np.float64)

    # Count occurrences of each cluster
    cluster_counts = {}
    for period_idx in range(n_original_periods):
        cluster_id = labels[period_idx * timesteps_per_cluster]
        cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1

    for cluster_id, count in cluster_counts.items():
        for pos in range(timesteps_per_cluster):
            rep_idx = cluster_id * timesteps_per_cluster + pos
            if rep_idx < n_representative_timesteps:
                representative_weights[rep_idx] = count

    return ManualBackend(
        timestep_mapping=xr.DataArray(timestep_mapping, dims=['original_time'], name='timestep_mapping'),
        representative_weights=xr.DataArray(representative_weights, dims=['time'], name='representative_weights'),
        timesteps_per_cluster=timesteps_per_cluster,
    )


def create_manual_backend_from_selection(
    selected_indices: np.ndarray,
    weights: np.ndarray,
    n_original_timesteps: int,
    timesteps_per_period: int | None = None,
) -> ManualBackend:
    """Create ManualBackend from selected representative timesteps.

    This is useful when you have a simple selection-based aggregation
    (e.g., select every Nth timestep, select specific representative days).

    Args:
        selected_indices: Indices of selected representative timesteps.
            These become the new time axis.
        weights: Weight for each selected timestep (how many originals it represents).
        n_original_timesteps: Total number of original timesteps.
        timesteps_per_period: Optional, for creating cluster structure.

    Returns:
        ManualBackend configured with the selection-based mapping.

    Example:
        >>> # Select every 7th day as representative
        >>> selected = np.arange(0, 365 * 24, 7 * 24)  # Weekly representatives
        >>> weights = np.ones(len(selected)) * 7  # Each represents 7 days
        >>> backend = create_manual_backend_from_selection(selected, weights, n_original_timesteps=365 * 24)
    """
    n_representatives = len(selected_indices)

    if len(weights) != n_representatives:
        raise ValueError(f'weights has {len(weights)} elements but selected_indices has {n_representatives}')

    # Build mapping: each original maps to nearest selected
    timestep_mapping = np.zeros(n_original_timesteps, dtype=np.int32)

    # Simple nearest-neighbor assignment
    for orig_idx in range(n_original_timesteps):
        # Find nearest selected index
        distances = np.abs(selected_indices - orig_idx)
        nearest_rep = np.argmin(distances)
        timestep_mapping[orig_idx] = nearest_rep

    return ManualBackend(
        timestep_mapping=xr.DataArray(timestep_mapping, dims=['original_time'], name='timestep_mapping'),
        representative_weights=xr.DataArray(weights, dims=['time'], name='representative_weights'),
        timesteps_per_cluster=timesteps_per_period,
    )
