"""
Clustering classes for time series aggregation.

This module provides wrapper classes around tsam's clustering functionality:
- `ClusteringResults`: Collection of tsam ClusteringResult objects for multi-dim (period, scenario) data
- `Clustering`: Top-level class stored on FlowSystem after clustering
"""

from __future__ import annotations

import json
from collections import Counter
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from pathlib import Path

    from tsam import AggregationResult
    from tsam import ClusteringResult as TsamClusteringResult

    from ..color_processing import ColorType
    from ..plot_result import PlotResult
    from ..statistics_accessor import SelectType

from ..statistics_accessor import _build_color_kwargs


def _apply_slot_defaults(plotly_kwargs: dict, defaults: dict[str, str | None]) -> None:
    """Apply default slot assignments to plotly kwargs.

    Args:
        plotly_kwargs: The kwargs dict to update (modified in place).
        defaults: Default slot assignments. None values block slots.
    """
    for slot, value in defaults.items():
        plotly_kwargs.setdefault(slot, value)


def _select_dims(da: xr.DataArray, period: Any = None, scenario: Any = None) -> xr.DataArray:
    """Select from DataArray by period/scenario if those dimensions exist."""
    if 'period' in da.dims and period is not None:
        da = da.sel(period=period)
    if 'scenario' in da.dims and scenario is not None:
        da = da.sel(scenario=scenario)
    return da


def combine_slices(
    slices: dict[tuple, np.ndarray],
    extra_dims: list[str],
    dim_coords: dict[str, list],
    output_dim: str,
    output_coord: Any,
    attrs: dict | None = None,
) -> xr.DataArray:
    """Combine {(dim_values): 1D_array} dict into a DataArray.

    This utility simplifies the common pattern of iterating over extra dimensions
    (like period, scenario), processing each slice, and combining results.

    Args:
        slices: Dict mapping dimension value tuples to 1D numpy arrays.
            Keys are tuples like ('period1', 'scenario1') matching extra_dims order.
        extra_dims: Dimension names in order (e.g., ['period', 'scenario']).
        dim_coords: Dict mapping dimension names to coordinate values.
        output_dim: Name of the output dimension (typically 'time').
        output_coord: Coordinate values for output dimension.
        attrs: Optional DataArray attributes.

    Returns:
        DataArray with dims [output_dim, *extra_dims].

    Example:
        >>> slices = {
        ...     ('P1', 'base'): np.array([1, 2, 3]),
        ...     ('P1', 'high'): np.array([4, 5, 6]),
        ...     ('P2', 'base'): np.array([7, 8, 9]),
        ...     ('P2', 'high'): np.array([10, 11, 12]),
        ... }
        >>> result = combine_slices(
        ...     slices,
        ...     extra_dims=['period', 'scenario'],
        ...     dim_coords={'period': ['P1', 'P2'], 'scenario': ['base', 'high']},
        ...     output_dim='time',
        ...     output_coord=[0, 1, 2],
        ... )
        >>> result.dims
        ('time', 'period', 'scenario')
    """
    n_output = len(next(iter(slices.values())))
    shape = [n_output] + [len(dim_coords[d]) for d in extra_dims]
    data = np.empty(shape)

    for combo in np.ndindex(*shape[1:]):
        key = tuple(dim_coords[d][i] for d, i in zip(extra_dims, combo, strict=True))
        data[(slice(None),) + combo] = slices[key]

    return xr.DataArray(
        data,
        dims=[output_dim] + extra_dims,
        coords={output_dim: output_coord, **dim_coords},
        attrs=attrs or {},
    )


def _cluster_occurrences(cr: TsamClusteringResult) -> np.ndarray:
    """Compute cluster occurrences from ClusteringResult."""
    counts = Counter(cr.cluster_assignments)
    return np.array([counts.get(i, 0) for i in range(cr.n_clusters)])


def _build_timestep_mapping(cr: TsamClusteringResult, n_timesteps: int) -> np.ndarray:
    """Build mapping from original timesteps to representative timestep indices.

    For segmented systems, the mapping uses segment_assignments from tsam to map
    each original timestep position to its corresponding segment index.
    """
    timesteps_per_cluster = cr.n_timesteps_per_period
    # For segmented systems, representative time dimension has n_segments entries
    # For non-segmented, it has timesteps_per_cluster entries
    n_segments = cr.n_segments
    is_segmented = n_segments is not None
    time_dim_size = n_segments if is_segmented else timesteps_per_cluster

    # For segmented systems, tsam provides segment_assignments which maps
    # each position within a period to its segment index
    segment_assignments = cr.segment_assignments if is_segmented else None

    mapping = np.zeros(n_timesteps, dtype=np.int32)
    for period_idx, cluster_id in enumerate(cr.cluster_assignments):
        for pos in range(timesteps_per_cluster):
            orig_idx = period_idx * timesteps_per_cluster + pos
            if orig_idx < n_timesteps:
                if is_segmented and segment_assignments is not None:
                    # For segmented: use tsam's segment_assignments to get segment index
                    # segment_assignments[cluster_id][pos] gives the segment index
                    segment_idx = segment_assignments[cluster_id][pos]
                    mapping[orig_idx] = int(cluster_id) * time_dim_size + segment_idx
                else:
                    # Non-segmented: direct position mapping
                    mapping[orig_idx] = int(cluster_id) * time_dim_size + pos
    return mapping


class ClusteringResults:
    """Collection of tsam ClusteringResult objects for multi-dimensional data.

    Manages multiple ClusteringResult objects keyed by (period, scenario) tuples
    and provides convenient access and multi-dimensional DataArray building.

    Follows xarray-like patterns with `.dims`, `.coords`, `.sel()`, and `.isel()`.

    Attributes:
        dims: Tuple of dimension names, e.g., ('period', 'scenario').
        coords: Dict mapping dimension names to their coordinate values.

    Example:
        >>> results = ClusteringResults({(): cr}, dim_names=[])
        >>> results.n_clusters
        2
        >>> results.cluster_assignments  # Returns DataArray
        <xarray.DataArray (original_cluster: 3)>

        >>> # Multi-dimensional case
        >>> results = ClusteringResults(
        ...     {(2024, 'high'): cr1, (2024, 'low'): cr2},
        ...     dim_names=['period', 'scenario'],
        ... )
        >>> results.dims
        ('period', 'scenario')
        >>> results.coords
        {'period': [2024], 'scenario': ['high', 'low']}
        >>> results.sel(period=2024, scenario='high')  # Label-based
        <tsam ClusteringResult>
        >>> results.isel(period=0, scenario=1)  # Index-based
        <tsam ClusteringResult>
    """

    def __init__(
        self,
        results: dict[tuple, TsamClusteringResult],
        dim_names: list[str],
    ):
        """Initialize ClusteringResults.

        Args:
            results: Dict mapping (period, scenario) tuples to tsam ClusteringResult objects.
                For simple cases without periods/scenarios, use {(): result}.
            dim_names: Names of extra dimensions, e.g., ['period', 'scenario'].
        """
        if not results:
            raise ValueError('results cannot be empty')
        self._results = results
        self._dim_names = dim_names

    # ==========================================================================
    # xarray-like interface
    # ==========================================================================

    @property
    def dims(self) -> tuple[str, ...]:
        """Dimension names as tuple (xarray-like)."""
        return tuple(self._dim_names)

    @property
    def dim_names(self) -> list[str]:
        """Dimension names as list (backwards compatibility)."""
        return list(self._dim_names)

    @property
    def coords(self) -> dict[str, list]:
        """Coordinate values for each dimension (xarray-like).

        Returns:
            Dict mapping dimension names to lists of coordinate values.
        """
        return {dim: self._get_dim_values(dim) for dim in self._dim_names}

    def sel(self, **kwargs: Any) -> TsamClusteringResult:
        """Select result by dimension labels (xarray-like).

        Args:
            **kwargs: Dimension name=value pairs, e.g., period=2024, scenario='high'.

        Returns:
            The tsam ClusteringResult for the specified combination.

        Raises:
            KeyError: If no result found for the specified combination.

        Example:
            >>> results.sel(period=2024, scenario='high')
            <tsam ClusteringResult>
        """
        key = self._make_key(**kwargs)
        if key not in self._results:
            raise KeyError(f'No result found for {kwargs}')
        return self._results[key]

    def isel(self, **kwargs: int) -> TsamClusteringResult:
        """Select result by dimension indices (xarray-like).

        Args:
            **kwargs: Dimension name=index pairs, e.g., period=0, scenario=1.

        Returns:
            The tsam ClusteringResult for the specified combination.

        Raises:
            IndexError: If index is out of range for a dimension.

        Example:
            >>> results.isel(period=0, scenario=1)
            <tsam ClusteringResult>
        """
        label_kwargs = {}
        for dim, idx in kwargs.items():
            coord_values = self._get_dim_values(dim)
            if coord_values is None:
                raise KeyError(f"Dimension '{dim}' not found in dims {self.dims}")
            if idx < 0 or idx >= len(coord_values):
                raise IndexError(f"Index {idx} out of range for dimension '{dim}' with {len(coord_values)} values")
            label_kwargs[dim] = coord_values[idx]
        return self.sel(**label_kwargs)

    def __getitem__(self, key: tuple) -> TsamClusteringResult:
        """Get result by key tuple."""
        return self._results[key]

    # === Iteration ===

    def __iter__(self):
        """Iterate over ClusteringResult objects."""
        return iter(self._results.values())

    def __len__(self) -> int:
        """Number of ClusteringResult objects."""
        return len(self._results)

    def items(self):
        """Iterate over (key, ClusteringResult) pairs."""
        return self._results.items()

    def keys(self):
        """Iterate over keys."""
        return self._results.keys()

    def values(self):
        """Iterate over ClusteringResult objects."""
        return self._results.values()

    # === Properties from first result ===

    @property
    def _first_result(self) -> TsamClusteringResult:
        """Get the first ClusteringResult (for structure info)."""
        return next(iter(self._results.values()))

    @property
    def n_clusters(self) -> int:
        """Number of clusters (same for all results)."""
        return self._first_result.n_clusters

    @property
    def timesteps_per_cluster(self) -> int:
        """Number of timesteps per cluster (same for all results)."""
        return self._first_result.n_timesteps_per_period

    @property
    def n_original_periods(self) -> int:
        """Number of original periods (same for all results)."""
        return self._first_result.n_original_periods

    @property
    def n_segments(self) -> int | None:
        """Number of segments per cluster, or None if not segmented."""
        return self._first_result.n_segments

    # === Multi-dim DataArrays ===

    @property
    def cluster_assignments(self) -> xr.DataArray:
        """Maps each original cluster to its typical cluster index.

        Returns:
            DataArray with dims [original_cluster, period?, scenario?].
        """
        # Note: No coords on original_cluster - they cause issues when used as isel() indexer
        return self._build_property_array(
            lambda cr: np.array(cr.cluster_assignments),
            base_dims=['original_cluster'],
            name='cluster_assignments',
        )

    @property
    def cluster_occurrences(self) -> xr.DataArray:
        """How many original clusters map to each typical cluster.

        Returns:
            DataArray with dims [cluster, period?, scenario?].
        """
        return self._build_property_array(
            _cluster_occurrences,
            base_dims=['cluster'],
            base_coords={'cluster': range(self.n_clusters)},
            name='cluster_occurrences',
        )

    @property
    def cluster_centers(self) -> xr.DataArray:
        """Which original cluster is the representative (center) for each typical cluster.

        Returns:
            DataArray with dims [cluster, period?, scenario?].
        """
        return self._build_property_array(
            lambda cr: np.array(cr.cluster_centers),
            base_dims=['cluster'],
            base_coords={'cluster': range(self.n_clusters)},
            name='cluster_centers',
        )

    @property
    def segment_assignments(self) -> xr.DataArray | None:
        """For each timestep within a cluster, which segment it belongs to.

        Returns:
            DataArray with dims [cluster, time, period?, scenario?], or None if not segmented.
        """
        if self._first_result.segment_assignments is None:
            return None
        timesteps = self._first_result.n_timesteps_per_period
        return self._build_property_array(
            lambda cr: np.array(cr.segment_assignments),
            base_dims=['cluster', 'time'],
            base_coords={'cluster': range(self.n_clusters), 'time': range(timesteps)},
            name='segment_assignments',
        )

    @property
    def segment_durations(self) -> xr.DataArray | None:
        """Duration of each segment in timesteps.

        Returns:
            DataArray with dims [cluster, segment, period?, scenario?], or None if not segmented.
        """
        if self._first_result.segment_durations is None:
            return None
        n_segments = self._first_result.n_segments

        def _get_padded_durations(cr: TsamClusteringResult) -> np.ndarray:
            """Pad ragged segment durations to uniform shape."""
            return np.array([list(d) + [np.nan] * (n_segments - len(d)) for d in cr.segment_durations])

        return self._build_property_array(
            _get_padded_durations,
            base_dims=['cluster', 'segment'],
            base_coords={'cluster': range(self.n_clusters), 'segment': range(n_segments)},
            name='segment_durations',
        )

    @property
    def segment_centers(self) -> xr.DataArray | None:
        """Center of each intra-period segment.

        Only available if segmentation was configured during clustering.

        Returns:
            DataArray or None if no segmentation.
        """
        first = self._first_result
        if first.segment_centers is None:
            return None

        # tsam's segment_centers may be None even with segments configured
        return None

    @property
    def position_within_segment(self) -> xr.DataArray | None:
        """Position of each timestep within its segment (0-indexed).

        For each (cluster, time) position, returns how many timesteps into the
        segment that position is. Used for interpolation within segments.

        Returns:
            DataArray with dims [cluster, time] or [cluster, time, period?, scenario?].
            Returns None if no segmentation.
        """
        segment_assignments = self.segment_assignments
        if segment_assignments is None:
            return None

        def _compute_positions(seg_assigns: np.ndarray) -> np.ndarray:
            """Compute position within segment for each (cluster, time)."""
            n_clusters, n_times = seg_assigns.shape
            positions = np.zeros_like(seg_assigns)
            for c in range(n_clusters):
                pos = 0
                prev_seg = -1
                for t in range(n_times):
                    seg = seg_assigns[c, t]
                    if seg != prev_seg:
                        pos = 0
                        prev_seg = seg
                    positions[c, t] = pos
                    pos += 1
            return positions

        # Handle extra dimensions by applying _compute_positions to each slice
        extra_dims = [d for d in segment_assignments.dims if d not in ('cluster', 'time')]

        if not extra_dims:
            positions = _compute_positions(segment_assignments.values)
            return xr.DataArray(
                positions,
                dims=['cluster', 'time'],
                coords=segment_assignments.coords,
                name='position_within_segment',
            )

        # Multi-dimensional case: compute for each period/scenario slice
        result = xr.apply_ufunc(
            _compute_positions,
            segment_assignments,
            input_core_dims=[['cluster', 'time']],
            output_core_dims=[['cluster', 'time']],
            vectorize=True,
        )
        return result.rename('position_within_segment')

    # === Serialization ===

    def to_dict(self) -> dict:
        """Serialize to dict.

        The dict can be used to reconstruct via from_dict().
        """
        return {
            'dim_names': list(self._dim_names),
            'results': {self._key_to_str(key): result.to_dict() for key, result in self._results.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> ClusteringResults:
        """Reconstruct from dict.

        Args:
            d: Dict from to_dict().

        Returns:
            Reconstructed ClusteringResults.
        """
        from tsam import ClusteringResult

        dim_names = d['dim_names']
        results = {}
        for key_str, result_dict in d['results'].items():
            key = cls._str_to_key(key_str, dim_names)
            results[key] = ClusteringResult.from_dict(result_dict)
        return cls(results, dim_names)

    # === Private helpers ===

    def _make_key(self, **kwargs: Any) -> tuple:
        """Create a key tuple from dimension keyword arguments."""
        key_parts = []
        for dim in self._dim_names:
            if dim in kwargs:
                key_parts.append(kwargs[dim])
        return tuple(key_parts)

    def _get_dim_values(self, dim: str) -> list | None:
        """Get unique values for a dimension, or None if dimension not present."""
        if dim not in self._dim_names:
            return None
        idx = self._dim_names.index(dim)
        return sorted(set(k[idx] for k in self._results.keys()))

    def _build_property_array(
        self,
        get_data: callable,
        base_dims: list[str],
        base_coords: dict | None = None,
        name: str | None = None,
    ) -> xr.DataArray:
        """Build a DataArray property, handling both single and multi-dimensional cases."""
        base_coords = base_coords or {}
        periods = self._get_dim_values('period')
        scenarios = self._get_dim_values('scenario')

        # Build list of (dim_name, values) for dimensions that exist
        extra_dims = []
        if periods is not None:
            extra_dims.append(('period', periods))
        if scenarios is not None:
            extra_dims.append(('scenario', scenarios))

        # Simple case: no extra dimensions
        if not extra_dims:
            return xr.DataArray(get_data(self._results[()]), dims=base_dims, coords=base_coords, name=name)

        # Multi-dimensional: stack data for each combination
        first_data = get_data(next(iter(self._results.values())))
        shape = list(first_data.shape) + [len(vals) for _, vals in extra_dims]
        data = np.empty(shape, dtype=first_data.dtype)  # Preserve dtype

        for combo in np.ndindex(*[len(vals) for _, vals in extra_dims]):
            key = tuple(extra_dims[i][1][idx] for i, idx in enumerate(combo))
            data[(...,) + combo] = get_data(self._results[key])

        dims = base_dims + [dim_name for dim_name, _ in extra_dims]
        coords = {**base_coords, **{dim_name: vals for dim_name, vals in extra_dims}}
        return xr.DataArray(data, dims=dims, coords=coords, name=name)

    @staticmethod
    def _key_to_str(key: tuple) -> str:
        """Convert key tuple to string for serialization."""
        if not key:
            return '__single__'
        return '|'.join(str(k) for k in key)

    @staticmethod
    def _str_to_key(key_str: str, dim_names: list[str]) -> tuple:
        """Convert string back to key tuple."""
        if key_str == '__single__':
            return ()
        parts = key_str.split('|')
        # Try to convert to int if possible (for period years)
        result = []
        for part in parts:
            try:
                result.append(int(part))
            except ValueError:
                result.append(part)
        return tuple(result)

    def __repr__(self) -> str:
        if not self.dims:
            return f'ClusteringResults(n_clusters={self.n_clusters})'
        coords_str = ', '.join(f'{k}: {len(v)}' for k, v in self.coords.items())
        return f'ClusteringResults(dims={self.dims}, coords=({coords_str}), n_clusters={self.n_clusters})'

    def apply(self, data: xr.Dataset) -> AggregationResults:
        """Apply clustering to dataset for all (period, scenario) combinations.

        Args:
            data: Dataset with time-varying data. Must have 'time' dimension.
                May have 'period' and/or 'scenario' dimensions matching this object.

        Returns:
            AggregationResults with full access to aggregated data.
            Use `.clustering` on the result to get ClusteringResults for IO.

        Example:
            >>> agg_results = clustering_results.apply(dataset)
            >>> agg_results.clustering  # Get ClusteringResults for IO
            >>> for key, result in agg_results:
            ...     print(result.cluster_representatives)
        """
        from ..core import drop_constant_arrays

        results = {}
        for key, cr in self._results.items():
            # Build selector for this key
            selector = dict(zip(self._dim_names, key, strict=False))

            # Select the slice for this (period, scenario)
            data_slice = data.sel(**selector, drop=True) if selector else data

            # Drop constant arrays and convert to DataFrame
            time_varying = drop_constant_arrays(data_slice, dim='time')
            df = time_varying.to_dataframe()

            # Apply clustering
            results[key] = cr.apply(df)

        return Clustering._from_aggregation_results(results, self._dim_names)


class Clustering:
    """Clustering information for a FlowSystem.

    Thin wrapper around tsam 3.0's AggregationResult objects, providing:
    1. Multi-dimensional access for (period, scenario) combinations
    2. Structure properties (n_clusters, dims, coords, cluster_assignments)
    3. JSON persistence via ClusteringResults

    Use ``sel()`` to access individual tsam AggregationResult objects for
    detailed analysis (cluster_representatives, accuracy, plotting).

    Attributes:
        results: ClusteringResults for structure access (works after JSON load).
        original_timesteps: Original timesteps before clustering.
        dims: Dimension names, e.g., ('period', 'scenario').
        coords: Coordinate values, e.g., {'period': [2024, 2025]}.

    Example:
        >>> clustering = fs_clustered.clustering
        >>> clustering.n_clusters
        8
        >>> clustering.dims
        ('period',)

        # Access tsam AggregationResult for detailed analysis
        >>> result = clustering.sel(period=2024)
        >>> result.cluster_representatives  # DataFrame
        >>> result.accuracy  # AccuracyMetrics
        >>> result.plot.compare()  # tsam's built-in plotting
    """

    # ==========================================================================
    # Core properties (delegated to ClusteringResults)
    # ==========================================================================

    @property
    def n_clusters(self) -> int:
        """Number of clusters (typical periods)."""
        return self.results.n_clusters

    @property
    def timesteps_per_cluster(self) -> int:
        """Number of timesteps in each cluster."""
        return self.results.timesteps_per_cluster

    @property
    def timesteps_per_period(self) -> int:
        """Alias for timesteps_per_cluster."""
        return self.timesteps_per_cluster

    @property
    def n_original_clusters(self) -> int:
        """Number of original periods (before clustering)."""
        return self.results.n_original_periods

    @property
    def dim_names(self) -> list[str]:
        """Names of extra dimensions, e.g., ['period', 'scenario']."""
        return self.results.dim_names

    @property
    def dims(self) -> tuple[str, ...]:
        """Dimension names as tuple (xarray-like)."""
        return self.results.dims

    @property
    def coords(self) -> dict[str, list]:
        """Coordinate values for each dimension (xarray-like).

        Returns:
            Dict mapping dimension names to lists of coordinate values.

        Example:
            >>> clustering.coords
            {'period': [2024, 2025], 'scenario': ['low', 'high']}
        """
        return self.results.coords

    def sel(
        self,
        period: int | str | None = None,
        scenario: str | None = None,
    ) -> AggregationResult:
        """Select AggregationResult by period and/or scenario.

        Access individual tsam AggregationResult objects for detailed analysis.

        Note:
            This method is only available before saving/loading the FlowSystem.
            After IO (to_dataset/from_dataset or to_json), the full AggregationResult
            data is not preserved. Use `results.sel()` for structure-only access
            after loading.

        Args:
            period: Period value (e.g., 2024). Required if clustering has periods.
            scenario: Scenario name (e.g., 'high'). Required if clustering has scenarios.

        Returns:
            The tsam AggregationResult for the specified combination.
            Access its properties like `cluster_representatives`, `accuracy`, etc.

        Raises:
            KeyError: If no result found for the specified combination.
            ValueError: If accessed on a Clustering loaded from JSON/NetCDF.

        Example:
            >>> result = clustering.sel(period=2024, scenario='high')
            >>> result.cluster_representatives  # DataFrame with aggregated data
            >>> result.accuracy  # AccuracyMetrics
            >>> result.plot.compare()  # tsam's built-in comparison plot
        """
        self._require_full_data('sel()')
        # Build key from provided args in dim order
        key_parts = []
        if 'period' in self._dim_names:
            if period is None:
                raise KeyError(f"'period' is required. Available: {self.coords.get('period', [])}")
            key_parts.append(period)
        if 'scenario' in self._dim_names:
            if scenario is None:
                raise KeyError(f"'scenario' is required. Available: {self.coords.get('scenario', [])}")
            key_parts.append(scenario)
        key = tuple(key_parts)
        if key not in self._aggregation_results:
            raise KeyError(f'No result found for period={period}, scenario={scenario}')
        return self._aggregation_results[key]

    @property
    def is_segmented(self) -> bool:
        """Whether intra-period segmentation was used.

        Segmented systems have variable timestep durations within each cluster,
        where each segment represents a different number of original timesteps.
        """
        return self.results.n_segments is not None

    @property
    def n_segments(self) -> int | None:
        """Number of segments per cluster, or None if not segmented."""
        return self.results.n_segments

    @property
    def cluster_assignments(self) -> xr.DataArray:
        """Mapping from original periods to cluster IDs.

        Returns:
            DataArray with dims [original_cluster] or [original_cluster, period?, scenario?].
        """
        return self.results.cluster_assignments

    @property
    def n_representatives(self) -> int:
        """Number of representative timesteps after clustering."""
        return self.n_clusters * self.timesteps_per_cluster

    # ==========================================================================
    # Derived properties
    # ==========================================================================

    @property
    def cluster_occurrences(self) -> xr.DataArray:
        """Count of how many original periods each cluster represents.

        Returns:
            DataArray with dims [cluster] or [cluster, period?, scenario?].
        """
        return self.results.cluster_occurrences

    @property
    def representative_weights(self) -> xr.DataArray:
        """Weight for each cluster (number of original periods it represents).

        This is the same as cluster_occurrences but named for API consistency.
        Used as cluster_weight in FlowSystem.
        """
        return self.cluster_occurrences.rename('representative_weights')

    @property
    def timestep_mapping(self) -> xr.DataArray:
        """Mapping from original timesteps to representative timestep indices.

        Each value indicates which representative timestep index (0 to n_representatives-1)
        corresponds to each original timestep.
        """
        return self._build_timestep_mapping()

    @property
    def metrics(self) -> xr.Dataset:
        """Clustering quality metrics (RMSE, MAE, etc.).

        Returns:
            Dataset with dims [time_series, period?, scenario?], or empty Dataset if no metrics.
        """
        if self._metrics is None:
            return xr.Dataset()
        return self._metrics

    @property
    def cluster_start_positions(self) -> np.ndarray:
        """Integer positions where clusters start in reduced timesteps.

        Returns:
            1D array: [0, T, 2T, ...] where T = timesteps_per_cluster.
        """
        n_timesteps = self.n_clusters * self.timesteps_per_cluster
        return np.arange(0, n_timesteps, self.timesteps_per_cluster)

    @property
    def cluster_centers(self) -> xr.DataArray:
        """Which original period is the representative (center) for each cluster.

        Returns:
            DataArray with dims [cluster] containing original period indices.
        """
        return self.results.cluster_centers

    @property
    def segment_assignments(self) -> xr.DataArray | None:
        """For each timestep within a cluster, which intra-period segment it belongs to.

        Only available if segmentation was configured during clustering.

        Returns:
            DataArray with dims [cluster, time] or None if no segmentation.
        """
        return self.results.segment_assignments

    @property
    def segment_durations(self) -> xr.DataArray | None:
        """Duration of each intra-period segment in hours.

        Only available if segmentation was configured during clustering.

        Returns:
            DataArray with dims [cluster, segment] or None if no segmentation.
        """
        return self.results.segment_durations

    @property
    def segment_centers(self) -> xr.DataArray | None:
        """Center of each intra-period segment.

        Only available if segmentation was configured during clustering.

        Returns:
            DataArray with dims [cluster, segment] or None if no segmentation.
        """
        return self.results.segment_centers

    # ==========================================================================
    # Methods
    # ==========================================================================

    def expand_data(
        self,
        aggregated: xr.DataArray,
        original_time: pd.DatetimeIndex | None = None,
    ) -> xr.DataArray:
        """Expand aggregated data back to original timesteps.

        Uses the timestep_mapping to map each original timestep to its
        representative value from the aggregated data. Fully vectorized using
        xarray's advanced indexing - no loops over period/scenario dimensions.

        Args:
            aggregated: DataArray with aggregated (cluster, time) or (time,) dimension.
            original_time: Original time coordinates. Defaults to self.original_timesteps.

        Returns:
            DataArray expanded to original timesteps.
        """
        if original_time is None:
            original_time = self.original_timesteps

        timestep_mapping = self.timestep_mapping  # Already multi-dimensional DataArray

        if 'cluster' not in aggregated.dims:
            # No cluster dimension: use mapping directly as time index
            expanded = aggregated.isel(time=timestep_mapping)
        else:
            # Has cluster dimension: compute cluster and time indices from mapping
            # For segmented systems, time dimension is n_segments, not timesteps_per_cluster
            if self.is_segmented and self.n_segments is not None:
                time_dim_size = self.n_segments
            else:
                time_dim_size = self.timesteps_per_cluster

            cluster_indices = timestep_mapping // time_dim_size
            time_indices = timestep_mapping % time_dim_size

            # xarray's advanced indexing handles broadcasting across period/scenario dims
            expanded = aggregated.isel(cluster=cluster_indices, time=time_indices)

        # Clean up: drop coordinate artifacts from isel, then rename original_time -> time
        # The isel operation may leave 'cluster' and 'time' as non-dimension coordinates
        expanded = expanded.drop_vars(['cluster', 'time'], errors='ignore')
        expanded = expanded.rename({'original_time': 'time'}).assign_coords(time=original_time)

        return expanded.transpose('time', ...).assign_attrs(aggregated.attrs)

    def build_expansion_divisor(
        self,
        original_time: pd.DatetimeIndex | None = None,
    ) -> xr.DataArray:
        """Build divisor for correcting segment totals when expanding to hourly.

        For segmented systems, each segment value is a total that gets repeated N times
        when expanded to hourly resolution (where N = segment duration in timesteps).
        This divisor allows converting those totals back to hourly rates during expansion.

        For each original timestep, returns the number of original timesteps that map
        to the same (cluster, segment) - i.e., the segment duration in timesteps.

        Fully vectorized using xarray's advanced indexing - no loops over period/scenario.

        Args:
            original_time: Original time coordinates. Defaults to self.original_timesteps.

        Returns:
            DataArray with dims ['time'] or ['time', 'period'?, 'scenario'?] containing
            the number of timesteps in each segment, aligned to original timesteps.
        """
        if not self.is_segmented or self.n_segments is None:
            raise ValueError('build_expansion_divisor requires a segmented clustering')

        if original_time is None:
            original_time = self.original_timesteps

        timestep_mapping = self.timestep_mapping  # Already multi-dimensional
        segment_assignments = self.results.segment_assignments  # [cluster, time, period?, scenario?]
        segment_durations = self.results.segment_durations  # [cluster, segment, period?, scenario?]

        # Decode cluster and time indices from timestep_mapping
        # For segmented systems, time dimension is n_segments
        time_dim_size = self.n_segments
        cluster_indices = timestep_mapping // time_dim_size
        time_indices = timestep_mapping % time_dim_size

        # Step 1: Get segment index for each original timestep
        # segment_assignments[cluster, time] -> segment index
        seg_indices = segment_assignments.isel(cluster=cluster_indices, time=time_indices)

        # Step 2: Get duration for each segment
        # segment_durations[cluster, segment] -> duration
        divisor = segment_durations.isel(cluster=cluster_indices, segment=seg_indices)

        # Clean up coordinates and rename
        divisor = divisor.drop_vars(['cluster', 'time', 'segment'], errors='ignore')
        divisor = divisor.rename({'original_time': 'time'}).assign_coords(time=original_time)

        return divisor.transpose('time', ...).rename('expansion_divisor')

    def get_result(
        self,
        period: Any = None,
        scenario: Any = None,
    ) -> TsamClusteringResult:
        """Get the tsam ClusteringResult for a specific (period, scenario).

        Args:
            period: Period label (if applicable).
            scenario: Scenario label (if applicable).

        Returns:
            The tsam ClusteringResult for the specified combination.
        """
        return self.results.sel(period=period, scenario=scenario)

    def apply(
        self,
        data: pd.DataFrame,
        period: Any = None,
        scenario: Any = None,
    ) -> AggregationResult:
        """Apply the saved clustering to new data.

        Args:
            data: DataFrame with time series data to cluster.
            period: Period label (if applicable).
            scenario: Scenario label (if applicable).

        Returns:
            tsam AggregationResult with the clustering applied.
        """
        return self.results.sel(period=period, scenario=scenario).apply(data)

    def to_json(self, path: str | Path) -> None:
        """Save the clustering for reuse.

        Uses ClusteringResults.to_dict() which preserves full tsam ClusteringResult.
        Can be loaded later with Clustering.from_json() and used with
        flow_system.transform.apply_clustering().

        Args:
            path: Path to save the JSON file.
        """
        data = {
            'results': self.results.to_dict(),
            'original_timesteps': [ts.isoformat() for ts in self.original_timesteps],
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        original_timesteps: pd.DatetimeIndex | None = None,
    ) -> Clustering:
        """Load a clustering from JSON.

        The loaded Clustering has full apply() support because ClusteringResult
        is fully preserved via tsam's serialization.

        Args:
            path: Path to the JSON file.
            original_timesteps: Original timesteps for the new FlowSystem.
                If None, uses the timesteps stored in the JSON.

        Returns:
            A Clustering that can be used with apply_clustering().
        """
        with open(path) as f:
            data = json.load(f)

        results = ClusteringResults.from_dict(data['results'])

        if original_timesteps is None:
            original_timesteps = pd.DatetimeIndex([pd.Timestamp(ts) for ts in data['original_timesteps']])

        return cls(
            results=results,
            original_timesteps=original_timesteps,
        )

    # ==========================================================================
    # Visualization
    # ==========================================================================

    @property
    def plot(self) -> ClusteringPlotAccessor:
        """Access plotting methods for clustering visualization.

        Returns:
            ClusteringPlotAccessor with compare(), heatmap(), and clusters() methods.
        """
        return ClusteringPlotAccessor(self)

    # ==========================================================================
    # Private helpers
    # ==========================================================================

    def _build_timestep_mapping(self) -> xr.DataArray:
        """Build timestep_mapping DataArray."""
        n_original = len(self.original_timesteps)
        original_time_coord = self.original_timesteps.rename('original_time')
        return self.results._build_property_array(
            lambda cr: _build_timestep_mapping(cr, n_original),
            base_dims=['original_time'],
            base_coords={'original_time': original_time_coord},
            name='timestep_mapping',
        )

    def _create_reference_structure(self, include_original_data: bool = True) -> tuple[dict, dict[str, xr.DataArray]]:
        """Create serialization structure for to_dataset().

        Args:
            include_original_data: Whether to include original_data in serialization.
                Set to False for smaller files when plot.compare() isn't needed after IO.
                Defaults to True.

        Returns:
            Tuple of (reference_dict, arrays_dict).
        """
        arrays = {}

        # Collect original_data arrays
        # Rename 'time' to 'original_time' to avoid conflict with clustered FlowSystem's time coord
        original_data_refs = None
        if include_original_data and self.original_data is not None:
            original_data_refs = []
            for name, da in self.original_data.data_vars.items():
                ref_name = f'original_data|{name}'
                # Rename time dim to avoid xarray alignment issues
                if 'time' in da.dims:
                    da = da.rename({'time': 'original_time'})
                arrays[ref_name] = da
                original_data_refs.append(f':::{ref_name}')

        # NOTE: aggregated_data is NOT serialized - it's identical to the FlowSystem's
        # main data arrays and would be redundant. After loading, aggregated_data is
        # reconstructed from the FlowSystem's dataset.

        # Collect metrics arrays
        metrics_refs = None
        if self._metrics is not None:
            metrics_refs = []
            for name, da in self._metrics.data_vars.items():
                ref_name = f'metrics|{name}'
                arrays[ref_name] = da
                metrics_refs.append(f':::{ref_name}')

        reference = {
            '__class__': 'Clustering',
            'results': self.results.to_dict(),  # Full ClusteringResults serialization
            'original_timesteps': [ts.isoformat() for ts in self.original_timesteps],
            '_original_data_refs': original_data_refs,
            '_metrics_refs': metrics_refs,
        }

        return reference, arrays

    def __init__(
        self,
        results: ClusteringResults | dict | None = None,
        original_timesteps: pd.DatetimeIndex | list[str] | None = None,
        original_data: xr.Dataset | None = None,
        aggregated_data: xr.Dataset | None = None,
        _metrics: xr.Dataset | None = None,
        # These are for reconstruction from serialization
        _original_data_refs: list[str] | None = None,
        _metrics_refs: list[str] | None = None,
        # Internal: AggregationResult dict for full data access
        _aggregation_results: dict[tuple, AggregationResult] | None = None,
        _dim_names: list[str] | None = None,
    ):
        """Initialize Clustering object.

        Args:
            results: ClusteringResults instance, or dict from to_dict() (for deserialization).
                Not needed if _aggregation_results is provided.
            original_timesteps: Original timesteps before clustering.
            original_data: Original dataset before clustering (for expand/plotting).
            aggregated_data: Aggregated dataset after clustering (for plotting).
                After loading from file, this is reconstructed from FlowSystem data.
            _metrics: Pre-computed metrics dataset.
            _original_data_refs: Internal: resolved DataArrays from serialization.
            _metrics_refs: Internal: resolved DataArrays from serialization.
            _aggregation_results: Internal: dict of AggregationResult for full data access.
            _dim_names: Internal: dimension names when using _aggregation_results.
        """
        # Handle ISO timestamp strings from serialization
        if (
            isinstance(original_timesteps, list)
            and len(original_timesteps) > 0
            and isinstance(original_timesteps[0], str)
        ):
            original_timesteps = pd.DatetimeIndex([pd.Timestamp(ts) for ts in original_timesteps])

        # Store AggregationResults if provided (full data access)
        self._aggregation_results = _aggregation_results
        self._dim_names = _dim_names or []

        # Handle results - only needed for serialization path
        if results is not None:
            if isinstance(results, dict):
                results = ClusteringResults.from_dict(results)
            self._results_cache = results
        else:
            self._results_cache = None

        # Flag indicating this was loaded from serialization (missing full AggregationResult data)
        self._from_serialization = _aggregation_results is None and results is not None

        self.original_timesteps = original_timesteps if original_timesteps is not None else pd.DatetimeIndex([])
        self._metrics = _metrics

        # Handle reconstructed data from refs (list of DataArrays)
        if _original_data_refs is not None and isinstance(_original_data_refs, list):
            # These are resolved DataArrays from the structure resolver
            if all(isinstance(da, xr.DataArray) for da in _original_data_refs):
                # Rename 'original_time' back to 'time' and strip 'original_data|' prefix
                data_vars = {}
                for da in _original_data_refs:
                    if 'original_time' in da.dims:
                        da = da.rename({'original_time': 'time'})
                    # Strip 'original_data|' prefix from name (added during serialization)
                    name = da.name
                    if name.startswith('original_data|'):
                        name = name[14:]  # len('original_data|') = 14
                    data_vars[name] = da.rename(name)
                self.original_data = xr.Dataset(data_vars)
            else:
                self.original_data = original_data
        else:
            self.original_data = original_data

        self.aggregated_data = aggregated_data

        if _metrics_refs is not None and isinstance(_metrics_refs, list):
            if all(isinstance(da, xr.DataArray) for da in _metrics_refs):
                # Strip 'metrics|' prefix from name (added during serialization)
                data_vars = {}
                for da in _metrics_refs:
                    name = da.name
                    if name.startswith('metrics|'):
                        name = name[8:]  # len('metrics|') = 8
                    data_vars[name] = da.rename(name)
                self._metrics = xr.Dataset(data_vars)

    @property
    def results(self) -> ClusteringResults:
        """ClusteringResults for structure access (derived from AggregationResults or cached)."""
        if self._results_cache is not None:
            return self._results_cache
        if self._aggregation_results is not None:
            # Derive from AggregationResults (cached on first access)
            self._results_cache = ClusteringResults(
                {k: r.clustering for k, r in self._aggregation_results.items()},
                self._dim_names,
            )
            return self._results_cache
        raise ValueError('No results available - neither AggregationResults nor ClusteringResults set')

    @classmethod
    def _from_aggregation_results(
        cls,
        aggregation_results: dict[tuple, AggregationResult],
        dim_names: list[str],
        original_timesteps: pd.DatetimeIndex | None = None,
        original_data: xr.Dataset | None = None,
    ) -> Clustering:
        """Create Clustering from AggregationResult dict.

        This is the primary way to create a Clustering with full data access.
        Called by ClusteringResults.apply() and TransformAccessor.

        Args:
            aggregation_results: Dict mapping (period, scenario) tuples to AggregationResult.
            dim_names: Dimension names, e.g., ['period', 'scenario'].
            original_timesteps: Original timesteps (optional, for expand).
            original_data: Original dataset (optional, for plotting).

        Returns:
            Clustering with full AggregationResult access.
        """
        return cls(
            original_timesteps=original_timesteps,
            original_data=original_data,
            _aggregation_results=aggregation_results,
            _dim_names=dim_names,
        )

    # ==========================================================================
    # Iteration over AggregationResults (for direct access to tsam results)
    # ==========================================================================

    def __iter__(self):
        """Iterate over (key, AggregationResult) pairs.

        Raises:
            ValueError: If accessed on a Clustering loaded from JSON.
        """
        self._require_full_data('iteration')
        return iter(self._aggregation_results.items())

    def __len__(self) -> int:
        """Number of (period, scenario) combinations."""
        if self._aggregation_results is not None:
            return len(self._aggregation_results)
        return len(list(self.results.keys()))

    def __getitem__(self, key: tuple) -> AggregationResult:
        """Get AggregationResult by (period, scenario) key.

        Raises:
            ValueError: If accessed on a Clustering loaded from JSON.
        """
        self._require_full_data('item access')
        return self._aggregation_results[key]

    def items(self):
        """Iterate over (key, AggregationResult) pairs.

        Raises:
            ValueError: If accessed on a Clustering loaded from JSON.
        """
        self._require_full_data('items()')
        return self._aggregation_results.items()

    def keys(self):
        """Iterate over (period, scenario) keys."""
        if self._aggregation_results is not None:
            return self._aggregation_results.keys()
        return self.results.keys()

    def values(self):
        """Iterate over AggregationResult objects.

        Raises:
            ValueError: If accessed on a Clustering loaded from JSON.
        """
        self._require_full_data('values()')
        return self._aggregation_results.values()

    def _require_full_data(self, operation: str) -> None:
        """Raise error if full AggregationResult data is not available."""
        if self._from_serialization:
            raise ValueError(
                f'{operation} requires full AggregationResult data, '
                f'but this Clustering was loaded from JSON. '
                f'Use apply_clustering() to get full results.'
            )

    def __repr__(self) -> str:
        return (
            f'Clustering(\n'
            f'  {self.n_original_clusters} periods → {self.n_clusters} clusters\n'
            f'  timesteps_per_cluster={self.timesteps_per_cluster}\n'
            f'  dims={self.dim_names}\n'
            f')'
        )


class ClusteringPlotAccessor:
    """Plot accessor for Clustering objects.

    Provides visualization methods for comparing original vs aggregated data
    and understanding the clustering structure.
    """

    def __init__(self, clustering: Clustering):
        self._clustering = clustering

    def compare(
        self,
        kind: str = 'timeseries',
        variables: str | list[str] | None = None,
        *,
        select: SelectType | None = None,
        colors: ColorType | None = None,
        show: bool | None = None,
        data_only: bool = False,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Compare original vs aggregated data.

        Args:
            kind: Type of comparison plot.
                - 'timeseries': Time series comparison (default)
                - 'duration_curve': Sorted duration curve comparison
            variables: Variable(s) to plot. Can be a string, list of strings,
                or None to plot all time-varying variables.
            select: xarray-style selection dict, e.g. {'scenario': 'Base Case'}.
            colors: Color specification (colorscale name, color list, or label-to-color dict).
            show: Whether to display the figure.
                Defaults to CONFIG.Plotting.default_show.
            data_only: If True, skip figure creation and return only data.
            **plotly_kwargs: Additional arguments passed to plotly (e.g., color, line_dash,
                facet_col, facet_row). Defaults: x='time'/'duration', color='variable',
                line_dash='representation', symbol=None.

        Returns:
            PlotResult containing the comparison figure and underlying data.
        """
        import plotly.graph_objects as go

        from ..config import CONFIG
        from ..plot_result import PlotResult
        from ..statistics_accessor import _apply_selection

        if kind not in ('timeseries', 'duration_curve'):
            raise ValueError(f"Unknown kind '{kind}'. Use 'timeseries' or 'duration_curve'.")

        clustering = self._clustering
        if clustering.original_data is None or clustering.aggregated_data is None:
            raise ValueError('No original/aggregated data available for comparison')

        resolved_variables = self._resolve_variables(variables)

        # Build Dataset with variables as data_vars
        data_vars = {}
        for var in resolved_variables:
            original = clustering.original_data[var]
            clustered = clustering.expand_data(clustering.aggregated_data[var])
            combined = xr.concat([original, clustered], dim=pd.Index(['Original', 'Clustered'], name='representation'))
            data_vars[var] = combined
        ds = xr.Dataset(data_vars)

        ds = _apply_selection(ds, select)

        if kind == 'duration_curve':
            sorted_vars = {}
            for var in ds.data_vars:
                for rep in ds.coords['representation'].values:
                    values = np.sort(ds[var].sel(representation=rep).values.flatten())[::-1]
                    sorted_vars[(var, rep)] = values
            # Get length from first sorted array
            n = len(next(iter(sorted_vars.values())))
            ds = xr.Dataset(
                {
                    var: xr.DataArray(
                        [sorted_vars[(var, r)] for r in ['Original', 'Clustered']],
                        dims=['representation', 'duration'],
                        coords={'representation': ['Original', 'Clustered'], 'duration': range(n)},
                    )
                    for var in resolved_variables
                }
            )

        title = (
            (
                'Original vs Clustered'
                if len(resolved_variables) > 1
                else f'Original vs Clustered: {resolved_variables[0]}'
            )
            if kind == 'timeseries'
            else ('Duration Curve' if len(resolved_variables) > 1 else f'Duration Curve: {resolved_variables[0]}')
        )

        # Early return for data_only mode
        if data_only:
            return PlotResult(data=ds, figure=go.Figure())

        # Apply slot defaults
        defaults = {
            'x': 'duration' if kind == 'duration_curve' else 'time',
            'color': 'variable',
            'line_dash': 'representation',
            'line_dash_map': {'Original': 'dot', 'Clustered': 'solid'},
            'symbol': None,  # Block symbol slot
        }
        _apply_slot_defaults(plotly_kwargs, defaults)

        color_kwargs = _build_color_kwargs(colors, list(ds.data_vars))
        fig = ds.plotly.line(
            title=title,
            **color_kwargs,
            **plotly_kwargs,
        )
        fig.update_yaxes(matches=None)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))

        plot_result = PlotResult(data=ds, figure=fig)

        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            plot_result.show()

        return plot_result

    def _get_time_varying_variables(self) -> list[str]:
        """Get list of time-varying variables from original data."""
        if self._clustering.original_data is None:
            return []
        return [
            name
            for name in self._clustering.original_data.data_vars
            if 'time' in self._clustering.original_data[name].dims
            and not np.isclose(
                self._clustering.original_data[name].min(),
                self._clustering.original_data[name].max(),
            )
        ]

    def _resolve_variables(self, variables: str | list[str] | None) -> list[str]:
        """Resolve variables parameter to a list of valid variable names."""
        time_vars = self._get_time_varying_variables()
        if not time_vars:
            raise ValueError('No time-varying variables found')

        if variables is None:
            return time_vars
        elif isinstance(variables, str):
            if variables not in time_vars:
                raise ValueError(f"Variable '{variables}' not found. Available: {time_vars}")
            return [variables]
        else:
            invalid = [v for v in variables if v not in time_vars]
            if invalid:
                raise ValueError(f'Variables {invalid} not found. Available: {time_vars}')
            return list(variables)

    def heatmap(
        self,
        *,
        select: SelectType | None = None,
        colors: str | list[str] | None = None,
        show: bool | None = None,
        data_only: bool = False,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot cluster assignments over time as a heatmap timeline.

        Shows which cluster each timestep belongs to as a horizontal color bar.
        The x-axis is time, color indicates cluster assignment. This visualization
        aligns with time series data, making it easy to correlate cluster
        assignments with other plots.

        For multi-period/scenario data, uses faceting and/or animation.

        Args:
            select: xarray-style selection dict, e.g. {'scenario': 'Base Case'}.
            colors: Colorscale name (str) or list of colors for heatmap coloring.
                Dicts are not supported for heatmaps.
                Defaults to plotly template's sequential colorscale.
            show: Whether to display the figure.
                Defaults to CONFIG.Plotting.default_show.
            data_only: If True, skip figure creation and return only data.
            **plotly_kwargs: Additional arguments passed to plotly (e.g., facet_col, animation_frame).

        Returns:
            PlotResult containing the heatmap figure and cluster assignment data.
            The data has 'cluster' variable with time dimension, matching original timesteps.
        """
        import plotly.graph_objects as go

        from ..config import CONFIG
        from ..plot_result import PlotResult
        from ..statistics_accessor import _apply_selection

        clustering = self._clustering
        cluster_assignments = clustering.cluster_assignments
        timesteps_per_cluster = clustering.timesteps_per_cluster
        original_time = clustering.original_timesteps

        if select:
            cluster_assignments = _apply_selection(cluster_assignments.to_dataset(name='cluster'), select)['cluster']

        # Expand cluster_assignments to per-timestep
        extra_dims = [d for d in cluster_assignments.dims if d != 'original_cluster']
        expanded_values = np.repeat(cluster_assignments.values, timesteps_per_cluster, axis=0)

        coords = {'time': original_time}
        coords.update({d: cluster_assignments.coords[d].values for d in extra_dims})
        cluster_da = xr.DataArray(expanded_values, dims=['time'] + extra_dims, coords=coords)
        cluster_da.name = 'cluster'

        # Early return for data_only mode
        if data_only:
            return PlotResult(data=xr.Dataset({'cluster': cluster_da}), figure=go.Figure())

        heatmap_da = cluster_da.expand_dims('y', axis=-1).assign_coords(y=['Cluster'])
        heatmap_da.name = 'cluster_assignment'
        heatmap_da = heatmap_da.transpose('time', 'y', ...)

        # Use plotly.imshow for heatmap
        # Only pass color_continuous_scale if explicitly provided (template handles default)
        if colors is not None:
            plotly_kwargs.setdefault('color_continuous_scale', colors)
        fig = heatmap_da.plotly.imshow(
            title='Cluster Assignments',
            aspect='auto',
            **plotly_kwargs,
        )

        fig.update_yaxes(showticklabels=False)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))

        # Data is exactly what we plotted (without dummy y dimension)
        data = xr.Dataset({'cluster': cluster_da})
        plot_result = PlotResult(data=data, figure=fig)

        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            plot_result.show()

        return plot_result

    def clusters(
        self,
        variables: str | list[str] | None = None,
        *,
        select: SelectType | None = None,
        colors: ColorType | None = None,
        show: bool | None = None,
        data_only: bool = False,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot each cluster's typical period profile.

        Shows each cluster as a separate faceted subplot with all variables
        colored differently. Useful for understanding what each cluster represents.

        Args:
            variables: Variable(s) to plot. Can be a string, list of strings,
                or None to plot all time-varying variables.
            select: xarray-style selection dict, e.g. {'scenario': 'Base Case'}.
            colors: Color specification (colorscale name, color list, or label-to-color dict).
            show: Whether to display the figure.
                Defaults to CONFIG.Plotting.default_show.
            data_only: If True, skip figure creation and return only data.
            **plotly_kwargs: Additional arguments passed to plotly (e.g., color, facet_col,
                facet_col_wrap). Defaults: x='time', color='variable', symbol=None.

        Returns:
            PlotResult containing the figure and underlying data.
        """
        import plotly.graph_objects as go

        from ..config import CONFIG
        from ..plot_result import PlotResult
        from ..statistics_accessor import _apply_selection

        clustering = self._clustering
        if clustering.aggregated_data is None:
            raise ValueError('No aggregated data available')

        aggregated_data = _apply_selection(clustering.aggregated_data, select)
        resolved_variables = self._resolve_variables(variables)

        n_clusters = clustering.n_clusters
        timesteps_per_cluster = clustering.timesteps_per_cluster
        cluster_occurrences = clustering.cluster_occurrences

        # Build cluster labels
        occ_extra_dims = [d for d in cluster_occurrences.dims if d != 'cluster']
        if occ_extra_dims:
            cluster_labels = [f'Cluster {c}' for c in range(n_clusters)]
        else:
            cluster_labels = [
                f'Cluster {c} (×{int(cluster_occurrences.sel(cluster=c).values)})' for c in range(n_clusters)
            ]

        data_vars = {}
        for var in resolved_variables:
            da = aggregated_data[var]
            if 'cluster' in da.dims:
                data_by_cluster = da.values
            else:
                data_by_cluster = da.values.reshape(n_clusters, timesteps_per_cluster)
            data_vars[var] = xr.DataArray(
                data_by_cluster,
                dims=['cluster', 'time'],
                coords={'cluster': cluster_labels, 'time': range(timesteps_per_cluster)},
            )

        ds = xr.Dataset(data_vars)

        # Early return for data_only mode (include occurrences in result)
        if data_only:
            data_vars['occurrences'] = cluster_occurrences
            return PlotResult(data=xr.Dataset(data_vars), figure=go.Figure())

        title = 'Clusters' if len(resolved_variables) > 1 else f'Clusters: {resolved_variables[0]}'

        # Apply slot defaults
        defaults = {
            'x': 'time',
            'color': 'variable',
            'symbol': None,  # Block symbol slot
        }
        _apply_slot_defaults(plotly_kwargs, defaults)

        color_kwargs = _build_color_kwargs(colors, list(ds.data_vars))
        fig = ds.plotly.line(
            title=title,
            **color_kwargs,
            **plotly_kwargs,
        )
        fig.update_yaxes(matches=None)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))

        data_vars['occurrences'] = cluster_occurrences
        result_data = xr.Dataset(data_vars)
        plot_result = PlotResult(data=result_data, figure=fig)

        if show is None:
            show = CONFIG.Plotting.default_show
        if show:
            plot_result.show()

        return plot_result


# Backwards compatibility alias
AggregationResults = Clustering


def _register_clustering_classes():
    """Register clustering classes for IO."""
    from ..structure import CLASS_REGISTRY

    CLASS_REGISTRY['Clustering'] = Clustering
