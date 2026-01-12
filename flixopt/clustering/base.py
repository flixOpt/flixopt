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


def _select_dims(da: xr.DataArray, period: Any = None, scenario: Any = None) -> xr.DataArray:
    """Select from DataArray by period/scenario if those dimensions exist."""
    if 'period' in da.dims and period is not None:
        da = da.sel(period=period)
    if 'scenario' in da.dims and scenario is not None:
        da = da.sel(scenario=scenario)
    return da


def _cluster_occurrences(cr: TsamClusteringResult) -> np.ndarray:
    """Compute cluster occurrences from ClusteringResult."""
    counts = Counter(cr.cluster_assignments)
    return np.array([counts.get(i, 0) for i in range(cr.n_clusters)])


def _build_timestep_mapping(cr: TsamClusteringResult, n_timesteps: int) -> np.ndarray:
    """Build mapping from original timesteps to representative timestep indices."""
    timesteps_per_cluster = cr.n_timesteps_per_period
    mapping = np.zeros(n_timesteps, dtype=np.int32)
    for period_idx, cluster_id in enumerate(cr.cluster_assignments):
        for pos in range(timesteps_per_cluster):
            orig_idx = period_idx * timesteps_per_cluster + pos
            if orig_idx < n_timesteps:
                mapping[orig_idx] = int(cluster_id) * timesteps_per_cluster + pos
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

    # === Multi-dim DataArrays ===

    @property
    def cluster_assignments(self) -> xr.DataArray:
        """Build multi-dimensional cluster_assignments DataArray.

        Returns:
            DataArray with dims [original_cluster] or [original_cluster, period?, scenario?].
        """
        if not self.dim_names:
            # Simple case: no extra dimensions
            # Note: Don't include coords - they cause issues when used as isel() indexer
            return xr.DataArray(
                np.array(self._results[()].cluster_assignments),
                dims=['original_cluster'],
                name='cluster_assignments',
            )

        # Multi-dimensional case
        # Note: Don't include coords - they cause issues when used as isel() indexer
        periods = self._get_dim_values('period')
        scenarios = self._get_dim_values('scenario')

        return self._build_multi_dim_array(
            lambda cr: np.array(cr.cluster_assignments),
            base_dims=['original_cluster'],
            base_coords={},  # No coords on original_cluster
            periods=periods,
            scenarios=scenarios,
            name='cluster_assignments',
        )

    @property
    def cluster_occurrences(self) -> xr.DataArray:
        """Build multi-dimensional cluster_occurrences DataArray.

        Returns:
            DataArray with dims [cluster] or [cluster, period?, scenario?].
        """
        if not self.dim_names:
            return xr.DataArray(
                _cluster_occurrences(self._results[()]),
                dims=['cluster'],
                coords={'cluster': range(self.n_clusters)},
                name='cluster_occurrences',
            )

        periods = self._get_dim_values('period')
        scenarios = self._get_dim_values('scenario')

        return self._build_multi_dim_array(
            _cluster_occurrences,
            base_dims=['cluster'],
            base_coords={'cluster': range(self.n_clusters)},
            periods=periods,
            scenarios=scenarios,
            name='cluster_occurrences',
        )

    @property
    def cluster_centers(self) -> xr.DataArray:
        """Which original period is the representative (center) for each cluster.

        Returns:
            DataArray with dims [cluster] containing original period indices.
        """
        if not self.dim_names:
            return xr.DataArray(
                np.array(self._results[()].cluster_centers),
                dims=['cluster'],
                coords={'cluster': range(self.n_clusters)},
                name='cluster_centers',
            )

        periods = self._get_dim_values('period')
        scenarios = self._get_dim_values('scenario')

        return self._build_multi_dim_array(
            lambda cr: np.array(cr.cluster_centers),
            base_dims=['cluster'],
            base_coords={'cluster': range(self.n_clusters)},
            periods=periods,
            scenarios=scenarios,
            name='cluster_centers',
        )

    @property
    def segment_assignments(self) -> xr.DataArray | None:
        """For each timestep within a cluster, which intra-period segment it belongs to.

        Only available if segmentation was configured during clustering.

        Returns:
            DataArray with dims [cluster, time] or None if no segmentation.
        """
        first = self._first_result
        if first.segment_assignments is None:
            return None

        if not self.dim_names:
            # segment_assignments is tuple of tuples: (cluster0_assignments, cluster1_assignments, ...)
            data = np.array(first.segment_assignments)
            return xr.DataArray(
                data,
                dims=['cluster', 'time'],
                coords={'cluster': range(self.n_clusters)},
                name='segment_assignments',
            )

        # Multi-dim case would need more complex handling
        # For now, return None for multi-dim
        return None

    @property
    def segment_durations(self) -> xr.DataArray | None:
        """Duration of each intra-period segment in hours.

        Only available if segmentation was configured during clustering.

        Returns:
            DataArray with dims [cluster, segment] or None if no segmentation.
        """
        first = self._first_result
        if first.segment_durations is None:
            return None

        if not self.dim_names:
            # segment_durations is tuple of tuples: (cluster0_durations, cluster1_durations, ...)
            # Each cluster may have different segment counts, so we need to handle ragged arrays
            durations = first.segment_durations
            n_segments = first.n_segments
            data = np.array([list(d) + [np.nan] * (n_segments - len(d)) for d in durations])
            return xr.DataArray(
                data,
                dims=['cluster', 'segment'],
                coords={'cluster': range(self.n_clusters), 'segment': range(n_segments)},
                name='segment_durations',
                attrs={'units': 'hours'},
            )

        return None

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

    def _build_multi_dim_array(
        self,
        get_data: callable,
        base_dims: list[str],
        base_coords: dict,
        periods: list | None,
        scenarios: list | None,
        name: str,
    ) -> xr.DataArray:
        """Build a multi-dimensional DataArray from per-result data."""
        has_periods = periods is not None
        has_scenarios = scenarios is not None

        slices = {}
        if has_periods and has_scenarios:
            for p in periods:
                for s in scenarios:
                    slices[(p, s)] = xr.DataArray(
                        get_data(self._results[(p, s)]),
                        dims=base_dims,
                        coords=base_coords,
                    )
        elif has_periods:
            for p in periods:
                slices[(p,)] = xr.DataArray(
                    get_data(self._results[(p,)]),
                    dims=base_dims,
                    coords=base_coords,
                )
        elif has_scenarios:
            for s in scenarios:
                slices[(s,)] = xr.DataArray(
                    get_data(self._results[(s,)]),
                    dims=base_dims,
                    coords=base_coords,
                )

        # Combine slices into multi-dimensional array
        if has_periods and has_scenarios:
            period_arrays = []
            for p in periods:
                scenario_arrays = [slices[(p, s)] for s in scenarios]
                period_arrays.append(xr.concat(scenario_arrays, dim=pd.Index(scenarios, name='scenario')))
            result = xr.concat(period_arrays, dim=pd.Index(periods, name='period'))
        elif has_periods:
            result = xr.concat([slices[(p,)] for p in periods], dim=pd.Index(periods, name='period'))
        else:
            result = xr.concat([slices[(s,)] for s in scenarios], dim=pd.Index(scenarios, name='scenario'))

        # Ensure base dims come first
        dim_order = base_dims + [d for d in result.dims if d not in base_dims]
        return result.transpose(*dim_order).rename(name)

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


class Clustering:
    """Clustering information for a FlowSystem.

    Uses ClusteringResults to manage tsam ClusteringResult objects and provides
    convenience accessors for common operations.

    This is a thin wrapper around tsam 3.0's API. The actual clustering
    logic is delegated to tsam, and this class only:
    1. Manages results for multiple (period, scenario) dimensions via ClusteringResults
    2. Provides xarray-based convenience properties
    3. Handles JSON persistence via ClusteringResults.to_dict()/from_dict()

    Attributes:
        results: ClusteringResults managing ClusteringResult objects for all (period, scenario) combinations.
        original_timesteps: Original timesteps before clustering.
        original_data: Original dataset before clustering (for expand/plotting).
        aggregated_data: Aggregated dataset after clustering (for plotting).

    Example:
        >>> fs_clustered = flow_system.transform.cluster(n_clusters=8, cluster_duration='1D')
        >>> fs_clustered.clustering.n_clusters
        8
        >>> fs_clustered.clustering.cluster_assignments
        <xarray.DataArray (original_cluster: 365)>
        >>> fs_clustered.clustering.plot.compare()
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
        representative value from the aggregated data.

        Args:
            aggregated: DataArray with aggregated (cluster, time) or (time,) dimension.
            original_time: Original time coordinates. Defaults to self.original_timesteps.

        Returns:
            DataArray expanded to original timesteps.
        """
        if original_time is None:
            original_time = self.original_timesteps

        timestep_mapping = self.timestep_mapping
        has_cluster_dim = 'cluster' in aggregated.dims
        timesteps_per_cluster = self.timesteps_per_cluster

        def _expand_slice(mapping: np.ndarray, data: xr.DataArray) -> np.ndarray:
            """Expand a single slice using the mapping."""
            if has_cluster_dim:
                cluster_ids = mapping // timesteps_per_cluster
                time_within = mapping % timesteps_per_cluster
                return data.values[cluster_ids, time_within]
            return data.values[mapping]

        # Simple case: no period/scenario dimensions
        extra_dims = [d for d in timestep_mapping.dims if d != 'original_time']
        if not extra_dims:
            expanded_values = _expand_slice(timestep_mapping.values, aggregated)
            return xr.DataArray(
                expanded_values,
                coords={'time': original_time},
                dims=['time'],
                attrs=aggregated.attrs,
            )

        # Multi-dimensional: expand each slice and recombine
        dim_coords = {d: list(timestep_mapping.coords[d].values) for d in extra_dims}
        expanded_slices = {}
        for combo in np.ndindex(*[len(v) for v in dim_coords.values()]):
            selector = {d: dim_coords[d][i] for d, i in zip(extra_dims, combo, strict=True)}
            mapping = _select_dims(timestep_mapping, **selector).values
            data_slice = (
                _select_dims(aggregated, **selector) if any(d in aggregated.dims for d in selector) else aggregated
            )
            expanded_slices[tuple(selector.values())] = xr.DataArray(
                _expand_slice(mapping, data_slice),
                coords={'time': original_time},
                dims=['time'],
            )

        # Concatenate along extra dimensions
        result_arrays = expanded_slices
        for dim in reversed(extra_dims):
            dim_vals = dim_coords[dim]
            grouped = {}
            for key, arr in result_arrays.items():
                rest_key = key[:-1] if len(key) > 1 else ()
                grouped.setdefault(rest_key, []).append(arr)
            result_arrays = {k: xr.concat(v, dim=pd.Index(dim_vals, name=dim)) for k, v in grouped.items()}
        result = list(result_arrays.values())[0]
        return result.transpose('time', ...).assign_attrs(aggregated.attrs)

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

        if not self.dim_names:
            # Simple case: no extra dimensions
            mapping = _build_timestep_mapping(self.results[()], n_original)
            return xr.DataArray(
                mapping,
                dims=['original_time'],
                coords={'original_time': original_time_coord},
                name='timestep_mapping',
            )

        # Multi-dimensional case: combine slices into multi-dim array
        return self.results._build_multi_dim_array(
            lambda cr: _build_timestep_mapping(cr, n_original),
            base_dims=['original_time'],
            base_coords={'original_time': original_time_coord},
            periods=self.results._get_dim_values('period'),
            scenarios=self.results._get_dim_values('scenario'),
            name='timestep_mapping',
        )

    def _create_reference_structure(self) -> tuple[dict, dict[str, xr.DataArray]]:
        """Create serialization structure for to_dataset().

        Returns:
            Tuple of (reference_dict, arrays_dict).
        """
        arrays = {}

        # Collect original_data arrays
        original_data_refs = None
        if self.original_data is not None:
            original_data_refs = []
            for name, da in self.original_data.data_vars.items():
                ref_name = f'original_data|{name}'
                arrays[ref_name] = da
                original_data_refs.append(f':::{ref_name}')

        # Collect aggregated_data arrays
        aggregated_data_refs = None
        if self.aggregated_data is not None:
            aggregated_data_refs = []
            for name, da in self.aggregated_data.data_vars.items():
                ref_name = f'aggregated_data|{name}'
                arrays[ref_name] = da
                aggregated_data_refs.append(f':::{ref_name}')

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
            '_aggregated_data_refs': aggregated_data_refs,
            '_metrics_refs': metrics_refs,
        }

        return reference, arrays

    def __init__(
        self,
        results: ClusteringResults | dict,
        original_timesteps: pd.DatetimeIndex | list[str],
        original_data: xr.Dataset | None = None,
        aggregated_data: xr.Dataset | None = None,
        _metrics: xr.Dataset | None = None,
        # These are for reconstruction from serialization
        _original_data_refs: list[str] | None = None,
        _aggregated_data_refs: list[str] | None = None,
        _metrics_refs: list[str] | None = None,
    ):
        """Initialize Clustering object.

        Args:
            results: ClusteringResults instance, or dict from to_dict() (for deserialization).
            original_timesteps: Original timesteps before clustering.
            original_data: Original dataset before clustering (for expand/plotting).
            aggregated_data: Aggregated dataset after clustering (for plotting).
            _metrics: Pre-computed metrics dataset.
            _original_data_refs: Internal: resolved DataArrays from serialization.
            _aggregated_data_refs: Internal: resolved DataArrays from serialization.
            _metrics_refs: Internal: resolved DataArrays from serialization.
        """
        # Handle ISO timestamp strings from serialization
        if (
            isinstance(original_timesteps, list)
            and len(original_timesteps) > 0
            and isinstance(original_timesteps[0], str)
        ):
            original_timesteps = pd.DatetimeIndex([pd.Timestamp(ts) for ts in original_timesteps])

        # Handle results as dict (from deserialization)
        if isinstance(results, dict):
            results = ClusteringResults.from_dict(results)

        self.results = results
        self.original_timesteps = original_timesteps
        self._metrics = _metrics

        # Handle reconstructed data from refs (list of DataArrays)
        if _original_data_refs is not None and isinstance(_original_data_refs, list):
            # These are resolved DataArrays from the structure resolver
            if all(isinstance(da, xr.DataArray) for da in _original_data_refs):
                self.original_data = xr.Dataset({da.name: da for da in _original_data_refs})
            else:
                self.original_data = original_data
        else:
            self.original_data = original_data

        if _aggregated_data_refs is not None and isinstance(_aggregated_data_refs, list):
            if all(isinstance(da, xr.DataArray) for da in _aggregated_data_refs):
                self.aggregated_data = xr.Dataset({da.name: da for da in _aggregated_data_refs})
            else:
                self.aggregated_data = aggregated_data
        else:
            self.aggregated_data = aggregated_data

        if _metrics_refs is not None and isinstance(_metrics_refs, list):
            if all(isinstance(da, xr.DataArray) for da in _metrics_refs):
                self._metrics = xr.Dataset({da.name: da for da in _metrics_refs})

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
        color: str | None = 'auto',
        line_dash: str | None = 'representation',
        facet_col: str | None = 'auto',
        facet_row: str | None = 'auto',
        show: bool | None = None,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Compare original vs aggregated data.

        Args:
            kind: Type of comparison plot ('timeseries' or 'duration_curve').
            variables: Variable(s) to plot. None for all time-varying variables.
            select: xarray-style selection dict.
            colors: Color specification.
            color: Dimension for line colors.
            line_dash: Dimension for line dash styles.
            facet_col: Dimension for subplot columns.
            facet_row: Dimension for subplot rows.
            show: Whether to display the figure.
            **plotly_kwargs: Additional arguments passed to plotly.

        Returns:
            PlotResult containing the comparison figure and underlying data.
        """
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
            n = len(values)
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

        line_kwargs = {}
        if line_dash is not None:
            line_kwargs['line_dash'] = line_dash
            if line_dash == 'representation':
                line_kwargs['line_dash_map'] = {'Original': 'dot', 'Clustered': 'solid'}

        fig = ds.fxplot.line(
            colors=colors,
            color=color,
            title=title,
            facet_col=facet_col,
            facet_row=facet_row,
            **line_kwargs,
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
        facet_col: str | None = 'auto',
        animation_frame: str | None = 'auto',
        show: bool | None = None,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot cluster assignments over time as a heatmap timeline."""
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

        heatmap_da = cluster_da.expand_dims('y', axis=-1).assign_coords(y=['Cluster'])
        heatmap_da.name = 'cluster_assignment'
        heatmap_da = heatmap_da.transpose('time', 'y', ...)

        fig = heatmap_da.fxplot.heatmap(
            colors=colors,
            title='Cluster Assignments',
            facet_col=facet_col,
            animation_frame=animation_frame,
            aspect='auto',
            **plotly_kwargs,
        )

        fig.update_yaxes(showticklabels=False)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))

        cluster_da.name = 'cluster'
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
        color: str | None = 'auto',
        facet_col: str | None = 'cluster',
        facet_cols: int | None = None,
        show: bool | None = None,
        **plotly_kwargs: Any,
    ) -> PlotResult:
        """Plot each cluster's typical period profile."""
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
        title = 'Clusters' if len(resolved_variables) > 1 else f'Clusters: {resolved_variables[0]}'

        fig = ds.fxplot.line(
            colors=colors,
            color=color,
            title=title,
            facet_col=facet_col,
            facet_cols=facet_cols,
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


# Backwards compatibility - keep these names for existing code
# TODO: Remove after migration
ClusteringResultCollection = Clustering  # Alias for backwards compat


def _register_clustering_classes():
    """Register clustering classes for IO."""
    from ..structure import CLASS_REGISTRY

    CLASS_REGISTRY['Clustering'] = Clustering
