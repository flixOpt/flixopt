"""
Clustering classes for time series aggregation.

This module provides the `Clustering` class stored on FlowSystem after clustering,
wrapping tsam_xarray's ClusteringInfo for structure access and AggregationResult
for full data access (pre-serialization only).
"""

from __future__ import annotations

import functools
import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from pathlib import Path

    from tsam import ClusteringResult as TsamClusteringResult
    from tsam_xarray import AggregationResult as TsamXarrayAggregationResult
    from tsam_xarray import ClusteringInfo

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


def _build_property_array(
    clustering_info: ClusteringInfo,
    get_data: callable,
    base_dims: list[str],
    base_coords: dict | None = None,
    name: str | None = None,
    unrename_map: dict[str, str] | None = None,
) -> xr.DataArray:
    """Build a DataArray property from per-slice ClusteringResult data.

    Used for custom properties not provided by ClusteringInfo (e.g., timestep_mapping).
    """
    dim_names = clustering_info.slice_dims
    results = clustering_info.clusterings

    slices = []
    for key, cr in results.items():
        da = xr.DataArray(get_data(cr), dims=base_dims, coords=base_coords or {}, name=name)
        for dim_name, coord_val in zip(dim_names, key, strict=True):
            da = da.expand_dims({dim_name: [coord_val]})
        slices.append(da)

    if len(slices) == 1:
        result = slices[0]
    else:
        combined = xr.combine_by_coords(slices)
        if isinstance(combined, xr.Dataset):
            result = combined[name]
        else:
            result = combined
    result = result.transpose(*base_dims, *dim_names)

    # Unrename dims (e.g., _period -> period)
    if unrename_map:
        renames = {k: v for k, v in unrename_map.items() if k in result.dims}
        if renames:
            result = result.rename(renames)

    return result


class Clustering:
    """Clustering information for a FlowSystem.

    Wraps tsam_xarray's ClusteringInfo for structure access and optionally
    AggregationResult for full data access (pre-serialization only).

    Attributes:
        original_timesteps: Original timesteps before clustering.
        dims: Dimension names, e.g., ('period', 'scenario').
        coords: Coordinate values, e.g., {'period': [2024, 2025]}.

    Example:
        >>> clustering = fs_clustered.clustering
        >>> clustering.n_clusters
        8
        >>> clustering.dims
        ('period',)

        # Access tsam_xarray AggregationResult for detailed analysis
        >>> clustering.aggregation_result.cluster_representatives  # DataArray
        >>> clustering.aggregation_result.accuracy  # AccuracyMetrics
    """

    def __init__(
        self,
        clustering_info: ClusteringInfo | dict | None = None,
        original_timesteps: pd.DatetimeIndex | list[str] | None = None,
        original_data: xr.Dataset | None = None,
        aggregated_data: xr.Dataset | None = None,
        _metrics: xr.Dataset | None = None,
        # These are for reconstruction from serialization
        _original_data_refs: list[str] | None = None,
        _metrics_refs: list[str] | None = None,
        # Internal: tsam_xarray AggregationResult for full data access
        _aggregation_result: TsamXarrayAggregationResult | None = None,
        # Internal: mapping from renamed dims back to originals (e.g., _period -> period)
        _unrename_map: dict[str, str] | None = None,
        # Legacy: accept 'results' kwarg for backwards compatibility during transition
        results: Any = None,
    ):
        from tsam_xarray import ClusteringInfo as ClusteringInfoClass

        # Handle ISO timestamp strings from serialization
        if (
            isinstance(original_timesteps, list)
            and len(original_timesteps) > 0
            and isinstance(original_timesteps[0], str)
        ):
            original_timesteps = pd.DatetimeIndex([pd.Timestamp(ts) for ts in original_timesteps])

        # Store tsam_xarray AggregationResult if provided (full data access)
        self._aggregation_result = _aggregation_result

        # Resolve ClusteringInfo from various sources
        if clustering_info is not None:
            if isinstance(clustering_info, dict):
                self._clustering_info = self._clustering_info_from_dict(clustering_info)
            else:
                self._clustering_info = clustering_info
        elif _aggregation_result is not None:
            self._clustering_info = _aggregation_result.clustering
        elif results is not None:
            # Legacy path: accept old ClusteringResults or dict
            if isinstance(results, dict):
                self._clustering_info = self._clustering_info_from_dict(results)
            elif hasattr(results, '_results') and hasattr(results, '_dim_names'):
                self._clustering_info = ClusteringInfoClass(
                    time_dim='time',
                    cluster_dim=['variable'],
                    slice_dims=list(results._dim_names),
                    clusterings=dict(results._results),
                )
            else:
                raise TypeError(f'Cannot create ClusteringInfo from {type(results)}')
        else:
            raise ValueError('Either clustering_info or _aggregation_result must be provided')

        # Resolve unrename_map: if not explicitly provided, infer from slice_dims
        # (e.g., '_period' in slice_dims → {'_period': 'period'})
        if _unrename_map:
            self._unrename_map = _unrename_map
        else:
            known_renames = {'_period': 'period', '_cluster': 'cluster'}
            self._unrename_map = {k: v for k, v in known_renames.items() if k in self._clustering_info.slice_dims}

        # Flag indicating this was loaded from serialization (missing full AggregationResult data)
        self._from_serialization = _aggregation_result is None

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

    @staticmethod
    def _clustering_info_from_dict(d: dict) -> ClusteringInfo:
        """Create ClusteringInfo from serialized dict."""
        from tsam_xarray import ClusteringInfo as ClusteringInfoClass

        return ClusteringInfoClass.from_dict(d)

    # ==========================================================================
    # Helper for dim unrenaming
    # ==========================================================================

    def _unrename(self, da: xr.DataArray) -> xr.DataArray:
        """Rename tsam_xarray output dims back to original names (e.g., _period -> period)."""
        if not self._unrename_map:
            return da
        renames = {k: v for k, v in self._unrename_map.items() if k in da.dims}
        return da.rename(renames) if renames else da

    # ==========================================================================
    # Core properties (delegated to ClusteringInfo)
    # ==========================================================================

    @property
    def clustering_info(self) -> ClusteringInfo:
        """tsam_xarray ClusteringInfo for reuse with apply_clustering()."""
        return self._clustering_info

    @property
    def n_clusters(self) -> int:
        """Number of clusters (typical periods)."""
        return self._clustering_info.n_clusters

    @property
    def timesteps_per_cluster(self) -> int:
        """Number of timesteps in each cluster."""
        return self._clustering_info.n_timesteps_per_period

    @property
    def n_original_clusters(self) -> int:
        """Number of original periods (before clustering)."""
        return self._clustering_info.n_original_periods

    @property
    def n_segments(self) -> int | None:
        """Number of segments per cluster, or None if not segmented."""
        return self._clustering_info.n_segments

    @property
    def is_segmented(self) -> bool:
        """Whether intra-period segmentation was used."""
        return self._clustering_info.n_segments is not None

    @property
    def dim_names(self) -> list[str]:
        """Names of extra dimensions, e.g., ['period', 'scenario']."""
        return [self._unrename_map.get(d, d) for d in self._clustering_info.slice_dims]

    @property
    def dims(self) -> tuple[str, ...]:
        """Dimension names as tuple (xarray-like)."""
        return tuple(self.dim_names)

    @property
    def coords(self) -> dict[str, list]:
        """Coordinate values for each dimension (xarray-like)."""
        raw_dims = self._clustering_info.slice_dims
        result = {}
        # Get unique values per dim from the clusterings dict keys
        for i, dim in enumerate(raw_dims):
            values = list(dict.fromkeys(k[i] for k in self._clustering_info.clusterings.keys()))
            display_name = self._unrename_map.get(dim, dim)
            result[display_name] = values
        return result

    # ==========================================================================
    # DataArray properties (delegated to ClusteringInfo with unrename)
    # ==========================================================================

    @property
    def cluster_assignments(self) -> xr.DataArray:
        """Mapping from original periods to cluster IDs.

        Returns:
            DataArray with dims [original_cluster, period?, scenario?].
        """
        da = self._clustering_info.cluster_assignments
        # Rename tsam_xarray's 'period' dim to our 'original_cluster' convention
        # (must happen before _unrename to avoid conflict with _period → period rename)
        if 'period' in da.dims:
            da = da.rename({'period': 'original_cluster'})
        da = self._unrename(da)
        # Ensure original_cluster is first dim (tsam_xarray puts slice dims first)
        if 'original_cluster' in da.dims and da.dims[0] != 'original_cluster':
            other_dims = [d for d in da.dims if d != 'original_cluster']
            da = da.transpose('original_cluster', *other_dims)
        return da

    @property
    def cluster_occurrences(self) -> xr.DataArray:
        """How many original clusters map to each typical cluster.

        Returns:
            DataArray with dims [cluster, period?, scenario?].
        """
        return self._unrename(self._clustering_info.cluster_occurrences)

    @property
    def cluster_centers(self) -> xr.DataArray:
        """Which original period is the representative (center) for each cluster.

        Returns:
            DataArray with dims [cluster, period?, scenario?].
        """
        return self._unrename(self._clustering_info.cluster_centers)

    @property
    def segment_assignments(self) -> xr.DataArray | None:
        """For each timestep within a cluster, which segment it belongs to.

        Returns:
            DataArray with dims [cluster, time, period?, scenario?], or None if not segmented.
        """
        result = self._clustering_info.segment_assignments
        if result is None:
            return None
        # tsam_xarray uses 'timestep', we use 'time'
        if 'timestep' in result.dims:
            result = result.rename({'timestep': 'time'})
        return self._unrename(result)

    @property
    def segment_durations(self) -> xr.DataArray | None:
        """Duration of each segment in timesteps.

        Returns:
            DataArray with dims [cluster, segment, period?, scenario?], or None if not segmented.
        """
        result = self._clustering_info.segment_durations
        if result is None:
            return None
        # tsam_xarray uses 'timestep', we use 'segment'
        if 'timestep' in result.dims:
            result = result.rename({'timestep': 'segment'})
        return self._unrename(result)

    @property
    def segment_centers(self) -> xr.DataArray | None:
        """Center of each intra-period segment.

        Returns:
            DataArray or None if no segmentation.
        """
        result = self._clustering_info.segment_centers
        if result is None:
            return None
        # tsam_xarray uses 'timestep', we use 'segment'
        if 'timestep' in result.dims:
            result = result.rename({'timestep': 'segment'})
        return self._unrename(result)

    @property
    def n_representatives(self) -> int:
        """Number of representative timesteps after clustering."""
        if self.is_segmented:
            return self.n_clusters * self.n_segments
        return self.n_clusters * self.timesteps_per_cluster

    @property
    def representative_weights(self) -> xr.DataArray:
        """Weight for each cluster (number of original periods it represents).

        Used as cluster_weight in FlowSystem.
        """
        return self.cluster_occurrences.rename('representative_weights')

    # ==========================================================================
    # Custom properties (not in ClusteringInfo)
    # ==========================================================================

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

    @functools.cached_property
    def timestep_mapping(self) -> xr.DataArray:
        """Mapping from original timesteps to representative timestep indices.

        Each value indicates which representative timestep index (0 to n_representatives-1)
        corresponds to each original timestep.

        Note: This property is cached for performance since it's accessed frequently
        during expand() operations.
        """
        n_original = len(self.original_timesteps)
        original_time_coord = self.original_timesteps.rename('original_time')
        return _build_property_array(
            self._clustering_info,
            lambda cr: _build_timestep_mapping(cr, n_original),
            base_dims=['original_time'],
            base_coords={'original_time': original_time_coord},
            name='timestep_mapping',
            unrename_map=self._unrename_map,
        )

    @property
    def metrics(self) -> xr.Dataset:
        """Clustering quality metrics (RMSE, MAE, etc.).

        Returns:
            Dataset with dims [time_series, period?, scenario?], or empty Dataset if no metrics.
        """
        if self._metrics is None:
            return xr.Dataset()
        return self._metrics

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

        # Align timestep_mapping coordinates with aggregated data to prevent
        # coordinate conflicts during isel (tsam_xarray may sort coords differently)
        shared_dims = set(timestep_mapping.dims) & set(aggregated.dims)
        for dim in shared_dims:
            if dim in timestep_mapping.coords and dim in aggregated.coords:
                timestep_mapping = timestep_mapping.reindex({dim: aggregated.coords[dim]})

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
        segment_durations = self.segment_durations  # [cluster, segment, period?, scenario?]

        # Align coordinates to prevent conflicts during isel
        shared_dims = set(timestep_mapping.dims) & set(segment_durations.dims)
        for dim in shared_dims:
            if dim in timestep_mapping.coords and dim in segment_durations.coords:
                segment_durations = segment_durations.reindex({dim: timestep_mapping.coords[dim]})

        # Decode cluster and segment indices from timestep_mapping
        # For segmented systems, encoding is: cluster_id * n_segments + segment_idx
        time_dim_size = self.n_segments
        cluster_indices = timestep_mapping // time_dim_size
        segment_indices = timestep_mapping % time_dim_size  # This IS the segment index

        # Get duration for each segment directly
        # segment_durations[cluster, segment] -> duration
        divisor = segment_durations.isel(cluster=cluster_indices, segment=segment_indices)

        # Clean up coordinates and rename
        divisor = divisor.drop_vars(['cluster', 'time', 'segment'], errors='ignore')
        divisor = divisor.rename({'original_time': 'time'}).assign_coords(time=original_time)

        return divisor.transpose('time', ...).rename('expansion_divisor')

    def apply(
        self,
        data: xr.DataArray,
    ) -> TsamXarrayAggregationResult:
        """Apply the saved clustering to new data.

        Args:
            data: DataArray with time series data to cluster.

        Returns:
            tsam_xarray AggregationResult with the clustering applied.
        """
        return self._clustering_info.apply(data)

    # ==========================================================================
    # Serialization
    # ==========================================================================

    def to_json(self, path: str | Path) -> None:
        """Save the clustering for reuse.

        Can be loaded later with Clustering.from_json() and used with
        flow_system.transform.apply_clustering().

        Args:
            path: Path to save the JSON file.
        """
        data = {
            'clustering_info': self._clustering_info.to_dict(),
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

        The loaded Clustering has full apply() support because ClusteringInfo
        is fully preserved via serialization.

        Args:
            path: Path to the JSON file.
            original_timesteps: Original timesteps for the new FlowSystem.
                If None, uses the timesteps stored in the JSON.

        Returns:
            A Clustering that can be used with apply_clustering().
        """
        with open(path) as f:
            data = json.load(f)

        # Support both new format (clustering_info) and legacy format (results)
        if 'clustering_info' in data:
            clustering_info = data['clustering_info']
        elif 'results' in data:
            clustering_info = data['results']  # Legacy format, handled by __init__
        else:
            raise ValueError('JSON file must contain "clustering_info" or "results" key')

        if original_timesteps is None:
            original_timesteps = pd.DatetimeIndex([pd.Timestamp(ts) for ts in data['original_timesteps']])

        return cls(
            clustering_info=clustering_info,
            original_timesteps=original_timesteps,
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
            # Use variables for faster access (avoids _construct_dataarray overhead)
            variables = self.original_data.variables
            for name in self.original_data.data_vars:
                var = variables[name]
                ref_name = f'original_data|{name}'
                # Rename time dim to avoid xarray alignment issues
                if 'time' in var.dims:
                    new_dims = tuple('original_time' if d == 'time' else d for d in var.dims)
                    arrays[ref_name] = xr.Variable(new_dims, var.values, attrs=var.attrs)
                else:
                    arrays[ref_name] = var
                original_data_refs.append(f':::{ref_name}')

        # NOTE: aggregated_data is NOT serialized - it's identical to the FlowSystem's
        # main data arrays and would be redundant. After loading, aggregated_data is
        # reconstructed from the FlowSystem's dataset.

        # Collect metrics arrays
        metrics_refs = None
        if self._metrics is not None:
            metrics_refs = []
            # Use variables for faster access (avoids _construct_dataarray overhead)
            metrics_vars = self._metrics.variables
            for name in self._metrics.data_vars:
                ref_name = f'metrics|{name}'
                arrays[ref_name] = metrics_vars[name]
                metrics_refs.append(f':::{ref_name}')

        reference = {
            '__class__': 'Clustering',
            'clustering_info': self._clustering_info.to_dict(),
            'original_timesteps': [ts.isoformat() for ts in self.original_timesteps],
            '_original_data_refs': original_data_refs,
            '_metrics_refs': metrics_refs,
        }

        return reference, arrays

    # ==========================================================================
    # Access to tsam_xarray AggregationResult
    # ==========================================================================

    @property
    def aggregation_result(self) -> TsamXarrayAggregationResult:
        """The tsam_xarray AggregationResult for full data access.

        Only available before serialization. After loading from file,
        use clustering_info for structure-only access.

        Raises:
            ValueError: If accessed on a Clustering loaded from JSON/NetCDF.
        """
        self._require_full_data('aggregation_result')
        return self._aggregation_result

    def __len__(self) -> int:
        """Number of (period, scenario) combinations."""
        return len(self._clustering_info.clusterings)

    def _require_full_data(self, operation: str) -> None:
        """Raise error if full AggregationResult data is not available."""
        if self._from_serialization or self._aggregation_result is None:
            raise ValueError(
                f'{operation} requires full AggregationResult data, '
                f'but this Clustering was loaded from JSON. '
                f'Use apply_clustering() to get full results.'
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
            # Use variables for faster access (avoids _construct_dataarray overhead)
            variables = ds.variables
            rep_values = ds.coords['representation'].values
            rep_idx = {rep: i for i, rep in enumerate(rep_values)}
            for var in ds.data_vars:
                data = variables[var].values
                for rep in rep_values:
                    # Direct numpy indexing instead of .sel()
                    values = np.sort(data[rep_idx[rep]].flatten())[::-1]
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
        """Get list of time-varying variables from original data that also exist in aggregated data."""
        if self._clustering.original_data is None:
            return []
        # Get variables that exist in both original and aggregated data
        aggregated_vars = (
            set(self._clustering.aggregated_data.data_vars)
            if self._clustering.aggregated_data is not None
            else set(self._clustering.original_data.data_vars)
        )
        return [
            name
            for name in self._clustering.original_data.data_vars
            if name in aggregated_vars
            and 'time' in self._clustering.original_data[name].dims
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


def _register_clustering_classes():
    """Register clustering classes for IO."""
    from ..structure import CLASS_REGISTRY

    CLASS_REGISTRY['Clustering'] = Clustering
