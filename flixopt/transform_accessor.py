"""
Transform accessor for FlowSystem.

This module provides the TransformAccessor class that enables
transformations on FlowSystem like clustering, selection, and resampling.
"""

from __future__ import annotations

import functools
import logging
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import xarray as xr

from .model_coordinates import ModelCoordinates
from .modeling import _scalar_safe_reduce
from .structure import NAME_TO_EXPANSION, ExpansionMode

if TYPE_CHECKING:
    from tsam import ClusterConfig, ExtremeConfig, SegmentConfig

    from .clustering import Clustering
    from .flow_system import FlowSystem

logger = logging.getLogger('flixopt')


def _combine_dataarray_slices(
    slices: list[xr.DataArray],
    base_dims: list[str],
    extra_dims: list[str],
    name: str | None = None,
) -> xr.DataArray:
    """Combine DataArray slices with extra dimensions into a single DataArray.

    Args:
        slices: List of DataArrays, each with extra dims already expanded.
        base_dims: Base dimension names (e.g., ['cluster', 'time']).
        extra_dims: Extra dimension names (e.g., ['period', 'scenario']).
        name: Optional name for the result.

    Returns:
        Combined DataArray with dims [*base_dims, *extra_dims].
    """
    if len(slices) == 1:
        result = slices[0]
    else:
        combined = xr.combine_by_coords(slices)
        # combine_by_coords returns Dataset when DataArrays have names
        if isinstance(combined, xr.Dataset):
            result = list(combined.data_vars.values())[0]
        else:
            result = combined

    # Ensure consistent dimension order for both single and multi-slice paths
    result = result.transpose(*base_dims, *extra_dims)

    if name is not None:
        result = result.rename(name)
    return result


def _expand_dims_for_key(da: xr.DataArray, dim_names: list[str], key: tuple) -> xr.DataArray:
    """Add dimensions to a DataArray based on key values.

    Args:
        da: DataArray without extra dimensions.
        dim_names: Names of dimensions to add (e.g., ['period', 'scenario']).
        key: Tuple of coordinate values matching dim_names.

    Returns:
        DataArray with extra dimensions added.
    """
    for dim_name, coord_val in zip(dim_names, key, strict=True):
        da = da.expand_dims({dim_name: [coord_val]})
    return da


class _ReducedFlowSystemBuilder:
    """Builds a reduced FlowSystem from tsam aggregation results.

    This class encapsulates the construction of reduced FlowSystem datasets,
    pre-computing shared coordinates and providing methods for building
    each component (weights, typical periods, segment durations, metrics).

    Args:
        fs: The original FlowSystem being reduced.
        aggregation_results: Dict mapping key tuples to tsam AggregationResult.
        timesteps_per_cluster: Number of timesteps per cluster.
        dt: Hours per timestep.
        dim_names: Names of extra dimensions (e.g., ['period', 'scenario']).
    """

    def __init__(
        self,
        fs: FlowSystem,
        aggregation_results: dict[tuple, Any],
        timesteps_per_cluster: int,
        dt: float,
        dim_names: list[str],
    ):
        self._fs = fs
        self._aggregation_results = aggregation_results
        self._timesteps_per_cluster = timesteps_per_cluster
        self._dt = dt
        self._dim_names = dim_names

        # Extract info from first result (all should be consistent)
        first_result = next(iter(aggregation_results.values()))
        self._n_reduced_timesteps = len(first_result.cluster_representatives)
        self._n_clusters = len(first_result.cluster_weights)
        self._is_segmented = first_result.n_segments is not None
        self._n_segments = first_result.n_segments

        # Pre-compute coordinates
        self._cluster_coords = np.arange(self._n_clusters)

        if self._is_segmented:
            self._n_time_points = self._n_segments
            self._time_coords = pd.RangeIndex(self._n_segments, name='time')
        else:
            self._n_time_points = timesteps_per_cluster
            self._time_coords = pd.date_range(
                start='2000-01-01',
                periods=timesteps_per_cluster,
                freq=pd.Timedelta(hours=dt),
                name='time',
            )

        self._base_coords = {'cluster': self._cluster_coords, 'time': self._time_coords}

    def _expand_and_combine(
        self,
        data_per_key: dict[tuple, xr.DataArray],
        base_dims: list[str],
        name: str | None = None,
    ) -> xr.DataArray:
        """Expand dims for each key and combine slices.

        Args:
            data_per_key: Dict mapping keys to DataArrays without extra dims.
            base_dims: Base dimension names (e.g., ['cluster'] or ['cluster', 'time']).
            name: Optional name for the result.

        Returns:
            Combined DataArray with dims [*base_dims, *dim_names].
        """
        slices = [_expand_dims_for_key(da, self._dim_names, key) for key, da in data_per_key.items()]
        return _combine_dataarray_slices(slices, base_dims, self._dim_names, name=name)

    def build_cluster_weights(self) -> xr.DataArray:
        """Build cluster_weight DataArray from aggregation results.

        Returns:
            DataArray with dims [cluster, *dim_names].
        """
        data_per_key = {}
        for key, result in self._aggregation_results.items():
            weights = np.array([result.cluster_weights.get(c, 0) for c in range(self._n_clusters)])
            data_per_key[key] = xr.DataArray(weights, dims=['cluster'], coords={'cluster': self._cluster_coords})
        return self._expand_and_combine(data_per_key, ['cluster'], name='cluster_weight')

    def build_typical_periods(self) -> dict[str, xr.DataArray]:
        """Build typical periods DataArrays with (cluster, time, *dim_names) shape.

        Returns:
            Dict mapping column names to combined DataArrays.
        """
        column_slices: dict[str, dict[tuple, xr.DataArray]] = {}

        for key, tsam_result in self._aggregation_results.items():
            typical_df = tsam_result.cluster_representatives
            if self._is_segmented:
                columns = typical_df.columns.tolist()
                reshaped = typical_df.values.reshape(self._n_clusters, self._n_time_points, -1)
                for col_idx, col in enumerate(columns):
                    da = xr.DataArray(reshaped[:, :, col_idx], dims=['cluster', 'time'], coords=self._base_coords)
                    column_slices.setdefault(col, {})[key] = da
            else:
                for col in typical_df.columns:
                    reshaped = typical_df[col].values.reshape(self._n_clusters, self._n_time_points)
                    da = xr.DataArray(reshaped, dims=['cluster', 'time'], coords=self._base_coords)
                    column_slices.setdefault(col, {})[key] = da

        return {
            col: self._expand_and_combine(data_per_key, ['cluster', 'time'])
            for col, data_per_key in column_slices.items()
        }

    def build_segment_durations(self) -> xr.DataArray:
        """Build timestep_duration DataArray from segment durations.

        Returns:
            DataArray with dims [cluster, time, *dim_names].

        Raises:
            ValueError: If not a segmented system.
        """
        if not self._is_segmented:
            raise ValueError('build_segment_durations() requires a segmented system')

        data_per_key = {}
        for key, tsam_result in self._aggregation_results.items():
            seg_durs = tsam_result.segment_durations
            data = np.array(
                [[seg_durs[c][s] * self._dt for s in range(self._n_segments)] for c in range(self._n_clusters)]
            )
            data_per_key[key] = xr.DataArray(data, dims=['cluster', 'time'], coords=self._base_coords)

        return self._expand_and_combine(data_per_key, ['cluster', 'time'], name='timestep_duration')

    def build_metrics(self) -> xr.Dataset:
        """Build clustering metrics Dataset from aggregation results.

        Returns:
            Dataset with RMSE, MAE, RMSE_duration metrics.
        """
        # Convert accuracy to DataFrames, filtering out failures
        metrics_dfs: dict[tuple, pd.DataFrame] = {}
        for key, result in self._aggregation_results.items():
            try:
                metrics_dfs[key] = _accuracy_to_dataframe(result.accuracy)
            except Exception as e:
                logger.warning(f'Failed to compute clustering metrics for {key}: {e}')
                metrics_dfs[key] = pd.DataFrame()

        non_empty_metrics = {k: v for k, v in metrics_dfs.items() if not v.empty}

        if not non_empty_metrics:
            return xr.Dataset()

        # Single slice case
        if len(metrics_dfs) == 1 and len(non_empty_metrics) == 1:
            metrics_df = next(iter(non_empty_metrics.values()))
            return xr.Dataset(
                {
                    col: xr.DataArray(
                        metrics_df[col].values,
                        dims=['time_series'],
                        coords={'time_series': metrics_df.index},
                    )
                    for col in metrics_df.columns
                }
            )

        # Multi-dim case - all periods have same time series
        sample_df = next(iter(non_empty_metrics.values()))
        time_series_index = list(sample_df.index)
        data_vars = {}

        for metric in sample_df.columns:
            data_per_key = {}
            for key, df in metrics_dfs.items():
                values = np.full(len(time_series_index), np.nan) if df.empty else df[metric].values
                data_per_key[key] = xr.DataArray(
                    values, dims=['time_series'], coords={'time_series': time_series_index}
                )
            data_vars[metric] = self._expand_and_combine(data_per_key, ['time_series'], name=metric)

        return xr.Dataset(data_vars)

    def build_reduced_dataset(self, ds: xr.Dataset, typical_das: dict[str, xr.DataArray]) -> xr.Dataset:
        """Build the reduced dataset with (cluster, time) structure.

        Args:
            ds: Original dataset.
            typical_das: Pre-combined DataArrays from build_typical_periods().

        Returns:
            Dataset with reduced timesteps and (cluster, time) structure.
        """
        from .core import TimeSeriesData

        ds_new_vars = {}
        variables = ds.variables
        coord_cache = {k: ds.coords[k].values for k in ds.coords}

        for name in ds.data_vars:
            var = variables[name]
            if 'time' not in var.dims:
                # No time dimension - copy as-is
                coords = {d: coord_cache[d] for d in var.dims if d in coord_cache}
                ds_new_vars[name] = xr.DataArray(var.values, dims=var.dims, coords=coords, attrs=var.attrs, name=name)
            elif name not in typical_das:
                # Time-dependent but constant: reshape to (cluster, time, ...)
                time_idx = var.dims.index('time')
                slices = [slice(None)] * len(var.dims)
                slices[time_idx] = slice(0, self._n_reduced_timesteps)
                sliced_values = var.values[tuple(slices)]

                other_dims = [d for d in var.dims if d != 'time']
                other_shape = [var.sizes[d] for d in other_dims]
                new_shape = [self._n_clusters, self._n_time_points] + other_shape
                reshaped = sliced_values.reshape(new_shape)
                new_coords = dict(self._base_coords)
                for dim in other_dims:
                    if dim in coord_cache:
                        new_coords[dim] = coord_cache[dim]
                ds_new_vars[name] = xr.DataArray(
                    reshaped,
                    dims=['cluster', 'time'] + other_dims,
                    coords=new_coords,
                    attrs=var.attrs,
                )
            else:
                # Time-varying: use pre-combined DataArray from typical_das
                da = typical_das[name].assign_attrs(var.attrs)
                if var.attrs.get('__timeseries_data__', False):
                    da = TimeSeriesData.from_dataarray(da)
                ds_new_vars[name] = da

        # Copy attrs but remove cluster_weight
        new_attrs = dict(ds.attrs)
        new_attrs.pop('cluster_weight', None)
        return xr.Dataset(ds_new_vars, attrs=new_attrs)

    def build(self, ds: xr.Dataset) -> FlowSystem:
        """Build the complete reduced FlowSystem.

        Args:
            ds: Original dataset.

        Returns:
            Reduced FlowSystem with clustering metadata attached.
        """
        from .clustering import Clustering
        from .core import drop_constant_arrays
        from .flow_system import FlowSystem

        # Build all components
        cluster_weight = self.build_cluster_weights()
        typical_das = self.build_typical_periods()
        metrics = self.build_metrics()
        ds_new = self.build_reduced_dataset(ds, typical_das)

        # Add segment durations if segmented
        if self._is_segmented:
            ds_new['timestep_duration'] = self.build_segment_durations()

        # Log reduction
        if self._is_segmented:
            logger.info(
                f'Reduced from {len(self._fs.timesteps)} to {self._n_clusters} clusters × {self._n_segments} segments'
            )
        else:
            logger.info(
                f'Reduced from {len(self._fs.timesteps)} to '
                f'{self._n_clusters} clusters × {self._timesteps_per_cluster} timesteps'
            )

        # Create FlowSystem
        reduced_fs = FlowSystem.from_dataset(ds_new)
        reduced_fs.cluster_weight = cluster_weight

        # Remove 'equals_final' from storages - doesn't make sense on reduced timesteps
        for storage in reduced_fs.storages.values():
            ics = storage.initial_charge_state
            if isinstance(ics, str) and ics == 'equals_final':
                storage.initial_charge_state = None

        # Create Clustering object with full AggregationResult access
        reduced_fs.clustering = Clustering(
            original_timesteps=self._fs.timesteps,
            original_data=drop_constant_arrays(ds, dim='time'),
            aggregated_data=drop_constant_arrays(ds_new, dim='time'),
            _metrics=metrics if metrics.data_vars else None,
            _aggregation_results=self._aggregation_results,
            _dim_names=self._dim_names,
        )

        return reduced_fs


def _accuracy_to_dataframe(accuracy: Any) -> pd.DataFrame:
    """Convert tsam ClusteringAccuracy to DataFrame with metrics.

    Args:
        accuracy: tsam ClusteringAccuracy object.

    Returns:
        DataFrame with RMSE, MAE, RMSE_duration columns indexed by time series name.
    """
    return pd.DataFrame(
        {
            'RMSE': accuracy.rmse,
            'MAE': accuracy.mae,
            'RMSE_duration': accuracy.rmse_duration,
        }
    )


class _Expander:
    """Handles expansion of clustered FlowSystem to original timesteps.

    This class encapsulates all expansion logic, pre-computing shared state
    once and providing methods for different expansion strategies.

    Args:
        fs: The clustered FlowSystem to expand.
        clustering: The Clustering object with cluster assignments and metadata.
    """

    def __init__(self, fs: FlowSystem, clustering: Clustering):
        self._fs = fs
        self._clustering = clustering

        # Pre-compute clustering dimensions
        self._timesteps_per_cluster = clustering.timesteps_per_cluster
        self._n_segments = clustering.n_segments
        self._time_dim_size = self._n_segments if self._n_segments else self._timesteps_per_cluster
        self._n_clusters = clustering.n_clusters
        self._n_original_clusters = clustering.n_original_clusters

        # Pre-compute timesteps
        self._original_timesteps = clustering.original_timesteps
        self._n_original_timesteps = len(self._original_timesteps)

        from .model_coordinates import ModelCoordinates

        self._original_timesteps_extra = ModelCoordinates._create_timesteps_with_extra(self._original_timesteps, None)

        # Index of last valid original cluster (for final state)
        self._last_original_cluster_idx = min(
            (self._n_original_timesteps - 1) // self._timesteps_per_cluster,
            self._n_original_clusters - 1,
        )

        # Build consume vars for intercluster post-processing
        from .structure import InterclusterStorageVarName

        soc_boundary_suffix = InterclusterStorageVarName.SOC_BOUNDARY
        solution_names = set(fs.solution)
        self._consume_vars: set[str] = {
            s for s in solution_names if s == soc_boundary_suffix or s.endswith(soc_boundary_suffix)
        }

        # Build expansion divisor for segmented systems
        self._expansion_divisor = None
        if clustering.is_segmented:
            self._expansion_divisor = clustering.build_expansion_divisor(original_time=self._original_timesteps)

    @functools.cached_property
    def _original_period_indices(self) -> np.ndarray:
        """Original period index for each original timestep."""
        return np.minimum(
            np.arange(self._n_original_timesteps) // self._timesteps_per_cluster,
            self._n_original_clusters - 1,
        )

    @functools.cached_property
    def _positions_in_period(self) -> np.ndarray:
        """Position within period for each original timestep."""
        return np.arange(self._n_original_timesteps) % self._timesteps_per_cluster

    @functools.cached_property
    def _original_period_da(self) -> xr.DataArray:
        """DataArray of original period indices."""
        return xr.DataArray(self._original_period_indices, dims=['original_time'])

    @functools.cached_property
    def _cluster_indices_per_timestep(self) -> xr.DataArray:
        """Cluster index for each original timestep."""
        return self._clustering.cluster_assignments.isel(original_cluster=self._original_period_da)

    @staticmethod
    def _get_mode(var_name: str) -> ExpansionMode:
        """Look up expansion mode for a variable name."""
        return NAME_TO_EXPANSION.get(var_name, ExpansionMode.REPEAT)

    def _append_final_state(self, expanded: xr.DataArray, da: xr.DataArray) -> xr.DataArray:
        """Append final state value from original data to expanded data."""
        cluster_assignments = self._clustering.cluster_assignments
        if cluster_assignments.ndim == 1:
            last_cluster = int(cluster_assignments.values[self._last_original_cluster_idx])
            extra_val = da.isel(cluster=last_cluster, time=-1)
        else:
            last_clusters = cluster_assignments.isel(original_cluster=self._last_original_cluster_idx)
            extra_val = da.isel(cluster=last_clusters, time=-1)
        extra_val = extra_val.drop_vars(['cluster', 'time'], errors='ignore')
        extra_val = extra_val.expand_dims(time=[self._original_timesteps_extra[-1]])
        return xr.concat([expanded, extra_val], dim='time')

    def _interpolate_charge_state_segmented(self, da: xr.DataArray) -> xr.DataArray:
        """Interpolate charge_state values within segments for segmented systems.

        For segmented systems, charge_state has values at segment boundaries (n_segments+1).
        This method interpolates between start and end boundary values to show the
        actual charge trajectory as the storage charges/discharges.

        Args:
            da: charge_state DataArray with dims (cluster, time) where time has n_segments+1 entries.

        Returns:
            Interpolated charge_state with dims (time, ...) for original timesteps.
        """
        clustering = self._clustering

        # Get multi-dimensional properties from Clustering
        segment_assignments = clustering.results.segment_assignments
        segment_durations = clustering.results.segment_durations
        position_within_segment = clustering.results.position_within_segment

        # Use cached period-to-cluster mapping
        position_in_period_da = xr.DataArray(self._positions_in_period, dims=['original_time'])
        cluster_indices = self._cluster_indices_per_timestep

        # Get segment index and position for each original timestep
        seg_indices = segment_assignments.isel(cluster=cluster_indices, time=position_in_period_da)
        positions = position_within_segment.isel(cluster=cluster_indices, time=position_in_period_da)
        durations = segment_durations.isel(cluster=cluster_indices, segment=seg_indices)

        # Calculate interpolation factor: position within segment (0 to 1)
        factor = xr.where(durations > 1, (positions + 0.5) / durations, 0.5)

        # Get start and end boundary values from charge_state
        start_vals = da.isel(cluster=cluster_indices, time=seg_indices)
        end_vals = da.isel(cluster=cluster_indices, time=seg_indices + 1)

        # Linear interpolation
        interpolated = start_vals + (end_vals - start_vals) * factor

        # Clean up coordinate artifacts and rename
        interpolated = interpolated.drop_vars(['cluster', 'time', 'segment'], errors='ignore')
        interpolated = interpolated.rename({'original_time': 'time'}).assign_coords(time=self._original_timesteps)

        return interpolated.transpose('time', ...).assign_attrs(da.attrs)

    def _expand_first_timestep_only(self, da: xr.DataArray) -> xr.DataArray:
        """Expand binary event variables to first timestep of each segment only.

        For segmented systems, binary event variables like startup and shutdown indicate
        that an event occurred somewhere in the segment. When expanded, the event is placed
        at the first timestep of each segment, with zeros elsewhere.

        Args:
            da: Binary event DataArray with dims including (cluster, time).

        Returns:
            Expanded DataArray with event values only at first timestep of each segment.
        """
        clustering = self._clustering

        # First expand normally (repeats values)
        expanded = clustering.expand_data(da, original_time=self._original_timesteps)

        # Build mask: True only at first timestep of each segment
        position_within_segment = clustering.results.position_within_segment

        # Use cached period-to-cluster mapping
        position_in_period_da = xr.DataArray(self._positions_in_period, dims=['original_time'])
        cluster_indices = self._cluster_indices_per_timestep
        pos_in_segment = position_within_segment.isel(cluster=cluster_indices, time=position_in_period_da)

        # Clean up and create mask
        pos_in_segment = pos_in_segment.drop_vars(['cluster', 'time'], errors='ignore')
        pos_in_segment = pos_in_segment.rename({'original_time': 'time'}).assign_coords(time=self._original_timesteps)

        # First timestep of segment has position 0
        is_first = pos_in_segment == 0

        # Apply mask: keep value at first timestep, zero elsewhere
        result = xr.where(is_first, expanded, 0)
        return result.assign_attrs(da.attrs)

    def expand_dataarray(self, da: xr.DataArray, var_name: str = '', is_solution: bool = False) -> xr.DataArray:
        """Expand a DataArray from clustered to original timesteps.

        Args:
            da: DataArray to expand.
            var_name: Variable name for category-based expansion handling.
            is_solution: Whether this is a solution variable (affects segment total handling).

        Returns:
            Expanded DataArray with original timesteps.
        """
        if 'time' not in da.dims:
            return da.copy()

        has_cluster = 'cluster' in da.dims
        mode = self._get_mode(var_name)

        match mode:
            case ExpansionMode.INTERPOLATE if has_cluster and self._clustering.is_segmented:
                expanded = self._interpolate_charge_state_segmented(da)
            case ExpansionMode.INTERPOLATE if has_cluster:
                expanded = self._clustering.expand_data(da, original_time=self._original_timesteps)
            case ExpansionMode.FIRST_TIMESTEP if has_cluster and is_solution and self._clustering.is_segmented:
                return self._expand_first_timestep_only(da)
            case ExpansionMode.DIVIDE if is_solution:
                expanded = self._clustering.expand_data(da, original_time=self._original_timesteps)
                if self._expansion_divisor is not None:
                    expanded = expanded / self._expansion_divisor
            case _:
                expanded = self._clustering.expand_data(da, original_time=self._original_timesteps)

        if mode == ExpansionMode.INTERPOLATE and has_cluster:
            expanded = self._append_final_state(expanded, da)

        return expanded

    def _fast_get_da(self, ds: xr.Dataset, name: str, coord_cache: dict) -> xr.DataArray:
        """Construct DataArray without slow _construct_dataarray calls."""
        variable = ds.variables[name]
        var_dims = set(variable.dims)
        coords = {k: v for k, v in coord_cache.items() if set(v.dims).issubset(var_dims)}
        return xr.DataArray(variable, coords=coords, name=name)

    def _combine_intercluster_charge_states(self, expanded_fs: FlowSystem, reduced_solution: xr.Dataset) -> None:
        """Combine charge_state with SOC_boundary for intercluster storages (in-place).

        For intercluster storages, charge_state is relative (delta-E) and can be negative.
        Per Blanke et al. (2022) Eq. 9, actual SOC at time t in period d is:
            SOC(t) = SOC_boundary[d] * (1 - loss)^t_within_period + charge_state(t)
        where t_within_period is hours from period start (accounts for self-discharge decay).

        Args:
            expanded_fs: The expanded FlowSystem (modified in-place).
            reduced_solution: The original reduced solution dataset.
        """
        n_original_timesteps_extra = len(self._original_timesteps_extra)
        soc_boundary_vars = list(self._consume_vars)

        for soc_boundary_name in soc_boundary_vars:
            storage_name = soc_boundary_name.rsplit('|', 1)[0]
            charge_state_name = f'{storage_name}|charge_state'
            if charge_state_name not in expanded_fs._solution:
                continue

            soc_boundary = reduced_solution[soc_boundary_name]
            expanded_charge_state = expanded_fs._solution[charge_state_name]

            # Map each original timestep to its original period index
            original_cluster_indices = np.minimum(
                np.arange(n_original_timesteps_extra) // self._timesteps_per_cluster,
                self._n_original_clusters - 1,
            )

            # Select SOC_boundary for each timestep
            soc_boundary_per_timestep = soc_boundary.isel(
                cluster_boundary=xr.DataArray(original_cluster_indices, dims=['time'])
            ).assign_coords(time=self._original_timesteps_extra)

            # Apply self-discharge decay
            soc_boundary_per_timestep = self._apply_soc_decay(
                soc_boundary_per_timestep, storage_name, original_cluster_indices
            )

            # Combine and clip to non-negative
            combined = (expanded_charge_state + soc_boundary_per_timestep).clip(min=0)
            expanded_fs._solution[charge_state_name] = combined.assign_attrs(expanded_charge_state.attrs)

        # Clean up SOC_boundary variables and orphaned coordinates
        for soc_boundary_name in soc_boundary_vars:
            if soc_boundary_name in expanded_fs._solution:
                del expanded_fs._solution[soc_boundary_name]
        if 'cluster_boundary' in expanded_fs._solution.coords:
            expanded_fs._solution = expanded_fs._solution.drop_vars('cluster_boundary')

    def _apply_soc_decay(
        self,
        soc_boundary_per_timestep: xr.DataArray,
        storage_name: str,
        original_cluster_indices: np.ndarray,
    ) -> xr.DataArray:
        """Apply self-discharge decay to SOC_boundary values.

        Args:
            soc_boundary_per_timestep: SOC boundary values mapped to each timestep.
            storage_name: Name of the storage component.
            original_cluster_indices: Mapping of timesteps to original cluster indices.

        Returns:
            SOC boundary values with decay applied.
        """
        storage = self._fs.storages.get(storage_name)
        if storage is None:
            return soc_boundary_per_timestep

        n_timesteps = len(self._original_timesteps_extra)

        # Time within period for each timestep (0, 1, 2, ..., T-1, 0, 1, ...)
        time_within_period = np.arange(n_timesteps) % self._timesteps_per_cluster
        time_within_period[-1] = self._timesteps_per_cluster  # Extra timestep gets full decay
        time_within_period_da = xr.DataArray(
            time_within_period, dims=['time'], coords={'time': self._original_timesteps_extra}
        )

        # Decay factor: (1 - loss)^t
        loss_value = _scalar_safe_reduce(storage.relative_loss_per_hour, 'time', 'mean')
        loss_arr = np.asarray(loss_value)
        if not np.any(loss_arr > 0):
            return soc_boundary_per_timestep

        decay_da = (1 - loss_arr) ** time_within_period_da

        # Handle cluster dimension if present
        if 'cluster' in decay_da.dims:
            cluster_assignments = self._clustering.cluster_assignments
            if cluster_assignments.ndim == 1:
                cluster_per_timestep = xr.DataArray(
                    cluster_assignments.values[original_cluster_indices],
                    dims=['time'],
                    coords={'time': self._original_timesteps_extra},
                )
            else:
                cluster_per_timestep = cluster_assignments.isel(
                    original_cluster=xr.DataArray(original_cluster_indices, dims=['time'])
                ).assign_coords(time=self._original_timesteps_extra)
            decay_da = decay_da.isel(cluster=cluster_per_timestep).drop_vars('cluster', errors='ignore')

        return soc_boundary_per_timestep * decay_da

    def expand_flow_system(self) -> FlowSystem:
        """Expand the clustered FlowSystem to full original timesteps.

        Returns:
            FlowSystem: A new FlowSystem with full timesteps and expanded solution.
        """
        from .flow_system import FlowSystem

        # 1. Expand FlowSystem data
        reduced_ds = self._fs.to_dataset(include_solution=False)
        clustering_attrs = {'is_clustered', 'n_clusters', 'timesteps_per_cluster', 'clustering', 'cluster_weight'}
        skip_vars = {'cluster_weight', 'timestep_duration'}  # These have special handling
        data_vars = {}
        coord_cache = {k: v for k, v in reduced_ds.coords.items()}
        coord_names = set(coord_cache)
        for name in reduced_ds.variables:
            if name in coord_names:
                continue
            if name in skip_vars or name.startswith('clustering|'):
                continue
            da = self._fast_get_da(reduced_ds, name, coord_cache)
            # Skip vars with cluster dim but no time dim - they don't make sense after expansion
            if 'cluster' in da.dims and 'time' not in da.dims:
                continue
            data_vars[name] = self.expand_dataarray(da, name)
        # Remove timestep_duration reference from attrs - let FlowSystem compute it from timesteps_extra
        attrs = {k: v for k, v in reduced_ds.attrs.items() if k not in clustering_attrs and k != 'timestep_duration'}
        expanded_ds = xr.Dataset(data_vars, attrs=attrs)

        expanded_fs = FlowSystem.from_dataset(expanded_ds)

        # 2. Expand solution (with segment total correction for segmented systems)
        reduced_solution = self._fs.solution
        sol_coord_cache = {k: v for k, v in reduced_solution.coords.items()}
        sol_coord_names = set(sol_coord_cache)
        expanded_sol_vars = {}
        for name in reduced_solution.variables:
            if name in sol_coord_names:
                continue
            da = self._fast_get_da(reduced_solution, name, sol_coord_cache)
            expanded_sol_vars[name] = self.expand_dataarray(da, name, is_solution=True)
        expanded_fs._solution = xr.Dataset(expanded_sol_vars, attrs=reduced_solution.attrs)
        expanded_fs._solution = expanded_fs._solution.reindex(time=self._original_timesteps_extra)

        # 3. Combine charge_state with SOC_boundary for intercluster storages
        self._combine_intercluster_charge_states(expanded_fs, reduced_solution)

        # Log expansion info
        has_periods = self._fs.periods is not None
        has_scenarios = self._fs.scenarios is not None
        n_combinations = (len(self._fs.periods) if has_periods else 1) * (
            len(self._fs.scenarios) if has_scenarios else 1
        )
        n_reduced_timesteps = self._n_clusters * self._time_dim_size
        segmented_info = f' ({self._n_segments} segments)' if self._n_segments else ''
        logger.info(
            f'Expanded FlowSystem from {n_reduced_timesteps} to {self._n_original_timesteps} timesteps '
            f'({self._n_clusters} clusters{segmented_info}'
            + (
                f', {n_combinations} period/scenario combinations)'
                if n_combinations > 1
                else f' → {self._n_original_clusters} original clusters)'
            )
        )

        return expanded_fs


class TransformAccessor:
    """
    Accessor for transformation methods on FlowSystem.

    This class provides transformations that create new FlowSystem instances
    with modified structure or data, accessible via `flow_system.transform`.

    Examples:
        Time series aggregation (8 typical days):

        >>> reduced_fs = flow_system.transform.cluster(n_clusters=8, cluster_duration='1D')
        >>> reduced_fs.optimize(solver)
        >>> expanded_fs = reduced_fs.transform.expand()

        Future MGA:

        >>> mga_fs = flow_system.transform.mga(alternatives=5)
        >>> mga_fs.optimize(solver)
    """

    def __init__(self, flow_system: FlowSystem) -> None:
        """
        Initialize the accessor with a reference to the FlowSystem.

        Args:
            flow_system: The FlowSystem to transform.
        """
        self._fs = flow_system

    @staticmethod
    def _calculate_clustering_weights(ds) -> dict[str, float]:
        """Calculate weights for clustering based on dataset attributes."""
        from collections import Counter

        import numpy as np

        groups = [da.attrs.get('clustering_group') for da in ds.data_vars.values() if 'clustering_group' in da.attrs]
        group_counts = Counter(groups)

        # Calculate weight for each group (1/count)
        group_weights = {group: 1 / count for group, count in group_counts.items()}

        weights = {}
        variables = ds.variables
        for name in ds.data_vars:
            var_attrs = variables[name].attrs
            clustering_group = var_attrs.get('clustering_group')
            group_weight = group_weights.get(clustering_group)
            if group_weight is not None:
                weights[name] = group_weight
            else:
                weights[name] = var_attrs.get('clustering_weight', 1)

        if np.all(np.isclose(list(weights.values()), 1, atol=1e-6)):
            logger.debug('All Clustering weights were set to 1')

        return weights

    @staticmethod
    def _build_cluster_config_with_weights(
        cluster: ClusterConfig | None,
        auto_weights: dict[str, float],
    ) -> ClusterConfig:
        """Merge auto-calculated weights into ClusterConfig.

        Args:
            cluster: Optional user-provided ClusterConfig.
            auto_weights: Automatically calculated weights based on data variance.

        Returns:
            ClusterConfig with weights set (either user-provided or auto-calculated).
        """
        from tsam import ClusterConfig

        # User provided ClusterConfig with weights - use as-is
        if cluster is not None and cluster.weights is not None:
            return cluster

        # No ClusterConfig provided - use defaults with auto-calculated weights
        if cluster is None:
            return ClusterConfig(weights=auto_weights)

        # ClusterConfig provided without weights - add auto-calculated weights
        return ClusterConfig(
            method=cluster.method,
            representation=cluster.representation,
            weights=auto_weights,
            normalize_column_means=cluster.normalize_column_means,
            use_duration_curves=cluster.use_duration_curves,
            include_period_sums=cluster.include_period_sums,
            solver=cluster.solver,
        )

    def sel(
        self,
        time: str | slice | list[str] | pd.Timestamp | pd.DatetimeIndex | None = None,
        period: int | slice | list[int] | pd.Index | None = None,
        scenario: str | slice | list[str] | pd.Index | None = None,
    ) -> FlowSystem:
        """
        Select a subset of the FlowSystem by label.

        Creates a new FlowSystem with data selected along the specified dimensions.
        The returned FlowSystem has no solution (it must be re-optimized).

        Args:
            time: Time selection (e.g., slice('2023-01-01', '2023-12-31'), '2023-06-15')
            period: Period selection (e.g., slice(2023, 2024), or list of periods)
            scenario: Scenario selection (e.g., 'scenario1', or list of scenarios)

        Returns:
            FlowSystem: New FlowSystem with selected data (no solution).

        Examples:
            >>> # Select specific time range
            >>> fs_jan = flow_system.transform.sel(time=slice('2023-01-01', '2023-01-31'))
            >>> fs_jan.optimize(solver)

            >>> # Select single scenario
            >>> fs_base = flow_system.transform.sel(scenario='Base Case')
        """
        from .flow_system import FlowSystem

        if time is None and period is None and scenario is None:
            result = self._fs.copy()
            result.solution = None
            return result

        if not self._fs.connected_and_transformed:
            self._fs.connect_and_transform()

        ds = self._fs.to_dataset()
        ds = self._dataset_sel(ds, time=time, period=period, scenario=scenario)
        return FlowSystem.from_dataset(ds)  # from_dataset doesn't include solution

    def isel(
        self,
        time: int | slice | list[int] | None = None,
        period: int | slice | list[int] | None = None,
        scenario: int | slice | list[int] | None = None,
    ) -> FlowSystem:
        """
        Select a subset of the FlowSystem by integer indices.

        Creates a new FlowSystem with data selected along the specified dimensions.
        The returned FlowSystem has no solution (it must be re-optimized).

        Args:
            time: Time selection by integer index (e.g., slice(0, 100), 50, or [0, 5, 10])
            period: Period selection by integer index
            scenario: Scenario selection by integer index

        Returns:
            FlowSystem: New FlowSystem with selected data (no solution).

        Examples:
            >>> # Select first 24 timesteps
            >>> fs_day1 = flow_system.transform.isel(time=slice(0, 24))
            >>> fs_day1.optimize(solver)

            >>> # Select first scenario
            >>> fs_first = flow_system.transform.isel(scenario=0)
        """
        from .flow_system import FlowSystem

        if time is None and period is None and scenario is None:
            result = self._fs.copy()
            result.solution = None
            return result

        if not self._fs.connected_and_transformed:
            self._fs.connect_and_transform()

        ds = self._fs.to_dataset()
        ds = self._dataset_isel(ds, time=time, period=period, scenario=scenario)
        return FlowSystem.from_dataset(ds)  # from_dataset doesn't include solution

    def resample(
        self,
        time: str,
        method: Literal['mean', 'sum', 'max', 'min', 'first', 'last', 'std', 'var', 'median', 'count'] = 'mean',
        hours_of_last_timestep: int | float | None = None,
        hours_of_previous_timesteps: int | float | np.ndarray | None = None,
        fill_gaps: Literal['ffill', 'bfill', 'interpolate'] | None = None,
        **kwargs: Any,
    ) -> FlowSystem:
        """
        Create a resampled FlowSystem by resampling data along the time dimension.

        Creates a new FlowSystem with resampled time series data.
        The returned FlowSystem has no solution (it must be re-optimized).

        Args:
            time: Resampling frequency (e.g., '3h', '2D', '1M')
            method: Resampling method. Recommended: 'mean', 'first', 'last', 'max', 'min'
            hours_of_last_timestep: Duration of the last timestep after resampling.
                If None, computed from the last time interval.
            hours_of_previous_timesteps: Duration of previous timesteps after resampling.
                If None, computed from the first time interval. Can be a scalar or array.
            fill_gaps: Strategy for filling gaps (NaN values) that arise when resampling
                irregular timesteps to regular intervals. Options: 'ffill' (forward fill),
                'bfill' (backward fill), 'interpolate' (linear interpolation).
                If None (default), raises an error when gaps are detected.
            **kwargs: Additional arguments passed to xarray.resample()

        Returns:
            FlowSystem: New resampled FlowSystem (no solution).

        Raises:
            ValueError: If resampling creates gaps and fill_gaps is not specified.

        Examples:
            >>> # Resample to 4-hour intervals
            >>> fs_4h = flow_system.transform.resample(time='4h', method='mean')
            >>> fs_4h.optimize(solver)

            >>> # Resample to daily with max values
            >>> fs_daily = flow_system.transform.resample(time='1D', method='max')
        """
        from .flow_system import FlowSystem

        if not self._fs.connected_and_transformed:
            self._fs.connect_and_transform()

        ds = self._fs.to_dataset()
        ds = self._dataset_resample(
            ds,
            freq=time,
            method=method,
            hours_of_last_timestep=hours_of_last_timestep,
            hours_of_previous_timesteps=hours_of_previous_timesteps,
            fill_gaps=fill_gaps,
            **kwargs,
        )
        return FlowSystem.from_dataset(ds)  # from_dataset doesn't include solution

    # --- Class methods for dataset operations (can be called without instance) ---

    @classmethod
    def _dataset_sel(
        cls,
        dataset: xr.Dataset,
        time: str | slice | list[str] | pd.Timestamp | pd.DatetimeIndex | None = None,
        period: int | slice | list[int] | pd.Index | None = None,
        scenario: str | slice | list[str] | pd.Index | None = None,
        hours_of_last_timestep: int | float | None = None,
        hours_of_previous_timesteps: int | float | np.ndarray | None = None,
    ) -> xr.Dataset:
        """
        Select subset of dataset by label.

        Args:
            dataset: xarray Dataset from FlowSystem.to_dataset()
            time: Time selection (e.g., '2020-01', slice('2020-01-01', '2020-06-30'))
            period: Period selection (e.g., 2020, slice(2020, 2022))
            scenario: Scenario selection (e.g., 'Base Case', ['Base Case', 'High Demand'])
            hours_of_last_timestep: Duration of the last timestep.
            hours_of_previous_timesteps: Duration of previous timesteps.

        Returns:
            xr.Dataset: Selected dataset
        """

        indexers = {}
        if time is not None:
            indexers['time'] = time
        if period is not None:
            indexers['period'] = period
        if scenario is not None:
            indexers['scenario'] = scenario

        if not indexers:
            return dataset

        result = dataset.sel(**indexers)

        if 'time' in indexers:
            result = ModelCoordinates._update_time_metadata(result, hours_of_last_timestep, hours_of_previous_timesteps)

        if 'period' in indexers:
            result = ModelCoordinates._update_period_metadata(result)

        if 'scenario' in indexers:
            result = ModelCoordinates._update_scenario_metadata(result)

        return result

    @classmethod
    def _dataset_isel(
        cls,
        dataset: xr.Dataset,
        time: int | slice | list[int] | None = None,
        period: int | slice | list[int] | None = None,
        scenario: int | slice | list[int] | None = None,
        hours_of_last_timestep: int | float | None = None,
        hours_of_previous_timesteps: int | float | np.ndarray | None = None,
    ) -> xr.Dataset:
        """
        Select subset of dataset by integer index.

        Args:
            dataset: xarray Dataset from FlowSystem.to_dataset()
            time: Time selection by index
            period: Period selection by index
            scenario: Scenario selection by index
            hours_of_last_timestep: Duration of the last timestep.
            hours_of_previous_timesteps: Duration of previous timesteps.

        Returns:
            xr.Dataset: Selected dataset
        """

        indexers = {}
        if time is not None:
            indexers['time'] = time
        if period is not None:
            indexers['period'] = period
        if scenario is not None:
            indexers['scenario'] = scenario

        if not indexers:
            return dataset

        result = dataset.isel(**indexers)

        if 'time' in indexers:
            result = ModelCoordinates._update_time_metadata(result, hours_of_last_timestep, hours_of_previous_timesteps)

        if 'period' in indexers:
            result = ModelCoordinates._update_period_metadata(result)

        if 'scenario' in indexers:
            result = ModelCoordinates._update_scenario_metadata(result)

        return result

    @classmethod
    def _dataset_resample(
        cls,
        dataset: xr.Dataset,
        freq: str,
        method: Literal['mean', 'sum', 'max', 'min', 'first', 'last', 'std', 'var', 'median', 'count'] = 'mean',
        hours_of_last_timestep: int | float | None = None,
        hours_of_previous_timesteps: int | float | np.ndarray | None = None,
        fill_gaps: Literal['ffill', 'bfill', 'interpolate'] | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """
        Resample dataset along time dimension.

        Args:
            dataset: xarray Dataset from FlowSystem.to_dataset()
            freq: Resampling frequency (e.g., '2h', '1D', '1M')
            method: Resampling method (e.g., 'mean', 'sum', 'first')
            hours_of_last_timestep: Duration of the last timestep after resampling.
            hours_of_previous_timesteps: Duration of previous timesteps after resampling.
            fill_gaps: Strategy for filling gaps (NaN values) that arise when resampling
                irregular timesteps to regular intervals. Options: 'ffill' (forward fill),
                'bfill' (backward fill), 'interpolate' (linear interpolation).
                If None (default), raises an error when gaps are detected.
            **kwargs: Additional arguments passed to xarray.resample()

        Returns:
            xr.Dataset: Resampled dataset

        Raises:
            ValueError: If resampling creates gaps and fill_gaps is not specified.
        """

        available_methods = ['mean', 'sum', 'max', 'min', 'first', 'last', 'std', 'var', 'median', 'count']
        if method not in available_methods:
            raise ValueError(f'Unsupported resampling method: {method}. Available: {available_methods}')

        original_attrs = dict(dataset.attrs)

        time_var_names = [v for v in dataset.data_vars if 'time' in dataset[v].dims]
        non_time_var_names = [v for v in dataset.data_vars if v not in time_var_names]

        # Handle case where no data variables have time dimension (all scalars)
        # We still need to resample the time coordinate itself
        if not time_var_names:
            if 'time' not in dataset.coords:
                raise ValueError('Dataset has no time dimension to resample')
            # Create a dummy variable to resample the time coordinate
            dummy = xr.DataArray(
                np.zeros(len(dataset.coords['time'])), dims=['time'], coords={'time': dataset.coords['time']}
            )
            dummy_ds = xr.Dataset({'__dummy__': dummy})
            resampled_dummy = getattr(dummy_ds.resample(time=freq, **kwargs), method)()
            # Get the resampled time coordinate
            resampled_time = resampled_dummy.coords['time']
            # Create result with all original scalar data and resampled time coordinate
            # Keep all existing coordinates (period, scenario, etc.) except time which gets resampled
            result = dataset.copy()
            result = result.assign_coords(time=resampled_time)
            result.attrs.update(original_attrs)
            return ModelCoordinates._update_time_metadata(result, hours_of_last_timestep, hours_of_previous_timesteps)

        time_dataset = dataset[time_var_names]
        resampled_time_dataset = cls._resample_by_dimension_groups(time_dataset, freq, method, **kwargs)

        # Handle NaN values that may arise from resampling irregular timesteps to regular intervals.
        # When irregular data (e.g., [00:00, 01:00, 03:00]) is resampled to regular intervals (e.g., '1h'),
        # bins without data (e.g., 02:00) get NaN.
        if resampled_time_dataset.isnull().any().to_array().any():
            if fill_gaps is None:
                # Find which variables have NaN values for a helpful error message
                vars_with_nans = [
                    name for name in resampled_time_dataset.data_vars if resampled_time_dataset[name].isnull().any()
                ]
                raise ValueError(
                    f'Resampling created gaps (NaN values) in variables: {vars_with_nans}. '
                    f'This typically happens when resampling irregular timesteps to regular intervals. '
                    f"Specify fill_gaps='ffill', 'bfill', or 'interpolate' to handle gaps, "
                    f'or resample to a coarser frequency.'
                )
            elif fill_gaps == 'ffill':
                resampled_time_dataset = resampled_time_dataset.ffill(dim='time').bfill(dim='time')
            elif fill_gaps == 'bfill':
                resampled_time_dataset = resampled_time_dataset.bfill(dim='time').ffill(dim='time')
            elif fill_gaps == 'interpolate':
                resampled_time_dataset = resampled_time_dataset.interpolate_na(dim='time', method='linear')
                # Handle edges that can't be interpolated
                resampled_time_dataset = resampled_time_dataset.ffill(dim='time').bfill(dim='time')

        if non_time_var_names:
            non_time_dataset = dataset[non_time_var_names]
            result = xr.merge([resampled_time_dataset, non_time_dataset])
        else:
            result = resampled_time_dataset

        # Preserve all original coordinates that aren't 'time' (e.g., period, scenario, cluster)
        # These may be lost during merge if no data variable uses them
        for coord_name, coord_val in dataset.coords.items():
            if coord_name != 'time' and coord_name not in result.coords:
                result = result.assign_coords({coord_name: coord_val})

        result.attrs.update(original_attrs)
        return ModelCoordinates._update_time_metadata(result, hours_of_last_timestep, hours_of_previous_timesteps)

    @staticmethod
    def _resample_by_dimension_groups(
        time_dataset: xr.Dataset,
        time: str,
        method: str,
        **kwargs: Any,
    ) -> xr.Dataset:
        """
        Resample variables grouped by their dimension structure to avoid broadcasting.

        Groups variables by their non-time dimensions before resampling for performance
        and to prevent xarray from broadcasting variables with different dimensions.

        Args:
            time_dataset: Dataset containing only variables with time dimension
            time: Resampling frequency (e.g., '2h', '1D', '1M')
            method: Resampling method name (e.g., 'mean', 'sum', 'first')
            **kwargs: Additional arguments passed to xarray.resample()

        Returns:
            Resampled dataset with original dimension structure preserved
        """
        dim_groups = defaultdict(list)
        variables = time_dataset.variables
        for var_name in time_dataset.data_vars:
            dims_key = tuple(sorted(d for d in variables[var_name].dims if d != 'time'))
            dim_groups[dims_key].append(var_name)

        # Note: defaultdict is always truthy, so we check length explicitly
        if len(dim_groups) == 0:
            return getattr(time_dataset.resample(time=time, **kwargs), method)()

        resampled_groups = []
        for var_names in dim_groups.values():
            if not var_names:
                continue

            stacked = xr.concat(
                [time_dataset[name] for name in var_names],
                dim=pd.Index(var_names, name='variable'),
                combine_attrs='drop_conflicts',
            )
            resampled = getattr(stacked.resample(time=time, **kwargs), method)()
            resampled_dataset = resampled.to_dataset(dim='variable')
            resampled_groups.append(resampled_dataset)

        if not resampled_groups:
            # No data variables to resample, but still resample coordinates
            return getattr(time_dataset.resample(time=time, **kwargs), method)()

        if len(resampled_groups) == 1:
            return resampled_groups[0]

        return xr.merge(resampled_groups, combine_attrs='drop_conflicts')

    def fix_sizes(
        self,
        sizes: xr.Dataset | dict[str, float] | None = None,
        decimal_rounding: int | None = 5,
    ) -> FlowSystem:
        """
        Create a new FlowSystem with investment sizes fixed to specified values.

        This is useful for two-stage optimization workflows:
        1. Solve a sizing problem (possibly resampled for speed)
        2. Fix sizes and solve dispatch at full resolution

        The returned FlowSystem has InvestParameters with fixed_size set,
        making those sizes mandatory rather than decision variables.

        Args:
            sizes: The sizes to fix. Can be:
                - None: Uses sizes from this FlowSystem's solution (must be solved)
                - xr.Dataset: Dataset with size variables (e.g., from statistics.sizes)
                - dict: Mapping of component names to sizes (e.g., {'Boiler(Q_fu)': 100})
            decimal_rounding: Number of decimal places to round sizes to.
                Rounding helps avoid numerical infeasibility. Set to None to disable.

        Returns:
            FlowSystem: New FlowSystem with fixed sizes (no solution).

        Raises:
            ValueError: If no sizes provided and FlowSystem has no solution.
            KeyError: If a specified size doesn't match any InvestParameters.

        Examples:
            Two-stage optimization:

            >>> # Stage 1: Size with resampled data
            >>> fs_sizing = flow_system.transform.resample('2h')
            >>> fs_sizing.optimize(solver)
            >>>
            >>> # Stage 2: Fix sizes and optimize at full resolution
            >>> fs_dispatch = flow_system.transform.fix_sizes(fs_sizing.stats.sizes)
            >>> fs_dispatch.optimize(solver)

            Using a dict:

            >>> fs_fixed = flow_system.transform.fix_sizes(
            ...     {
            ...         'Boiler(Q_fu)': 100,
            ...         'Storage': 500,
            ...     }
            ... )
            >>> fs_fixed.optimize(solver)
        """
        from .flow_system import FlowSystem
        from .interface import InvestParameters

        # Get sizes from solution if not provided
        if sizes is None:
            if self._fs.solution is None:
                raise ValueError(
                    'No sizes provided and FlowSystem has no solution. '
                    'Either provide sizes or optimize the FlowSystem first.'
                )
            sizes = self._fs.stats.sizes

        # Convert dict to Dataset format
        if isinstance(sizes, dict):
            sizes = xr.Dataset({k: xr.DataArray(v) for k, v in sizes.items()})

        # Apply rounding
        if decimal_rounding is not None:
            sizes = sizes.round(decimal_rounding)

        # Create copy of FlowSystem
        if not self._fs.connected_and_transformed:
            self._fs.connect_and_transform()

        ds = self._fs.to_dataset()
        new_fs = FlowSystem.from_dataset(ds)

        # Fix sizes in the new FlowSystem's InvestParameters
        # Note: statistics.sizes returns keys without '|size' suffix (e.g., 'Boiler(Q_fu)')
        # but dicts may have either format
        for size_var in sizes.data_vars:
            # Normalize: strip '|size' suffix if present
            base_name = size_var.replace('|size', '') if size_var.endswith('|size') else size_var
            fixed_value = float(sizes[size_var].item())

            # Find matching element with InvestParameters
            found = False

            # Check flows
            for flow in new_fs.flows.values():
                if flow.label_full == base_name and isinstance(flow.size, InvestParameters):
                    flow.size.fixed_size = fixed_value
                    flow.size.mandatory = True
                    found = True
                    logger.debug(f'Fixed size of {base_name} to {fixed_value}')
                    break

            # Check storage capacity
            if not found:
                for component in new_fs.components.values():
                    if hasattr(component, 'capacity_in_flow_hours'):
                        if component.label == base_name and isinstance(
                            component.capacity_in_flow_hours, InvestParameters
                        ):
                            component.capacity_in_flow_hours.fixed_size = fixed_value
                            component.capacity_in_flow_hours.mandatory = True
                            found = True
                            logger.debug(f'Fixed size of {base_name} to {fixed_value}')
                            break

            if not found:
                logger.warning(
                    f'Size variable "{base_name}" not found as InvestParameters in FlowSystem. '
                    f'It may be a fixed-size component or the name may not match.'
                )

        return new_fs

    def clustering_data(
        self,
        period: Any | None = None,
        scenario: Any | None = None,
    ) -> xr.Dataset:
        """
        Get the time-varying data that would be used for clustering.

        This method extracts only the data arrays that vary over time, which is
        the data that clustering algorithms use to identify typical periods.
        Constant arrays (same value for all timesteps) are excluded since they
        don't contribute to pattern identification.

        Use this to inspect or pre-process the data before clustering, or to
        understand which variables influence the clustering result.

        Args:
            period: Optional period label to select. If None and the FlowSystem
                has multiple periods, returns data for all periods.
            scenario: Optional scenario label to select. If None and the FlowSystem
                has multiple scenarios, returns data for all scenarios.

        Returns:
            xr.Dataset containing only time-varying data arrays. The dataset
            includes arrays like demand profiles, price profiles, and other
            time series that vary over the time dimension.

        Examples:
            Inspect clustering input data:

            >>> data = flow_system.transform.clustering_data()
            >>> print(f'Variables used for clustering: {list(data.data_vars)}')
            >>> data['HeatDemand(Q)|fixed_relative_profile'].plot()

            Get data for a specific period/scenario:

            >>> data_2024 = flow_system.transform.clustering_data(period=2024)
            >>> data_high = flow_system.transform.clustering_data(scenario='high')

            Convert to DataFrame for external tools:

            >>> df = flow_system.transform.clustering_data().to_dataframe()
        """
        from .core import drop_constant_arrays

        if not self._fs.connected_and_transformed:
            self._fs.connect_and_transform()

        ds = self._fs.to_dataset(include_solution=False)

        # Build selector for period/scenario
        selector = {}
        if period is not None:
            selector['period'] = period
        if scenario is not None:
            selector['scenario'] = scenario

        # Apply selection if specified
        if selector:
            ds = ds.sel(**selector, drop=True)

        # Filter to only time-varying arrays
        result = drop_constant_arrays(ds, dim='time')

        # Guard against empty dataset (all variables are constant)
        if not result.data_vars:
            selector_info = f' for {selector}' if selector else ''
            raise ValueError(
                f'No time-varying data found{selector_info}. '
                f'All variables are constant over time. Check your period/scenario filter or input data.'
            )

        # Remove attrs for cleaner output
        result.attrs = {}
        for var in result.data_vars:
            result[var].attrs = {}

        return result

    def cluster(
        self,
        n_clusters: int,
        cluster_duration: str | float,
        data_vars: list[str] | None = None,
        cluster: ClusterConfig | None = None,
        extremes: ExtremeConfig | None = None,
        segments: SegmentConfig | None = None,
        preserve_column_means: bool = True,
        rescale_exclude_columns: list[str] | None = None,
        round_decimals: int | None = None,
        numerical_tolerance: float = 1e-13,
        **tsam_kwargs: Any,
    ) -> FlowSystem:
        """
        Create a FlowSystem with reduced timesteps using typical clusters.

        This method creates a new FlowSystem optimized for sizing studies by reducing
        the number of timesteps to only the typical (representative) clusters identified
        through time series aggregation using the tsam package.

        The method:
        1. Performs time series clustering using tsam (hierarchical by default)
        2. Extracts only the typical clusters (not all original timesteps)
        3. Applies timestep weighting for accurate cost representation
        4. Handles storage states between clusters based on each Storage's ``cluster_mode``

        Use this for initial sizing optimization, then use ``fix_sizes()`` to re-optimize
        at full resolution for accurate dispatch results.

        To reuse an existing clustering on different data, use ``apply_clustering()`` instead.

        Args:
            n_clusters: Number of clusters (typical periods) to extract (e.g., 8 typical days).
            cluster_duration: Duration of each cluster. Can be a pandas-style string
                ('1D', '24h', '6h') or a numeric value in hours.
            data_vars: Optional list of variable names to use for clustering. If specified,
                only these variables are used to determine cluster assignments, but the
                clustering is then applied to ALL time-varying data in the FlowSystem.
                Use ``transform.clustering_data()`` to see available variables.
                Example: ``data_vars=['HeatDemand(Q)|fixed_relative_profile']`` to cluster
                based only on heat demand patterns.
            cluster: Optional tsam ``ClusterConfig`` object specifying clustering algorithm,
                representation method, and weights. If None, uses default settings (hierarchical
                clustering with medoid representation) and automatically calculated weights
                based on data variance.
            extremes: Optional tsam ``ExtremeConfig`` object specifying how to handle
                extreme periods (peaks). Use this to ensure peak demand days are captured.
                Example: ``ExtremeConfig(method='new_cluster', max_value=['demand'])``.
            segments: Optional tsam ``SegmentConfig`` object specifying intra-period
                segmentation. Segments divide each cluster period into variable-duration
                sub-segments. Example: ``SegmentConfig(n_segments=4)``.
            preserve_column_means: Rescale typical periods so each column's weighted mean
                matches the original data's mean. Ensures total energy/load is preserved
                when weights represent occurrence counts. Default is True.
            rescale_exclude_columns: Column names to exclude from rescaling when
                ``preserve_column_means=True``. Useful for binary/indicator columns (0/1 values)
                that should not be rescaled.
            round_decimals: Round output values to this many decimal places.
                If None (default), no rounding is applied.
            numerical_tolerance: Tolerance for numerical precision issues. Controls when
                warnings are raised for aggregated values exceeding original time series bounds.
                Default is 1e-13.
            **tsam_kwargs: Additional keyword arguments passed to ``tsam.aggregate()``
                for forward compatibility. See tsam documentation for all options.

        Returns:
            A new FlowSystem with reduced timesteps (only typical clusters).
            The FlowSystem has metadata stored in ``clustering`` for expansion.

        Raises:
            ValueError: If timestep sizes are inconsistent.
            ValueError: If cluster_duration is not a multiple of timestep size.

        Examples:
            Basic clustering with peak preservation:

            >>> from tsam import ExtremeConfig
            >>> fs_clustered = flow_system.transform.cluster(
            ...     n_clusters=8,
            ...     cluster_duration='1D',
            ...     extremes=ExtremeConfig(
            ...         method='new_cluster',
            ...         max_value=['HeatDemand(Q_th)|fixed_relative_profile'],
            ...     ),
            ... )
            >>> fs_clustered.optimize(solver)

            Clustering based on specific variables only:

            >>> # See available variables for clustering
            >>> print(flow_system.transform.clustering_data().data_vars)
            >>>
            >>> # Cluster based only on demand profile
            >>> fs_clustered = flow_system.transform.cluster(
            ...     n_clusters=8,
            ...     cluster_duration='1D',
            ...     data_vars=['HeatDemand(Q)|fixed_relative_profile'],
            ... )

        Note:
            - This is best suited for initial sizing, not final dispatch optimization
            - Use ``extremes`` to ensure peak demand clusters are captured
            - A 5-10% safety margin on sizes is recommended for the dispatch stage
            - For seasonal storage (e.g., hydrogen, thermal storage), set
              ``Storage.cluster_mode='intercluster'`` or ``'intercluster_cyclic'``
        """
        import tsam

        from .clustering import ClusteringResults
        from .core import drop_constant_arrays

        # Parse cluster_duration to hours
        hours_per_cluster = (
            pd.Timedelta(cluster_duration).total_seconds() / 3600
            if isinstance(cluster_duration, str)
            else float(cluster_duration)
        )

        # Validation
        dt = float(self._fs.timestep_duration.min().item())
        if not np.isclose(dt, float(self._fs.timestep_duration.max().item())):
            raise ValueError(
                f'cluster() requires uniform timestep sizes, got min={dt}h, '
                f'max={float(self._fs.timestep_duration.max().item())}h.'
            )
        if not np.isclose(hours_per_cluster / dt, round(hours_per_cluster / dt), atol=1e-9):
            raise ValueError(f'cluster_duration={hours_per_cluster}h must be a multiple of timestep size ({dt}h).')

        timesteps_per_cluster = int(round(hours_per_cluster / dt))
        has_periods = self._fs.periods is not None
        has_scenarios = self._fs.scenarios is not None

        # Determine iteration dimensions
        periods = list(self._fs.periods) if has_periods else [None]
        scenarios = list(self._fs.scenarios) if has_scenarios else [None]

        ds = self._fs.to_dataset(include_solution=False)

        # Validate and prepare data_vars for clustering
        if data_vars is not None:
            missing = set(data_vars) - set(ds.data_vars)
            if missing:
                raise ValueError(
                    f'data_vars not found in FlowSystem: {missing}. '
                    f'Available time-varying variables can be found via transform.clustering_data().'
                )
            ds_for_clustering = ds[list(data_vars)]
        else:
            ds_for_clustering = ds

        # Filter constant arrays once on the full dataset (not per slice)
        # This ensures all slices have the same variables for consistent metrics
        ds_for_clustering = drop_constant_arrays(ds_for_clustering, dim='time')

        # Guard against empty dataset after removing constant arrays
        if not ds_for_clustering.data_vars:
            filter_info = f'data_vars={data_vars}' if data_vars else 'all variables'
            raise ValueError(
                f'No time-varying data found for clustering ({filter_info}). '
                f'All variables are constant over time. Check your data_vars filter or input data.'
            )

        # Validate tsam_kwargs doesn't override explicit parameters
        reserved_tsam_keys = {
            'n_clusters',
            'period_duration',  # exposed as cluster_duration
            'timestep_duration',  # computed automatically
            'cluster',
            'segments',
            'extremes',
            'preserve_column_means',
            'rescale_exclude_columns',
            'round_decimals',
            'numerical_tolerance',
        }
        conflicts = reserved_tsam_keys & set(tsam_kwargs.keys())
        if conflicts:
            raise ValueError(
                f'Cannot override explicit parameters via tsam_kwargs: {conflicts}. '
                f'Use the corresponding cluster() parameters instead.'
            )

        # Validate ExtremeConfig compatibility with multi-period/scenario systems
        # Without preserve_n_clusters=True, methods can produce different n_clusters per period,
        # which breaks the xarray structure that requires uniform dimensions
        total_slices = len(periods) * len(scenarios)
        if total_slices > 1 and extremes is not None:
            if not extremes.preserve_n_clusters:
                raise ValueError(
                    'ExtremeConfig must have preserve_n_clusters=True for multi-period '
                    'or multi-scenario systems to ensure consistent cluster counts across all slices. '
                    'Example: ExtremeConfig(method="new_cluster", max_value=[...], preserve_n_clusters=True)'
                )

        # Build dim_names and clean key helper
        dim_names: list[str] = []
        if has_periods:
            dim_names.append('period')
        if has_scenarios:
            dim_names.append('scenario')

        def to_clean_key(period_label, scenario_label) -> tuple:
            """Convert (period, scenario) to clean key based on which dims exist."""
            key_parts = []
            if has_periods:
                key_parts.append(period_label)
            if has_scenarios:
                key_parts.append(scenario_label)
            return tuple(key_parts)

        # Cluster each (period, scenario) combination using tsam directly
        aggregation_results: dict[tuple, Any] = {}

        for period_label in periods:
            for scenario_label in scenarios:
                key = to_clean_key(period_label, scenario_label)
                selector = {k: v for k, v in [('period', period_label), ('scenario', scenario_label)] if v is not None}

                # Select data slice for clustering
                ds_slice = ds_for_clustering.sel(**selector, drop=True) if selector else ds_for_clustering
                df_for_clustering = ds_slice.to_dataframe()

                if selector:
                    logger.info(f'Clustering {", ".join(f"{k}={v}" for k, v in selector.items())}...')

                # Suppress tsam warning about minimal value constraints (informational, not actionable)
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning, message='.*minimal value.*exceeds.*')

                    # Build ClusterConfig with auto-calculated weights
                    clustering_weights = self._calculate_clustering_weights(ds_slice)
                    filtered_weights = {
                        name: w for name, w in clustering_weights.items() if name in df_for_clustering.columns
                    }
                    cluster_config = self._build_cluster_config_with_weights(cluster, filtered_weights)

                    # Perform clustering based on selected data_vars (or all if not specified)
                    aggregation_results[key] = tsam.aggregate(
                        df_for_clustering,
                        n_clusters=n_clusters,
                        period_duration=hours_per_cluster,
                        temporal_resolution=dt,
                        cluster=cluster_config,
                        extremes=extremes,
                        segments=segments,
                        preserve_column_means=preserve_column_means,
                        rescale_exclude_columns=rescale_exclude_columns,
                        round_decimals=round_decimals,
                        numerical_tolerance=numerical_tolerance,
                        **tsam_kwargs,
                    )

        # If data_vars was specified, apply clustering to FULL data
        if data_vars is not None:
            # Build ClusteringResults from subset clustering
            clustering_results = ClusteringResults(
                {k: r.clustering for k, r in aggregation_results.items()},
                dim_names,
            )
            # Apply to full data and replace results
            aggregation_results = dict(clustering_results.apply(ds))

        # Build and return the reduced FlowSystem
        builder = _ReducedFlowSystemBuilder(self._fs, aggregation_results, timesteps_per_cluster, dt, dim_names)
        return builder.build(ds)

    def apply_clustering(
        self,
        clustering: Clustering,
    ) -> FlowSystem:
        """
        Apply an existing clustering to this FlowSystem.

        This method applies a previously computed clustering (from another FlowSystem)
        to the current FlowSystem's data. The clustering structure (cluster assignments,
        number of clusters, etc.) is preserved while the time series data is aggregated
        according to the existing cluster assignments.

        Use this to:
        - Compare different scenarios with identical cluster assignments
        - Apply a reference clustering to new data

        Args:
            clustering: A ``Clustering`` object from a previously clustered FlowSystem.
                Obtain this via ``fs.clustering`` from a clustered FlowSystem.

        Returns:
            A new FlowSystem with reduced timesteps (only typical clusters).
            The FlowSystem has metadata stored in ``clustering`` for expansion.

        Raises:
            ValueError: If the clustering dimensions don't match this FlowSystem's
                periods/scenarios.

        Examples:
            Apply clustering from one FlowSystem to another:

            >>> fs_reference = fs_base.transform.cluster(n_clusters=8, cluster_duration='1D')
            >>> fs_other = fs_high.transform.apply_clustering(fs_reference.clustering)
        """
        # Validation
        dt = float(self._fs.timestep_duration.min().item())
        if not np.isclose(dt, float(self._fs.timestep_duration.max().item())):
            raise ValueError(
                f'apply_clustering() requires uniform timestep sizes, got min={dt}h, '
                f'max={float(self._fs.timestep_duration.max().item())}h.'
            )

        # Get timesteps_per_cluster from the clustering object (survives serialization)
        timesteps_per_cluster = clustering.timesteps_per_cluster
        dim_names = clustering.results.dim_names

        ds = self._fs.to_dataset(include_solution=False)

        # Validate that timesteps match the clustering expectations
        current_timesteps = len(self._fs.timesteps)
        expected_timesteps = clustering.n_original_clusters * clustering.timesteps_per_cluster
        if current_timesteps != expected_timesteps:
            raise ValueError(
                f'Timestep count mismatch in apply_clustering(): '
                f'FlowSystem has {current_timesteps} timesteps, but clustering expects '
                f'{expected_timesteps} timesteps ({clustering.n_original_clusters} clusters × '
                f'{clustering.timesteps_per_cluster} timesteps/cluster). '
                f'Ensure self._fs.timesteps matches the original data used for clustering.results.apply(ds).'
            )

        # Apply existing clustering to all (period, scenario) combinations at once
        logger.info('Applying clustering...')
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, message='.*minimal value.*exceeds.*')
            agg_results = clustering.results.apply(ds)

        # Convert AggregationResults to dict format
        aggregation_results = dict(agg_results)

        # Build and return the reduced FlowSystem
        builder = _ReducedFlowSystemBuilder(self._fs, aggregation_results, timesteps_per_cluster, dt, dim_names)
        return builder.build(ds)

    def _validate_for_expansion(self) -> Clustering:
        """Validate FlowSystem can be expanded and return clustering info.

        Returns:
            The Clustering object.

        Raises:
            ValueError: If FlowSystem wasn't created with cluster() or has no solution.
        """

        if self._fs.clustering is None:
            raise ValueError(
                'expand() requires a FlowSystem created with cluster(). This FlowSystem has no aggregation info.'
            )
        if self._fs.solution is None:
            raise ValueError('FlowSystem has no solution. Run optimize() or solve() first.')

        return self._fs.clustering

    def expand(self) -> FlowSystem:
        """Expand a clustered FlowSystem back to full original timesteps.

        After solving a FlowSystem created with ``cluster()``, this method
        disaggregates the FlowSystem by:
        1. Expanding all time series data from typical clusters to full timesteps
        2. Expanding the solution by mapping each typical cluster back to all
           original clusters it represents

        For FlowSystems with periods and/or scenarios, each (period, scenario)
        combination is expanded using its own cluster assignment.

        This enables using all existing solution accessors (``statistics``, ``plot``, etc.)
        with full time resolution, where both the data and solution are consistently
        expanded from the typical clusters.

        Returns:
            FlowSystem: A new FlowSystem with full timesteps and expanded solution.

        Raises:
            ValueError: If the FlowSystem was not created with ``cluster()``.
            ValueError: If the FlowSystem has no solution.

        Examples:
            Two-stage optimization with expansion:

            >>> # Stage 1: Size with reduced timesteps
            >>> fs_reduced = flow_system.transform.cluster(
            ...     n_clusters=8,
            ...     cluster_duration='1D',
            ... )
            >>> fs_reduced.optimize(solver)
            >>>
            >>> # Expand to full resolution FlowSystem
            >>> fs_expanded = fs_reduced.transform.expand()
            >>>
            >>> # Use all existing accessors with full timesteps
            >>> fs_expanded.stats.flow_rates  # Full 8760 timesteps
            >>> fs_expanded.stats.plot.balance('HeatBus')  # Full resolution plots
            >>> fs_expanded.stats.plot.heatmap('Boiler(Q_th)|flow_rate')

        Note:
            The expanded FlowSystem repeats the typical cluster values for all
            original clusters belonging to the same cluster. Both input data and solution
            are consistently expanded, so they match. This is an approximation -
            the actual dispatch at full resolution would differ due to
            intra-cluster variations in time series data.

            For accurate dispatch results, use ``fix_sizes()`` to fix the sizes
            from the reduced optimization and re-optimize at full resolution.

            **Segmented Systems Variable Handling:**

            For systems clustered with ``SegmentConfig``, special handling is applied
            to time-varying solution variables. Variables without a ``time`` dimension
            are unaffected by segment expansion. This includes:

            - Investment: ``{component}|size``, ``{component}|exists``
            - Storage boundaries: ``{storage}|SOC_boundary``
            - Aggregated totals: ``{flow}|total_flow_hours``, ``{flow}|active_hours``
            - Effect totals: ``{effect}``, ``{effect}(temporal)``, ``{effect}(periodic)``

            Time-varying variables are categorized and handled as follows:

            1. **State variables** - Interpolated within segments:

               - ``{storage}|charge_state``: Linear interpolation between segment
                 boundary values to show the charge trajectory during charge/discharge.

            2. **Segment totals** - Divided by segment duration:

               These variables represent values summed over the segment. Division
               converts them back to hourly rates for correct plotting and analysis.

               - ``{effect}(temporal)|per_timestep``: Per-timestep effect contributions
               - ``{flow}->{effect}(temporal)``: Flow contributions (includes both
                 ``effects_per_flow_hour`` and ``effects_per_startup``)
               - ``{component}->{effect}(temporal)``: Component-level contributions
               - ``{source}(temporal)->{target}(temporal)``: Effect-to-effect shares

            3. **Rate/average variables** - Expanded as-is:

               These variables represent average values within the segment. tsam
               already provides properly averaged values, so no correction needed.

               - ``{flow}|flow_rate``: Average flow rate during segment
               - ``{storage}|netto_discharge``: Net discharge rate (discharge - charge)

            4. **Binary status variables** - Constant within segment:

               These variables cannot be meaningfully interpolated. The status
               indicates the dominant state during the segment.

               - ``{flow}|status``: On/off status (0 or 1), repeated for all timesteps

            5. **Binary event variables** (segmented systems only) - First timestep of segment:

               For segmented systems, these variables indicate that an event occurred
               somewhere during the segment. When expanded, the event is placed at the
               first timestep of each segment, with zeros elsewhere. This preserves the
               total count of events while providing a reasonable temporal placement.

               For non-segmented systems, the timing within the cluster is preserved
               by normal expansion (no special handling needed).

               - ``{flow}|startup``: Startup event
               - ``{flow}|shutdown``: Shutdown event
        """
        clustering = self._validate_for_expansion()
        expander = _Expander(self._fs, clustering)
        return expander.expand_flow_system()
