"""
Transform accessor for FlowSystem.

This module provides the TransformAccessor class that enables
transformations on FlowSystem like clustering, selection, and resampling.
"""

from __future__ import annotations

import logging
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import xarray as xr

from .modeling import _scalar_safe_reduce
from .structure import EXPAND_DIVIDE, EXPAND_FIRST_TIMESTEP, EXPAND_INTERPOLATE, VariableCategory

if TYPE_CHECKING:
    from tsam import ClusterConfig, ExtremeConfig, SegmentConfig

    from .clustering import Clustering
    from .flow_system import FlowSystem

logger = logging.getLogger('flixopt')


class _ReducedFlowSystemBuilder:
    """Builds a reduced FlowSystem from a tsam_xarray AggregationResult.

    This class encapsulates the construction of reduced FlowSystem datasets,
    extracting cluster representatives, weights, and metrics from the
    tsam_xarray result.

    Args:
        fs: The original FlowSystem being reduced.
        agg_result: tsam_xarray AggregationResult with DataArray-based results.
        timesteps_per_cluster: Number of timesteps per cluster.
        dt: Hours per timestep.
    """

    def __init__(
        self,
        fs: FlowSystem,
        agg_result: Any,  # tsam_xarray.AggregationResult
        timesteps_per_cluster: int,
        dt: float,
        unrename_map: dict[str, str] | None = None,
    ):
        self._fs = fs
        self._agg_result = agg_result
        self._timesteps_per_cluster = timesteps_per_cluster
        self._dt = dt
        self._unrename_map = unrename_map or {}

        self._n_clusters = agg_result.n_clusters
        self._is_segmented = agg_result.n_segments is not None
        self._n_segments = agg_result.n_segments

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

    def _unrename(self, da: xr.DataArray) -> xr.DataArray:
        """Rename tsam_xarray output dims back to original names (e.g., _period -> period)."""
        renames = {k: v for k, v in self._unrename_map.items() if k in da.dims}
        return da.rename(renames) if renames else da

    def _unrename_ds(self, ds: xr.Dataset) -> xr.Dataset:
        """Rename tsam_xarray output dims back to original names in Dataset."""
        renames = {k: v for k, v in self._unrename_map.items() if k in ds.dims}
        return ds.rename(renames) if renames else ds

    def build_cluster_weights(self) -> xr.DataArray:
        """Build cluster_weight DataArray from aggregation result.

        Returns:
            DataArray with dims [cluster, period?, scenario?].
        """
        weights = self._agg_result.cluster_weights.rename(cluster='cluster')
        return self._unrename(weights.rename('cluster_weight'))

    def build_typical_periods(self) -> dict[str, xr.DataArray]:
        """Build typical periods DataArrays with (cluster, time, ...) shape.

        Returns:
            Dict mapping column names to DataArrays.
        """
        representatives = self._agg_result.cluster_representatives
        # representatives has dims: (cluster, timestep, variable, _period?, scenario?)
        # We need to split by variable and rename timestep -> time
        result = {}
        # Exclude known dims (including renamed variants like _period, _cluster)
        known_dims = {'cluster', 'timestep', 'period', 'scenario'} | set(self._unrename_map.keys())
        unknown_dims = [d for d in representatives.dims if d not in known_dims]
        assert len(unknown_dims) == 1, (
            f'Expected exactly 1 variable dim, got {unknown_dims} (known: {known_dims}, all: {representatives.dims})'
        )
        variable_dim = unknown_dims[0]
        for var_name in representatives.coords[variable_dim].values:
            da = representatives.sel({variable_dim: var_name}, drop=True)
            # Rename timestep -> time and assign our coordinates
            da = da.rename({'timestep': 'time'})
            da = da.assign_coords(cluster=self._cluster_coords, time=self._time_coords)
            # Ensure cluster and time are first two dims
            other_dims = [d for d in da.dims if d not in ('cluster', 'time')]
            da = da.transpose('cluster', 'time', *other_dims)
            result[str(var_name)] = self._unrename(da)
        return result

    def build_segment_durations(self) -> xr.DataArray:
        """Build timestep_duration DataArray from segment durations.

        Returns:
            DataArray with dims [cluster, time, period?, scenario?].
        """
        if not self._is_segmented:
            raise ValueError('build_segment_durations() requires a segmented system')

        seg_durs = self._agg_result.segment_durations
        # Convert from timestep counts to hours
        da = seg_durs * self._dt
        # Rename dims to match our convention
        da = da.rename({'timestep': 'time'})
        da = da.assign_coords(cluster=self._cluster_coords, time=self._time_coords)
        other_dims = [d for d in da.dims if d not in ('cluster', 'time')]
        return self._unrename(da.transpose('cluster', 'time', *other_dims).rename('timestep_duration'))

    def build_metrics(self) -> xr.Dataset:
        """Build clustering metrics Dataset from aggregation result.

        Returns:
            Dataset with RMSE, MAE, RMSE_duration metrics.
        """
        accuracy = self._agg_result.accuracy
        try:
            data_vars = {}
            for metric_name, metric_da in [
                ('RMSE', accuracy.rmse),
                ('MAE', accuracy.mae),
                ('RMSE_duration', accuracy.rmse_duration),
            ]:
                # Rename the variable dimension to 'time_series'
                known_metric_dims = {'period', 'scenario'} | set(self._unrename_map.keys())
                unknown_dims = [d for d in metric_da.dims if d not in known_metric_dims]
                assert len(unknown_dims) == 1, f'Expected 1 variable dim in {metric_name}, got {unknown_dims}'
                variable_dim = unknown_dims[0]
                da = metric_da.rename({variable_dim: 'time_series'})
                data_vars[metric_name] = da
            return self._unrename_ds(xr.Dataset(data_vars))
        except Exception as e:
            logger.warning(f'Failed to compute clustering metrics: {e}')
            return xr.Dataset()

    def build_reduced_dataset(self, ds: xr.Dataset, typical_das: dict[str, xr.DataArray]) -> xr.Dataset:
        """Build the reduced dataset with (cluster, time) structure.

        Args:
            ds: Original dataset.
            typical_das: Pre-combined DataArrays from build_typical_periods().

        Returns:
            Dataset with reduced timesteps and (cluster, time) structure.
        """
        from .core import TimeSeriesData

        n_reduced_timesteps = self._n_clusters * self._n_time_points

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
                slices[time_idx] = slice(0, n_reduced_timesteps)
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
            _aggregation_result=self._agg_result,
            _unrename_map=self._unrename_map,
        )

        return reduced_fs


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
        self._n_clusters = clustering.n_clusters
        self._n_original_clusters = clustering.n_original_clusters

        # Pre-compute timesteps
        self._original_timesteps = clustering.original_timesteps
        self._n_original_timesteps = len(self._original_timesteps)

        # Import here to avoid circular import
        from .flow_system import FlowSystem

        self._original_timesteps_extra = FlowSystem._create_timesteps_with_extra(self._original_timesteps, None)

        # Index of last valid original cluster (for final state)
        self._last_original_cluster_idx = min(
            (self._n_original_timesteps - 1) // self._timesteps_per_cluster,
            self._n_original_clusters - 1,
        )

        # Build variable category sets from registered categories
        variable_categories = fs._variable_categories
        self._state_vars = {name for name, cat in variable_categories.items() if cat in EXPAND_INTERPOLATE}
        self._first_timestep_vars = {name for name, cat in variable_categories.items() if cat in EXPAND_FIRST_TIMESTEP}
        self._segment_total_vars = {name for name, cat in variable_categories.items() if cat in EXPAND_DIVIDE}

        # Pre-compute expansion divisor for segmented systems (segment durations on original time)
        self._expansion_divisor = None
        if clustering.is_segmented:
            self._expansion_divisor = clustering.disaggregate(clustering.segment_durations).ffill(dim='time')

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

    def expand_dataarray(self, da: xr.DataArray, var_name: str = '', is_solution: bool = False) -> xr.DataArray:
        """Expand a DataArray from clustered to original timesteps.

        Uses clustering.disaggregate() as the core expansion, then applies
        post-processing based on variable category:
        - State variables (segmented): interpolate within segments
        - First-timestep variables (segmented): value at segment start, zero elsewhere
        - Segment totals: divide by segment duration for hourly rate

        Args:
            da: DataArray to expand.
            var_name: Variable name for category-based expansion handling.
            is_solution: Whether this is a solution variable (affects segment total handling).

        Returns:
            Expanded DataArray with original timesteps.
        """
        if 'time' not in da.dims:
            return da.copy()

        clustering = self._clustering
        has_cluster_dim = 'cluster' in da.dims
        is_state = var_name in self._state_vars and has_cluster_dim
        is_first_timestep = var_name in self._first_timestep_vars and has_cluster_dim
        is_segment_total = is_solution and var_name in self._segment_total_vars

        # Solution variables have n+1 timesteps (extra boundary value).
        # Strip it before disaggregating — it will be appended back for state variables.
        expected_time = clustering.n_segments if clustering.is_segmented else clustering.timesteps_per_cluster
        has_extra = has_cluster_dim and da.sizes.get('time', 0) > expected_time
        da_for_disagg = da.isel(time=slice(None, expected_time)) if has_extra else da

        # Disaggregate: map (cluster, time) back to original time axis.
        # For non-segmented: values are repeated. For segmented: NaN between boundaries.
        expanded = clustering.disaggregate(da_for_disagg)

        # Post-processing for segmented systems
        if clustering.is_segmented and has_cluster_dim:
            if is_state:
                expanded = expanded.interpolate_na(dim='time')
            elif is_first_timestep and is_solution:
                return expanded.fillna(0).assign_attrs(da.attrs)
            else:
                expanded = expanded.ffill(dim='time')
                if is_segment_total and self._expansion_divisor is not None:
                    expanded = expanded / self._expansion_divisor

        # State variables need final state appended
        if is_state:
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
        soc_boundary_vars = self._fs.get_variables_by_category(VariableCategory.SOC_BOUNDARY)

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
        if reduced_solution is not None:
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
        n_segments = self._clustering.n_segments
        time_dim_size = n_segments if n_segments else self._timesteps_per_cluster
        n_reduced_timesteps = self._n_clusters * time_dim_size
        segmented_info = f' ({n_segments} segments)' if n_segments else ''
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
        from .flow_system import FlowSystem

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
            result = FlowSystem._update_time_metadata(result, hours_of_last_timestep, hours_of_previous_timesteps)

        if 'period' in indexers:
            result = FlowSystem._update_period_metadata(result)

        if 'scenario' in indexers:
            result = FlowSystem._update_scenario_metadata(result)

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
        from .flow_system import FlowSystem

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
            result = FlowSystem._update_time_metadata(result, hours_of_last_timestep, hours_of_previous_timesteps)

        if 'period' in indexers:
            result = FlowSystem._update_period_metadata(result)

        if 'scenario' in indexers:
            result = FlowSystem._update_scenario_metadata(result)

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
        from .flow_system import FlowSystem

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
            return FlowSystem._update_time_metadata(result, hours_of_last_timestep, hours_of_previous_timesteps)

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
        return FlowSystem._update_time_metadata(result, hours_of_last_timestep, hours_of_previous_timesteps)

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

    def cluster(
        self,
        n_clusters: int,
        cluster_duration: str | float,
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
            cluster: Optional tsam ``ClusterConfig`` object specifying clustering algorithm,
                representation method, and weights. Use ``weights={var: 0}`` to exclude
                specific variables from influencing cluster assignments while still
                aggregating them. If None, uses default settings (hierarchical clustering
                with medoid representation).
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

            Clustering based on specific variables only (zero-weight the rest):

            >>> from tsam import ClusterConfig
            >>> fs_clustered = flow_system.transform.cluster(
            ...     n_clusters=8,
            ...     cluster_duration='1D',
            ...     cluster=ClusterConfig(
            ...         weights={
            ...             'HeatDemand(Q)|fixed_relative_profile': 1,
            ...             'GasSource(Gas)|costs|per_flow_hour': 0,  # ignored for clustering
            ...         }
            ...     ),
            ... )

        Note:
            - This is best suited for initial sizing, not final dispatch optimization
            - Use ``extremes`` to ensure peak demand clusters are captured
            - A 5-10% safety margin on sizes is recommended for the dispatch stage
            - For seasonal storage (e.g., hydrogen, thermal storage), set
              ``Storage.cluster_mode='intercluster'`` or ``'intercluster_cyclic'``
        """
        import tsam_xarray

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

        ds = self._fs.to_dataset(include_solution=False)

        # Only keep variables with a time dimension for clustering
        ds_for_clustering = ds[[name for name in ds.data_vars if 'time' in ds[name].dims]]

        if not ds_for_clustering.data_vars:
            raise ValueError('No time-varying data found for clustering. Check your input data.')

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
        has_slices = has_periods or has_scenarios
        if has_slices and extremes is not None:
            if extremes.method != 'replace':
                raise ValueError(
                    f"ExtremeConfig method='{extremes.method}' is not supported for multi-period "
                    "or multi-scenario systems. Only method='replace' reliably produces consistent "
                    'cluster counts across all slices. Use: '
                    "ExtremeConfig(..., method='replace')"
                )

        # Rename reserved dimension names to avoid conflict with tsam_xarray
        # tsam_xarray reserves: 'period', 'cluster', 'timestep'
        reserved_renames = {'period': '_period', 'cluster': '_cluster'}
        # Check against full ds dims (period/cluster may only exist as coords, not in ds_for_clustering)
        rename_map = {k: v for k, v in reserved_renames.items() if k in ds.dims}
        unrename_map = {v: k for k, v in rename_map.items()}

        if rename_map:
            # Only rename dims that exist in each dataset
            clustering_renames = {k: v for k, v in rename_map.items() if k in ds_for_clustering.dims}
            if clustering_renames:
                ds_for_clustering = ds_for_clustering.rename(clustering_renames)
            ds = ds.rename(rename_map)

        # Stack Dataset into a single DataArray with 'variable' dimension
        da_for_clustering = ds_for_clustering.to_dataarray(dim='variable')

        # Ensure period/scenario dimensions are present in the DataArray
        # even if the data doesn't vary across them (tsam_xarray needs them for slicing)
        extra_dims = []
        if has_periods:
            extra_dims.append(rename_map.get('period', 'period'))
        if has_scenarios:
            extra_dims.append(rename_map.get('scenario', 'scenario'))
        for dim_name in extra_dims:
            if dim_name not in da_for_clustering.dims and dim_name in ds.dims:
                # Drop as non-dim coordinate first (to_dataarray may keep it as scalar coord)
                if dim_name in da_for_clustering.coords:
                    da_for_clustering = da_for_clustering.drop_vars(dim_name)
                da_for_clustering = da_for_clustering.expand_dims({dim_name: ds.coords[dim_name].values})

        # Pass user-specified weights to tsam_xarray (validates unknown keys)
        if cluster is not None and cluster.weights is not None:
            weights = dict(cluster.weights)
        else:
            weights = {}

        # Build tsam_kwargs with explicit parameters
        tsam_kwargs_full = {
            'period_duration': hours_per_cluster,
            'temporal_resolution': dt,
            'extremes': extremes,
            'segments': segments,
            'preserve_column_means': preserve_column_means,
            'rescale_exclude_columns': rescale_exclude_columns,
            'round_decimals': round_decimals,
            'numerical_tolerance': numerical_tolerance,
            **tsam_kwargs,
        }

        # Pass cluster config settings (without weights, which go to tsam_xarray directly)
        if cluster is not None:
            from tsam import ClusterConfig

            cluster_config = ClusterConfig(
                method=cluster.method,
                representation=cluster.representation,
                normalize_column_means=cluster.normalize_column_means,
                use_duration_curves=cluster.use_duration_curves,
                include_period_sums=cluster.include_period_sums,
                solver=cluster.solver,
            )
            tsam_kwargs_full['cluster'] = cluster_config

        # Suppress tsam warning about minimal value constraints (informational, not actionable)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, message='.*minimal value.*exceeds.*')

            # Single call: tsam_xarray handles (period, scenario) slicing automatically
            agg_result = tsam_xarray.aggregate(
                da_for_clustering,
                time_dim='time',
                cluster_dim='variable',
                n_clusters=n_clusters,
                weights=weights,
                **tsam_kwargs_full,
            )

        # Rename reserved dims back to original names in the dataset
        if unrename_map:
            ds = ds.rename(unrename_map)

        # Build and return the reduced FlowSystem
        builder = _ReducedFlowSystemBuilder(self._fs, agg_result, timesteps_per_cluster, dt, unrename_map)
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
                f'Ensure self._fs.timesteps matches the original data used for clustering.'
            )

        # Rename reserved dimension names to avoid conflict with tsam_xarray
        reserved_renames = {'period': '_period', 'cluster': '_cluster'}
        rename_map = {k: v for k, v in reserved_renames.items() if k in ds.dims}
        unrename_map = {v: k for k, v in rename_map.items()}

        if rename_map:
            ds = ds.rename(rename_map)

        # Apply existing clustering to full data
        logger.info('Applying clustering...')
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, message='.*minimal value.*exceeds.*')
            da_full = ds.to_dataarray(dim='variable')

            # Ensure extra dims are present in DataArray
            for _orig_name, renamed in rename_map.items():
                if renamed not in da_full.dims and renamed in ds.dims:
                    if renamed in da_full.coords:
                        da_full = da_full.drop_vars(renamed)
                    da_full = da_full.expand_dims({renamed: ds.coords[renamed].values})

            # Get clustering result with correct dim names for the renamed data
            from tsam_xarray import ClusteringResult as ClusteringResultClass

            cr_result = clustering.clustering_result
            # Map dim names to renamed versions (e.g., period → _period)
            slice_dims = [rename_map.get(d, d) for d in clustering.dim_names]
            cr_result = ClusteringResultClass(
                time_dim='time',
                cluster_dim=['variable'],
                slice_dims=slice_dims,
                clusterings=dict(cr_result.clusterings),
            )
            # TODO(tsam_xarray): Same workaround as in cluster() above — remove
            # once tsam_xarray handles mismatched weights in apply().
            for cr in cr_result.clusterings.values():
                object.__setattr__(cr, 'weights', {})
            agg_result = cr_result.apply(da_full)

        # Rename back
        if unrename_map:
            ds = ds.rename(unrename_map)

        # Build and return the reduced FlowSystem
        builder = _ReducedFlowSystemBuilder(self._fs, agg_result, timesteps_per_cluster, dt, unrename_map)
        return builder.build(ds)

    def _validate_for_expansion(self) -> Clustering:
        """Validate FlowSystem can be expanded and return clustering info.

        Returns:
            The Clustering object.

        Raises:
            ValueError: If FlowSystem wasn't created with cluster().
        """

        if self._fs.clustering is None:
            raise ValueError(
                'expand() requires a FlowSystem created with cluster(). This FlowSystem has no aggregation info.'
            )

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
