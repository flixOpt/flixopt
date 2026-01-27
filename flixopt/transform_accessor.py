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

from .clustering.aggregation import (
    accuracy_to_dataframe,
    build_cluster_config_with_weights,
    build_cluster_weights,
    build_clustering_metrics,
    build_segment_durations,
    build_typical_dataarrays,
    calculate_clustering_weights,
    combine_slices_to_dataarray,
)
from .clustering.expansion import (
    VariableExpansionHandler,
    append_final_state,
    expand_first_timestep_only,
    interpolate_charge_state_segmented,
)
from .clustering.intercluster_helpers import combine_intercluster_charge_states
from .clustering.iteration import DimInfo, iter_dim_slices
from .structure import VariableCategory

if TYPE_CHECKING:
    from tsam import ClusterConfig, ExtremeConfig, SegmentConfig

    from .clustering import Clustering
    from .flow_system import FlowSystem

logger = logging.getLogger('flixopt')


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

    # Note: Static methods _calculate_clustering_weights, _build_cluster_config_with_weights,
    # and _accuracy_to_dataframe have been moved to flixopt/clustering/aggregation.py
    # for better modularity. They are imported at the top of this module.

    # Note: The following methods have been moved to flixopt/clustering/aggregation.py:
    # - _build_cluster_weight_da -> build_cluster_weights()
    # - _build_typical_das -> build_typical_dataarrays()
    # - _build_segment_durations_da -> build_segment_durations()
    # - _build_clustering_metrics -> build_clustering_metrics()
    # They are imported and used directly from the clustering module.

    def _build_reduced_flow_system(
        self,
        ds: xr.Dataset,
        tsam_aggregation_results: dict[tuple, Any],
        cluster_occurrences_all: dict[tuple, dict],
        clustering_metrics_all: dict[tuple, pd.DataFrame],
        timesteps_per_cluster: int,
        dt: float,
        dim_info: DimInfo,
        n_clusters_requested: int | None = None,
    ) -> FlowSystem:
        """Build a reduced FlowSystem from tsam aggregation results.

        This is the shared implementation used by both cluster() and apply_clustering().

        Args:
            ds: Original dataset.
            tsam_aggregation_results: Dict mapping (period, scenario) to tsam AggregationResult.
            cluster_occurrences_all: Dict mapping (period, scenario) to cluster occurrence counts.
            clustering_metrics_all: Dict mapping (period, scenario) to accuracy metrics.
            timesteps_per_cluster: Number of timesteps per cluster.
            dt: Hours per timestep.
            dim_info: DimInfo with period/scenario information.
            n_clusters_requested: Requested number of clusters (for logging). None to skip.

        Returns:
            Reduced FlowSystem with clustering metadata attached.
        """
        from .clustering import Clustering
        from .core import drop_constant_arrays
        from .flow_system import FlowSystem

        # Build dict keyed by ClusteringResults format (without None)
        aggregation_results: dict[tuple, Any] = {}
        for (p, s), result in tsam_aggregation_results.items():
            key = dim_info.to_clustering_key(p, s)
            aggregation_results[key] = result

        # Use first result for structure
        first_key = (dim_info.periods[0], dim_info.scenarios[0])
        first_tsam = tsam_aggregation_results[first_key]

        # Build metrics using extracted helper
        clustering_metrics = build_clustering_metrics(clustering_metrics_all, dim_info)

        n_reduced_timesteps = len(first_tsam.cluster_representatives)
        actual_n_clusters = len(first_tsam.cluster_weights)

        # Create coordinates for the 2D cluster structure
        cluster_coords = np.arange(actual_n_clusters)

        # Detect if segmentation was used
        is_segmented = first_tsam.n_segments is not None
        n_segments = first_tsam.n_segments if is_segmented else None

        # Determine time dimension based on segmentation
        if is_segmented:
            n_time_points = n_segments
            time_coords = pd.RangeIndex(n_time_points, name='time')
        else:
            n_time_points = timesteps_per_cluster
            time_coords = pd.date_range(
                start='2000-01-01',
                periods=timesteps_per_cluster,
                freq=pd.Timedelta(hours=dt),
                name='time',
            )

        # Build cluster_weight using extracted helper
        cluster_weight = build_cluster_weights(cluster_occurrences_all, actual_n_clusters, cluster_coords, dim_info)

        # Logging
        if is_segmented:
            logger.info(
                f'Reduced from {len(self._fs.timesteps)} to {actual_n_clusters} clusters × {n_segments} segments'
            )
        else:
            logger.info(
                f'Reduced from {len(self._fs.timesteps)} to {actual_n_clusters} clusters × {timesteps_per_cluster} timesteps'
            )

        # Build typical periods DataArrays with (cluster, time) shape
        typical_das = build_typical_dataarrays(
            tsam_aggregation_results, actual_n_clusters, n_time_points, cluster_coords, time_coords, is_segmented
        )

        # Build reduced dataset with (cluster, time) dimensions
        ds_new = self._build_reduced_dataset(
            ds,
            typical_das,
            actual_n_clusters,
            n_reduced_timesteps,
            n_time_points,
            cluster_coords,
            time_coords,
            dim_info,
        )

        # For segmented systems, build timestep_duration from segment_durations
        if is_segmented:
            segment_durations_da = build_segment_durations(
                tsam_aggregation_results,
                actual_n_clusters,
                n_segments,
                cluster_coords,
                time_coords,
                dt,
                dim_info,
            )
            ds_new['timestep_duration'] = segment_durations_da

        reduced_fs = FlowSystem.from_dataset(ds_new)
        reduced_fs.cluster_weight = cluster_weight

        # Remove 'equals_final' from storages - doesn't make sense on reduced timesteps
        for storage in reduced_fs.storages.values():
            ics = storage.initial_charge_state
            if isinstance(ics, str) and ics == 'equals_final':
                storage.initial_charge_state = None

        # Create Clustering object with full AggregationResult access
        # Only store time-varying data (constant arrays are clutter for plotting)
        reduced_fs.clustering = Clustering(
            original_timesteps=self._fs.timesteps,
            original_data=drop_constant_arrays(ds, dim='time'),
            aggregated_data=drop_constant_arrays(ds_new, dim='time'),
            _metrics=clustering_metrics if clustering_metrics.data_vars else None,
            _aggregation_results=aggregation_results,
            _dim_names=dim_info.dim_names,
        )

        return reduced_fs

    def _build_reduced_dataset(
        self,
        ds: xr.Dataset,
        typical_das: dict[str, dict[tuple, xr.DataArray]],
        actual_n_clusters: int,
        n_reduced_timesteps: int,
        n_time_points: int,
        cluster_coords: np.ndarray,
        time_coords: pd.DatetimeIndex | pd.RangeIndex,
        dim_info: DimInfo,
    ) -> xr.Dataset:
        """Build the reduced dataset with (cluster, time) structure.

        Args:
            ds: Original dataset.
            typical_das: Typical periods DataArrays from build_typical_dataarrays().
            actual_n_clusters: Number of clusters.
            n_reduced_timesteps: Total reduced timesteps (n_clusters * n_time_points).
            n_time_points: Number of time points per cluster (timesteps or segments).
            cluster_coords: Cluster coordinate values.
            time_coords: Time coordinate values.
            dim_info: DimInfo with period/scenario information.

        Returns:
            Dataset with reduced timesteps and (cluster, time) structure.
        """
        from .core import TimeSeriesData

        all_keys = {(p, s) for p in dim_info.periods for s in dim_info.scenarios}
        ds_new_vars = {}

        # Use ds.variables to avoid _construct_dataarray overhead
        variables = ds.variables
        coord_cache = {k: ds.coords[k].values for k in ds.coords}

        for name in ds.data_vars:
            var = variables[name]
            if 'time' not in var.dims:
                # No time dimension - wrap Variable in DataArray
                coords = {d: coord_cache[d] for d in var.dims if d in coord_cache}
                ds_new_vars[name] = xr.DataArray(var.values, dims=var.dims, coords=coords, attrs=var.attrs, name=name)
            elif name not in typical_das:
                # Time-dependent but constant: reshape to (cluster, time, ...)
                # Use numpy slicing instead of .isel()
                time_idx = var.dims.index('time')
                slices = [slice(None)] * len(var.dims)
                slices[time_idx] = slice(0, n_reduced_timesteps)
                sliced_values = var.values[tuple(slices)]

                other_dims = [d for d in var.dims if d != 'time']
                other_shape = [var.sizes[d] for d in other_dims]
                new_shape = [actual_n_clusters, n_time_points] + other_shape
                reshaped = sliced_values.reshape(new_shape)
                new_coords = {'cluster': cluster_coords, 'time': time_coords}
                for dim in other_dims:
                    if dim in coord_cache:
                        new_coords[dim] = coord_cache[dim]
                ds_new_vars[name] = xr.DataArray(
                    reshaped,
                    dims=['cluster', 'time'] + other_dims,
                    coords=new_coords,
                    attrs=var.attrs,
                )
            elif set(typical_das[name].keys()) != all_keys:
                # Partial typical slices: fill missing keys with constant values
                # For multi-period/scenario data, we need to select the right slice for each key

                # Exclude 'period' and 'scenario' - they're handled by combine_slices_to_dataarray
                other_dims = [d for d in var.dims if d not in ('time', 'period', 'scenario')]
                other_shape = [var.sizes[d] for d in other_dims]
                new_shape = [actual_n_clusters, n_time_points] + other_shape

                new_coords = {'cluster': cluster_coords, 'time': time_coords}
                for dim in other_dims:
                    if dim in coord_cache:
                        new_coords[dim] = coord_cache[dim]

                # Build filled slices dict: use typical where available, constant otherwise
                filled_slices = {}
                for key in all_keys:
                    if key in typical_das[name]:
                        filled_slices[key] = typical_das[name][key]
                    else:
                        # Select the specific period/scenario slice, then reshape
                        period_label, scenario_label = key
                        selector = {}
                        if period_label is not None and 'period' in var.dims:
                            selector['period'] = period_label
                        if scenario_label is not None and 'scenario' in var.dims:
                            selector['scenario'] = scenario_label

                        # Select per-key slice if needed, otherwise use full variable
                        if selector:
                            var_slice = ds[name].sel(**selector, drop=True)
                        else:
                            var_slice = ds[name]

                        # Now slice time and reshape
                        time_idx = var_slice.dims.index('time')
                        slices_list = [slice(None)] * len(var_slice.dims)
                        slices_list[time_idx] = slice(0, n_reduced_timesteps)
                        sliced_values = var_slice.values[tuple(slices_list)]
                        reshaped_constant = sliced_values.reshape(new_shape)

                        filled_slices[key] = xr.DataArray(
                            reshaped_constant,
                            dims=['cluster', 'time'] + other_dims,
                            coords=new_coords,
                        )

                da = combine_slices_to_dataarray(
                    slices=filled_slices,
                    dim_info=dim_info,
                    base_dims=['cluster', 'time'],
                    attrs=var.attrs,
                )
                if var.attrs.get('__timeseries_data__', False):
                    da = TimeSeriesData.from_dataarray(da.assign_attrs(var.attrs))
                ds_new_vars[name] = da
            else:
                # Time-varying: combine per-(period, scenario) slices
                da = combine_slices_to_dataarray(
                    slices=typical_das[name],
                    dim_info=dim_info,
                    base_dims=['cluster', 'time'],
                    attrs=var.attrs,
                )
                if var.attrs.get('__timeseries_data__', False):
                    da = TimeSeriesData.from_dataarray(da.assign_attrs(var.attrs))
                ds_new_vars[name] = da

        # Copy attrs but remove cluster_weight
        new_attrs = dict(ds.attrs)
        new_attrs.pop('cluster_weight', None)
        return xr.Dataset(ds_new_vars, attrs=new_attrs)

    def _build_cluster_assignments_da(
        self,
        cluster_assignmentss: dict[tuple, np.ndarray],
        periods: list,
        scenarios: list,
    ) -> xr.DataArray:
        """Build cluster_assignments DataArray from cluster assignments.

        Args:
            cluster_assignmentss: Dict mapping (period, scenario) to cluster assignment arrays.
            periods: List of period labels ([None] if no periods dimension).
            scenarios: List of scenario labels ([None] if no scenarios dimension).

        Returns:
            DataArray with dims [original_cluster] or [original_cluster, period?, scenario?].
        """
        has_periods = periods != [None]
        has_scenarios = scenarios != [None]

        if has_periods or has_scenarios:
            # Multi-dimensional case
            cluster_assignments_slices = {}
            for p in periods:
                for s in scenarios:
                    key = (p, s)
                    cluster_assignments_slices[key] = xr.DataArray(
                        cluster_assignmentss[key], dims=['original_cluster'], name='cluster_assignments'
                    )
            return self._combine_slices_to_dataarray_generic(
                cluster_assignments_slices, ['original_cluster'], periods, scenarios, 'cluster_assignments'
            )
        else:
            # Simple case
            first_key = (periods[0], scenarios[0])
            return xr.DataArray(cluster_assignmentss[first_key], dims=['original_cluster'], name='cluster_assignments')

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

        # Build DimInfo for standardized iteration
        dim_info = DimInfo.from_flow_system(self._fs)

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
        # Methods 'new_cluster' and 'append' can produce different n_clusters per period,
        # which breaks the xarray structure that requires uniform dimensions
        is_multi_dimensional = dim_info.has_periods or dim_info.has_scenarios
        if is_multi_dimensional and extremes is not None:
            extreme_method = getattr(extremes, 'method', None)
            if extreme_method in ('new_cluster', 'append'):
                raise ValueError(
                    f'ExtremeConfig with method="{extreme_method}" is not supported for multi-period '
                    f'or multi-scenario systems because it can produce different cluster counts per '
                    f'period/scenario. Use method="replace" instead, which replaces existing clusters '
                    f'with extreme periods while maintaining the requested n_clusters.'
                )

        # Cluster each (period, scenario) combination using tsam directly
        tsam_aggregation_results: dict[tuple, Any] = {}  # AggregationResult objects
        tsam_clustering_results: dict[tuple, Any] = {}  # ClusteringResult objects for persistence
        cluster_assignmentss: dict[tuple, np.ndarray] = {}
        cluster_occurrences_all: dict[tuple, dict] = {}

        # Collect metrics per (period, scenario) slice
        clustering_metrics_all: dict[tuple, pd.DataFrame] = {}

        for ctx in iter_dim_slices(dim_info):
            # Select data for clustering (may be subset if data_vars specified)
            ds_slice_for_clustering = (
                ds_for_clustering.sel(**ctx.selector, drop=True) if ctx.selector else ds_for_clustering
            )
            temporaly_changing_ds_for_clustering = drop_constant_arrays(ds_slice_for_clustering, dim='time')

            # Guard against empty dataset after removing constant arrays
            if not temporaly_changing_ds_for_clustering.data_vars:
                filter_info = f'data_vars={data_vars}' if data_vars else 'all variables'
                selector_info = f', selector={ctx.selector}' if ctx.selector else ''
                raise ValueError(
                    f'No time-varying data found for clustering ({filter_info}{selector_info}). '
                    f'All variables are constant over time. Check your data_vars filter or input data.'
                )

            df_for_clustering = temporaly_changing_ds_for_clustering.to_dataframe()

            if ctx.selector:
                logger.info(f'Clustering {", ".join(f"{k}={v}" for k, v in ctx.selector.items())}...')

            # Suppress tsam warning about minimal value constraints (informational, not actionable)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning, message='.*minimal value.*exceeds.*')

                # Build ClusterConfig with auto-calculated weights
                clustering_weights = calculate_clustering_weights(temporaly_changing_ds_for_clustering)
                filtered_weights = {
                    name: w for name, w in clustering_weights.items() if name in df_for_clustering.columns
                }
                cluster_config = build_cluster_config_with_weights(cluster, filtered_weights)

                # Perform clustering based on selected data_vars (or all if not specified)
                tsam_result = tsam.aggregate(
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

            tsam_aggregation_results[ctx.key] = tsam_result
            tsam_clustering_results[ctx.key] = tsam_result.clustering
            cluster_assignmentss[ctx.key] = tsam_result.cluster_assignments
            cluster_occurrences_all[ctx.key] = tsam_result.cluster_weights
            try:
                clustering_metrics_all[ctx.key] = accuracy_to_dataframe(tsam_result.accuracy)
            except Exception as e:
                logger.warning(f'Failed to compute clustering metrics for {ctx.key}: {e}')
                clustering_metrics_all[ctx.key] = pd.DataFrame()

        # If data_vars was specified, apply clustering to FULL data
        if data_vars is not None:
            # Build ClusteringResults from subset clustering
            clustering_results = ClusteringResults(
                {dim_info.to_clustering_key(p, s): cr for (p, s), cr in tsam_clustering_results.items()},
                dim_info.dim_names,
            )

            # Apply to full data - this returns AggregationResults
            agg_results = clustering_results.apply(ds)

            # Update tsam_aggregation_results with full data results
            for cr_key, result in agg_results:
                # Convert back to (period, scenario) format
                full_key = dim_info.from_clustering_key(cr_key)
                tsam_aggregation_results[full_key] = result
                cluster_occurrences_all[full_key] = result.cluster_weights

        # Build and return the reduced FlowSystem
        return self._build_reduced_flow_system(
            ds=ds,
            tsam_aggregation_results=tsam_aggregation_results,
            cluster_occurrences_all=cluster_occurrences_all,
            clustering_metrics_all=clustering_metrics_all,
            timesteps_per_cluster=timesteps_per_cluster,
            dt=dt,
            dim_info=dim_info,
            n_clusters_requested=n_clusters,
        )

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

        # Build DimInfo for standardized iteration
        dim_info = DimInfo.from_flow_system(self._fs)

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

        # Convert AggregationResults to the dict format expected by _build_reduced_flow_system
        tsam_aggregation_results: dict[tuple, Any] = {}
        cluster_occurrences_all: dict[tuple, dict] = {}
        clustering_metrics_all: dict[tuple, pd.DataFrame] = {}

        for cr_key, result in agg_results:
            # Convert ClusteringResults key to (period, scenario) format
            full_key = dim_info.from_clustering_key(cr_key)

            tsam_aggregation_results[full_key] = result
            cluster_occurrences_all[full_key] = result.cluster_weights
            try:
                clustering_metrics_all[full_key] = accuracy_to_dataframe(result.accuracy)
            except Exception as e:
                logger.warning(f'Failed to compute clustering metrics for {full_key}: {e}')
                clustering_metrics_all[full_key] = pd.DataFrame()

        # Build and return the reduced FlowSystem
        return self._build_reduced_flow_system(
            ds=ds,
            tsam_aggregation_results=tsam_aggregation_results,
            cluster_occurrences_all=cluster_occurrences_all,
            clustering_metrics_all=clustering_metrics_all,
            timesteps_per_cluster=timesteps_per_cluster,
            dt=dt,
            dim_info=dim_info,
        )

    # Note: _combine_slices_to_dataarray_generic and _combine_slices_to_dataarray_2d
    # have been unified into combine_slices_to_dataarray() in flixopt/clustering/aggregation.py

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

    # Note: _combine_intercluster_charge_states, _apply_soc_decay, _build_segment_total_varnames,
    # _interpolate_charge_state_segmented, and _expand_first_timestep_only have been moved to:
    # - flixopt/clustering/intercluster_helpers.py (combine_intercluster_charge_states, apply_soc_decay)
    # - flixopt/clustering/expansion.py (interpolate_charge_state_segmented, expand_first_timestep_only, build_segment_total_varnames)

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
        from .flow_system import FlowSystem

        # Validate and extract clustering info
        clustering = self._validate_for_expansion()

        timesteps_per_cluster = clustering.timesteps_per_cluster
        # For segmented systems, the time dimension has n_segments entries
        n_segments = clustering.n_segments
        time_dim_size = n_segments if n_segments is not None else timesteps_per_cluster
        n_clusters = clustering.n_clusters
        n_original_clusters = clustering.n_original_clusters

        # Get original timesteps and dimensions
        original_timesteps = clustering.original_timesteps
        n_original_timesteps = len(original_timesteps)
        original_timesteps_extra = FlowSystem._create_timesteps_with_extra(original_timesteps, None)

        # Create expansion handler for variable-specific expansion logic
        variable_categories = getattr(self._fs, '_variable_categories', {})
        handler = VariableExpansionHandler(variable_categories, clustering, self._fs)

        def expand_da(da: xr.DataArray, var_name: str = '', is_solution: bool = False) -> xr.DataArray:
            """Expand a DataArray from clustered to original timesteps."""
            if 'time' not in da.dims:
                return da.copy()

            is_state = handler.is_state_variable(var_name) and 'cluster' in da.dims
            is_first_timestep = handler.is_first_timestep_variable(var_name) and 'cluster' in da.dims

            # State variables in segmented systems: interpolate within segments
            if is_state and clustering.is_segmented:
                expanded = interpolate_charge_state_segmented(da, clustering, original_timesteps)
                return append_final_state(expanded, da, clustering, original_timesteps_extra)

            # Binary events (startup/shutdown) in segmented systems: first timestep of each segment
            # For non-segmented systems, timing within cluster is preserved, so normal expansion is correct
            if is_first_timestep and is_solution and clustering.is_segmented:
                return expand_first_timestep_only(da, clustering, original_timesteps)

            expanded = clustering.expand_data(da, original_time=original_timesteps)

            # Segment totals: divide by expansion divisor
            if is_solution and handler.expansion_divisor is not None and handler.is_segment_total_variable(var_name):
                expanded = expanded / handler.expansion_divisor

            # State variables: append final state
            if is_state:
                expanded = append_final_state(expanded, da, clustering, original_timesteps_extra)

            return expanded

        # Helper to construct DataArray without slow _construct_dataarray
        def _fast_get_da(ds: xr.Dataset, name: str, coord_cache: dict) -> xr.DataArray:
            variable = ds.variables[name]
            var_dims = set(variable.dims)
            coords = {k: v for k, v in coord_cache.items() if set(v.dims).issubset(var_dims)}
            return xr.DataArray(variable, coords=coords, name=name)

        # 1. Expand FlowSystem data
        reduced_ds = self._fs.to_dataset(include_solution=False)
        clustering_attrs = {'is_clustered', 'n_clusters', 'timesteps_per_cluster', 'clustering', 'cluster_weight'}
        skip_vars = {'cluster_weight', 'timestep_duration'}  # These have special handling
        data_vars = {}
        # Use ds.variables pattern to avoid slow _construct_dataarray calls
        coord_cache = {k: v for k, v in reduced_ds.coords.items()}
        coord_names = set(coord_cache)
        for name in reduced_ds.variables:
            if name in coord_names:
                continue
            if name in skip_vars or name.startswith('clustering|'):
                continue
            da = _fast_get_da(reduced_ds, name, coord_cache)
            # Skip vars with cluster dim but no time dim - they don't make sense after expansion
            # (e.g., representative_weights with dims ('cluster',) or ('cluster', 'period'))
            if 'cluster' in da.dims and 'time' not in da.dims:
                continue
            data_vars[name] = expand_da(da, name)
        # Remove timestep_duration reference from attrs - let FlowSystem compute it from timesteps_extra
        # This ensures proper time coordinates for xarray alignment with N+1 solution timesteps
        attrs = {k: v for k, v in reduced_ds.attrs.items() if k not in clustering_attrs and k != 'timestep_duration'}
        expanded_ds = xr.Dataset(data_vars, attrs=attrs)

        expanded_fs = FlowSystem.from_dataset(expanded_ds)

        # 2. Expand solution (with segment total correction for segmented systems)
        reduced_solution = self._fs.solution
        # Use ds.variables pattern to avoid slow _construct_dataarray calls
        sol_coord_cache = {k: v for k, v in reduced_solution.coords.items()}
        sol_coord_names = set(sol_coord_cache)
        expanded_sol_vars = {}
        for name in reduced_solution.variables:
            if name in sol_coord_names:
                continue
            da = _fast_get_da(reduced_solution, name, sol_coord_cache)
            expanded_sol_vars[name] = expand_da(da, name, is_solution=True)
        expanded_fs._solution = xr.Dataset(expanded_sol_vars, attrs=reduced_solution.attrs)
        expanded_fs._solution = expanded_fs._solution.reindex(time=original_timesteps_extra)

        # 3. Combine charge_state with SOC_boundary for intercluster storages
        soc_boundary_vars = self._fs.get_variables_by_category(VariableCategory.SOC_BOUNDARY)
        combine_intercluster_charge_states(
            expanded_fs,
            reduced_solution,
            clustering,
            original_timesteps_extra,
            timesteps_per_cluster,
            n_original_clusters,
            soc_boundary_vars,
        )

        # Log expansion info
        has_periods = self._fs.periods is not None
        has_scenarios = self._fs.scenarios is not None
        n_combinations = (len(self._fs.periods) if has_periods else 1) * (
            len(self._fs.scenarios) if has_scenarios else 1
        )
        n_reduced_timesteps = n_clusters * time_dim_size
        segmented_info = f' ({n_segments} segments)' if n_segments else ''
        logger.info(
            f'Expanded FlowSystem from {n_reduced_timesteps} to {n_original_timesteps} timesteps '
            f'({n_clusters} clusters{segmented_info}'
            + (
                f', {n_combinations} period/scenario combinations)'
                if n_combinations > 1
                else f' → {n_original_clusters} original clusters)'
            )
        )

        return expanded_fs
