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

from .structure import VariableCategory

if TYPE_CHECKING:
    from tsam.config import ClusterConfig, ExtremeConfig, SegmentConfig

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
        for name, da in ds.data_vars.items():
            clustering_group = da.attrs.get('clustering_group')
            group_weight = group_weights.get(clustering_group)
            if group_weight is not None:
                weights[name] = group_weight
            else:
                weights[name] = da.attrs.get('clustering_weight', 1)

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
        from tsam.config import ClusterConfig

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

    @staticmethod
    def _accuracy_to_dataframe(accuracy) -> pd.DataFrame:
        """Convert tsam AccuracyMetrics to DataFrame.

        Args:
            accuracy: tsam AccuracyMetrics object.

        Returns:
            DataFrame with RMSE, MAE, and RMSE_duration columns.
        """
        return pd.DataFrame(
            {
                'RMSE': accuracy.rmse,
                'MAE': accuracy.mae,
                'RMSE_duration': accuracy.rmse_duration,
            }
        )

    def _build_cluster_weight_da(
        self,
        cluster_occurrences_all: dict[tuple, dict],
        n_clusters: int,
        cluster_coords: np.ndarray,
        periods: list,
        scenarios: list,
    ) -> xr.DataArray:
        """Build cluster_weight DataArray from occurrence counts.

        Args:
            cluster_occurrences_all: Dict mapping (period, scenario) tuples to
                dicts of {cluster_id: occurrence_count}.
            n_clusters: Number of clusters.
            cluster_coords: Cluster coordinate values.
            periods: List of period labels ([None] if no periods dimension).
            scenarios: List of scenario labels ([None] if no scenarios dimension).

        Returns:
            DataArray with dims [cluster] or [cluster, period?, scenario?].
        """

        def _weight_for_key(key: tuple) -> xr.DataArray:
            occurrences = cluster_occurrences_all[key]
            weights = np.array([occurrences.get(c, 1) for c in range(n_clusters)])
            return xr.DataArray(weights, dims=['cluster'], coords={'cluster': cluster_coords})

        weight_slices = {key: _weight_for_key(key) for key in cluster_occurrences_all}
        return self._combine_slices_to_dataarray_generic(
            weight_slices, ['cluster'], periods, scenarios, 'cluster_weight'
        )

    def _build_typical_das(
        self,
        tsam_aggregation_results: dict[tuple, Any],
        actual_n_clusters: int,
        n_time_points: int,
        cluster_coords: np.ndarray,
        time_coords: pd.DatetimeIndex | pd.RangeIndex,
        is_segmented: bool = False,
    ) -> dict[str, dict[tuple, xr.DataArray]]:
        """Build typical periods DataArrays with (cluster, time) shape.

        Args:
            tsam_aggregation_results: Dict mapping (period, scenario) to tsam results.
            actual_n_clusters: Number of clusters.
            n_time_points: Number of time points per cluster (timesteps or segments).
            cluster_coords: Cluster coordinate values.
            time_coords: Time coordinate values.
            is_segmented: Whether segmentation was used.

        Returns:
            Nested dict: {column_name: {(period, scenario): DataArray}}.
        """
        typical_das: dict[str, dict[tuple, xr.DataArray]] = {}
        for key, tsam_result in tsam_aggregation_results.items():
            typical_df = tsam_result.cluster_representatives
            if is_segmented:
                # Segmented data: MultiIndex (Segment Step, Segment Duration)
                # Need to extract by cluster (first level of index)
                for col in typical_df.columns:
                    data = np.zeros((actual_n_clusters, n_time_points))
                    for cluster_id in range(actual_n_clusters):
                        cluster_data = typical_df.loc[cluster_id, col]
                        data[cluster_id, :] = cluster_data.values[:n_time_points]
                    typical_das.setdefault(col, {})[key] = xr.DataArray(
                        data,
                        dims=['cluster', 'time'],
                        coords={'cluster': cluster_coords, 'time': time_coords},
                    )
            else:
                # Non-segmented: flat data that can be reshaped
                for col in typical_df.columns:
                    flat_data = typical_df[col].values
                    reshaped = flat_data.reshape(actual_n_clusters, n_time_points)
                    typical_das.setdefault(col, {})[key] = xr.DataArray(
                        reshaped,
                        dims=['cluster', 'time'],
                        coords={'cluster': cluster_coords, 'time': time_coords},
                    )
        return typical_das

    def _build_segment_durations_da(
        self,
        tsam_aggregation_results: dict[tuple, Any],
        actual_n_clusters: int,
        n_segments: int,
        cluster_coords: np.ndarray,
        time_coords: pd.RangeIndex,
        dt: float,
        periods: list,
        scenarios: list,
    ) -> xr.DataArray:
        """Build timestep_duration DataArray from segment durations.

        For segmented systems, each segment represents multiple original timesteps.
        The duration is segment_duration_in_original_timesteps * dt (hours per original timestep).

        Args:
            tsam_aggregation_results: Dict mapping (period, scenario) to tsam results.
            actual_n_clusters: Number of clusters.
            n_segments: Number of segments per cluster.
            cluster_coords: Cluster coordinate values.
            time_coords: Time coordinate values (RangeIndex for segments).
            dt: Hours per original timestep.
            periods: List of period labels ([None] if no periods dimension).
            scenarios: List of scenario labels ([None] if no scenarios dimension).

        Returns:
            DataArray with dims [cluster, time] or [cluster, time, period?, scenario?]
            containing duration in hours for each segment.
        """
        segment_duration_slices: dict[tuple, xr.DataArray] = {}

        for key, tsam_result in tsam_aggregation_results.items():
            # segment_durations is tuple of tuples: ((dur1, dur2, ...), (dur1, dur2, ...), ...)
            # Each inner tuple is durations for one cluster
            seg_durs = tsam_result.segment_durations

            # Build 2D array (cluster, segment) of durations in hours
            data = np.zeros((actual_n_clusters, n_segments))
            for cluster_id in range(actual_n_clusters):
                cluster_seg_durs = seg_durs[cluster_id]
                for seg_id in range(n_segments):
                    # Duration in hours = number of original timesteps * dt
                    data[cluster_id, seg_id] = cluster_seg_durs[seg_id] * dt

            segment_duration_slices[key] = xr.DataArray(
                data,
                dims=['cluster', 'time'],
                coords={'cluster': cluster_coords, 'time': time_coords},
            )

        return self._combine_slices_to_dataarray_generic(
            segment_duration_slices, ['cluster', 'time'], periods, scenarios, 'timestep_duration'
        )

    def _build_clustering_metrics(
        self,
        clustering_metrics_all: dict[tuple, pd.DataFrame],
        periods: list,
        scenarios: list,
    ) -> xr.Dataset:
        """Build clustering metrics Dataset from per-slice DataFrames.

        Args:
            clustering_metrics_all: Dict mapping (period, scenario) to metric DataFrames.
            periods: List of period labels ([None] if no periods dimension).
            scenarios: List of scenario labels ([None] if no scenarios dimension).

        Returns:
            Dataset with RMSE, MAE, RMSE_duration metrics.
        """
        non_empty_metrics = {k: v for k, v in clustering_metrics_all.items() if not v.empty}

        if not non_empty_metrics:
            return xr.Dataset()

        first_key = (periods[0], scenarios[0])

        if len(non_empty_metrics) == 1 or len(clustering_metrics_all) == 1:
            metrics_df = non_empty_metrics.get(first_key)
            if metrics_df is None:
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

        # Multi-dim case
        sample_df = next(iter(non_empty_metrics.values()))
        metric_names = list(sample_df.columns)
        data_vars = {}

        for metric in metric_names:
            slices = {}
            for (p, s), df in clustering_metrics_all.items():
                if df.empty:
                    slices[(p, s)] = xr.DataArray(
                        np.full(len(sample_df.index), np.nan),
                        dims=['time_series'],
                        coords={'time_series': list(sample_df.index)},
                    )
                else:
                    slices[(p, s)] = xr.DataArray(
                        df[metric].values,
                        dims=['time_series'],
                        coords={'time_series': list(df.index)},
                    )
            data_vars[metric] = self._combine_slices_to_dataarray_generic(
                slices, ['time_series'], periods, scenarios, metric
            )

        return xr.Dataset(data_vars)

    def _build_reduced_flow_system(
        self,
        ds: xr.Dataset,
        tsam_aggregation_results: dict[tuple, Any],
        cluster_occurrences_all: dict[tuple, dict],
        clustering_metrics_all: dict[tuple, pd.DataFrame],
        timesteps_per_cluster: int,
        dt: float,
        periods: list,
        scenarios: list,
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
            periods: List of period labels ([None] if no periods).
            scenarios: List of scenario labels ([None] if no scenarios).
            n_clusters_requested: Requested number of clusters (for logging). None to skip.

        Returns:
            Reduced FlowSystem with clustering metadata attached.
        """
        from .clustering import Clustering
        from .core import drop_constant_arrays
        from .flow_system import FlowSystem

        has_periods = periods != [None]
        has_scenarios = scenarios != [None]

        # Build dim_names for Clustering
        dim_names = []
        if has_periods:
            dim_names.append('period')
        if has_scenarios:
            dim_names.append('scenario')

        # Build dict keyed by (period?, scenario?) tuples (without None)
        aggregation_results: dict[tuple, Any] = {}
        for (p, s), result in tsam_aggregation_results.items():
            key_parts = []
            if has_periods:
                key_parts.append(p)
            if has_scenarios:
                key_parts.append(s)
            key = tuple(key_parts)
            aggregation_results[key] = result

        # Use first result for structure
        first_key = (periods[0], scenarios[0])
        first_tsam = tsam_aggregation_results[first_key]

        # Build metrics
        clustering_metrics = self._build_clustering_metrics(clustering_metrics_all, periods, scenarios)

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

        # Build cluster_weight
        cluster_weight = self._build_cluster_weight_da(
            cluster_occurrences_all, actual_n_clusters, cluster_coords, periods, scenarios
        )

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
        typical_das = self._build_typical_das(
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
            periods,
            scenarios,
        )

        # For segmented systems, build timestep_duration from segment_durations
        if is_segmented:
            segment_durations = self._build_segment_durations_da(
                tsam_aggregation_results,
                actual_n_clusters,
                n_segments,
                cluster_coords,
                time_coords,
                dt,
                periods,
                scenarios,
            )
            ds_new['timestep_duration'] = segment_durations

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
            _dim_names=dim_names,
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
        periods: list,
        scenarios: list,
    ) -> xr.Dataset:
        """Build the reduced dataset with (cluster, time) structure.

        Args:
            ds: Original dataset.
            typical_das: Typical periods DataArrays from _build_typical_das().
            actual_n_clusters: Number of clusters.
            n_reduced_timesteps: Total reduced timesteps (n_clusters * n_time_points).
            n_time_points: Number of time points per cluster (timesteps or segments).
            cluster_coords: Cluster coordinate values.
            time_coords: Time coordinate values.
            periods: List of period labels.
            scenarios: List of scenario labels.

        Returns:
            Dataset with reduced timesteps and (cluster, time) structure.
        """
        from .core import TimeSeriesData

        all_keys = {(p, s) for p in periods for s in scenarios}
        ds_new_vars = {}

        for name, original_da in ds.data_vars.items():
            if 'time' not in original_da.dims:
                ds_new_vars[name] = original_da.copy()
            elif name not in typical_das or set(typical_das[name].keys()) != all_keys:
                # Time-dependent but constant: reshape to (cluster, time, ...)
                sliced = original_da.isel(time=slice(0, n_reduced_timesteps))
                other_dims = [d for d in sliced.dims if d != 'time']
                other_shape = [sliced.sizes[d] for d in other_dims]
                new_shape = [actual_n_clusters, n_time_points] + other_shape
                reshaped = sliced.values.reshape(new_shape)
                new_coords = {'cluster': cluster_coords, 'time': time_coords}
                for dim in other_dims:
                    new_coords[dim] = sliced.coords[dim].values
                ds_new_vars[name] = xr.DataArray(
                    reshaped,
                    dims=['cluster', 'time'] + other_dims,
                    coords=new_coords,
                    attrs=original_da.attrs,
                )
            else:
                # Time-varying: combine per-(period, scenario) slices
                da = self._combine_slices_to_dataarray_2d(
                    slices=typical_das[name],
                    original_da=original_da,
                    periods=periods,
                    scenarios=scenarios,
                )
                if TimeSeriesData.is_timeseries_data(original_da):
                    da = TimeSeriesData.from_dataarray(da.assign_attrs(original_da.attrs))
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
        for var_name, var in time_dataset.data_vars.items():
            dims_key = tuple(sorted(d for d in var.dims if d != 'time'))
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
            >>> fs_dispatch = flow_system.transform.fix_sizes(fs_sizing.statistics.sizes)
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
            sizes = self._fs.statistics.sizes

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
            **tsam_kwargs: Additional keyword arguments passed to ``tsam.aggregate()``.
                See tsam documentation for all options (e.g., ``preserve_column_means``).

        Returns:
            A new FlowSystem with reduced timesteps (only typical clusters).
            The FlowSystem has metadata stored in ``clustering`` for expansion.

        Raises:
            ValueError: If timestep sizes are inconsistent.
            ValueError: If cluster_duration is not a multiple of timestep size.

        Examples:
            Basic clustering with peak preservation:

            >>> from tsam.config import ExtremeConfig
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

        # Validate tsam_kwargs doesn't override explicit parameters
        reserved_tsam_keys = {
            'n_periods',
            'period_hours',
            'resolution',
            'cluster',  # ClusterConfig object (weights are passed through this)
            'extremes',  # ExtremeConfig object
        }
        conflicts = reserved_tsam_keys & set(tsam_kwargs.keys())
        if conflicts:
            raise ValueError(
                f'Cannot override explicit parameters via tsam_kwargs: {conflicts}. '
                f'Use the corresponding cluster() parameters instead.'
            )

        # Cluster each (period, scenario) combination using tsam directly
        tsam_aggregation_results: dict[tuple, Any] = {}  # AggregationResult objects
        tsam_clustering_results: dict[tuple, Any] = {}  # ClusteringResult objects for persistence
        cluster_assignmentss: dict[tuple, np.ndarray] = {}
        cluster_occurrences_all: dict[tuple, dict] = {}

        # Collect metrics per (period, scenario) slice
        clustering_metrics_all: dict[tuple, pd.DataFrame] = {}

        for period_label in periods:
            for scenario_label in scenarios:
                key = (period_label, scenario_label)
                selector = {k: v for k, v in [('period', period_label), ('scenario', scenario_label)] if v is not None}

                # Select data for clustering (may be subset if data_vars specified)
                ds_slice_for_clustering = (
                    ds_for_clustering.sel(**selector, drop=True) if selector else ds_for_clustering
                )
                temporaly_changing_ds_for_clustering = drop_constant_arrays(ds_slice_for_clustering, dim='time')
                df_for_clustering = temporaly_changing_ds_for_clustering.to_dataframe()

                if selector:
                    logger.info(f'Clustering {", ".join(f"{k}={v}" for k, v in selector.items())}...')

                # Suppress tsam warning about minimal value constraints (informational, not actionable)
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning, message='.*minimal value.*exceeds.*')

                    # Build ClusterConfig with auto-calculated weights
                    clustering_weights = self._calculate_clustering_weights(temporaly_changing_ds_for_clustering)
                    filtered_weights = {
                        name: w for name, w in clustering_weights.items() if name in df_for_clustering.columns
                    }
                    cluster_config = self._build_cluster_config_with_weights(cluster, filtered_weights)

                    # Perform clustering based on selected data_vars (or all if not specified)
                    tsam_result = tsam.aggregate(
                        df_for_clustering,
                        n_clusters=n_clusters,
                        period_duration=hours_per_cluster,
                        timestep_duration=dt,
                        cluster=cluster_config,
                        extremes=extremes,
                        segments=segments,
                        **tsam_kwargs,
                    )

                tsam_aggregation_results[key] = tsam_result
                tsam_clustering_results[key] = tsam_result.clustering
                cluster_assignmentss[key] = tsam_result.cluster_assignments
                cluster_occurrences_all[key] = tsam_result.cluster_weights
                try:
                    clustering_metrics_all[key] = self._accuracy_to_dataframe(tsam_result.accuracy)
                except Exception as e:
                    logger.warning(f'Failed to compute clustering metrics for {key}: {e}')
                    clustering_metrics_all[key] = pd.DataFrame()

        # If data_vars was specified, apply clustering to FULL data
        if data_vars is not None:
            # Build dim_names for ClusteringResults
            dim_names = []
            if has_periods:
                dim_names.append('period')
            if has_scenarios:
                dim_names.append('scenario')

            # Convert (period, scenario) keys to ClusteringResults format
            def to_cr_key(p, s):
                key_parts = []
                if has_periods:
                    key_parts.append(p)
                if has_scenarios:
                    key_parts.append(s)
                return tuple(key_parts)

            # Build ClusteringResults from subset clustering
            clustering_results = ClusteringResults(
                {to_cr_key(p, s): cr for (p, s), cr in tsam_clustering_results.items()},
                dim_names,
            )

            # Apply to full data - this returns AggregationResults
            agg_results = clustering_results.apply(ds)

            # Update tsam_aggregation_results with full data results
            for cr_key, result in agg_results:
                # Convert back to (period, scenario) format
                if has_periods and has_scenarios:
                    full_key = (cr_key[0], cr_key[1])
                elif has_periods:
                    full_key = (cr_key[0], None)
                elif has_scenarios:
                    full_key = (None, cr_key[0])
                else:
                    full_key = (None, None)
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
            periods=periods,
            scenarios=scenarios,
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
        has_periods = self._fs.periods is not None
        has_scenarios = self._fs.scenarios is not None

        # Determine iteration dimensions
        periods = list(self._fs.periods) if has_periods else [None]
        scenarios = list(self._fs.scenarios) if has_scenarios else [None]

        ds = self._fs.to_dataset(include_solution=False)

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
            if has_periods and has_scenarios:
                full_key = (cr_key[0], cr_key[1])
            elif has_periods:
                full_key = (cr_key[0], None)
            elif has_scenarios:
                full_key = (None, cr_key[0])
            else:
                full_key = (None, None)

            tsam_aggregation_results[full_key] = result
            cluster_occurrences_all[full_key] = result.cluster_weights
            try:
                clustering_metrics_all[full_key] = self._accuracy_to_dataframe(result.accuracy)
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
            periods=periods,
            scenarios=scenarios,
        )

    @staticmethod
    def _combine_slices_to_dataarray_generic(
        slices: dict[tuple, xr.DataArray],
        base_dims: list[str],
        periods: list,
        scenarios: list,
        name: str,
    ) -> xr.DataArray:
        """Combine per-(period, scenario) slices into a multi-dimensional DataArray.

        Generic version that works with any base dimension (not just 'time').

        Args:
            slices: Dict mapping (period, scenario) tuples to DataArrays.
            base_dims: Base dimensions of each slice (e.g., ['original_cluster'] or ['original_time']).
            periods: List of period labels ([None] if no periods dimension).
            scenarios: List of scenario labels ([None] if no scenarios dimension).
            name: Name for the resulting DataArray.

        Returns:
            DataArray with dimensions [base_dims..., period?, scenario?].
        """
        first_key = (periods[0], scenarios[0])
        has_periods = periods != [None]
        has_scenarios = scenarios != [None]

        # Simple case: no period/scenario dimensions
        if not has_periods and not has_scenarios:
            return slices[first_key].rename(name)

        # Multi-dimensional: use xr.concat to stack along period/scenario dims
        # Use join='outer' to handle cases where different periods/scenarios have different
        # coordinate values (e.g., different time_series after drop_constant_arrays)
        if has_periods and has_scenarios:
            # Stack scenarios first, then periods
            period_arrays = []
            for p in periods:
                scenario_arrays = [slices[(p, s)] for s in scenarios]
                period_arrays.append(
                    xr.concat(
                        scenario_arrays, dim=pd.Index(scenarios, name='scenario'), join='outer', fill_value=np.nan
                    )
                )
            result = xr.concat(period_arrays, dim=pd.Index(periods, name='period'), join='outer', fill_value=np.nan)
        elif has_periods:
            result = xr.concat(
                [slices[(p, None)] for p in periods],
                dim=pd.Index(periods, name='period'),
                join='outer',
                fill_value=np.nan,
            )
        else:
            result = xr.concat(
                [slices[(None, s)] for s in scenarios],
                dim=pd.Index(scenarios, name='scenario'),
                join='outer',
                fill_value=np.nan,
            )

        # Put base dimension first (standard order)
        result = result.transpose(base_dims[0], ...)

        return result.rename(name)

    @staticmethod
    def _combine_slices_to_dataarray_2d(
        slices: dict[tuple, xr.DataArray],
        original_da: xr.DataArray,
        periods: list,
        scenarios: list,
    ) -> xr.DataArray:
        """Combine per-(period, scenario) slices into a multi-dimensional DataArray with (cluster, time) dims.

        Args:
            slices: Dict mapping (period, scenario) tuples to DataArrays with (cluster, time) dims.
            original_da: Original DataArray to get attrs from.
            periods: List of period labels ([None] if no periods dimension).
            scenarios: List of scenario labels ([None] if no scenarios dimension).

        Returns:
            DataArray with dimensions (cluster, time, period?, scenario?).
        """
        first_key = (periods[0], scenarios[0])
        has_periods = periods != [None]
        has_scenarios = scenarios != [None]

        # Simple case: no period/scenario dimensions
        if not has_periods and not has_scenarios:
            return slices[first_key].assign_attrs(original_da.attrs)

        # Multi-dimensional: use xr.concat to stack along period/scenario dims
        if has_periods and has_scenarios:
            # Stack scenarios first, then periods
            period_arrays = []
            for p in periods:
                scenario_arrays = [slices[(p, s)] for s in scenarios]
                period_arrays.append(xr.concat(scenario_arrays, dim=pd.Index(scenarios, name='scenario')))
            result = xr.concat(period_arrays, dim=pd.Index(periods, name='period'))
        elif has_periods:
            result = xr.concat([slices[(p, None)] for p in periods], dim=pd.Index(periods, name='period'))
        else:
            result = xr.concat([slices[(None, s)] for s in scenarios], dim=pd.Index(scenarios, name='scenario'))

        # Put cluster and time first (standard order for clustered data)
        result = result.transpose('cluster', 'time', ...)

        return result.assign_attrs(original_da.attrs)

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

    def _combine_intercluster_charge_states(
        self,
        expanded_fs: FlowSystem,
        reduced_solution: xr.Dataset,
        clustering: Clustering,
        original_timesteps_extra: pd.DatetimeIndex,
        timesteps_per_cluster: int,
        n_original_clusters: int,
    ) -> None:
        """Combine charge_state with SOC_boundary for intercluster storages (in-place).

        For intercluster storages, charge_state is relative (delta-E) and can be negative.
        Per Blanke et al. (2022) Eq. 9, actual SOC at time t in period d is:
            SOC(t) = SOC_boundary[d] * (1 - loss)^t_within_period + charge_state(t)
        where t_within_period is hours from period start (accounts for self-discharge decay).

        Args:
            expanded_fs: The expanded FlowSystem (modified in-place).
            reduced_solution: The original reduced solution dataset.
            clustering: Clustering with cluster order info.
            original_timesteps_extra: Original timesteps including the extra final timestep.
            timesteps_per_cluster: Number of timesteps per cluster.
            n_original_clusters: Number of original clusters before aggregation.
        """
        n_original_timesteps_extra = len(original_timesteps_extra)
        soc_boundary_vars = [name for name in reduced_solution.data_vars if name.endswith('|SOC_boundary')]

        for soc_boundary_name in soc_boundary_vars:
            storage_name = soc_boundary_name.rsplit('|', 1)[0]
            charge_state_name = f'{storage_name}|charge_state'
            if charge_state_name not in expanded_fs._solution:
                continue

            soc_boundary = reduced_solution[soc_boundary_name]
            expanded_charge_state = expanded_fs._solution[charge_state_name]

            # Map each original timestep to its original period index
            original_cluster_indices = np.minimum(
                np.arange(n_original_timesteps_extra) // timesteps_per_cluster,
                n_original_clusters - 1,
            )

            # Select SOC_boundary for each timestep
            soc_boundary_per_timestep = soc_boundary.isel(
                cluster_boundary=xr.DataArray(original_cluster_indices, dims=['time'])
            ).assign_coords(time=original_timesteps_extra)

            # Apply self-discharge decay
            soc_boundary_per_timestep = self._apply_soc_decay(
                soc_boundary_per_timestep,
                storage_name,
                clustering,
                original_timesteps_extra,
                original_cluster_indices,
                timesteps_per_cluster,
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
        clustering: Clustering,
        original_timesteps_extra: pd.DatetimeIndex,
        original_cluster_indices: np.ndarray,
        timesteps_per_cluster: int,
    ) -> xr.DataArray:
        """Apply self-discharge decay to SOC_boundary values.

        Args:
            soc_boundary_per_timestep: SOC boundary values mapped to each timestep.
            storage_name: Name of the storage component.
            clustering: Clustering with cluster order info.
            original_timesteps_extra: Original timesteps including final extra timestep.
            original_cluster_indices: Mapping of timesteps to original cluster indices.
            timesteps_per_cluster: Number of timesteps per cluster.

        Returns:
            SOC boundary values with decay applied.
        """
        storage = self._fs.storages.get(storage_name)
        if storage is None:
            return soc_boundary_per_timestep

        n_timesteps = len(original_timesteps_extra)

        # Time within period for each timestep (0, 1, 2, ..., T-1, 0, 1, ...)
        time_within_period = np.arange(n_timesteps) % timesteps_per_cluster
        time_within_period[-1] = timesteps_per_cluster  # Extra timestep gets full decay
        time_within_period_da = xr.DataArray(
            time_within_period, dims=['time'], coords={'time': original_timesteps_extra}
        )

        # Decay factor: (1 - loss)^t
        loss_value = storage.relative_loss_per_hour.mean('time')
        if not np.any(loss_value.values > 0):
            return soc_boundary_per_timestep

        decay_da = (1 - loss_value) ** time_within_period_da

        # Handle cluster dimension if present
        if 'cluster' in decay_da.dims:
            cluster_assignments = clustering.cluster_assignments
            if cluster_assignments.ndim == 1:
                cluster_per_timestep = xr.DataArray(
                    cluster_assignments.values[original_cluster_indices],
                    dims=['time'],
                    coords={'time': original_timesteps_extra},
                )
            else:
                cluster_per_timestep = cluster_assignments.isel(
                    original_cluster=xr.DataArray(original_cluster_indices, dims=['time'])
                ).assign_coords(time=original_timesteps_extra)
            decay_da = decay_da.isel(cluster=cluster_per_timestep).drop_vars('cluster', errors='ignore')

        return soc_boundary_per_timestep * decay_da

    def _build_segment_total_varnames(self) -> set[str]:
        """Build the set of solution variable names that represent segment totals.

        For segmented systems, these variables contain values that are summed over
        segments. When expanded to hourly resolution, they need to be divided by
        segment duration to get correct hourly rates.

        Derives variable names directly from FlowSystem structure (effects, flows,
        components) rather than pattern matching, ensuring robustness.

        Returns:
            Set of variable names that should be divided by expansion divisor.
        """
        segment_total_vars: set[str] = set()

        # Get all effect names
        effect_names = list(self._fs.effects.keys())

        # 1. Per-timestep totals for each effect: {effect}(temporal)|per_timestep
        for effect in effect_names:
            segment_total_vars.add(f'{effect}(temporal)|per_timestep')

        # 2. Flow contributions to effects: {flow}->{effect}(temporal)
        #    (from effects_per_flow_hour on Flow elements)
        for flow_label in self._fs.flows:
            for effect in effect_names:
                segment_total_vars.add(f'{flow_label}->{effect}(temporal)')

        # 3. Component contributions to effects: {component}->{effect}(temporal)
        #    (from effects_per_startup, effects_per_active_hour on OnOffParameters)
        for component_label in self._fs.components:
            for effect in effect_names:
                segment_total_vars.add(f'{component_label}->{effect}(temporal)')

        # 4. Effect-to-effect contributions (from share_from_temporal)
        #    {source_effect}(temporal)->{target_effect}(temporal)
        for target_effect_name, target_effect in self._fs.effects.items():
            if target_effect.share_from_temporal:
                for source_effect_name in target_effect.share_from_temporal:
                    segment_total_vars.add(f'{source_effect_name}(temporal)->{target_effect_name}(temporal)')

        return segment_total_vars

    def _interpolate_charge_state_segmented(
        self,
        da: xr.DataArray,
        clustering: Clustering,
        original_timesteps: pd.DatetimeIndex,
    ) -> xr.DataArray:
        """Interpolate charge_state values within segments for segmented systems.

        For segmented systems, charge_state has values at segment boundaries (n_segments+1).
        Instead of repeating the start boundary value for all timesteps in a segment,
        this method interpolates between start and end boundary values to show the
        actual charge trajectory as the storage charges/discharges.

        Args:
            da: charge_state DataArray with dims (cluster, time) where time has n_segments+1 entries.
            clustering: Clustering object with segment info.
            original_timesteps: Original timesteps to expand to.

        Returns:
            Interpolated charge_state with dims (time, ...) for original timesteps.
        """
        timesteps_per_cluster = clustering.timesteps_per_cluster
        n_original_clusters = clustering.n_original_clusters
        cluster_assignments = clustering.cluster_assignments

        # Get segment assignments and durations from clustering results
        extra_dims = clustering.results.dim_names

        def _interpolate_slice(
            charge_state_data: np.ndarray,
            assignments: np.ndarray,
            clustering_result,
        ) -> np.ndarray:
            """Interpolate charge_state for a single period/scenario slice."""
            n_timesteps = n_original_clusters * timesteps_per_cluster
            result = np.zeros(n_timesteps)

            seg_assignments = clustering_result.segment_assignments
            seg_durations = clustering_result.segment_durations

            for orig_cluster_idx in range(n_original_clusters):
                typical_cluster_idx = int(assignments[orig_cluster_idx])
                cluster_seg_assigns = seg_assignments[typical_cluster_idx]
                cluster_seg_durs = seg_durations[typical_cluster_idx]

                # Build cumulative positions within cluster for interpolation
                for t in range(timesteps_per_cluster):
                    seg_idx = int(cluster_seg_assigns[t])
                    # Count how many timesteps before this one are in the same segment
                    seg_start_t = 0
                    for prev_t in range(t):
                        if cluster_seg_assigns[prev_t] == seg_idx:
                            break
                        seg_start_t = prev_t + 1
                    t_within_seg = t - seg_start_t
                    seg_duration = cluster_seg_durs[seg_idx]

                    # Interpolation factor: position within segment (0 to 1)
                    # At t_within_seg=0, factor=0 (start of segment)
                    # At t_within_seg=seg_duration-1, factor approaches 1 (end of segment)
                    if seg_duration > 1:
                        factor = (t_within_seg + 0.5) / seg_duration
                    else:
                        factor = 0.5

                    # Get start and end boundary values
                    start_val = charge_state_data[typical_cluster_idx, seg_idx]
                    end_val = charge_state_data[typical_cluster_idx, seg_idx + 1]

                    # Linear interpolation
                    result[orig_cluster_idx * timesteps_per_cluster + t] = start_val + (end_val - start_val) * factor

            return result

        # Handle extra dimensions (period, scenario)
        if not extra_dims:
            # Simple case: no period/scenario dimensions
            result = clustering.results.sel()
            interpolated = _interpolate_slice(da.values, cluster_assignments.values, result)
            return xr.DataArray(
                interpolated,
                dims=['time'],
                coords={'time': original_timesteps},
                attrs=da.attrs,
            )

        # Multi-dimensional case
        dim_coords = clustering.results.coords
        interpolated_slices = {}

        for combo in np.ndindex(*[len(v) for v in dim_coords.values()]):
            selector = {d: dim_coords[d][i] for d, i in zip(extra_dims, combo, strict=True)}

            # Get cluster assignments for this period/scenario
            if any(d in cluster_assignments.dims for d in selector):
                from .clustering.base import _select_dims

                assignments = _select_dims(cluster_assignments, **selector).values
            else:
                assignments = cluster_assignments.values

            # Get charge_state data for this period/scenario
            da_slice = da
            for dim, val in selector.items():
                if dim in da.dims:
                    da_slice = da_slice.sel({dim: val})

            # Get clustering result for this period/scenario
            result = clustering.results.sel(**selector)
            interpolated = _interpolate_slice(da_slice.values, assignments, result)

            interpolated_slices[tuple(selector.values())] = xr.DataArray(
                interpolated,
                dims=['time'],
                coords={'time': original_timesteps},
            )

        # Concatenate along extra dimensions
        result_arrays = interpolated_slices
        for dim in reversed(extra_dims):
            dim_vals = dim_coords[dim]
            grouped = {}
            for key, arr in result_arrays.items():
                rest_key = key[:-1] if len(key) > 1 else ()
                grouped.setdefault(rest_key, []).append(arr)
            result_arrays = {k: xr.concat(v, dim=pd.Index(dim_vals, name=dim)) for k, v in grouped.items()}

        result = list(result_arrays.values())[0]
        return result.transpose('time', ...).assign_attrs(da.attrs)

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
            >>> fs_expanded.statistics.flow_rates  # Full 8760 timesteps
            >>> fs_expanded.statistics.plot.balance('HeatBus')  # Full resolution plots
            >>> fs_expanded.statistics.plot.heatmap('Boiler(Q_th)|flow_rate')

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

               These variables cannot be meaningfully interpolated. They indicate
               the dominant state or whether an event occurred during the segment.

               - ``{flow}|status``: On/off status (0 or 1)
               - ``{flow}|startup``: Startup event occurred in segment
               - ``{flow}|shutdown``: Shutdown event occurred in segment
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

        # For charge_state expansion: index of last valid original cluster
        last_original_cluster_idx = min(
            (n_original_timesteps - 1) // timesteps_per_cluster,
            n_original_clusters - 1,
        )

        # For segmented systems: build expansion divisor and identify segment total variables
        expansion_divisor = None
        segment_total_vars: set[str] = set()
        variable_categories = getattr(self._fs, '_variable_categories', {})
        if clustering.is_segmented:
            expansion_divisor = clustering.build_expansion_divisor(original_time=original_timesteps)
            # Build segment total vars using registry first, fall back to pattern matching
            segment_total_vars = {
                name for name, cat in variable_categories.items() if cat == VariableCategory.SEGMENT_TOTAL
            }
            # Fall back to pattern matching for backwards compatibility (old FlowSystems without categories)
            if not segment_total_vars:
                segment_total_vars = self._build_segment_total_varnames()

        def _is_state_variable(var_name: str) -> bool:
            """Check if a variable is a state variable (should be interpolated)."""
            if var_name in variable_categories:
                return variable_categories[var_name] == VariableCategory.STATE
            # Fall back to pattern matching for backwards compatibility
            return var_name.endswith('|charge_state')

        def expand_da(da: xr.DataArray, var_name: str = '', is_solution: bool = False) -> xr.DataArray:
            """Expand a DataArray from clustered to original timesteps.

            Args:
                da: DataArray to expand.
                var_name: Variable name for segment total lookup.
                is_solution: True if this is a solution variable (may need segment correction).
                    FlowSystem data (is_solution=False) is never corrected for segments.
            """
            if 'time' not in da.dims:
                return da.copy()

            # For state variables (like charge_state) in segmented systems: interpolate within segments
            # to show the actual state trajectory as the storage charges/discharges
            if _is_state_variable(var_name) and 'cluster' in da.dims and clustering.is_segmented:
                expanded = self._interpolate_charge_state_segmented(da, clustering, original_timesteps)
                # Append the extra timestep value (final charge state)
                cluster_assignments = clustering.cluster_assignments
                if cluster_assignments.ndim == 1:
                    last_cluster = int(cluster_assignments[last_original_cluster_idx])
                    extra_val = da.isel(cluster=last_cluster, time=-1)
                else:
                    last_clusters = cluster_assignments.isel(original_cluster=last_original_cluster_idx)
                    extra_val = da.isel(cluster=last_clusters, time=-1)
                extra_val = extra_val.drop_vars(['cluster', 'time'], errors='ignore')
                extra_val = extra_val.expand_dims(time=[original_timesteps_extra[-1]])
                expanded = xr.concat([expanded, extra_val], dim='time')
                return expanded

            expanded = clustering.expand_data(da, original_time=original_timesteps)

            # For segmented systems: divide segment totals by expansion divisor
            # ONLY for solution variables explicitly identified as segment totals
            if is_solution and expansion_divisor is not None and var_name in segment_total_vars:
                expanded = expanded / expansion_divisor

            # For state variables (like charge_state) with cluster dim (non-segmented), append the extra timestep value
            if _is_state_variable(var_name) and 'cluster' in da.dims:
                cluster_assignments = clustering.cluster_assignments
                if cluster_assignments.ndim == 1:
                    last_cluster = int(cluster_assignments[last_original_cluster_idx])
                    extra_val = da.isel(cluster=last_cluster, time=-1)
                else:
                    last_clusters = cluster_assignments.isel(original_cluster=last_original_cluster_idx)
                    extra_val = da.isel(cluster=last_clusters, time=-1)
                extra_val = extra_val.drop_vars(['cluster', 'time'], errors='ignore')
                extra_val = extra_val.expand_dims(time=[original_timesteps_extra[-1]])
                expanded = xr.concat([expanded, extra_val], dim='time')

            return expanded

        # 1. Expand FlowSystem data
        reduced_ds = self._fs.to_dataset(include_solution=False)
        clustering_attrs = {'is_clustered', 'n_clusters', 'timesteps_per_cluster', 'clustering', 'cluster_weight'}
        data_vars = {
            name: expand_da(da, name)
            for name, da in reduced_ds.data_vars.items()
            if name != 'cluster_weight' and not name.startswith('clustering|')
        }
        attrs = {k: v for k, v in reduced_ds.attrs.items() if k not in clustering_attrs}
        expanded_ds = xr.Dataset(data_vars, attrs=attrs)

        # Update timestep_duration for original timesteps
        timestep_duration = FlowSystem.calculate_timestep_duration(original_timesteps_extra)
        expanded_ds.attrs['timestep_duration'] = timestep_duration.values.tolist()

        expanded_fs = FlowSystem.from_dataset(expanded_ds)

        # 2. Expand solution (with segment total correction for segmented systems)
        reduced_solution = self._fs.solution
        expanded_fs._solution = xr.Dataset(
            {name: expand_da(da, name, is_solution=True) for name, da in reduced_solution.data_vars.items()},
            attrs=reduced_solution.attrs,
        )
        expanded_fs._solution = expanded_fs._solution.reindex(time=original_timesteps_extra)

        # 3. Combine charge_state with SOC_boundary for intercluster storages
        self._combine_intercluster_charge_states(
            expanded_fs,
            reduced_solution,
            clustering,
            original_timesteps_extra,
            timesteps_per_cluster,
            n_original_clusters,
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
