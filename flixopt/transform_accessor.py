"""
Transform accessor for FlowSystem.

This module provides the TransformAccessor class that enables
transformations on FlowSystem like clustering, selection, and resampling.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from .clustering import ClusteringParameters
    from .flow_system import FlowSystem

logger = logging.getLogger('flixopt')


class TransformAccessor:
    """
    Accessor for transformation methods on FlowSystem.

    This class provides transformations that create new FlowSystem instances
    with modified structure or data, accessible via `flow_system.transform`.

    Examples:
        Clustered optimization (8 typical days):

        >>> clustered_fs = flow_system.transform.cluster(n_clusters=8, cluster_duration='1D')
        >>> clustered_fs.optimize(solver)
        >>> print(clustered_fs.solution)

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

    def cluster(
        self,
        n_clusters: int | None,
        cluster_duration: str | float,
        n_segments: int | None = None,
        aggregate_data: bool = True,
        include_storage: bool = True,
        flexibility_percent: float = 0,
        flexibility_penalty: float = 0,
        time_series_for_high_peaks: list | None = None,
        time_series_for_low_peaks: list | None = None,
        components_to_clusterize: list | None = None,
    ) -> FlowSystem:
        """
        Create a clustered FlowSystem for time series aggregation.

        This method creates a new FlowSystem that can be optimized with
        clustered time series data. The clustering reduces computational
        complexity by identifying representative time segments (e.g., typical days).

        For FlowSystems with multiple periods or scenarios, clustering is performed
        independently for each period/scenario combination.

        The returned FlowSystem:
        - Has the same timesteps as the original (clustering works via constraints, not reduction)
        - Has aggregated time series data (if ``aggregate_data=True``)
        - Will have clustering constraints added during ``build_model()``

        Args:
            n_clusters: Number of clusters (typical segments) to create.
                E.g., 8 for 8 typical days from a year of data.
                Set to None to skip inter-period clustering (only do segmentation).
            cluster_duration: Duration of each cluster segment. Can be a pandas-style
                string ('1D', '24h', '6h') or a numeric value in hours.
            n_segments: Number of segments within each cluster (inner-period clustering).
                For example, n_segments=4 with cluster_duration='1D' will reduce
                24 hourly timesteps to 4 representative segments per day.
                Default is None (no inner-period segmentation).
            aggregate_data: If True (default), aggregate time series data and fix
                all time-dependent variables. If False, only fix binary variables.
            include_storage: Whether to include storage flows in clustering constraints.
                Default is True.
            flexibility_percent: Maximum percentage (0-100) of binary values that can
                deviate from the clustered pattern. Default is 0 (no flexibility).
            flexibility_penalty: Penalty added to objective for each deviation.
                Only applies when flexibility_percent > 0. Default is 0.
            time_series_for_high_peaks: List of TimeSeriesData to force inclusion of
                segments with high values.
            time_series_for_low_peaks: List of TimeSeriesData to force inclusion of
                segments with low values.
            components_to_clusterize: List of components to apply clustering to.
                If None, all components are clustered.

        Returns:
            A new FlowSystem configured for clustered optimization.

        Raises:
            ValueError: If timestep sizes are inconsistent.
            ValueError: If cluster_duration is not a multiple of timestep size.

        Examples:
            Basic clustered optimization (8 typical days):

            >>> clustered_fs = flow_system.transform.cluster(
            ...     n_clusters=8,
            ...     cluster_duration='1D',
            ... )
            >>> clustered_fs.optimize(solver)

            With inner-period segmentation (8 typical days × 4 segments = 32 timesteps):

            >>> clustered_fs = flow_system.transform.cluster(
            ...     n_clusters=8,
            ...     cluster_duration='1D',
            ...     n_segments=4,  # Reduce 24 hours to 4 segments
            ... )

            Segmentation only (no clustering, reduce each day to 4 segments):

            >>> clustered_fs = flow_system.transform.cluster(
            ...     n_clusters=None,  # Skip inter-period clustering
            ...     cluster_duration='1D',
            ...     n_segments=4,
            ... )

            Multi-period FlowSystem (each year clustered independently):

            >>> multi_year_fs = fx.FlowSystem(timesteps, periods=pd.Index([2025, 2026, 2027]))
            >>> clustered_fs = multi_year_fs.transform.cluster(
            ...     n_clusters=8,
            ...     cluster_duration='1D',
            ... )
        """
        from .clustering import ClusteringParameters

        # Create ClusteringParameters from keyword arguments
        params = ClusteringParameters(
            n_clusters=n_clusters,
            cluster_duration=cluster_duration,
            n_segments=n_segments,
            aggregate_data=aggregate_data,
            include_storage=include_storage,
            flexibility_percent=flexibility_percent,
            flexibility_penalty=flexibility_penalty,
            time_series_for_high_peaks=time_series_for_high_peaks,
            time_series_for_low_peaks=time_series_for_low_peaks,
        )

        # Check for multi-period/scenario dimensions
        has_periods = self._fs.periods is not None
        has_scenarios = self._fs.scenarios is not None

        if not has_periods and not has_scenarios:
            # Simple case: no extra dimensions
            return self._cluster_simple(params, components_to_clusterize)
        else:
            # Multi-dimensional case: cluster independently per period/scenario
            return self._cluster_multi_dimensional(params, components_to_clusterize)

    def _cluster_simple(
        self,
        params: ClusteringParameters,
        components_to_clusterize: list | None,
    ) -> FlowSystem:
        """Perform clustering for simple case (no periods/scenarios)."""
        import numpy as np

        from .clustering import Clustering
        from .core import DataConverter, TimeSeriesData, drop_constant_arrays

        # Validation
        dt_min = float(self._fs.timestep_duration.min().item())
        dt_max = float(self._fs.timestep_duration.max().item())
        if dt_min != dt_max:
            raise ValueError(
                f'Clustering failed due to inconsistent time step sizes: '
                f'delta_t varies from {dt_min} to {dt_max} hours.'
            )
        ratio = params.cluster_duration_hours / dt_max
        if not np.isclose(ratio, round(ratio), atol=1e-9):
            raise ValueError(
                f'The selected cluster_duration={params.cluster_duration_hours}h does not match the time '
                f'step size of {dt_max} hours. It must be an integer multiple of {dt_max} hours.'
            )

        logger.info(f'{"":#^80}')
        logger.info(f'{" Clustering TimeSeries Data ":#^80}')

        # Get dataset representation
        ds = self._fs.to_dataset(include_solution=False)
        temporaly_changing_ds = drop_constant_arrays(ds, dim='time')

        # Perform clustering
        clustering = Clustering(
            original_data=temporaly_changing_ds.to_dataframe(),
            hours_per_time_step=float(dt_min),
            hours_per_period=params.cluster_duration_hours,
            nr_of_periods=params.n_clusters,
            n_segments=params.n_segments,
            weights=self._calculate_clustering_weights(temporaly_changing_ds),
            time_series_for_high_peaks=params.labels_for_high_peaks,
            time_series_for_low_peaks=params.labels_for_low_peaks,
        )
        clustering.cluster()

        # Create new FlowSystem (with aggregated data if requested)
        if params.aggregate_data:
            ds = self._fs.to_dataset()
            for name, series in clustering.aggregated_data.items():
                da = DataConverter.to_dataarray(series, self._fs.coords).rename(name).assign_attrs(ds[name].attrs)
                if TimeSeriesData.is_timeseries_data(da):
                    da = TimeSeriesData.from_dataarray(da)
                ds[name] = da

            from .flow_system import FlowSystem

            clustered_fs = FlowSystem.from_dataset(ds)
        else:
            clustered_fs = self._fs.copy()

        # Store clustering info for later use
        clustered_fs._clustering_info = {
            'parameters': params,
            'clustering': clustering,
            'components_to_clusterize': components_to_clusterize,
            'original_fs': self._fs,
        }

        return clustered_fs

    def _cluster_multi_dimensional(
        self,
        params: ClusteringParameters,
        components_to_clusterize: list | None,
    ) -> FlowSystem:
        """Perform clustering independently for each period/scenario combination."""
        import numpy as np

        from .clustering import Clustering
        from .core import DataConverter, TimeSeriesData, drop_constant_arrays

        # Validation
        dt_min = float(self._fs.timestep_duration.min().item())
        dt_max = float(self._fs.timestep_duration.max().item())
        if dt_min != dt_max:
            raise ValueError(
                f'Clustering failed due to inconsistent time step sizes: '
                f'delta_t varies from {dt_min} to {dt_max} hours.'
            )
        ratio = params.cluster_duration_hours / dt_max
        if not np.isclose(ratio, round(ratio), atol=1e-9):
            raise ValueError(
                f'The selected cluster_duration={params.cluster_duration_hours}h does not match the time '
                f'step size of {dt_max} hours. It must be an integer multiple of {dt_max} hours.'
            )

        logger.info(f'{"":#^80}')
        logger.info(f'{" Clustering TimeSeries Data (Multi-dimensional) ":#^80}')

        # Determine iteration dimensions
        periods = list(self._fs.periods) if self._fs.periods is not None else [None]
        scenarios = list(self._fs.scenarios) if self._fs.scenarios is not None else [None]

        ds = self._fs.to_dataset(include_solution=False).copy(deep=True)  # Deep copy to allow in-place modifications
        clustering_results: dict[tuple, Clustering] = {}

        # Cluster each period x scenario combination independently
        for period_label in periods:
            for scenario_label in scenarios:
                # Select slice for this combination
                selector = {}
                if period_label is not None:
                    selector['period'] = period_label
                if scenario_label is not None:
                    selector['scenario'] = scenario_label

                if selector:
                    ds_slice = ds.sel(**selector, drop=True)
                else:
                    ds_slice = ds

                # Drop constant arrays for clustering
                temporaly_changing_ds = drop_constant_arrays(ds_slice, dim='time')

                # Skip if no time-varying data
                if len(temporaly_changing_ds.data_vars) == 0:
                    logger.warning(f'No time-varying data for period={period_label}, scenario={scenario_label}')
                    continue

                dim_info = []
                if period_label is not None:
                    dim_info.append(f'period={period_label}')
                if scenario_label is not None:
                    dim_info.append(f'scenario={scenario_label}')
                logger.info(f'Clustering {", ".join(dim_info) or "data"}...')

                # Perform clustering on this slice
                clustering = Clustering(
                    original_data=temporaly_changing_ds.to_dataframe(),
                    hours_per_time_step=float(dt_min),
                    hours_per_period=params.cluster_duration_hours,
                    nr_of_periods=params.n_clusters,
                    n_segments=params.n_segments,
                    weights=self._calculate_clustering_weights(temporaly_changing_ds),
                    time_series_for_high_peaks=params.labels_for_high_peaks,
                    time_series_for_low_peaks=params.labels_for_low_peaks,
                )
                clustering.cluster()
                clustering_results[(period_label, scenario_label)] = clustering

                # Apply aggregated data if requested
                if params.aggregate_data:
                    for name, series in clustering.aggregated_data.items():
                        if name not in ds.data_vars:
                            continue
                        # Get the original data array to update
                        original_da = ds[name]
                        # Create aggregated data array
                        agg_da = DataConverter.to_dataarray(series, {'time': ds_slice.indexes['time']})

                        # Update the slice in the full dataset
                        if selector:
                            # Need to update just this slice in the full array
                            # Use xr.where or direct assignment
                            if 'period' in original_da.dims and period_label is not None:
                                if 'scenario' in original_da.dims and scenario_label is not None:
                                    original_da.loc[{'period': period_label, 'scenario': scenario_label}] = (
                                        agg_da.values
                                    )
                                else:
                                    original_da.loc[{'period': period_label}] = agg_da.values
                            elif 'scenario' in original_da.dims and scenario_label is not None:
                                original_da.loc[{'scenario': scenario_label}] = agg_da.values

        # Create new FlowSystem
        from .flow_system import FlowSystem

        if params.aggregate_data:
            # Ensure TimeSeriesData is preserved
            for name in ds.data_vars:
                da = ds[name]
                if TimeSeriesData.is_timeseries_data(da):
                    ds[name] = TimeSeriesData.from_dataarray(da)
            clustered_fs = FlowSystem.from_dataset(ds)
        else:
            clustered_fs = self._fs.copy()

        # Store clustering info for later use
        clustered_fs._clustering_info = {
            'parameters': params,
            'clustering': clustering_results,  # Required by _add_clustering_constraints
            'clustering_results': clustering_results,  # Dict of Clustering objects per dimension
            'components_to_clusterize': components_to_clusterize,
            'original_fs': self._fs,
            'has_periods': self._fs.periods is not None,
            'has_scenarios': self._fs.scenarios is not None,
        }

        return clustered_fs

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
            logger.info('All Clustering weights were set to 1')

        return weights

    def add_clustering(
        self,
        parameters: ClusteringParameters,
        components_to_clusterize: list | None = None,
    ) -> FlowSystem:
        """Add clustering constraints using ClusteringParameters with pre-set indices.

        This method allows applying clustering to a FlowSystem using indices
        computed outside of flixopt. This is useful when:
        - You want to cluster based on a subset of time series data (faster tsam)
        - You have custom clustering logic or algorithms
        - You want to reuse clustering results across multiple FlowSystems

        The clustering indices define equality constraints that equate variable values
        at specific timestep pairs. The parameters must have `cluster_order` and
        `period_length` set (either directly or via `populate_from_tsam()`).

        Args:
            parameters: ClusteringParameters with clustering indices set.
                Must have `cluster_order` and `period_length` populated.
            components_to_clusterize: Components to apply clustering to.
                If None, all components are clustered.

        Returns:
            A new FlowSystem with clustering constraints configured.

        Examples:
            External clustering with tsam on subset of data:

            >>> import tsam.timeseriesaggregation as tsam
            >>> # Extract subset of timeseries for clustering
            >>> subset_df = pd.DataFrame(
            ...     {
            ...         'price': flow_system['prices'].values,
            ...         'demand': flow_system['heat_demand'].values,
            ...     },
            ...     index=flow_system.timesteps,
            ... )
            >>>
            >>> # Run tsam on subset
            >>> aggregation = tsam.TimeSeriesAggregation(subset_df, noTypicalPeriods=8, hoursPerPeriod=24)
            >>> aggregation.createTypicalPeriods()
            >>>
            >>> # Create parameters and populate from tsam
            >>> params = fx.ClusteringParameters(n_clusters=8, cluster_duration='1D')
            >>> params.populate_from_tsam(aggregation)
            >>>
            >>> # Apply to FlowSystem
            >>> clustered_fs = flow_system.transform.add_clustering(params)
            >>> clustered_fs.optimize(solver)

            With pre-computed cluster assignments:

            >>> import xarray as xr
            >>> params = fx.ClusteringParameters(
            ...     n_clusters=8,
            ...     cluster_duration='1D',
            ...     cluster_order=xr.DataArray([0, 1, 2, 0, 1, 2, 0, 1], dims=['cluster_period']),
            ...     period_length=24,
            ...     flexibility_percent=5,  # Allow 5% binary deviation
            ... )
            >>> clustered_fs = flow_system.transform.add_clustering(params)
        """
        from .clustering import ClusteringParameters
        from .core import DataConverter, TimeSeriesData

        # Validate parameters type
        if not isinstance(parameters, ClusteringParameters):
            raise TypeError(f'parameters must be ClusteringParameters, got {type(parameters).__name__}')

        # Validate that indices are set
        if not parameters.has_indices:
            raise ValueError(
                'ClusteringParameters must have indices set. '
                'Either provide cluster_order/period_length directly, pass tsam_aggregation, or call populate_from_tsam().'
            )

        # Aggregate data if tsam_aggregation is provided and aggregate_data=True
        if parameters.aggregate_data and parameters.tsam_aggregation is not None:
            ds = self._fs.to_dataset()
            tsam_agg = parameters.tsam_aggregation

            # Get aggregated data from tsam (this is pre-computed for the subset that was clustered)
            aggregated_df = tsam_agg.predictOriginalData()

            # For variables not in the clustering subset, compute aggregation manually
            # using the cluster assignments
            period_length = parameters.period_length
            cluster_order = parameters.cluster_order.values
            n_timesteps = len(self._fs.timesteps)

            for name in ds.data_vars:
                da = ds[name]
                if 'time' not in da.dims:
                    continue

                if name in aggregated_df.columns:
                    # Use tsam's aggregated result for columns that were clustered
                    series = aggregated_df[name]
                    da_new = DataConverter.to_dataarray(series, self._fs.coords).rename(name).assign_attrs(da.attrs)
                else:
                    # Manually aggregate using cluster assignments
                    # For each timestep, replace with mean of corresponding timesteps in same cluster
                    import numpy as np

                    values = da.values.copy()
                    aggregated_values = np.zeros_like(values)

                    # Build mapping: for each cluster, collect all timestep indices
                    n_clusters = int(cluster_order.max()) + 1
                    cluster_to_timesteps: dict[int, list[int]] = {c: [] for c in range(n_clusters)}
                    for period_idx, cluster_id in enumerate(cluster_order):
                        for pos in range(period_length):
                            ts_idx = period_idx * period_length + pos
                            if ts_idx < n_timesteps:
                                cluster_to_timesteps[int(cluster_id)].append((ts_idx, pos))

                    # For each cluster, compute mean for each position
                    for _cluster_id, ts_list in cluster_to_timesteps.items():
                        # Group by position within period
                        position_values: dict[int, list] = {}
                        for ts_idx, pos in ts_list:
                            position_values.setdefault(pos, []).append(values[ts_idx])

                        # Compute mean for each position and assign back
                        for ts_idx, pos in ts_list:
                            aggregated_values[ts_idx] = np.mean(position_values[pos])

                    da_new = da.copy(data=aggregated_values)

                if TimeSeriesData.is_timeseries_data(da_new):
                    da_new = TimeSeriesData.from_dataarray(da_new)
                ds[name] = da_new

            from .flow_system import FlowSystem

            clustered_fs = FlowSystem.from_dataset(ds)
        else:
            # No data aggregation - just copy
            clustered_fs = self._fs.copy()

        # Store clustering info
        clustered_fs._clustering_info = {
            'parameters': parameters,
            'components_to_clusterize': components_to_clusterize,
        }

        return clustered_fs

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

    def cluster_reduce(
        self,
        n_clusters: int,
        cluster_duration: str | float,
        weights: dict[str, float] | None = None,
        time_series_for_high_peaks: list[str] | None = None,
        time_series_for_low_peaks: list[str] | None = None,
        storage_inter_period_linking: bool = True,
        storage_cyclic: bool = True,
    ) -> FlowSystem:
        """
        Create a FlowSystem with reduced timesteps using typical clusters.

        This method creates a new FlowSystem optimized for sizing studies by reducing
        the number of timesteps to only the typical (representative) clusters identified
        through time series aggregation. Unlike `cluster()` which uses equality constraints,
        this method actually reduces the problem size for faster solving.

        The method:
        1. Performs time series clustering using tsam
        2. Extracts only the typical clusters (not all original timesteps)
        3. Applies timestep weighting for accurate cost representation
        4. Optionally links storage states between clusters via boundary variables

        Use this for initial sizing optimization, then use `fix_sizes()` to re-optimize
        at full resolution for accurate dispatch results.

        Args:
            n_clusters: Number of clusters (typical segments) to extract (e.g., 8 typical days).
            cluster_duration: Duration of each cluster. Can be a pandas-style string
                ('1D', '24h', '6h') or a numeric value in hours.
            weights: Optional clustering weights per time series. Keys are time series labels.
            time_series_for_high_peaks: Time series labels for explicitly selecting high-value
                clusters. **Recommended** for demand time series to capture peak demand days.
            time_series_for_low_peaks: Time series labels for explicitly selecting low-value clusters.
            storage_inter_period_linking: If True, link storage states between clusters using
                boundary variables. This preserves long-term storage behavior. Default: True.
            storage_cyclic: If True, enforce SOC_boundary[0] = SOC_boundary[end] for storages.
                Only used when storage_inter_period_linking=True. Default: True.

        Returns:
            A new FlowSystem with reduced timesteps (only typical clusters).
            The FlowSystem has metadata stored in `_cluster_info` for weighting.

        Raises:
            ValueError: If timestep sizes are inconsistent.
            ValueError: If cluster_duration is not a multiple of timestep size.

        Examples:
            Two-stage sizing optimization:

            >>> # Stage 1: Size with reduced timesteps (fast)
            >>> fs_sizing = flow_system.transform.cluster_reduce(
            ...     n_clusters=8,
            ...     cluster_duration='1D',
            ...     time_series_for_high_peaks=['HeatDemand(Q_th)|fixed_relative_profile'],
            ... )
            >>> fs_sizing.optimize(solver)
            >>>
            >>> # Apply safety margin (typical clusters may smooth peaks)
            >>> sizes_with_margin = {
            ...     name: float(size.item()) * 1.05 for name, size in fs_sizing.statistics.sizes.items()
            ... }
            >>>
            >>> # Stage 2: Fix sizes and re-optimize at full resolution
            >>> fs_dispatch = flow_system.transform.fix_sizes(sizes_with_margin)
            >>> fs_dispatch.optimize(solver)

        Note:
            - This is best suited for initial sizing, not final dispatch optimization
            - Use `time_series_for_high_peaks` to ensure peak demand clusters are captured
            - A 5-10% safety margin on sizes is recommended for the dispatch stage
            - Storage linking adds SOC_boundary variables to track state between clusters
        """
        from .clustering import Clustering
        from .core import TimeSeriesData, drop_constant_arrays
        from .flow_system import FlowSystem

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
                f'cluster_reduce() requires uniform timestep sizes, got min={dt}h, '
                f'max={float(self._fs.timestep_duration.max().item())}h.'
            )
        if not np.isclose(hours_per_cluster / dt, round(hours_per_cluster / dt), atol=1e-9):
            raise ValueError(f'cluster_duration={hours_per_cluster}h must be a multiple of timestep size ({dt}h).')

        timesteps_per_cluster = int(round(hours_per_cluster / dt))
        has_periods = self._fs.periods is not None
        has_scenarios = self._fs.scenarios is not None

        logger.info(f'{"":#^80}')
        logger.info(f'{" Creating Typical Clusters ":#^80}')

        # Determine iteration dimensions
        periods = list(self._fs.periods) if has_periods else [None]
        scenarios = list(self._fs.scenarios) if has_scenarios else [None]

        ds = self._fs.to_dataset(include_solution=False)

        # Cluster each (period, scenario) combination
        clustering_results: dict[tuple, Clustering] = {}
        cluster_orders: dict[tuple, np.ndarray] = {}
        cluster_occurrences_all: dict[tuple, dict] = {}

        for period_label in periods:
            for scenario_label in scenarios:
                key = (period_label, scenario_label)
                selector = {k: v for k, v in [('period', period_label), ('scenario', scenario_label)] if v is not None}
                ds_slice = ds.sel(**selector, drop=True) if selector else ds
                temporaly_changing_ds = drop_constant_arrays(ds_slice, dim='time')

                if selector:
                    logger.info(f'Clustering {", ".join(f"{k}={v}" for k, v in selector.items())}...')

                clustering = Clustering(
                    original_data=temporaly_changing_ds.to_dataframe(),
                    hours_per_time_step=dt,
                    hours_per_period=hours_per_cluster,
                    nr_of_periods=n_clusters,
                    weights=weights or self._calculate_clustering_weights(temporaly_changing_ds),
                    time_series_for_high_peaks=time_series_for_high_peaks or [],
                    time_series_for_low_peaks=time_series_for_low_peaks or [],
                )
                clustering.cluster()

                clustering_results[key] = clustering
                cluster_orders[key] = clustering.tsam.clusterOrder
                cluster_occurrences_all[key] = clustering.tsam.clusterPeriodNoOccur

        # Use first clustering for structure
        first_key = (periods[0], scenarios[0])
        first_clustering = clustering_results[first_key]
        n_reduced_timesteps = len(first_clustering.tsam.typicalPeriods)
        actual_n_clusters = len(first_clustering.tsam.clusterPeriodNoOccur)

        # Create timestep weights from cluster occurrences
        cluster_occurrences = cluster_occurrences_all[first_key]
        timestep_weights = np.repeat(
            [cluster_occurrences.get(c, 1) for c in range(actual_n_clusters)], timesteps_per_cluster
        )

        logger.info(f'Reduced from {len(self._fs.timesteps)} to {n_reduced_timesteps} timesteps')
        logger.info(f'Clusters: {actual_n_clusters} (requested: {n_clusters})')

        # Create new time index
        new_time_index = pd.date_range(
            start=self._fs.timesteps[0], periods=n_reduced_timesteps, freq=pd.Timedelta(hours=dt)
        )

        # Build typical periods DataArrays keyed by (variable_name, (period, scenario))
        typical_das: dict[str, dict[tuple, xr.DataArray]] = {}
        for key, clustering in clustering_results.items():
            typical_df = clustering.tsam.typicalPeriods
            for col in typical_df.columns:
                typical_das.setdefault(col, {})[key] = xr.DataArray(
                    typical_df[col].values, dims=['time'], coords={'time': new_time_index}
                )

        # Build reduced dataset
        all_keys = {(p, s) for p in periods for s in scenarios}
        ds_new_vars = {}
        for name, original_da in ds.data_vars.items():
            if 'time' not in original_da.dims:
                ds_new_vars[name] = original_da.copy()
            elif name not in typical_das or set(typical_das[name].keys()) != all_keys:
                # Time-dependent but constant (or not present in all clustering results): slice to new time length
                ds_new_vars[name] = original_da.isel(time=slice(0, n_reduced_timesteps)).assign_coords(
                    time=new_time_index
                )
            else:
                # Time-varying: combine per-(period, scenario) slices
                da = self._combine_slices_to_dataarray(
                    slices=typical_das[name],
                    original_da=original_da,
                    new_time_index=new_time_index,
                    periods=periods,
                    scenarios=scenarios,
                )
                if TimeSeriesData.is_timeseries_data(original_da):
                    da = TimeSeriesData.from_dataarray(da.assign_attrs(original_da.attrs))
                ds_new_vars[name] = da

        ds_new = xr.Dataset(ds_new_vars, attrs=ds.attrs)
        ds_new.attrs['timesteps_per_cluster'] = timesteps_per_cluster
        ds_new.attrs['timestep_duration'] = dt

        reduced_fs = FlowSystem.from_dataset(ds_new)
        reduced_fs.cluster_weight = reduced_fs.fit_to_model_coords('cluster_weight', timestep_weights, dims=['time'])

        reduced_fs._cluster_info = {
            'clustering_results': clustering_results,
            'cluster_orders': cluster_orders,
            'cluster_occurrences': cluster_occurrences_all,
            'timestep_weights': timestep_weights,
            'n_clusters': actual_n_clusters,
            'timesteps_per_cluster': timesteps_per_cluster,
            'storage_inter_period_linking': storage_inter_period_linking,
            'storage_cyclic': storage_cyclic,
            'original_fs': self._fs,
            'has_periods': has_periods,
            'has_scenarios': has_scenarios,
            'cluster_order': cluster_orders[first_key],
            'clustering': first_clustering,
        }

        return reduced_fs

    @staticmethod
    def _combine_slices_to_dataarray(
        slices: dict[tuple, xr.DataArray],
        original_da: xr.DataArray,
        new_time_index: pd.DatetimeIndex,
        periods: list,
        scenarios: list,
    ) -> xr.DataArray:
        """Combine per-(period, scenario) slices into a multi-dimensional DataArray using xr.concat.

        Args:
            slices: Dict mapping (period, scenario) tuples to 1D DataArrays (time only).
            original_da: Original DataArray to get dimension order and attrs from.
            new_time_index: New time coordinate for the output.
            periods: List of period labels ([None] if no periods dimension).
            scenarios: List of scenario labels ([None] if no scenarios dimension).

        Returns:
            DataArray with dimensions matching original_da but reduced time.
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

        # Match original dimension order
        target_dims = [d for d in original_da.dims if d in result.dims]
        if target_dims and tuple(target_dims) != result.dims:
            result = result.transpose(*target_dims)

        return result.assign_attrs(original_da.attrs)

    def expand_solution(self) -> FlowSystem:
        """Expand a reduced (clustered) FlowSystem back to full original timesteps.

        After solving a FlowSystem created with ``cluster_reduce()``, this method
        disaggregates the FlowSystem by:
        1. Expanding all time series data from typical clusters to full timesteps
        2. Expanding the solution by mapping each typical cluster back to all
           original segments it represents

        For FlowSystems with periods and/or scenarios, each (period, scenario)
        combination is expanded using its own cluster assignment.

        This enables using all existing solution accessors (``statistics``, ``plot``, etc.)
        with full time resolution, where both the data and solution are consistently
        expanded from the typical clusters.

        Returns:
            FlowSystem: A new FlowSystem with full timesteps and expanded solution.

        Raises:
            ValueError: If the FlowSystem was not created with ``cluster_reduce()``.
            ValueError: If the FlowSystem has no solution.

        Examples:
            Two-stage optimization with solution expansion:

            >>> # Stage 1: Size with reduced timesteps
            >>> fs_reduced = flow_system.transform.cluster_reduce(
            ...     n_clusters=8,
            ...     cluster_duration='1D',
            ... )
            >>> fs_reduced.optimize(solver)
            >>>
            >>> # Expand to full resolution FlowSystem
            >>> fs_expanded = fs_reduced.transform.expand_solution()
            >>>
            >>> # Use all existing accessors with full timesteps
            >>> fs_expanded.statistics.flow_rates  # Full 8760 timesteps
            >>> fs_expanded.statistics.plot.balance('HeatBus')  # Full resolution plots
            >>> fs_expanded.statistics.plot.heatmap('Boiler(Q_th)|flow_rate')

        Note:
            The expanded FlowSystem repeats the typical cluster values for all
            segments belonging to the same cluster. Both input data and solution
            are consistently expanded, so they match. This is an approximation -
            the actual dispatch at full resolution would differ due to
            intra-cluster variations in time series data.

            For accurate dispatch results, use ``fix_sizes()`` to fix the sizes
            from the reduced optimization and re-optimize at full resolution.
        """
        from .flow_system import FlowSystem

        # Validate
        if not hasattr(self._fs, '_cluster_info') or self._fs._cluster_info is None:
            raise ValueError(
                'expand_solution() requires a FlowSystem created with cluster_reduce(). '
                'This FlowSystem has no cluster info.'
            )
        if self._fs.solution is None:
            raise ValueError('FlowSystem has no solution. Run optimize() or solve() first.')

        info = self._fs._cluster_info
        timesteps_per_cluster = info['timesteps_per_cluster']
        original_fs: FlowSystem = info['original_fs']
        n_clusters = info['n_clusters']
        has_periods = info.get('has_periods', False)
        has_scenarios = info.get('has_scenarios', False)
        cluster_orders = info.get('cluster_orders', {(None, None): info['cluster_order']})

        periods = list(original_fs.periods) if has_periods else [None]
        scenarios = list(original_fs.scenarios) if has_scenarios else [None]

        original_timesteps = original_fs.timesteps
        n_original_timesteps = len(original_timesteps)
        n_reduced_timesteps = n_clusters * timesteps_per_cluster
        first_key = (periods[0], scenarios[0])

        # Build expansion mappings per (period, scenario)
        mappings = {
            key: self._build_expansion_mapping(order, timesteps_per_cluster, n_original_timesteps)
            for key, order in cluster_orders.items()
        }

        # Expand function for DataArrays
        def expand_da(da: xr.DataArray) -> xr.DataArray:
            if 'time' not in da.dims:
                return da.copy()
            return self._expand_dataarray(da, mappings, original_timesteps, periods, scenarios)

        # 1. Expand FlowSystem data
        reduced_ds = self._fs.to_dataset(include_solution=False)
        expanded_ds = xr.Dataset(
            {name: expand_da(da) for name, da in reduced_ds.data_vars.items()}, attrs=reduced_ds.attrs
        )
        expanded_ds.attrs['timestep_duration'] = original_fs.timestep_duration.values.tolist()

        expanded_fs = FlowSystem.from_dataset(expanded_ds)

        # 2. Expand solution
        reduced_solution = self._fs.solution
        expanded_fs._solution = xr.Dataset(
            {name: expand_da(da) for name, da in reduced_solution.data_vars.items()},
            attrs=reduced_solution.attrs,
        )

        n_combinations = len(periods) * len(scenarios)
        logger.info(
            f'Expanded FlowSystem from {n_reduced_timesteps} to {n_original_timesteps} timesteps '
            f'({n_clusters} clusters'
            + (
                f', {n_combinations} period/scenario combinations)'
                if n_combinations > 1
                else f' → {len(cluster_orders[first_key])} original segments)'
            )
        )

        return expanded_fs

    @staticmethod
    def _build_expansion_mapping(
        cluster_order: np.ndarray, timesteps_per_cluster: int, n_original_timesteps: int
    ) -> np.ndarray:
        """Build mapping from original timesteps to reduced (typical) timesteps.

        Args:
            cluster_order: Array mapping each original segment to its cluster ID.
            timesteps_per_cluster: Number of timesteps per cluster.
            n_original_timesteps: Total number of original timesteps.

        Returns:
            Array where mapping[i] gives the reduced timestep index for original timestep i.
        """
        n_reduced = len(set(cluster_order)) * timesteps_per_cluster
        segment_indices = np.arange(n_original_timesteps) // timesteps_per_cluster
        pos_in_segment = np.arange(n_original_timesteps) % timesteps_per_cluster
        # Handle edge case where segment_indices exceed cluster_order length
        safe_segment_indices = np.minimum(segment_indices, len(cluster_order) - 1)
        cluster_ids = cluster_order[safe_segment_indices]
        mapping = cluster_ids * timesteps_per_cluster + pos_in_segment
        return np.minimum(mapping, n_reduced - 1).astype(np.int32)

    @staticmethod
    def _expand_dataarray(
        da: xr.DataArray,
        mappings: dict[tuple, np.ndarray],
        original_timesteps: pd.DatetimeIndex,
        periods: list,
        scenarios: list,
    ) -> xr.DataArray:
        """Expand a DataArray from reduced to original timesteps using cluster mappings.

        Args:
            da: DataArray with reduced time dimension.
            mappings: Dict mapping (period, scenario) tuples to expansion index arrays.
            original_timesteps: Original time coordinates.
            periods: List of period labels ([None] if no periods).
            scenarios: List of scenario labels ([None] if no scenarios).

        Returns:
            DataArray with expanded time dimension.
        """
        first_key = (periods[0], scenarios[0])
        has_periods = periods != [None]
        has_scenarios = scenarios != [None]

        # Simple case: no period/scenario dimensions in the data
        if (not has_periods and not has_scenarios) or ('period' not in da.dims and 'scenario' not in da.dims):
            mapping = mappings[first_key]
            expanded = da.isel(time=xr.DataArray(mapping, dims=['time']))
            return expanded.assign_coords(time=original_timesteps).assign_attrs(da.attrs)

        # Multi-dimensional: expand each (period, scenario) slice and recombine
        expanded_slices: dict[tuple, xr.DataArray] = {}
        for p in periods:
            for s in scenarios:
                key = (p, s)
                mapping = mappings[key]

                # Select the slice for this (period, scenario) combination
                selector = {}
                if p is not None and 'period' in da.dims:
                    selector['period'] = p
                if s is not None and 'scenario' in da.dims:
                    selector['scenario'] = s

                slice_da = da.sel(**selector, drop=True) if selector else da
                expanded = slice_da.isel(time=xr.DataArray(mapping, dims=['time']))
                expanded_slices[key] = expanded.assign_coords(time=original_timesteps)

        # Recombine slices using _combine_slices_to_dataarray
        return TransformAccessor._combine_slices_to_dataarray(
            slices=expanded_slices,
            original_da=da,
            new_time_index=original_timesteps,
            periods=periods,
            scenarios=scenarios,
        )

    # Future methods can be added here:
    #
    # def mga(self, alternatives: int = 5) -> FlowSystem:
    #     """Create a FlowSystem configured for Modeling to Generate Alternatives."""
    #     ...
