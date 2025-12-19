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
        >>> expanded_fs = reduced_fs.transform.expand_solution()

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
            logger.info('All Clustering weights were set to 1')

        return weights

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

    def cluster(
        self,
        n_clusters: int,
        cluster_duration: str | float,
        weights: dict[str, float] | None = None,
        time_series_for_high_peaks: list[str] | None = None,
        time_series_for_low_peaks: list[str] | None = None,
        storage: Literal['independent', 'cyclic', 'intercluster', 'intercluster_cyclic'] = 'intercluster_cyclic',
    ) -> FlowSystem:
        """
        Create a FlowSystem with reduced timesteps using typical clusters.

        This method creates a new FlowSystem optimized for sizing studies by reducing
        the number of timesteps to only the typical (representative) clusters identified
        through time series aggregation using the tsam package.

        The method:
        1. Performs time series clustering using tsam (k-means)
        2. Extracts only the typical clusters (not all original timesteps)
        3. Applies timestep weighting for accurate cost representation
        4. Handles storage states between clusters based on the ``storage`` mode

        Use this for initial sizing optimization, then use ``fix_sizes()`` to re-optimize
        at full resolution for accurate dispatch results.

        Args:
            n_clusters: Number of clusters (typical periods) to extract (e.g., 8 typical days).
            cluster_duration: Duration of each cluster. Can be a pandas-style string
                ('1D', '24h', '6h') or a numeric value in hours.
            weights: Optional clustering weights per time series. Keys are time series labels.
            time_series_for_high_peaks: Time series labels for explicitly selecting high-value
                clusters. **Recommended** for demand time series to capture peak demand days.
            time_series_for_low_peaks: Time series labels for explicitly selecting low-value clusters.
            storage: How storages are treated during clustering. Options:

                - ``'independent'``: Clusters are fully decoupled. No constraints between
                  clusters, each cluster has free start/end SOC. Fast but ignores
                  seasonal storage value.
                - ``'cyclic'``: Each cluster is self-contained. The SOC at the start of
                  each cluster equals its end (cluster returns to initial state).
                  Good for "average day" modeling.
                - ``'intercluster'``: Link storage state across the original timeline using
                  SOC boundary variables (Kotzur et al. approach). Properly values
                  seasonal storage patterns. Overall SOC can drift.
                - ``'intercluster_cyclic'`` (default): Like 'intercluster' but also enforces
                  that overall SOC returns to initial state (yearly cyclic).

        Returns:
            A new FlowSystem with reduced timesteps (only typical clusters).
            The FlowSystem has metadata stored in ``clustering`` for expansion.

        Raises:
            ValueError: If timestep sizes are inconsistent.
            ValueError: If cluster_duration is not a multiple of timestep size.

        Examples:
            Two-stage sizing optimization:

            >>> # Stage 1: Size with reduced timesteps (fast)
            >>> fs_sizing = flow_system.transform.cluster(
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
            - Use ``time_series_for_high_peaks`` to ensure peak demand clusters are captured
            - A 5-10% safety margin on sizes is recommended for the dispatch stage
            - For seasonal storage (e.g., hydrogen, thermal storage), use 'intercluster' or
              'intercluster_cyclic' to properly value long-term storage
        """
        import tsam.timeseriesaggregation as tsam

        from .aggregation import Clustering, ClusterResult, ClusterStructure
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
                f'cluster() requires uniform timestep sizes, got min={dt}h, '
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

        # Cluster each (period, scenario) combination using tsam directly
        tsam_results: dict[tuple, tsam.TimeSeriesAggregation] = {}
        cluster_orders: dict[tuple, np.ndarray] = {}
        cluster_occurrences_all: dict[tuple, dict] = {}
        use_extreme_periods = bool(time_series_for_high_peaks or time_series_for_low_peaks)

        for period_label in periods:
            for scenario_label in scenarios:
                key = (period_label, scenario_label)
                selector = {k: v for k, v in [('period', period_label), ('scenario', scenario_label)] if v is not None}
                ds_slice = ds.sel(**selector, drop=True) if selector else ds
                temporaly_changing_ds = drop_constant_arrays(ds_slice, dim='time')
                df = temporaly_changing_ds.to_dataframe()

                if selector:
                    logger.info(f'Clustering {", ".join(f"{k}={v}" for k, v in selector.items())}...')

                # Use tsam directly
                clustering_weights = weights or self._calculate_clustering_weights(temporaly_changing_ds)
                tsam_agg = tsam.TimeSeriesAggregation(
                    df,
                    noTypicalPeriods=n_clusters,
                    hoursPerPeriod=hours_per_cluster,
                    resolution=dt,
                    clusterMethod='k_means',
                    extremePeriodMethod='new_cluster_center' if use_extreme_periods else 'None',
                    weightDict={name: w for name, w in clustering_weights.items() if name in df.columns},
                    addPeakMax=time_series_for_high_peaks or [],
                    addPeakMin=time_series_for_low_peaks or [],
                )
                tsam_agg.createTypicalPeriods()

                tsam_results[key] = tsam_agg
                cluster_orders[key] = tsam_agg.clusterOrder
                cluster_occurrences_all[key] = tsam_agg.clusterPeriodNoOccur

        # Use first result for structure
        first_key = (periods[0], scenarios[0])
        first_tsam = tsam_results[first_key]
        n_reduced_timesteps = len(first_tsam.typicalPeriods)
        actual_n_clusters = len(first_tsam.clusterPeriodNoOccur)

        # Create new time index (needed for weights and typical periods)
        new_time_index = pd.date_range(
            start=self._fs.timesteps[0], periods=n_reduced_timesteps, freq=pd.Timedelta(hours=dt)
        )

        # Create timestep weights from cluster occurrences (per period/scenario)
        def _build_weights_for_key(key: tuple) -> xr.DataArray:
            occurrences = cluster_occurrences_all[key]
            weights = np.repeat([occurrences.get(c, 1) for c in range(actual_n_clusters)], timesteps_per_cluster)
            return xr.DataArray(weights, dims=['time'], coords={'time': new_time_index})

        # Build weights - use _combine_slices_to_dataarray for consistent multi-dim handling
        weights_slices = {key: _build_weights_for_key(key) for key in cluster_occurrences_all}
        # Create a dummy 1D DataArray as template for _combine_slices_to_dataarray
        dummy_template = xr.DataArray(np.zeros(n_reduced_timesteps), dims=['time'])
        timestep_weights = self._combine_slices_to_dataarray(
            weights_slices, dummy_template, new_time_index, periods, scenarios
        )

        logger.info(f'Reduced from {len(self._fs.timesteps)} to {n_reduced_timesteps} timesteps')
        logger.info(f'Clusters: {actual_n_clusters} (requested: {n_clusters})')

        # Build typical periods DataArrays keyed by (variable_name, (period, scenario))
        typical_das: dict[str, dict[tuple, xr.DataArray]] = {}
        for key, tsam_agg in tsam_results.items():
            typical_df = tsam_agg.typicalPeriods
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
        # Set cluster_weight - might have period/scenario dimensions
        reduced_fs.cluster_weight = reduced_fs.fit_to_model_coords(
            'cluster_weight', timestep_weights, dims=['scenario', 'period', 'time']
        )

        # Remove 'equals_final' from storages - doesn't make sense on reduced timesteps
        for storage in reduced_fs.storages.values():
            # Handle both scalar and xarray cases
            ics = storage.initial_charge_state
            if isinstance(ics, str) and ics == 'equals_final':
                storage.initial_charge_state = 0

        # Build Clustering for inter-cluster linking and solution expansion
        n_original_timesteps = len(self._fs.timesteps)

        # Build per-slice cluster_order and timestep_mapping as multi-dimensional DataArrays
        # This is needed because each (period, scenario) combination may have different clustering

        def _build_timestep_mapping_for_key(key: tuple) -> np.ndarray:
            """Build timestep_mapping for a single (period, scenario) slice."""
            mapping = np.zeros(n_original_timesteps, dtype=np.int32)
            for period_idx, cluster_id in enumerate(cluster_orders[key]):
                for pos in range(timesteps_per_cluster):
                    original_idx = period_idx * timesteps_per_cluster + pos
                    if original_idx < n_original_timesteps:
                        representative_idx = cluster_id * timesteps_per_cluster + pos
                        mapping[original_idx] = representative_idx
            return mapping

        def _build_cluster_occurrences_for_key(key: tuple) -> np.ndarray:
            """Build cluster_occurrences array for a single (period, scenario) slice."""
            occurrences = cluster_occurrences_all[key]
            return np.array([occurrences.get(c, 0) for c in range(actual_n_clusters)])

        # Build multi-dimensional arrays
        if has_periods or has_scenarios:
            # Multi-dimensional case: build arrays for each (period, scenario) combination
            # cluster_order: dims [original_period, period?, scenario?]
            cluster_order_slices = {}
            timestep_mapping_slices = {}
            cluster_occurrences_slices = {}

            for p in periods:
                for s in scenarios:
                    key = (p, s)
                    cluster_order_slices[key] = xr.DataArray(
                        cluster_orders[key], dims=['original_period'], name='cluster_order'
                    )
                    timestep_mapping_slices[key] = xr.DataArray(
                        _build_timestep_mapping_for_key(key), dims=['original_time'], name='timestep_mapping'
                    )
                    cluster_occurrences_slices[key] = xr.DataArray(
                        _build_cluster_occurrences_for_key(key), dims=['cluster'], name='cluster_occurrences'
                    )

            # Combine slices into multi-dimensional DataArrays
            cluster_order_da = self._combine_slices_to_dataarray_generic(
                cluster_order_slices, ['original_period'], periods, scenarios, 'cluster_order'
            )
            timestep_mapping_da = self._combine_slices_to_dataarray_generic(
                timestep_mapping_slices, ['original_time'], periods, scenarios, 'timestep_mapping'
            )
            cluster_occurrences_da = self._combine_slices_to_dataarray_generic(
                cluster_occurrences_slices, ['cluster'], periods, scenarios, 'cluster_occurrences'
            )
        else:
            # Simple case: single (None, None) slice
            cluster_order_da = xr.DataArray(cluster_orders[first_key], dims=['original_period'], name='cluster_order')
            timestep_mapping_da = xr.DataArray(
                _build_timestep_mapping_for_key(first_key), dims=['original_time'], name='timestep_mapping'
            )
            cluster_occurrences_da = xr.DataArray(
                _build_cluster_occurrences_for_key(first_key), dims=['cluster'], name='cluster_occurrences'
            )

        cluster_structure = ClusterStructure(
            cluster_order=cluster_order_da,
            cluster_occurrences=cluster_occurrences_da,
            n_clusters=actual_n_clusters,
            timesteps_per_cluster=timesteps_per_cluster,
        )

        aggregation_result = ClusterResult(
            timestep_mapping=timestep_mapping_da,
            n_representatives=n_reduced_timesteps,
            representative_weights=timestep_weights.rename('representative_weights'),
            cluster_structure=cluster_structure,
            original_data=ds,
            aggregated_data=ds_new,
        )

        reduced_fs.clustering = Clustering(
            result=aggregation_result,
            original_flow_system=self._fs,
            backend_name='tsam',
            storage_mode=storage,
        )

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

        # Put time dimension first (standard order), preserve other dims
        result = result.transpose('time', ...)

        return result.assign_attrs(original_da.attrs)

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
            base_dims: Base dimensions of each slice (e.g., ['original_period'] or ['original_time']).
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

        # Put base dimension first (standard order)
        result = result.transpose(base_dims[0], ...)

        return result.rename(name)

    def expand_solution(self) -> FlowSystem:
        """Expand a reduced (clustered) FlowSystem back to full original timesteps.

        After solving a FlowSystem created with ``cluster()``, this method
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
            ValueError: If the FlowSystem was not created with ``cluster()``.
            ValueError: If the FlowSystem has no solution.

        Examples:
            Two-stage optimization with solution expansion:

            >>> # Stage 1: Size with reduced timesteps
            >>> fs_reduced = flow_system.transform.cluster(
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
        if self._fs.clustering is None:
            raise ValueError(
                'expand_solution() requires a FlowSystem created with cluster(). '
                'This FlowSystem has no aggregation info.'
            )
        if self._fs.solution is None:
            raise ValueError('FlowSystem has no solution. Run optimize() or solve() first.')

        info = self._fs.clustering
        cluster_structure = info.result.cluster_structure
        if cluster_structure is None:
            raise ValueError('No cluster structure available for expansion.')

        timesteps_per_cluster = cluster_structure.timesteps_per_cluster
        original_fs: FlowSystem = info.original_flow_system
        n_clusters = (
            int(cluster_structure.n_clusters)
            if isinstance(cluster_structure.n_clusters, (int, np.integer))
            else int(cluster_structure.n_clusters.values)
        )
        has_periods = original_fs.periods is not None
        has_scenarios = original_fs.scenarios is not None

        periods = list(original_fs.periods) if has_periods else [None]
        scenarios = list(original_fs.scenarios) if has_scenarios else [None]

        original_timesteps = original_fs.timesteps
        n_original_timesteps = len(original_timesteps)
        n_reduced_timesteps = n_clusters * timesteps_per_cluster

        # Expand function using ClusterResult.expand_data() - handles multi-dimensional cases
        def expand_da(da: xr.DataArray) -> xr.DataArray:
            if 'time' not in da.dims:
                return da.copy()
            return info.result.expand_data(da, original_time=original_timesteps)

        # 1. Expand FlowSystem data (with cluster_weight set to 1.0 for all timesteps)
        reduced_ds = self._fs.to_dataset(include_solution=False)
        expanded_ds = xr.Dataset(
            {name: expand_da(da) for name, da in reduced_ds.data_vars.items() if name != 'cluster_weight'},
            attrs=reduced_ds.attrs,
        )
        expanded_ds.attrs['timestep_duration'] = original_fs.timestep_duration.values.tolist()

        # Create cluster_weight with value 1.0 for all timesteps (no weighting needed for expanded)
        # Use _combine_slices_to_dataarray for consistent multi-dim handling
        ones_da = xr.DataArray(np.ones(n_original_timesteps), dims=['time'], coords={'time': original_timesteps})
        ones_slices = {(p, s): ones_da for p in periods for s in scenarios}
        cluster_weight = self._combine_slices_to_dataarray(
            ones_slices, ones_da, original_timesteps, periods, scenarios
        ).rename('cluster_weight')
        expanded_ds['cluster_weight'] = cluster_weight

        expanded_fs = FlowSystem.from_dataset(expanded_ds)

        # 2. Expand solution
        reduced_solution = self._fs.solution
        expanded_fs._solution = xr.Dataset(
            {name: expand_da(da) for name, da in reduced_solution.data_vars.items()},
            attrs=reduced_solution.attrs,
        )

        n_combinations = len(periods) * len(scenarios)
        n_original_segments = cluster_structure.n_original_periods
        logger.info(
            f'Expanded FlowSystem from {n_reduced_timesteps} to {n_original_timesteps} timesteps '
            f'({n_clusters} clusters'
            + (
                f', {n_combinations} period/scenario combinations)'
                if n_combinations > 1
                else f' → {n_original_segments} original segments)'
            )
        )

        return expanded_fs
