"""Statistics accessor for individual nodes (ComponentResults/BusResults).

This module provides the NodeStatisticsAccessor class which adds statistics
methods to individual component and bus results. Each method is decorated
with @MethodHandlerWrapper to return StatisticPlotter objects that provide
both data access and plotting capabilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr

from ..plotting_accessor import MethodHandlerWrapper, StatisticPlotter

if TYPE_CHECKING:
    from ..results import BusResults, ComponentResults


class NodeStatisticsAccessor:
    """Accessor for calculating statistics on individual node results.

    This class provides methods to calculate various statistics from
    ComponentResults or BusResults, all of which return StatisticPlotter
    objects for easy visualization and data access.

    The accessor integrates seamlessly with node results:
    - Access via: results['NodeName'].statistics.method_name()
    - Each method returns StatisticPlotter with .plot interface
    - Call the plotter to get raw xarray data: plotter()
    - Create visualizations: plotter.plot.bar(), plotter.plot.line(), etc.

    Args:
        parent: The parent ComponentResults or BusResults object

    Examples:
        >>> results = CalculationResults.from_file('results', 'optimization')
        >>>
        >>> # Get flow summary for a component
        >>> data = results['Boiler'].statistics.flow_summary()()
        >>>
        >>> # Create visualization
        >>> fig = results['Boiler'].statistics.flow_summary().plot.bar()
        >>> fig.show()
        >>>
        >>> # Capacity utilization over time
        >>> fig = results['CHP'].statistics.capacity_utilization().plot.line()
        >>>
        >>> # Storage-specific metrics
        >>> fig = results['Battery'].statistics.storage_metrics().plot.bar()
    """

    def __init__(self, parent: ComponentResults | BusResults):
        """Initialize the accessor with parent node results.

        Args:
            parent: The parent ComponentResults or BusResults object
        """
        self._parent = parent

    @MethodHandlerWrapper(handler_class=StatisticPlotter)
    def flow_summary(
        self,
        unit_type: Literal['flow_rate', 'flow_hours'] = 'flow_hours',
        aggregate_time: bool = True,
    ) -> xr.Dataset:
        """Calculate summary statistics of all flows at this node.

        Provides min, max, mean, sum, and std statistics for all flows
        connected to this node (inputs and outputs).

        Args:
            unit_type: Whether to analyze 'flow_rate' or 'flow_hours'. Defaults to 'flow_hours'.
            aggregate_time: If True, aggregate over time dimension. Defaults to True.

        Returns:
            Dataset with flow statistics for this node

        Examples:
            >>> # Total flow hours for all flows
            >>> results['Boiler'].statistics.flow_summary().plot.bar()
            >>>
            >>> # Time series of flow rates
            >>> results['Boiler'].statistics.flow_summary(unit_type='flow_rate', aggregate_time=False).plot.line()
        """
        # Get node balance dataset
        ds = self._parent.node_balance(
            with_last_timestep=False, unit_type=unit_type, drop_suffix=True, negate_inputs=False
        )

        if aggregate_time:
            # Calculate statistics over time for each variable
            results = []
            for var in ds.data_vars:
                var_stats = xr.Dataset(
                    {
                        f'{var}_sum': ds[var].sum('time'),
                        f'{var}_mean': ds[var].mean('time'),
                        f'{var}_max': ds[var].max('time'),
                        f'{var}_min': ds[var].min('time'),
                        f'{var}_std': ds[var].std('time'),
                    }
                )
                results.append(var_stats)

            # Merge all variable stats
            result = xr.merge(results)
        else:
            # Keep time dimension
            result = ds

        return result

    @MethodHandlerWrapper(handler_class=StatisticPlotter)
    def flow_hours(self) -> xr.Dataset:
        """Calculate total flow hours for each flow at this node.

        Convenient shorthand for flow_summary with flow_hours and aggregation.

        Returns:
            Dataset with total flow hours for each flow

        Examples:
            >>> # Total flow hours
            >>> results['Boiler'].statistics.flow_hours().plot.bar()
        """
        # Get node balance in flow_hours
        ds = self._parent.node_balance(
            with_last_timestep=False, unit_type='flow_hours', drop_suffix=True, negate_inputs=False
        )

        # Sum over time to get total flow hours
        result = ds.sum('time')

        return result

    @MethodHandlerWrapper(handler_class=StatisticPlotter)
    def capacity_utilization(self, nominal_capacity: dict[str, float] | None = None) -> xr.Dataset:
        """Calculate capacity utilization for flows at this node.

        Shows what percentage of nominal capacity is being used.

        Args:
            nominal_capacity: Dict mapping flow names to nominal capacities.
                If None, attempts to extract from flow_system data.

        Returns:
            Dataset with capacity utilization statistics (mean, max)

        Examples:
            >>> # Capacity utilization with auto-detected capacities
            >>> results['Boiler'].statistics.capacity_utilization().plot.bar()
            >>>
            >>> # With custom capacities
            >>> results['Boiler'].statistics.capacity_utilization(
            ...     nominal_capacity={'Q_th': 100.0, 'Q_fu': 120.0}
            ... ).plot.bar()
        """
        # Get flow rates
        ds = self._parent.node_balance(
            with_last_timestep=False, unit_type='flow_rate', drop_suffix=True, negate_inputs=False
        )

        # Try to get nominal capacities from flow_system if not provided
        if nominal_capacity is None:
            nominal_capacity = {}
            try:
                flow_system = self._parent._calculation_results.flow_system
                # Extract capacities from flows
                for flow_name in ds.data_vars:
                    # Try to find corresponding flow in flow_system
                    # This is a simplified approach - may need refinement
                    for flow in flow_system.flows.values():
                        if flow.label in flow_name:
                            if hasattr(flow, 'size') and flow.size is not None:
                                nominal_capacity[flow_name] = float(flow.size.max())
            except Exception:
                pass

        # Calculate utilization
        utilization = xr.Dataset()

        for var in ds.data_vars:
            if var in nominal_capacity and nominal_capacity[var] > 0:
                util = (ds[var] / nominal_capacity[var]) * 100.0
                utilization[f'{var}_utilization'] = xr.Dataset(
                    {
                        'mean': util.mean('time'),
                        'max': util.max('time'),
                    }
                ).to_array(dim='metric')

        if not utilization.data_vars:
            # No capacities available, just return relative values
            utilization = (
                xr.Dataset(
                    {
                        'mean_flow': ds.mean('time'),
                        'max_flow': ds.max('time'),
                    }
                )
                .to_array(dim='metric')
                .to_dataset(dim='variable')
            )

        return utilization

    @MethodHandlerWrapper(handler_class=StatisticPlotter)
    def peak_analysis(self, n_peaks: int = 10) -> xr.Dataset:
        """Identify peak flow periods and their magnitudes.

        Args:
            n_peaks: Number of top peaks to identify per flow. Defaults to 10.

        Returns:
            Dataset with peak flow information

        Examples:
            >>> # Top 10 peaks per flow
            >>> results['Boiler'].statistics.peak_analysis().plot.bar()
        """
        # Get flow rates
        ds = self._parent.node_balance(
            with_last_timestep=False, unit_type='flow_rate', drop_suffix=True, negate_inputs=False
        )

        peaks = xr.Dataset()

        for var in ds.data_vars:
            # Get absolute values to handle negative flows
            abs_values = np.abs(ds[var].values)

            # Find peak magnitude
            peak_value = float(np.nanmax(abs_values)) if abs_values.size > 0 else 0.0

            # Find average of top n_peaks
            if abs_values.size >= n_peaks:
                top_n = np.partition(abs_values.flatten(), -n_peaks)[-n_peaks:]
                avg_top_n = float(np.nanmean(top_n))
            else:
                avg_top_n = float(np.nanmean(abs_values))

            peaks[var] = xr.DataArray(
                [peak_value, avg_top_n], dims=['metric'], coords={'metric': ['peak', f'avg_top_{n_peaks}']}
            )

        return peaks

    @MethodHandlerWrapper(handler_class=StatisticPlotter)
    def storage_metrics(self) -> xr.Dataset:
        """Calculate storage-specific metrics (cycles, utilization, efficiency).

        Only available for storage components.

        Returns:
            Dataset with storage metrics

        Raises:
            ValueError: If node is not a storage

        Examples:
            >>> # Storage metrics
            >>> results['Battery'].statistics.storage_metrics().plot.bar()
        """
        if not self._parent.is_storage:
            raise ValueError(f'Node "{self._parent.label}" is not a storage component')

        # Get charge state
        charge_state = self._parent.charge_state

        # Calculate metrics
        metrics = xr.Dataset()

        # Capacity utilization
        max_capacity = float(charge_state.max())
        min_capacity = float(charge_state.min())
        mean_capacity = float(charge_state.mean())

        if max_capacity > 0:
            utilization = (mean_capacity / max_capacity) * 100.0
        else:
            utilization = 0.0

        # Estimate cycles (simplified: count sign changes in delta)
        delta = charge_state.diff('time')
        sign_changes = (np.diff(np.sign(delta.values)) != 0).sum()
        cycles = sign_changes / 2.0  # Full cycle = charge + discharge

        # Create metrics dataset
        metrics['storage_metrics'] = xr.DataArray(
            [max_capacity, min_capacity, mean_capacity, utilization, cycles],
            dims=['metric'],
            coords={
                'metric': [
                    'max_capacity',
                    'min_capacity',
                    'mean_capacity',
                    'utilization_percent',
                    'estimated_cycles',
                ]
            },
        )

        return metrics

    @MethodHandlerWrapper(handler_class=StatisticPlotter)
    def temporal_patterns(
        self, freq: Literal['h', 'D', 'W', 'MS'] = 'D', stat: Literal['mean', 'sum', 'max', 'min'] = 'mean'
    ) -> xr.Dataset:
        """Analyze temporal patterns in flows (hourly, daily, weekly, monthly).

        Args:
            freq: Frequency for grouping ('h'=hourly, 'D'=daily, 'W'=weekly, 'MS'=monthly).
                Defaults to 'D' (daily).
            stat: Statistic to calculate ('mean', 'sum', 'max', 'min'). Defaults to 'mean'.

        Returns:
            Dataset with temporal patterns

        Examples:
            >>> # Daily mean flow patterns
            >>> results['Boiler'].statistics.temporal_patterns(freq='D').plot.line()
            >>>
            >>> # Weekly sum patterns
            >>> results['Boiler'].statistics.temporal_patterns(freq='W', stat='sum').plot.bar()
        """
        # Get flow rates
        ds = self._parent.node_balance(
            with_last_timestep=False, unit_type='flow_rate', drop_suffix=True, negate_inputs=False
        )

        # Resample based on frequency
        resampled = ds.resample(time=freq)

        # Apply statistic
        if stat == 'mean':
            result = resampled.mean()
        elif stat == 'sum':
            result = resampled.sum()
        elif stat == 'max':
            result = resampled.max()
        elif stat == 'min':
            result = resampled.min()
        else:
            raise ValueError(f'Invalid stat: {stat}. Use mean, sum, max, or min.')

        return result

    @MethodHandlerWrapper(handler_class=StatisticPlotter)
    def flow_statistics(self, flow_names: str | list[str] | None = None) -> xr.Dataset:
        """Get detailed statistics for specific flows.

        Args:
            flow_names: Flow name(s) to analyze. If None, analyzes all flows.

        Returns:
            Dataset with comprehensive statistics per flow

        Examples:
            >>> # Single flow analysis
            >>> results['Boiler'].statistics.flow_statistics('Boiler(Q_th)').plot.bar()
            >>>
            >>> # Compare multiple flows
            >>> results['Boiler'].statistics.flow_statistics(['Boiler(Q_th)', 'Boiler(Q_fu)']).plot.bar()
        """
        # Get node balance
        ds = self._parent.node_balance(
            with_last_timestep=False, unit_type='flow_rate', drop_suffix=True, negate_inputs=False
        )

        # Filter to specific flows if requested
        if flow_names is not None:
            if isinstance(flow_names, str):
                flow_names = [flow_names]
            ds = ds[[var for var in ds.data_vars if any(fn in var for fn in flow_names)]]

        # Calculate comprehensive statistics
        stats = xr.Dataset()
        for var in ds.data_vars:
            stats[var] = xr.DataArray(
                [
                    float(ds[var].sum('time')),
                    float(ds[var].mean('time')),
                    float(ds[var].max('time')),
                    float(ds[var].min('time')),
                    float(ds[var].std('time')),
                    float(ds[var].median('time')),
                ],
                dims=['statistic'],
                coords={'statistic': ['sum', 'mean', 'max', 'min', 'std', 'median']},
            )

        return stats

    @MethodHandlerWrapper(handler_class=StatisticPlotter)
    def flow_duration_curve(self, flow_name: str | None = None) -> xr.Dataset:
        """Generate load/flow duration curve for flows.

        Sorts flow values in descending order to show how often different
        load levels occur.

        Args:
            flow_name: Specific flow to analyze. If None, analyzes all flows.

        Returns:
            Dataset with sorted flow values

        Examples:
            >>> # Duration curve for specific flow
            >>> results['Boiler'].statistics.flow_duration_curve('Boiler(Q_th)').plot.line()
            >>>
            >>> # All flows duration curves
            >>> results['Boiler'].statistics.flow_duration_curve().plot.line()
        """
        # Get node balance
        ds = self._parent.node_balance(
            with_last_timestep=False, unit_type='flow_rate', drop_suffix=True, negate_inputs=False
        )

        # Filter to specific flow if requested
        if flow_name is not None:
            ds = ds[[var for var in ds.data_vars if flow_name in var]]

        # Sort each flow in descending order
        result = xr.Dataset()
        for var in ds.data_vars:
            sorted_values = np.sort(ds[var].values.flatten())[::-1]
            # Create new dimension for sorted index
            result[var] = xr.DataArray(
                sorted_values, dims=['duration_step'], coords={'duration_step': np.arange(len(sorted_values))}
            )

        return result

    @MethodHandlerWrapper(handler_class=StatisticPlotter)
    def effect_contributions(self, effects: str | list[str] | None = None) -> xr.Dataset:
        """Calculate how flows at this node contribute to effects (costs, CO2, etc.).

        Args:
            effects: Effect name(s) to analyze. If None, analyzes all effects.

        Returns:
            Dataset with effect contributions per flow

        Examples:
            >>> # Cost contributions from each flow
            >>> results['Boiler'].statistics.effect_contributions('costs').plot.bar()
            >>>
            >>> # Multiple effects
            >>> results['Boiler'].statistics.effect_contributions(['costs', 'CO2']).plot.bar()
        """
        try:
            calc_results = self._parent._calculation_results

            # Get this node's flows
            flow_names = self._parent.flows

            # Filter effects if specified
            if effects is not None:
                if isinstance(effects, str):
                    effects = [effects]

            # Calculate contributions
            contributions = xr.Dataset()

            for flow_name in flow_names:
                # Get effect variables for this flow
                effect_vars = [var for var in calc_results.solution.data_vars if flow_name in var and '->' in var]

                for effect_var in effect_vars:
                    # Extract effect name from variable name (format: flow->effect)
                    parts = effect_var.split('->')
                    if len(parts) == 2:
                        effect_name = parts[1]

                        # Skip if filtering and not in list
                        if effects is not None and effect_name not in effects:
                            continue

                        # Get the effect values
                        effect_data = calc_results.solution[effect_var]

                        # Sum over time if it has time dimension
                        if 'time' in effect_data.dims:
                            total = float(effect_data.sum('time'))
                        else:
                            total = float(effect_data.sum())

                        contributions[f'{flow_name}_{effect_name}'] = xr.DataArray(total)

            if not contributions.data_vars:
                # No effect data found, return empty dataset with message
                contributions = xr.Dataset()
                contributions.attrs['note'] = 'No effect contributions found for this node'

            return contributions

        except Exception as e:
            # Return empty dataset with error info
            result = xr.Dataset()
            result.attrs['error'] = str(e)
            return result

    @MethodHandlerWrapper(handler_class=StatisticPlotter)
    def effect_summary(self, aggregate_time: bool = True) -> xr.Dataset:
        """Summarize total effects attributable to this node.

        Args:
            aggregate_time: If True, sum over time. If False, keep time series.

        Returns:
            Dataset with effect totals

        Examples:
            >>> # Total effects for this node
            >>> results['Boiler'].statistics.effect_summary().plot.bar()
            >>>
            >>> # Time series of effects
            >>> results['Boiler'].statistics.effect_summary(aggregate_time=False).plot.line()
        """
        try:
            calc_results = self._parent._calculation_results

            # Get this node's flows
            flow_names = self._parent.flows

            # Collect all effect variables for these flows
            effect_data = {}

            for flow_name in flow_names:
                effect_vars = [var for var in calc_results.solution.data_vars if flow_name in var and '->' in var]

                for effect_var in effect_vars:
                    # Extract effect name
                    parts = effect_var.split('->')
                    if len(parts) == 2:
                        effect_name = parts[1]

                        # Add to cumulative effect
                        if effect_name not in effect_data:
                            effect_data[effect_name] = calc_results.solution[effect_var]
                        else:
                            effect_data[effect_name] = effect_data[effect_name] + calc_results.solution[effect_var]

            # Create dataset
            result = xr.Dataset(effect_data)

            # Aggregate if requested
            if aggregate_time and 'time' in result.dims:
                result = result.sum('time')

            if not result.data_vars:
                result.attrs['note'] = 'No effects found for this node'

            return result

        except Exception as e:
            result = xr.Dataset()
            result.attrs['error'] = str(e)
            return result

    @MethodHandlerWrapper(handler_class=StatisticPlotter)
    def flow_comparison(self, metric: Literal['total', 'mean', 'max', 'peak'] = 'total') -> xr.Dataset:
        """Compare all flows at this node using a specific metric.

        Args:
            metric: Metric to compare - 'total' (sum), 'mean', 'max', or 'peak' (top 10 avg).

        Returns:
            Dataset with comparison values

        Examples:
            >>> # Compare total flow hours
            >>> results['Bus'].statistics.flow_comparison(metric='total').plot.bar()
            >>>
            >>> # Compare peak loads
            >>> results['Bus'].statistics.flow_comparison(metric='peak').plot.bar()
        """
        # Get node balance
        ds = self._parent.node_balance(
            with_last_timestep=False,
            unit_type='flow_hours' if metric == 'total' else 'flow_rate',
            drop_suffix=True,
            negate_inputs=False,
        )

        result = xr.Dataset()

        for var in ds.data_vars:
            if metric == 'total':
                value = float(ds[var].sum('time'))
            elif metric == 'mean':
                value = float(ds[var].mean('time'))
            elif metric == 'max':
                value = float(ds[var].max('time'))
            elif metric == 'peak':
                # Average of top 10 values
                abs_values = np.abs(ds[var].values.flatten())
                if abs_values.size >= 10:
                    top_10 = np.partition(abs_values, -10)[-10:]
                    value = float(np.mean(top_10))
                else:
                    value = float(np.mean(abs_values))
            else:
                raise ValueError(f'Invalid metric: {metric}')

            result[var] = xr.DataArray(value)

        return result

    def __repr__(self) -> str:
        """String representation of the accessor."""
        return f"NodeStatisticsAccessor(node='{self._parent.label}')"
