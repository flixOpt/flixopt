"""Statistics accessor for CalculationResults.

This module provides the StatisticsAccessor class which adds PyPSA-style
statistics methods to CalculationResults objects. Each method is decorated
with @MethodHandlerWrapper to return StatisticPlotter objects that provide
both data access and plotting capabilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import xarray as xr

from ..plotting_accessor import MethodHandlerWrapper, StatisticPlotter

if TYPE_CHECKING:
    from ..results import CalculationResults


class StatisticsAccessor:
    """Accessor for calculating statistics on optimization results.

    This class provides methods to calculate various statistics from
    CalculationResults, all of which return StatisticPlotter objects
    for easy visualization and data access.

    The accessor integrates seamlessly with CalculationResults:
    - Access via: results.statistics.method_name()
    - Each method returns StatisticPlotter with .plot interface
    - Call the plotter to get raw xarray data: plotter()
    - Create visualizations: plotter.plot.bar(), plotter.plot.line(), etc.

    Args:
        parent: The parent CalculationResults object containing optimization results

    Examples:
        >>> results = CalculationResults.from_file('results', 'optimization')
        >>>
        >>> # Get raw data
        >>> data = results.statistics.flow_summary()()
        >>>
        >>> # Create visualization
        >>> fig = results.statistics.flow_summary().plot.bar()
        >>> fig.show()
        >>>
        >>> # With filtering
        >>> fig = results.statistics.energy_balance(components=['Boiler_01', 'HeatPump_01']).plot.bar(
        ...     x='component', y='energy'
        ... )
    """

    def __init__(self, parent: CalculationResults):
        """Initialize the accessor with parent CalculationResults.

        Args:
            parent: The parent results object
        """
        self._parent = parent

    @MethodHandlerWrapper(handler_class=StatisticPlotter)
    def flow_summary(
        self, components: list[str] | None = None, aggregate_time: bool = True, aggregate_scenarios: bool = True
    ) -> xr.Dataset:
        """Calculate summary of flow rates across components.

        Provides aggregated flow rates for all or selected components,
        useful for understanding overall system operation.

        Args:
            components: Component labels to include. If None, includes all components.
            aggregate_time: If True, sum flow rates over time dimension. Defaults to True.
            aggregate_scenarios: If True, average over scenario dimension (if present). Defaults to True.

        Returns:
            Dataset with flow rates, with dimensions based on aggregation settings

        Examples:
            >>> # Total flows per component
            >>> results.statistics.flow_summary().plot.bar()
            >>>
            >>> # Time series of selected components
            >>> results.statistics.flow_summary(
            ...     components=['Boiler_01', 'HeatPump_01'], aggregate_time=False
            ... ).plot.line(x='time', y='flow_rate')
        """
        # Get flow rate variables from solution
        flow_vars = [var for var in self._parent.solution.data_vars if 'flow_rate' in var]

        if not flow_vars:
            raise ValueError('No flow_rate variables found in solution')

        # Filter by components if specified
        if components is not None:
            flow_vars = [var for var in flow_vars if any(comp in var for comp in components)]

        # Extract flow rates
        flows = self._parent.solution[flow_vars]

        # Aggregate over time if requested
        if aggregate_time and 'time' in flows.dims:
            flows = flows.sum(dim='time')

        # Aggregate over scenarios if requested
        if aggregate_scenarios and 'scenario' in flows.dims:
            flows = flows.mean(dim='scenario')

        return flows

    @MethodHandlerWrapper(handler_class=StatisticPlotter)
    def energy_balance(
        self, components: list[str] | None = None, aggregate_time: bool = True, aggregate_scenarios: bool = True
    ) -> xr.Dataset:
        """Calculate energy balance across components.

        Computes total energy (flow_rate * hours_per_timestep) for each component,
        providing insight into total energy flows in the system.

        Args:
            components: Component labels to include
            aggregate_time: If True, sum energy over time dimension. Defaults to True.
            aggregate_scenarios: If True, average over scenario dimension. Defaults to True.

        Returns:
            Dataset with energy values (flow_rate * time)

        Examples:
            >>> # Total energy per component
            >>> results.statistics.energy_balance().plot.bar()
            >>>
            >>> # Energy balance for specific components
            >>> results.statistics.energy_balance(components=['Generator_01', 'Storage_01']).plot.bar(
            ...     x='component', y='energy'
            ... )
        """
        # Get flow rate variables
        flow_vars = [var for var in self._parent.solution.data_vars if 'flow_rate' in var]

        if components is not None:
            flow_vars = [var for var in flow_vars if any(comp in var for comp in components)]

        # Calculate energy (flow_rate * hours)
        flows = self._parent.solution[flow_vars]
        energy = flows * self._parent.hours_per_timestep

        # Aggregate over time if requested
        if aggregate_time and 'time' in energy.dims:
            energy = energy.sum(dim='time')

        # Aggregate over scenarios if requested
        if aggregate_scenarios and 'scenario' in energy.dims:
            energy = energy.mean(dim='scenario')

        return energy

    @MethodHandlerWrapper(handler_class=StatisticPlotter)
    def storage_states(
        self, storages: list[str] | None = None, aggregate_scenarios: bool = True, normalize: bool = False
    ) -> xr.Dataset:
        """Get storage charge states over time.

        Extracts charge state time series for storage components,
        useful for understanding storage operation patterns.

        Args:
            storages: Storage component labels to include. If None, includes all storages.
            aggregate_scenarios: If True, average over scenario dimension. Defaults to True.
            normalize: If True, normalize charge states to [0, 1] range. Defaults to False.

        Returns:
            Dataset with charge state time series

        Examples:
            >>> # Charge state time series
            >>> results.statistics.storage_states().plot.line(x='time', y='charge_state')
            >>>
            >>> # Normalized for comparison
            >>> results.statistics.storage_states(normalize=True).plot.line()
        """
        # Get charge state variables
        charge_vars = [var for var in self._parent.solution.data_vars if 'charge_state' in var]

        if not charge_vars:
            raise ValueError('No charge_state variables found in solution (no storages?)')

        # Filter by storage labels if specified
        if storages is not None:
            charge_vars = [var for var in charge_vars if any(stor in var for stor in storages)]

        # Extract charge states
        states = self._parent.solution[charge_vars]

        # Normalize if requested
        if normalize:
            states = states / states.max()

        # Aggregate over scenarios if requested
        if aggregate_scenarios and 'scenario' in states.dims:
            states = states.mean(dim='scenario')

        return states

    @MethodHandlerWrapper(handler_class=StatisticPlotter)
    def component_effects(self, effect_mode: str = 'total', components: list[str] | None = None) -> xr.Dataset:
        """Calculate effects (costs, emissions, etc.) per component.

        Aggregates system effects by component, useful for understanding
        which components contribute most to costs, emissions, etc.

        Args:
            effect_mode: Which effect mode to use. One of:
                - 'total': Total effects across all time
                - 'temporal': Time-resolved effects
                - 'periodic': Effects per period
                Defaults to 'total'.
            components: Component labels to include

        Returns:
            Dataset with effects per component

        Examples:
            >>> # Total costs per component
            >>> results.statistics.component_effects().plot.bar(x='component', y='costs')
            >>>
            >>> # Temporal cost evolution
            >>> results.statistics.component_effects(effect_mode='temporal').plot.line(x='time', y='costs')
        """
        # Get effects per component from CalculationResults
        effects = self._parent.effects_per_component

        # Select the requested mode
        if effect_mode not in effects.data_vars:
            raise ValueError(f"Effect mode '{effect_mode}' not found. Available: {list(effects.data_vars)}")

        mode_data = effects[[effect_mode]]

        # Filter components if specified
        if components is not None and 'component' in mode_data.dims:
            mode_data = mode_data.sel(component=components)

        return mode_data

    @MethodHandlerWrapper(handler_class=StatisticPlotter)
    def objective_breakdown(self) -> xr.Dataset:
        """Get breakdown of objective function value.

        Provides insight into which components or effects contribute
        to the total objective value.

        Returns:
            Dataset with objective contributions

        Examples:
            >>> # Objective breakdown
            >>> results.statistics.objective_breakdown().plot.bar()
        """
        # Get the objective value and effects
        objective = self._parent.objective

        # Get all effect results
        effect_data = {}
        for effect_name, effect_results in self._parent.effects.items():
            # Get total effect value
            effect_vars = [var for var in effect_results.solution.data_vars]
            if effect_vars:
                effect_data[effect_name] = effect_results.solution[effect_vars].to_array().sum()

        # Create dataset
        breakdown = xr.Dataset(effect_data)
        breakdown.attrs['objective'] = objective

        return breakdown

    @MethodHandlerWrapper(handler_class=StatisticPlotter)
    def component_sizes(self, components: list[str] | None = None) -> xr.Dataset:
        """Get installed/optimized component sizes.

        Extracts the size variables (capacities) for components,
        useful for capacity planning analysis.

        Args:
            components: Component labels to include

        Returns:
            Dataset with component sizes

        Examples:
            >>> # Component capacities
            >>> results.statistics.component_sizes().plot.bar(x='component', y='size')
        """
        # Get size variables
        size_vars = [var for var in self._parent.solution.data_vars if 'size' in var]

        if not size_vars:
            raise ValueError('No size variables found in solution')

        # Filter by components if specified
        if components is not None:
            size_vars = [var for var in size_vars if any(comp in var for comp in components)]

        # Extract sizes
        sizes = self._parent.solution[size_vars]

        return sizes

    def __repr__(self) -> str:
        """String representation of the accessor."""
        return f"StatisticsAccessor(parent='{self._parent.name}')"
