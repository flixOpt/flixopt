"""Plot accessor classes for results objects.

This module provides .plot accessor classes that integrate with CalculationResults,
ComponentResults, and BusResults. These accessors provide convenient methods that
return plotter objects for visualization.

Architecture:
- ComponentPlotAccessor: For ComponentResults.plot
- BusPlotAccessor: For BusResults.plot
- CalculationResultsPlotAccessor: For CalculationResults.plot
- SegmentedCalculationResultsPlotAccessor: For SegmentedCalculationResults.plot

Each accessor provides methods that prepare data and return appropriate plotter instances.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any, Literal

from .results_plotters import ChargeStatePlotter, HeatmapPlotter, NodeBalancePlotter, PieChartPlotter

if TYPE_CHECKING:
    import xarray as xr

    from ..results import BusResults, CalculationResults, ComponentResults, SegmentedCalculationResults


class ComponentPlotAccessor:
    """Plot accessor for ComponentResults.

    Provides plotting methods via the .plot property on ComponentResults objects.

    Examples:
        >>> # Access via component results
        >>> plotter = results['Boiler'].plot.node_balance()
        >>> fig = plotter.bar()
        >>>
        >>> # Storage components also have charge_state
        >>> plotter = results['Storage'].plot.charge_state()
        >>> fig = plotter.area()
    """

    def __init__(self, component_results: ComponentResults):
        """Initialize accessor with component results object.

        Args:
            component_results: Parent ComponentResults object
        """
        self._component = component_results

    def node_balance(
        self,
        unit_type: Literal['flow_rate', 'flow_hours'] = 'flow_rate',
        drop_suffix: bool = True,
        select: dict[str, Any] | None = None,
    ) -> NodeBalancePlotter:
        """Get node balance plotter for this component.

        Args:
            unit_type: Whether to plot 'flow_rate' or 'flow_hours'
            drop_suffix: Whether to drop variable name suffixes
            select: Optional data selection dict (applied before plotting)

        Returns:
            NodeBalancePlotter instance for creating visualizations

        Examples:
            >>> plotter = results['Boiler'].plot.node_balance()
            >>> fig = plotter.bar()
            >>> fig = plotter.area(facet_by='scenario')
            >>> fig = plotter.line(animate_by='period')
        """
        # Import here to avoid circular dependency
        from ..results import _apply_selection_to_data

        # Get node balance data
        ds = self._component.node_balance(with_last_timestep=False, unit_type=unit_type, drop_suffix=drop_suffix)

        # Apply selection if provided
        if select is not None:
            ds, _ = _apply_selection_to_data(ds, select=select, drop=True)

        # Create and return plotter
        return NodeBalancePlotter(
            data=ds,
            parent=self._component,
            name=self._component.label,
            folder=self._component._calculation_results.folder,
            unit_type=unit_type,
        )

    def node_balance_pie(
        self,
        select: dict[str, Any] | None = None,
        threshold: float | None = 1e-5,
    ) -> PieChartPlotter:
        """Get pie chart plotter for node balance flow hours.

        Args:
            select: Optional data selection dict (applied before plotting)
            threshold: Threshold for dropping small values

        Returns:
            PieChartPlotter instance for creating visualizations

        Examples:
            >>> plotter = results['Bus'].plot.node_balance_pie()
            >>> fig = plotter.pie()
            >>> fig = plotter.donut(hole=0.4)
        """
        # Import here to avoid circular dependency
        from ..results import _apply_selection_to_data, sanitize_dataset

        # Get input and output flow hours
        inputs = sanitize_dataset(
            ds=self._component.solution[self._component.inputs]
            * self._component._calculation_results.hours_per_timestep,
            threshold=threshold,
            drop_small_vars=True,
            zero_small_values=True,
            drop_suffix='|',
        )
        outputs = sanitize_dataset(
            ds=self._component.solution[self._component.outputs]
            * self._component._calculation_results.hours_per_timestep,
            threshold=threshold,
            drop_small_vars=True,
            zero_small_values=True,
            drop_suffix='|',
        )

        # Apply selection if provided
        if select is not None:
            inputs, _ = _apply_selection_to_data(inputs, select=select, drop=True)
            outputs, _ = _apply_selection_to_data(outputs, select=select, drop=True)

        # Sum over time
        inputs = inputs.sum('time')
        outputs = outputs.sum('time')

        # Auto-select first value for any remaining dimensions
        for dim in set(list(inputs.dims) + list(outputs.dims)):
            if dim != 'time':
                if dim in inputs.coords:
                    inputs = inputs.sel({dim: inputs.coords[dim].values[0]})
                if dim in outputs.coords:
                    outputs = outputs.sel({dim: outputs.coords[dim].values[0]})

        # Create and return plotter
        return PieChartPlotter(
            data_left=inputs,
            data_right=outputs,
            parent=self._component,
            name=self._component.label,
            folder=self._component._calculation_results.folder,
        )

    def charge_state(
        self,
        select: dict[str, Any] | None = None,
    ) -> ChargeStatePlotter:
        """Get charge state plotter for this storage component.

        Args:
            select: Optional data selection dict (applied before plotting)

        Returns:
            ChargeStatePlotter instance for creating visualizations

        Raises:
            ValueError: If component is not a storage

        Examples:
            >>> plotter = results['Storage'].plot.charge_state()
            >>> fig = plotter.area()
            >>> fig = plotter.overlay(overlay_color='red')
        """
        if not self._component.is_storage:
            raise ValueError(f'Cant plot charge_state. "{self._component.label}" is not a storage')

        # Import here to avoid circular dependency
        from ..results import _apply_selection_to_data

        # Get node balance and charge state
        ds = self._component.node_balance(with_last_timestep=True).fillna(0)
        charge_state_da = self._component.charge_state

        # Apply selection if provided
        if select is not None:
            ds, _ = _apply_selection_to_data(ds, select=select, drop=True)
            charge_state_da, _ = _apply_selection_to_data(charge_state_da, select=select, drop=True)

        # Create and return plotter
        return ChargeStatePlotter(
            flow_data=ds,
            charge_state_data=charge_state_da,
            parent=self._component,
            name=self._component.label,
            folder=self._component._calculation_results.folder,
            charge_state_var_name=self._component._charge_state,
        )


class BusPlotAccessor:
    """Plot accessor for BusResults.

    Provides plotting methods via the .plot property on BusResults objects.
    BusResults have the same plotting capabilities as ComponentResults.

    Examples:
        >>> # Access via bus results
        >>> plotter = results['ElectricityBus'].plot.node_balance()
        >>> fig = plotter.bar()
    """

    def __init__(self, bus_results: BusResults):
        """Initialize accessor with bus results object.

        Args:
            bus_results: Parent BusResults object
        """
        self._bus = bus_results
        # Reuse ComponentPlotAccessor implementation since BusResults inherits from ComponentResults
        self._component_accessor = ComponentPlotAccessor(bus_results)

    def node_balance(
        self,
        unit_type: Literal['flow_rate', 'flow_hours'] = 'flow_rate',
        drop_suffix: bool = True,
        select: dict[str, Any] | None = None,
    ) -> NodeBalancePlotter:
        """Get node balance plotter for this bus.

        Args:
            unit_type: Whether to plot 'flow_rate' or 'flow_hours'
            drop_suffix: Whether to drop variable name suffixes
            select: Optional data selection dict (applied before plotting)

        Returns:
            NodeBalancePlotter instance for creating visualizations

        Examples:
            >>> plotter = results['ElectricityBus'].plot.node_balance()
            >>> fig = plotter.bar()
            >>> fig = plotter.area(facet_by='scenario')
        """
        return self._component_accessor.node_balance(unit_type=unit_type, drop_suffix=drop_suffix, select=select)

    def node_balance_pie(
        self,
        select: dict[str, Any] | None = None,
        threshold: float | None = 1e-5,
    ) -> PieChartPlotter:
        """Get pie chart plotter for bus flow hours.

        Args:
            select: Optional data selection dict (applied before plotting)
            threshold: Threshold for dropping small values

        Returns:
            PieChartPlotter instance for creating visualizations

        Examples:
            >>> plotter = results['ElectricityBus'].plot.node_balance_pie()
            >>> fig = plotter.pie()
        """
        return self._component_accessor.node_balance_pie(select=select, threshold=threshold)


class CalculationResultsPlotAccessor:
    """Plot accessor for CalculationResults.

    Provides plotting methods via the .plot property on CalculationResults objects.

    Examples:
        >>> # Access via calculation results
        >>> plotter = results.plot.heatmap('Boiler(Gas)|flow_rate')
        >>> fig = plotter.heatmap(reshape_time=('D', 'h'))
    """

    def __init__(self, calc_results: CalculationResults):
        """Initialize accessor with calculation results object.

        Args:
            calc_results: Parent CalculationResults object
        """
        self._results = calc_results

    def heatmap(
        self,
        variable_name: str | list[str],
        select: dict[str, Any] | None = None,
    ) -> HeatmapPlotter:
        """Get heatmap plotter for variable(s).

        Args:
            variable_name: Variable name or list of variable names to plot
            select: Optional data selection dict (applied before plotting)

        Returns:
            HeatmapPlotter instance for creating visualizations

        Examples:
            >>> plotter = results.plot.heatmap('Boiler(Gas)|flow_rate')
            >>> fig = plotter.heatmap(reshape_time=('D', 'h'))
            >>>
            >>> # Multiple variables
            >>> plotter = results.plot.heatmap(['Var1', 'Var2', 'Var3'])
            >>> fig = plotter.heatmap(facet_by='variable')
        """
        # Import here to avoid circular dependency
        from ..results import _apply_selection_to_data

        # Get data
        data = self._results.solution[variable_name]

        # Apply selection if provided
        if select is not None:
            data, _ = _apply_selection_to_data(data, select=select, drop=True)

        # Get name for title
        name = variable_name if isinstance(variable_name, str) else f'{len(variable_name)} variables'

        # Create and return plotter
        return HeatmapPlotter(
            data=data,
            parent=self._results,
            name=name,
            folder=self._results.folder,
        )


class SegmentedCalculationResultsPlotAccessor:
    """Plot accessor for SegmentedCalculationResults.

    Provides plotting methods via the .plot property on SegmentedCalculationResults objects.

    Examples:
        >>> # Access via segmented calculation results
        >>> plotter = segmented_results.plot.heatmap('Variable')
        >>> fig = plotter.heatmap()
    """

    def __init__(self, segmented_results: SegmentedCalculationResults):
        """Initialize accessor with segmented calculation results object.

        Args:
            segmented_results: Parent SegmentedCalculationResults object
        """
        self._results = segmented_results

    def heatmap(
        self,
        variable_name: str,
    ) -> HeatmapPlotter:
        """Get heatmap plotter for variable across segments.

        Args:
            variable_name: Variable name to plot

        Returns:
            HeatmapPlotter instance for creating visualizations

        Examples:
            >>> plotter = segmented_results.plot.heatmap('Variable')
            >>> fig = plotter.heatmap(reshape_time=('D', 'h'))
        """
        # Get solution without overlap
        data = self._results.solution_without_overlap(variable_name)

        # Create and return plotter
        return HeatmapPlotter(
            data=data,
            parent=self._results,
            name=variable_name,
            folder=self._results.folder,
        )
