"""Statistics accessor for CalculationResults.

This module provides the StatisticsAccessor class which adds statistics
methods to CalculationResults objects. Each method is decorated
with @MethodHandlerWrapper to return StatisticPlotter objects that provide
both data access and plotting capabilities.

Start with minimal functionality - build up gradually as needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..plotting_accessor import MethodHandlerWrapper, StatisticPlotter

if TYPE_CHECKING:
    import xarray as xr

    from ..results import CalculationResults


class StatisticsAccessor:
    """Accessor for calculating statistics on optimization results.

    Provides minimal statistics methods that return StatisticPlotter objects
    for easy visualization and data access. Start simple - add more methods
    as needed.

    Args:
        parent: The parent CalculationResults object containing optimization results

    Examples:
        >>> results = CalculationResults.from_file('results', 'optimization')
        >>>
        >>> # Get raw data
        >>> data = results.statistics.flow_summary().data
        >>>
        >>> # Create visualization
        >>> fig = results.statistics.flow_summary().plot.bar()
        >>> fig.show()
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
            >>> results.statistics.flow_summary(components=['Boiler_01'], aggregate_time=False).plot.line()
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

    def __repr__(self) -> str:
        """String representation of the accessor."""
        return f"StatisticsAccessor(parent='{self._parent.name}')"
