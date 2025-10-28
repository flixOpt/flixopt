"""Statistics accessor for individual nodes (ComponentResults/BusResults).

This module provides the NodeStatisticsAccessor class which adds statistics
methods to individual component and bus results. Each method is decorated
with @MethodHandlerWrapper to return StatisticPlotter objects that provide
both data access and plotting capabilities.

Start with minimal functionality - build up gradually as needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..plotting_accessor import MethodHandlerWrapper, StatisticPlotter

if TYPE_CHECKING:
    import xarray as xr

    from ..results import BusResults, ComponentResults


class NodeStatisticsAccessor:
    """Accessor for calculating statistics on individual node results.

    Provides minimal statistics methods that return StatisticPlotter objects
    for easy visualization. Start simple - add more methods as needed.

    Args:
        parent: The parent ComponentResults or BusResults object

    Examples:
        >>> results = CalculationResults.from_file('results', 'optimization')
        >>>
        >>> # Get flow hours for a component
        >>> data = results['Boiler'].statistics.flow_hours().data
        >>>
        >>> # Create visualization
        >>> fig = results['Boiler'].statistics.flow_hours().plot.bar()
        >>> fig.show()
    """

    def __init__(self, parent: ComponentResults | BusResults):
        """Initialize the accessor with parent node results.

        Args:
            parent: The parent ComponentResults or BusResults object
        """
        self._parent = parent

    @MethodHandlerWrapper(handler_class=StatisticPlotter)
    def flow_hours(self) -> xr.Dataset:
        """Calculate total flow hours for each flow at this node.

        Replicates original flow_hours property but with plotting capabilities.

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

        # Add better defaults for plotting
        result.attrs['title'] = f'Total Flow Hours - {self._parent.label}'
        result.attrs['ylabel'] = 'Flow Hours'

        return result

    def __repr__(self) -> str:
        """String representation of the accessor."""
        return f"NodeStatisticsAccessor(node='{self._parent.label}')"
