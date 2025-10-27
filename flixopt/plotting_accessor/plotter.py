"""Statistic plotter wrapper providing plotting interface for statistics.

This module provides the StatisticPlotter class which acts as a wrapper around
statistics calculation methods, providing a clean plotting interface while
supporting lazy evaluation and result caching.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    import xarray as xr

    from .plotly_charts import InteractivePlotter


class StatisticPlotter:
    """Wrapper around statistics methods that provides plotting capabilities.

    This class is the bridge between statistics calculation and visualization.
    It stores a bound method (the statistics calculation) and provides a plotting
    interface through the `.plot` and `.iplot` properties. The statistics are only
    calculated when actually needed, enabling efficient lazy evaluation.

    The class maintains a clear separation between computation and visualization:
    - Call the plotter as a function to get raw xarray data: `plotter()`
    - Access `.plot` property to get visualization interface: `plotter.plot.bar()`

    Attributes:
        data: Property to access the calculated statistics as xarray.Dataset (recommended)
        plot: Plotly-based plotting interface providing bar(), line(), scatter(), area(), etc.
        iplot: Alias for plot property (for consistency with other libraries)

    Args:
        bound_method: The statistics calculation method to call when data is needed.
            This is a lambda that captures the original method call and arguments.
        parent_object: The parent CalculationResults object, providing access to configuration
            like colors, model settings, etc.
        method_name: Name of the statistics method (used for default plot titles and caching)

    Examples:
        >>> # Typically created automatically via @MethodHandlerWrapper
        >>> plotter = results.statistics.energy_balance()
        >>>
        >>> # Get raw data (recommended way)
        >>> data = plotter.data
        >>> print(data)
        >>>
        >>> # Alternative: call it (less intuitive)
        >>> data = plotter()
        >>>
        >>> # Create visualization
        >>> fig = plotter.plot.bar(x='component', y='energy')
        >>> fig.show()
        >>>
        >>> # Access plot via iplot (same as plot)
        >>> fig = plotter.iplot.line(x='time', y='power')
    """

    def __init__(self, bound_method: Callable, parent_object: Any, method_name: str):
        """Initialize the plotter with bound method and parent reference.

        Args:
            bound_method: Function that returns xarray.Dataset when called. This captures
                the statistics calculation with all its arguments.
            parent_object: Parent CalculationResults object for accessing configuration
            method_name: Name of the statistics method for labeling and caching
        """
        self._bound_method = bound_method
        self._parent = parent_object
        self._method_name = method_name
        self._cached_result = None

    def __call__(self, *args, **kwargs) -> xr.Dataset:
        """Execute the statistics method and return the raw data.

        Note: Using the `.data` property is more intuitive than calling `()`.

        This allows using the plotter as a function to get the underlying
        xarray.Dataset without plotting. Results are cached to avoid
        recomputation on subsequent calls.

        Returns:
            The calculated statistics as xarray Dataset

        Examples:
            >>> plotter = results.statistics.energy_balance()
            >>> # Recommended:
            >>> data = plotter.data
            >>> # Alternative (works but less intuitive):
            >>> data = plotter()
        """
        if self._cached_result is None:
            self._cached_result = self._bound_method(*args, **kwargs)
        return self._cached_result

    def _get_data(self) -> xr.Dataset:
        """Get the data, computing if necessary.

        Internal method used by plotting interface to access data.
        Ensures data is computed only once and cached.

        Returns:
            The calculated statistics
        """
        if self._cached_result is None:
            self._cached_result = self._bound_method()
        return self._cached_result

    @property
    def data(self) -> xr.Dataset:
        """Get the calculated statistics as xarray.Dataset.

        This is the recommended way to access the raw data instead of
        using the callable interface `()`.

        Returns:
            The calculated statistics

        Examples:
            >>> plotter = results.statistics.energy_balance()
            >>> data = plotter.data  # Cleaner than plotter()
            >>> print(data)
        """
        return self._get_data()

    @property
    def plot(self) -> InteractivePlotter:
        """Access Plotly-based interactive plotting methods.

        Returns a plotting interface object with methods for creating
        various types of interactive visualizations using Plotly.

        The plotter class is automatically selected based on the statistic method:
        - Generic methods get InteractivePlotter (bar, line, area, scatter, plot)
        - Specialized methods get domain-specific plotters with extra methods
          (e.g., storage_states gets StorageStatePlotter with charge_state_overlay)

        Returns:
            Plotter object providing bar(), line(), scatter(), area(), plot(), etc.
            May also provide specialized methods depending on the statistic type.

        Examples:
            >>> # Generic plotting methods (available for all statistics)
            >>> results.statistics.energy_balance().plot.bar()
            >>> results.statistics.flow_summary().plot.line()
            >>> results.statistics.component_effects().plot.area()
            >>>
            >>> # Specialized methods for specific statistics
            >>> results.statistics.storage_states().plot.charge_state_overlay()
        """
        from .plotly_charts import get_plotter_class

        # Select the appropriate plotter class based on method name
        plotter_class = get_plotter_class(self._method_name)

        return plotter_class(data_getter=self._get_data, method_name=self._method_name, parent=self._parent)

    @property
    def iplot(self) -> InteractivePlotter:
        """Alias for plot property (interactive plotting).

        Provides the same functionality as `.plot` for consistency with
        other libraries that use `iplot` naming convention.

        Returns:
            Same as `.plot` property

        Examples:
            >>> # These are equivalent:
            >>> results.statistics.energy_balance().plot.bar()
            >>> results.statistics.energy_balance().iplot.bar()
        """
        return self.plot

    def __repr__(self) -> str:
        """String representation showing method name and data status."""
        cached_status = 'cached' if self._cached_result is not None else 'not computed'
        return f"StatisticPlotter(method='{self._method_name}', data={cached_status})"
