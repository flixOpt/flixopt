"""Interactive Plotly-based plotting for xarray statistics.

This module provides the InteractivePlotter class which implements actual
visualization methods using Plotly for creating interactive charts from
xarray datasets.

This integrates with flixopt's existing plotting infrastructure to provide
faceting, animation, and advanced color processing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import plotly.express as px

from .. import plotting

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd
    import plotly.graph_objects as go
    import xarray as xr


class InteractivePlotter:
    """Plotly-based plotting interface for xarray statistics.

    This class integrates with flixopt's plotting.with_plotly() infrastructure
    to provide interactive visualizations with advanced features including:
    - Multi-dimensional faceting (subplots)
    - Animation over dimensions
    - Sophisticated color processing
    - Automatic handling of xarray datasets

    The plotter integrates with the parent CalculationResults object to access
    color configurations and other settings, ensuring consistent styling across
    all visualizations.

    Args:
        data_getter: Function that returns the xarray.Dataset to plot
        method_name: Name of the statistics method (for default titles)
        parent: Parent CalculationResults object (for accessing colors, config, etc.)

    Examples:
        >>> # Via statistics accessor (recommended)
        >>> fig = results.statistics.energy_balance().plot.bar()
        >>> fig.show()
        >>>
        >>> # With faceting and animation
        >>> fig = results.statistics.flow_summary().plot.bar(
        ...     facet_by='scenario', animate_by='time', ylabel='Energy [MWh]'
        ... )
        >>> fig.show()
    """

    def __init__(self, data_getter: Callable[[], xr.Dataset], method_name: str, parent: Any):
        """Initialize the plotter with data source and configuration.

        Args:
            data_getter: Function returning the data to visualize
            method_name: Name of the statistics method
            parent: Parent object for configuration access
        """
        self._data_getter = data_getter
        self._method_name = method_name
        self._parent = parent

    def _get_dataset(self, data: xr.Dataset | None = None) -> xr.Dataset:
        """Get the dataset, fetching if necessary.

        Args:
            data: Data to use. If None, fetches from data_getter.

        Returns:
            xarray.Dataset ready for plotting
        """
        import xarray as xr

        if data is None:
            data = self._data_getter()

        # Ensure we have a Dataset
        if isinstance(data, xr.DataArray):
            data = data.to_dataset()
        elif not isinstance(data, xr.Dataset):
            raise TypeError(f'Expected xarray Dataset or DataArray, got {type(data)}')

        return data

    def _prepare_data(self, data: xr.Dataset | None = None) -> pd.DataFrame:
        """Convert xarray data to pandas DataFrame for Plotly.

        Deprecated: Use _get_dataset() and plotting.with_plotly() instead.

        Args:
            data: Data to convert. If None, fetches from data_getter.

        Returns:
            Data ready for Plotly visualization
        """
        import pandas as pd
        import xarray as xr

        if data is None:
            data = self._data_getter()

        # Convert to DataFrame
        if isinstance(data, xr.Dataset):
            # If multiple data variables, stack them
            df = data.to_dataframe().reset_index()
        elif isinstance(data, xr.DataArray):
            df = data.to_dataframe().reset_index()
        else:
            raise TypeError(f'Expected xarray Dataset or DataArray, got {type(data)}')

        return df

    def _get_colors(self, color_by: str | None = None) -> dict[str, str] | None:
        """Get color mapping from parent configuration.

        Args:
            color_by: Dimension to color by (e.g., 'carrier', 'component')

        Returns:
            Mapping from category to color hex code, or None if not available
        """
        if color_by is None:
            return None

        # Try to get colors from parent object if it has a _colors attribute
        if hasattr(self._parent, '_colors'):
            return self._parent._colors.get(color_by, None)

        return None

    def _make_title(self, title: str | None = None) -> str:
        """Create default title from method name if not provided.

        Args:
            title: User-provided title

        Returns:
            Final title to use
        """
        if title is not None:
            return title
        return self._method_name.replace('_', ' ').title()

    def bar(
        self,
        mode: Literal['stacked', 'grouped'] = 'stacked',
        colors: plotting.ColorType | None = None,
        title: str | None = None,
        ylabel: str = '',
        xlabel: str = '',
        facet_by: str | list[str] | None = None,
        animate_by: str | None = None,
        facet_cols: int | None = None,
        shared_yaxes: bool = True,
        shared_xaxes: bool = True,
        **kwargs,
    ) -> go.Figure:
        """Create an interactive bar chart with faceting and animation support.

        Uses flixopt's plotting.with_plotly() infrastructure for advanced features.

        Args:
            mode: Bar chart mode. Defaults to 'stacked'.
                - 'stacked': Stacked bars
                - 'grouped': Grouped bars side-by-side
            colors: Color specification. Can be:
                - None: Use parent's colors if available
                - str: Single color for all bars
                - dict: Mapping of categories to colors
                - Sequence: List of colors to cycle through
            title: Plot title. If None, derived from method name.
            ylabel: Y-axis label
            xlabel: X-axis label
            facet_by: Dimension(s) to create subplots for. Can be:
                - str: Single dimension (e.g., 'scenario')
                - list[str]: Multiple dimensions (e.g., ['scenario', 'carrier'])
            animate_by: Dimension to animate over (e.g., 'time')
            facet_cols: Number of columns for facet grid. Defaults to auto.
            shared_yaxes: Share y-axis range across facets. Defaults to True.
            shared_xaxes: Share x-axis range across facets. Defaults to True.
            **kwargs: Additional arguments passed to plotly.express

        Returns:
            Interactive bar chart figure

        Examples:
            >>> # Simple stacked bar chart
            >>> fig = plotter.bar()
            >>> fig.show()
            >>>
            >>> # Grouped bars with custom colors
            >>> fig = plotter.bar(mode='grouped', colors={'coal': 'black', 'gas': 'blue'})
            >>>
            >>> # Faceted by scenario with animation over time
            >>> fig = plotter.bar(facet_by='scenario', animate_by='time')
        """
        # Get dataset
        data = self._get_dataset()

        # Use parent colors if not specified
        if colors is None:
            colors = getattr(self._parent, 'colors', None)

        # Create title if not provided
        if title is None:
            title = self._make_title()

        # Map mode to plotting.with_plotly mode
        plotly_mode = 'stacked_bar' if mode == 'stacked' else 'grouped_bar'

        # Create figure using flixopt's plotting infrastructure
        fig = plotting.with_plotly(
            data=data,
            mode=plotly_mode,
            colors=colors,
            title=title,
            ylabel=ylabel,
            xlabel=xlabel,
            facet_by=facet_by,
            animate_by=animate_by,
            facet_cols=facet_cols,
            shared_yaxes=shared_yaxes,
            shared_xaxes=shared_xaxes,
            **kwargs,
        )

        return fig

    def line(
        self,
        colors: plotting.ColorType | None = None,
        title: str | None = None,
        ylabel: str = '',
        xlabel: str = '',
        facet_by: str | list[str] | None = None,
        animate_by: str | None = None,
        facet_cols: int | None = None,
        shared_yaxes: bool = True,
        shared_xaxes: bool = True,
        **kwargs,
    ) -> go.Figure:
        """Create an interactive line chart with faceting and animation support.

        Particularly useful for time series data and multi-dimensional analysis.

        Args:
            colors: Color specification. Can be:
                - None: Use parent's colors if available
                - str: Single color for all lines
                - dict: Mapping of categories to colors
                - Sequence: List of colors to cycle through
            title: Plot title. If None, derived from method name.
            ylabel: Y-axis label
            xlabel: X-axis label
            facet_by: Dimension(s) to create subplots for. Can be:
                - str: Single dimension (e.g., 'component')
                - list[str]: Multiple dimensions (e.g., ['component', 'carrier'])
            animate_by: Dimension to animate over (e.g., 'scenario')
            facet_cols: Number of columns for facet grid. Defaults to auto.
            shared_yaxes: Share y-axis range across facets. Defaults to True.
            shared_xaxes: Share x-axis range across facets. Defaults to True.
            **kwargs: Additional arguments passed to plotly.express

        Returns:
            Interactive line chart figure

        Examples:
            >>> # Simple time series line chart
            >>> fig = plotter.line()
            >>> fig.show()
            >>>
            >>> # Multi-faceted time series by component
            >>> fig = plotter.line(facet_by='component', ylabel='Power [MW]')
            >>>
            >>> # Animated time series
            >>> fig = plotter.line(animate_by='scenario')
        """
        # Get dataset
        data = self._get_dataset()

        # Use parent colors if not specified
        if colors is None:
            colors = getattr(self._parent, 'colors', None)

        # Create title if not provided
        if title is None:
            title = self._make_title()

        # Create figure using flixopt's plotting infrastructure
        fig = plotting.with_plotly(
            data=data,
            mode='line',
            colors=colors,
            title=title,
            ylabel=ylabel,
            xlabel=xlabel,
            facet_by=facet_by,
            animate_by=animate_by,
            facet_cols=facet_cols,
            shared_yaxes=shared_yaxes,
            shared_xaxes=shared_xaxes,
            **kwargs,
        )

        return fig

    def scatter(
        self,
        x: str | None = None,
        y: str | None = None,
        size: str | None = None,
        colors: plotting.ColorType | None = None,
        title: str | None = None,
        ylabel: str = '',
        xlabel: str = '',
        **kwargs,
    ) -> go.Figure:
        """Create an interactive scatter plot.

        Note: This method uses px.scatter directly as scatter plots are not
        supported by plotting.with_plotly(). Limited faceting/animation support.
        Requires multi-dimensional data.

        Args:
            x: Column/dimension name for x-axis. If None, auto-detected.
            y: Column/dimension name for y-axis. If None, auto-detected.
            size: Column/dimension name for point sizes (optional).
            colors: Color specification. Can be:
                - None: Use parent's colors if available
                - str: Single color for all points
                - dict: Mapping of categories to colors
            title: Plot title. If None, derived from method name.
            ylabel: Y-axis label
            xlabel: X-axis label
            **kwargs: Additional arguments passed to plotly.express.scatter
                (e.g., facet_row, facet_col, color)

        Returns:
            Interactive scatter plot figure

        Raises:
            ValueError: If data is 0-dimensional (no dimensions to plot)

        Examples:
            >>> # Simple scatter plot (requires dimensional data)
            >>> fig = plotter.scatter()
            >>> fig.show()
            >>>
            >>> # With explicit axes
            >>> fig = plotter.scatter(x='capacity', y='cost', size='efficiency')
        """
        # Get dataset and check dimensions
        data = self._get_dataset()

        # Check if data has dimensions
        if not data.dims:
            raise ValueError(
                'Scatter plot requires multi-dimensional data. '
                'The current dataset has no dimensions. '
                'Consider using aggregate_time=False or aggregate_scenarios=False '
                'in the statistics method to preserve dimensions.'
            )

        # Convert to DataFrame for scatter plot
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            df = data
        else:
            df = data.to_dataframe().reset_index()

        # Auto-detect axes if not provided
        if x is None:
            x = df.columns[0]
        if y is None:
            numeric_cols = df.select_dtypes(include=['number']).columns
            y = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[1]

        # Process colors
        color_discrete_map = None
        if colors is None:
            colors = getattr(self._parent, 'colors', None)
        if isinstance(colors, dict):
            color_discrete_map = colors
            colors = None  # px.scatter expects color_discrete_map separately

        # Create title if not provided
        if title is None:
            title = self._make_title()

        # Prepare labels
        labels = {}
        if xlabel:
            labels[x] = xlabel
        if ylabel:
            labels[y] = ylabel

        # Create figure
        fig = px.scatter(
            df,
            x=x,
            y=y,
            size=size,
            color=kwargs.pop('color', None),
            color_discrete_map=color_discrete_map,
            title=title,
            labels=labels if labels else None,
            **kwargs,
        )

        return fig

    def area(
        self,
        colors: plotting.ColorType | None = None,
        title: str | None = None,
        ylabel: str = '',
        xlabel: str = '',
        facet_by: str | list[str] | None = None,
        animate_by: str | None = None,
        facet_cols: int | None = None,
        shared_yaxes: bool = True,
        shared_xaxes: bool = True,
        **kwargs,
    ) -> go.Figure:
        """Create a stacked area chart with faceting and animation support.

        Useful for showing generation dispatch, component contributions over time,
        or energy flows across different dimensions.

        Args:
            colors: Color specification. Can be:
                - None: Use parent's colors if available
                - str: Single color for all areas
                - dict: Mapping of categories to colors
                - Sequence: List of colors to cycle through
            title: Plot title. If None, derived from method name.
            ylabel: Y-axis label
            xlabel: X-axis label
            facet_by: Dimension(s) to create subplots for. Can be:
                - str: Single dimension (e.g., 'scenario')
                - list[str]: Multiple dimensions (e.g., ['scenario', 'carrier'])
            animate_by: Dimension to animate over (e.g., 'period')
            facet_cols: Number of columns for facet grid. Defaults to auto.
            shared_yaxes: Share y-axis range across facets. Defaults to True.
            shared_xaxes: Share x-axis range across facets. Defaults to True.
            **kwargs: Additional arguments passed to plotly.express

        Returns:
            Interactive stacked area chart figure

        Examples:
            >>> # Simple stacked area chart (e.g., dispatch over time)
            >>> fig = plotter.area()
            >>> fig.show()
            >>>
            >>> # Faceted area chart by scenario
            >>> fig = plotter.area(facet_by='scenario', ylabel='Energy [MWh]')
            >>>
            >>> # Animated area chart with custom colors
            >>> fig = plotter.area(animate_by='period', colors={'coal': 'black', 'gas': 'blue', 'wind': 'green'})
        """
        # Get dataset
        data = self._get_dataset()

        # Use parent colors if not specified
        if colors is None:
            colors = getattr(self._parent, 'colors', None)

        # Create title if not provided
        if title is None:
            title = self._make_title()

        # Create figure using flixopt's plotting infrastructure
        fig = plotting.with_plotly(
            data=data,
            mode='area',
            colors=colors,
            title=title,
            ylabel=ylabel,
            xlabel=xlabel,
            facet_by=facet_by,
            animate_by=animate_by,
            facet_cols=facet_cols,
            shared_yaxes=shared_yaxes,
            shared_xaxes=shared_xaxes,
            **kwargs,
        )

        return fig

    def __repr__(self) -> str:
        """String representation of the plotter."""
        return f"InteractivePlotter(method='{self._method_name}')"
