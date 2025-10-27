"""Interactive Plotly-based plotting for xarray statistics.

This module provides the InteractivePlotter class which implements actual
visualization methods using Plotly for creating interactive charts from
xarray datasets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import plotly.express as px

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd
    import plotly.graph_objects as go
    import xarray as xr


class InteractivePlotter:
    """Plotly-based plotting interface for xarray statistics.

    This class implements various plot types using Plotly Express and Plotly
    Graph Objects. It automatically converts xarray datasets to pandas DataFrames
    and creates interactive visualizations with intelligent defaults for axes,
    colors, and styling.

    The plotter integrates with the parent CalculationResults object to access
    color configurations and other settings, ensuring consistent styling across
    all visualizations.

    Args:
        data_getter: Function that returns the xarray.Dataset to plot
        method_name: Name of the statistics method (for default titles)
        parent: Parent CalculationResults object (for accessing colors, config, etc.)

    Examples:
        >>> plotter = InteractivePlotter(
        ...     data_getter=lambda: results.solution, method_name='energy_balance', parent=results
        ... )
        >>> fig = plotter.bar(x='component', y='energy', color='carrier')
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

    def _prepare_data(self, data: xr.Dataset | None = None) -> pd.DataFrame:
        """Convert xarray data to pandas DataFrame for Plotly.

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
        x: str | None = None,
        y: str | None = None,
        color: str | None = None,
        facet_row: str | None = None,
        facet_col: str | None = None,
        title: str | None = None,
        labels: dict[str, str] | None = None,
        color_discrete_map: dict[str, str] | None = None,
        barmode: str = 'group',
        height: int = 500,
        width: int | None = None,
        **kwargs,
    ) -> go.Figure:
        """Create an interactive bar chart.

        Parameters
        ----------
        x : str, optional
            Column name for x-axis. If None, uses first column.
        y : str, optional
            Column name for y-axis. If None, uses first numeric column.
        color : str, optional
            Column name to color bars by
        facet_row : str, optional
            Column name for row facets
        facet_col : str, optional
            Column name for column facets
        title : str, optional
            Plot title. If None, derived from method name.
        labels : dict, optional
            Dictionary mapping column names to display labels
        color_discrete_map : dict, optional
            Dictionary mapping color values to color codes
        barmode : {'group', 'stack', 'overlay', 'relative'}
            How to display bars when color is specified
        height : int, default 500
            Figure height in pixels
        width : int, optional
            Figure width in pixels
        **kwargs
            Additional arguments passed to plotly.express.bar

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive bar chart
        """
        df = self._prepare_data()

        # Auto-detect x if not provided
        if x is None:
            x = df.columns[0]

        # Auto-detect y if not provided
        if y is None:
            numeric_cols = df.select_dtypes(include=['number']).columns
            y = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[1]

        # Get colors if not provided
        if color_discrete_map is None and color is not None:
            color_discrete_map = self._get_colors(color)

        # Create title
        title = self._make_title(title)

        # Create figure
        fig = px.bar(
            df,
            x=x,
            y=y,
            color=color,
            facet_row=facet_row,
            facet_col=facet_col,
            title=title,
            labels=labels,
            color_discrete_map=color_discrete_map,
            barmode=barmode,
            height=height,
            width=width,
            **kwargs,
        )

        return fig

    def line(
        self,
        x: str | None = None,
        y: str | None = None,
        color: str | None = None,
        facet_row: str | None = None,
        facet_col: str | None = None,
        title: str | None = None,
        labels: dict[str, str] | None = None,
        color_discrete_map: dict[str, str] | None = None,
        height: int = 500,
        width: int | None = None,
        **kwargs,
    ) -> go.Figure:
        """Create an interactive line chart.

        Particularly useful for time series data.

        Parameters
        ----------
        x : str, optional
            Column name for x-axis. If None, tries to find 'time' column, else uses first column.
        y : str, optional
            Column name for y-axis. If None, uses first numeric column.
        color : str, optional
            Column name to color lines by
        facet_row : str, optional
            Column name for row facets
        facet_col : str, optional
            Column name for column facets
        title : str, optional
            Plot title
        labels : dict, optional
            Dictionary mapping column names to display labels
        color_discrete_map : dict, optional
            Dictionary mapping color values to color codes
        height : int, default 500
            Figure height in pixels
        width : int, optional
            Figure width in pixels
        **kwargs
            Additional arguments passed to plotly.express.line

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive line chart
        """
        df = self._prepare_data()

        # Auto-detect x (prefer time columns)
        if x is None:
            time_cols = [col for col in df.columns if 'time' in col.lower()]
            x = time_cols[0] if time_cols else df.columns[0]

        # Auto-detect y
        if y is None:
            numeric_cols = df.select_dtypes(include=['number']).columns
            y = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[1]

        # Get colors
        if color_discrete_map is None and color is not None:
            color_discrete_map = self._get_colors(color)

        # Create title
        title = self._make_title(title)

        # Create figure
        fig = px.line(
            df,
            x=x,
            y=y,
            color=color,
            facet_row=facet_row,
            facet_col=facet_col,
            title=title,
            labels=labels,
            color_discrete_map=color_discrete_map,
            height=height,
            width=width,
            **kwargs,
        )

        return fig

    def scatter(
        self,
        x: str | None = None,
        y: str | None = None,
        size: str | None = None,
        color: str | None = None,
        facet_row: str | None = None,
        facet_col: str | None = None,
        title: str | None = None,
        labels: dict[str, str] | None = None,
        color_discrete_map: dict[str, str] | None = None,
        height: int = 500,
        width: int | None = None,
        **kwargs,
    ) -> go.Figure:
        """Create an interactive scatter plot.

        Parameters
        ----------
        x : str, optional
            Column name for x-axis
        y : str, optional
            Column name for y-axis
        size : str, optional
            Column name for point sizes
        color : str, optional
            Column name to color points by
        facet_row : str, optional
            Column name for row facets
        facet_col : str, optional
            Column name for column facets
        title : str, optional
            Plot title
        labels : dict, optional
            Dictionary mapping column names to display labels
        color_discrete_map : dict, optional
            Dictionary mapping color values to color codes
        height : int, default 500
            Figure height in pixels
        width : int, optional
            Figure width in pixels
        **kwargs
            Additional arguments passed to plotly.express.scatter

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive scatter plot
        """
        df = self._prepare_data()

        # Auto-detect axes
        if x is None:
            x = df.columns[0]
        if y is None:
            numeric_cols = df.select_dtypes(include=['number']).columns
            y = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[1]

        # Get colors
        if color_discrete_map is None and color is not None:
            color_discrete_map = self._get_colors(color)

        # Create title
        title = self._make_title(title)

        # Create figure
        fig = px.scatter(
            df,
            x=x,
            y=y,
            size=size,
            color=color,
            facet_row=facet_row,
            facet_col=facet_col,
            title=title,
            labels=labels,
            color_discrete_map=color_discrete_map,
            height=height,
            width=width,
            **kwargs,
        )

        return fig

    def area(
        self,
        x: str | None = None,
        y: str | None = None,
        color: str | None = None,
        title: str | None = None,
        labels: dict[str, str] | None = None,
        color_discrete_map: dict[str, str] | None = None,
        groupnorm: str | None = None,
        height: int = 500,
        width: int | None = None,
        **kwargs,
    ) -> go.Figure:
        """Create a stacked area chart.

        Useful for showing generation dispatch or component contributions over time.

        Parameters
        ----------
        x : str, optional
            Column for x-axis (usually time)
        y : str, optional
            Column for y-axis (values to stack)
        color : str, optional
            Column to color by (different areas)
        title : str, optional
            Plot title
        labels : dict, optional
            Dictionary mapping column names to display labels
        color_discrete_map : dict, optional
            Dictionary mapping color values to color codes
        groupnorm : str, optional
            If 'percent', normalize to 100%
        height : int, default 500
            Figure height in pixels
        width : int, optional
            Figure width in pixels
        **kwargs
            Additional arguments passed to plotly.express.area

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive area chart
        """
        df = self._prepare_data()

        # Auto-detect x (prefer time)
        if x is None:
            time_cols = [col for col in df.columns if 'time' in col.lower()]
            x = time_cols[0] if time_cols else df.columns[0]

        # Auto-detect y
        if y is None:
            numeric_cols = df.select_dtypes(include=['number']).columns
            y = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[1]

        # Get colors
        if color_discrete_map is None and color is not None:
            color_discrete_map = self._get_colors(color)

        # Create title
        title = self._make_title(title)

        # Create figure
        fig = px.area(
            df,
            x=x,
            y=y,
            color=color,
            title=title,
            labels=labels,
            color_discrete_map=color_discrete_map,
            groupnorm=groupnorm,
            height=height,
            width=width,
            **kwargs,
        )

        return fig

    def __repr__(self) -> str:
        """String representation of the plotter."""
        return f"InteractivePlotter(method='{self._method_name}')"
