"""Interactive Plotly-based plotting for xarray statistics.

This module provides plotter classes for creating interactive visualizations.
Uses a base class with common functionality and specialized subclasses for
domain-specific plot types, following the DRY (Don't Repeat Yourself) principle.

Architecture:
- InteractivePlotter: Base class with common plotting logic
- Specialized plotters: Inherit and add domain-specific methods
- Automatic plotter selection based on statistic method name
- DataTransformer integration for clean data handling
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import plotly.express as px

from .. import plotting
from .data_transformer import DataTransformer

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
        self._transformer = DataTransformer()

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

        Deprecated: Use _to_tidy() or _to_wide() instead, which leverage DataTransformer.

        Args:
            data: Data to convert. If None, fetches from data_getter.

        Returns:
            Data ready for Plotly visualization
        """
        import pandas as pd
        import xarray as xr

        if data is None:
            data = self._data_getter()

        # Use DataTransformer for conversion
        return self._transformer.to_tidy_dataframe(data)

    def _to_tidy(self, data: xr.Dataset | xr.DataArray | None = None, value_name: str = 'value') -> pd.DataFrame:
        """Convert data to tidy/long format DataFrame.

        This is the recommended format for Plotly Express. Each row represents
        one observation, dimensions become columns.

        Args:
            data: Data to convert. If None, uses data_getter.
            value_name: Name for value column (for DataArrays)

        Returns:
            Tidy DataFrame ready for Plotly Express

        Examples:
            >>> df = self._to_tidy(value_name='generation')
            >>> fig = px.line(df, x='time', y='generation', color='generator')
        """
        if data is None:
            data = self._get_dataset()
        return self._transformer.to_tidy_dataframe(data, value_name=value_name)

    def _to_wide(
        self, data: xr.DataArray | None = None, index_dim: str | None = None, columns_dim: str | None = None
    ) -> pd.DataFrame:
        """Convert data to wide format DataFrame.

        Wide format has one dimension as index and another as columns.
        Useful for some chart types.

        Args:
            data: DataArray to convert. If None, uses data_getter (must return DataArray).
            index_dim: Dimension for index (usually 'time')
            columns_dim: Dimension for columns (usually category like 'generator')

        Returns:
            Wide DataFrame

        Examples:
            >>> df = self._to_wide(index_dim='time', columns_dim='generator')
            >>> # time as index, generators as columns
        """
        import xarray as xr

        if data is None:
            fetched = self._get_dataset()
            # If Dataset with single variable, extract it
            if isinstance(fetched, xr.Dataset) and len(fetched.data_vars) == 1:
                data = fetched[list(fetched.data_vars)[0]]
            else:
                data = fetched

        return self._transformer.to_wide_dataframe(data, index_dim=index_dim, columns_dim=columns_dim)

    def _aggregate(
        self,
        data: xr.DataArray | xr.Dataset | None = None,
        dim: str = 'time',
        method: Literal['sum', 'mean', 'max', 'min', 'std', 'median'] = 'sum',
    ) -> xr.DataArray | xr.Dataset:
        """Aggregate data along a dimension.

        Args:
            data: Data to aggregate. If None, uses data_getter.
            dim: Dimension to aggregate along
            method: Aggregation method

        Returns:
            Aggregated data

        Examples:
            >>> # Sum over time
            >>> total = self._aggregate(dim='time', method='sum')
        """
        if data is None:
            data = self._get_dataset()
        return self._transformer.aggregate_dimension(data, dim=dim, method=method)

    def _select(self, data: xr.DataArray | xr.Dataset | None = None, **selectors: Any) -> xr.DataArray | xr.Dataset:
        """Select subset of data using dimension selectors.

        Args:
            data: Data to select from. If None, uses data_getter.
            **selectors: Dimension selectors

        Returns:
            Selected subset

        Examples:
            >>> # Select specific generator
            >>> subset = self._select(generator='solar')
            >>> # Select time range
            >>> subset = self._select(time=slice(0, 10))
        """
        if data is None:
            data = self._get_dataset()
        return self._transformer.select_subset(data, **selectors)

    def _melt(
        self, data: xr.Dataset | None = None, var_name: str = 'variable', value_name: str = 'value'
    ) -> pd.DataFrame:
        """Convert Dataset to ultra-tidy format with variable names as column.

        Useful for plotting multiple variables together distinguished by color/facet.

        Args:
            data: Dataset to melt. If None, uses data_getter.
            var_name: Name for variable column
            value_name: Name for value column

        Returns:
            Melted DataFrame

        Examples:
            >>> df = self._melt()
            >>> fig = px.line(df, x='time', y='value', color='variable')
        """
        if data is None:
            data = self._get_dataset()
        return self._transformer.melt_dataset(data, var_name=var_name, value_name=value_name)

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

    def _setup_plot_params(
        self, colors: plotting.ColorType | None = None, title: str | None = None, **kwargs
    ) -> tuple[plotting.ColorType | None, str, dict]:
        """Setup common plot parameters with intelligent defaults.

        This is a DRY helper that handles color and title setup consistently.

        Args:
            colors: Color specification (None uses parent colors)
            title: Plot title (None auto-generates from method name)
            **kwargs: Additional parameters to pass through

        Returns:
            Tuple of (colors, title, kwargs)
        """
        if colors is None:
            colors = getattr(self._parent, 'colors', None)
        if title is None:
            title = self._make_title()
        return colors, title, kwargs

    def _combine_figures(
        self,
        base_fig: go.Figure,
        overlay_fig: go.Figure,
        overlay_styling: dict[str, Any] | None = None,
    ) -> go.Figure:
        """Combine two figures by adding overlay traces to base figure.

        This is a DRY helper that handles trace and animation frame merging,
        used for creating mixed plot types (e.g., area + line overlay).

        Args:
            base_fig: Base figure to add traces to
            overlay_fig: Figure whose traces will be added as overlay
            overlay_styling: Optional styling to apply to overlay traces.
                Can contain 'line', 'marker', etc. properties.

        Returns:
            Modified base figure with overlay traces added

        Examples:
            >>> # Add line overlay to area chart
            >>> fig = self._combine_figures(
            ...     area_fig, line_fig, overlay_styling={'line': {'width': 2, 'color': 'black'}}
            ... )
        """
        import plotly.graph_objects as go

        # Add overlay traces to base figure
        for trace in overlay_fig.data:
            # Apply custom styling if provided
            if overlay_styling:
                for attr, style in overlay_styling.items():
                    if hasattr(trace, attr):
                        # Update the attribute (e.g., trace.line)
                        current = getattr(trace, attr)
                        if isinstance(current, dict):
                            current.update(style)
                        else:
                            # Create new dict-like object with updates
                            for key, value in style.items():
                                setattr(current, key, value)

            base_fig.add_trace(trace)

        # Handle animation frames if they exist
        if hasattr(overlay_fig, 'frames') and overlay_fig.frames:
            if not hasattr(base_fig, 'frames') or not base_fig.frames:
                # Base fig has no frames, add them
                base_fig.frames = overlay_fig.frames
            else:
                # Both have frames, merge them
                for i, frame in enumerate(overlay_fig.frames):
                    if i < len(base_fig.frames):
                        # Apply styling to frame traces too
                        for trace in frame.data:
                            if overlay_styling:
                                for attr, style in overlay_styling.items():
                                    if hasattr(trace, attr):
                                        current = getattr(trace, attr)
                                        for key, value in style.items():
                                            setattr(current, key, value)

                            # Add trace to frame
                            base_fig.frames[i].data = base_fig.frames[i].data + (trace,)

        return base_fig

    def plot(
        self,
        mode: Literal['stacked_bar', 'grouped_bar', 'line', 'area'] = 'stacked_bar',
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
        """Create an interactive plot with the specified mode.

        This is the main generic plotting method that wraps plotting.with_plotly().
        Specialized plotters can use this as a building block.

        Args:
            mode: Plot type - 'stacked_bar', 'grouped_bar', 'line', or 'area'
            colors: Color specification (None uses parent colors)
            title: Plot title (None auto-generates)
            ylabel: Y-axis label
            xlabel: X-axis label
            facet_by: Dimension(s) for subplots
            animate_by: Dimension for animation
            facet_cols: Number of subplot columns
            shared_yaxes: Share y-axis across facets
            shared_xaxes: Share x-axis across facets
            **kwargs: Additional arguments for plotting.with_plotly()

        Returns:
            Interactive plotly figure

        Examples:
            >>> # Generic plotting
            >>> fig = plotter.plot(mode='area', ylabel='Energy [MWh]')
            >>>
            >>> # With faceting
            >>> fig = plotter.plot(mode='line', facet_by='scenario')
        """
        data = self._get_dataset()
        colors, title, kwargs = self._setup_plot_params(colors, title, **kwargs)

        return plotting.with_plotly(
            data=data,
            mode=mode,
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

    def bar(
        self,
        mode: Literal['stacked', 'grouped'] = 'stacked',
        **kwargs,
    ) -> go.Figure:
        """Create an interactive bar chart (convenience method).

        Args:
            mode: Bar chart mode - 'stacked' or 'grouped'. Defaults to 'stacked'.
            **kwargs: All arguments from plot() method

        Returns:
            Interactive bar chart figure

        Examples:
            >>> fig = plotter.bar()
            >>> fig = plotter.bar(mode='grouped', colors={'coal': 'black'})
            >>> fig = plotter.bar(facet_by='scenario', animate_by='time')
        """
        plotly_mode = 'stacked_bar' if mode == 'stacked' else 'grouped_bar'
        return self.plot(mode=plotly_mode, **kwargs)

    def line(self, **kwargs) -> go.Figure:
        """Create an interactive line chart (convenience method).

        Particularly useful for time series data.

        Args:
            **kwargs: All arguments from plot() method

        Returns:
            Interactive line chart figure

        Examples:
            >>> fig = plotter.line()
            >>> fig = plotter.line(facet_by='component', ylabel='Power [MW]')
            >>> fig = plotter.line(animate_by='scenario')
        """
        return self.plot(mode='line', **kwargs)

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

        # Convert to tidy DataFrame using DataTransformer
        df = self._to_tidy(data)

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

    def area(self, **kwargs) -> go.Figure:
        """Create a stacked area chart (convenience method).

        Useful for showing generation dispatch or energy flows over time.

        Args:
            **kwargs: All arguments from plot() method

        Returns:
            Interactive stacked area chart figure

        Examples:
            >>> fig = plotter.area()
            >>> fig = plotter.area(facet_by='scenario', ylabel='Energy [MWh]')
            >>> fig = plotter.area(animate_by='period', colors={'coal': 'black'})
        """
        return self.plot(mode='area', **kwargs)

    def __repr__(self) -> str:
        """String representation of the plotter."""
        return f"InteractivePlotter(method='{self._method_name}')"


# ============================================================================
# Specialized Plotter Classes
# ============================================================================
# These inherit from InteractivePlotter and add domain-specific methods
# following the DRY principle - all common logic stays in the base class.


class StorageStatePlotter(InteractivePlotter):
    """Specialized plotter for storage state statistics.

    Inherits all base plotting methods and adds storage-specific
    visualization methods like charge state overlays.

    Examples:
        >>> # Via statistics accessor
        >>> fig = results.statistics.storage_states().plot.charge_state_overlay()
        >>> fig.show()
    """

    def charge_state_overlay(
        self,
        mode: Literal['area', 'stacked_bar'] = 'area',
        overlay_color: str = 'black',
        overlay_width: float = 2.0,
        flow_vars: list[str] | None = None,
        charge_var: str = 'charge_state',
        **kwargs,
    ) -> go.Figure:
        """Plot flows with charge state as line overlay.

        Creates a mixed visualization:
        - Flow variables shown as area/stacked_bar
        - Charge state shown as line overlay

        This replicates the pattern from plot_charge_state() in results.py.

        Args:
            mode: Plot mode for flows - 'area' or 'stacked_bar'. Defaults to 'area'.
            overlay_color: Color for charge state line. Defaults to 'black'.
            overlay_width: Width of charge state line. Defaults to 2.0.
            flow_vars: List of flow variable names. If None, auto-detects
                (all vars except charge_var).
            charge_var: Name of charge state variable. Defaults to 'charge_state'.
            **kwargs: Additional arguments passed to plot() method
                (colors, title, ylabel, xlabel, facet_by, animate_by, etc.)

        Returns:
            Interactive figure with flows and charge state overlay

        Raises:
            ValueError: If charge_var not found in dataset

        Examples:
            >>> # Basic usage
            >>> fig = plotter.charge_state_overlay()
            >>>
            >>> # Custom styling
            >>> fig = plotter.charge_state_overlay(mode='stacked_bar', overlay_color='red', overlay_width=3.0)
            >>>
            >>> # With faceting and animation
            >>> fig = plotter.charge_state_overlay(facet_by='scenario', animate_by='period', ylabel='Energy [MWh]')
        """
        import xarray as xr

        data = self._get_dataset()

        # Validate charge state variable exists
        if charge_var not in data.data_vars:
            raise ValueError(
                f"charge_state_overlay requires '{charge_var}' variable in dataset. "
                f'Available variables: {list(data.data_vars)}'
            )

        # Auto-detect flow variables if not provided
        if flow_vars is None:
            flow_vars = [v for v in data.data_vars if v != charge_var]

        # Split data: flows vs charge_state
        flows_ds = data[flow_vars] if flow_vars else None
        charge_state_ds = data[[charge_var]]

        # Extract facet/animate parameters for both plots
        facet_by = kwargs.get('facet_by', None)
        animate_by = kwargs.get('animate_by', None)

        # Create base figure with flows (if any)
        if flows_ds is not None and len(flows_ds.data_vars) > 0:
            # Use the base plot() method (DRY!)
            base_fig = self.plot(mode=mode, **kwargs)
        else:
            # No flows, create empty figure
            import plotly.graph_objects as go

            base_fig = go.Figure()
            colors, title, _ = self._setup_plot_params(kwargs.get('colors'), kwargs.get('title'))
            base_fig.update_layout(title=title)

        # Create temporary plotter for charge state
        # (We need a separate plotter instance to plot just the charge state)
        charge_plotter = InteractivePlotter(
            data_getter=lambda: charge_state_ds,
            method_name=self._method_name,
            parent=self._parent,
        )

        # Create charge state figure (always as line)
        charge_fig = charge_plotter.plot(
            mode='line',
            colors={charge_var: overlay_color},
            title='',  # No title needed
            facet_by=facet_by,
            animate_by=animate_by,
            facet_cols=kwargs.get('facet_cols', None),
            shared_yaxes=kwargs.get('shared_yaxes', True),
            shared_xaxes=kwargs.get('shared_xaxes', True),
        )

        # Combine figures using DRY helper!
        overlay_styling = {'line': {'width': overlay_width, 'shape': 'linear', 'color': overlay_color}}

        return self._combine_figures(base_fig, charge_fig, overlay_styling)


# ============================================================================
# Plotter Selection Mapping
# ============================================================================
# Map statistic method names to specialized plotter classes.
# This enables automatic selection of the right plotter for each statistic.

PLOTTER_CLASS_MAP: dict[str, type[InteractivePlotter]] = {
    'storage_states': StorageStatePlotter,
    # Add more mappings as you create specialized plotters:
    # 'energy_balance': EnergyBalancePlotter,
    # 'flow_summary': FlowSummaryPlotter,
    # etc.
}


def get_plotter_class(method_name: str) -> type[InteractivePlotter]:
    """Get the appropriate plotter class for a statistic method.

    Args:
        method_name: Name of the statistics method

    Returns:
        Plotter class (specialized if mapped, otherwise base InteractivePlotter)

    Examples:
        >>> plotter_cls = get_plotter_class('storage_states')
        >>> # Returns StorageStatePlotter
        >>>
        >>> plotter_cls = get_plotter_class('energy_balance')
        >>> # Returns InteractivePlotter (default)
    """
    return PLOTTER_CLASS_MAP.get(method_name, InteractivePlotter)
