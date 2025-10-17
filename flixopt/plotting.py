"""Comprehensive visualization toolkit for flixopt optimization results and data analysis.

This module provides a unified plotting interface supporting both Plotly (interactive)
and Matplotlib (static) backends for visualizing energy system optimization results.
It offers specialized plotting functions for time series, heatmaps, network diagrams,
and statistical analyses commonly needed in energy system modeling.

Key Features:
    **Dual Backend Support**: Seamless switching between Plotly and Matplotlib
    **Energy System Focus**: Specialized plots for power flows, storage states, emissions
    **Color Management**: Intelligent color processing and palette management
    **Export Capabilities**: High-quality export for reports and publications
    **Integration Ready**: Designed for use with CalculationResults and standalone analysis

Main Plot Types:
    - **Time Series**: Flow rates, power profiles, storage states over time
    - **Heatmaps**: High-resolution temporal data visualization with customizable aggregation
    - **Network Diagrams**: System topology with flow visualization
    - **Statistical Plots**: Distribution analysis, correlation studies, performance metrics
    - **Comparative Analysis**: Multi-scenario and sensitivity study visualizations

The module integrates seamlessly with flixopt's result classes while remaining
accessible for standalone data visualization tasks.
"""

from __future__ import annotations

import itertools
import logging
import os
import pathlib
from typing import TYPE_CHECKING, Any, Literal

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline
import xarray as xr
from plotly.exceptions import PlotlyError

if TYPE_CHECKING:
    import pyvis

logger = logging.getLogger('flixopt')

# Define the colors for the 'portland' colormap in matplotlib
_portland_colors = [
    [12 / 255, 51 / 255, 131 / 255],  # Dark blue
    [10 / 255, 136 / 255, 186 / 255],  # Light blue
    [242 / 255, 211 / 255, 56 / 255],  # Yellow
    [242 / 255, 143 / 255, 56 / 255],  # Orange
    [217 / 255, 30 / 255, 30 / 255],  # Red
]

# Check if the colormap already exists before registering it
if hasattr(plt, 'colormaps'):  # Matplotlib >= 3.7
    registry = plt.colormaps
    if 'portland' not in registry:
        registry.register(mcolors.LinearSegmentedColormap.from_list('portland', _portland_colors))
else:  # Matplotlib < 3.7
    if 'portland' not in [c for c in plt.colormaps()]:
        plt.register_cmap(name='portland', cmap=mcolors.LinearSegmentedColormap.from_list('portland', _portland_colors))


ColorType = str | list[str] | dict[str, str]
"""Flexible color specification type supporting multiple input formats for visualization.

Color specifications can take several forms to accommodate different use cases:

**Named Colormaps** (str):
    - Standard colormaps: 'viridis', 'plasma', 'cividis', 'tab10', 'Set1'
    - Energy-focused: 'portland' (custom flixopt colormap for energy systems)
    - Backend-specific maps available in Plotly and Matplotlib

**Color Lists** (list[str]):
    - Explicit color sequences: ['red', 'blue', 'green', 'orange']
    - HEX codes: ['#FF0000', '#0000FF', '#00FF00', '#FFA500']
    - Mixed formats: ['red', '#0000FF', 'green', 'orange']

**Label-to-Color Mapping** (dict[str, str]):
    - Explicit associations: {'Wind': 'skyblue', 'Solar': 'gold', 'Gas': 'brown'}
    - Ensures consistent colors across different plots and datasets
    - Ideal for energy system components with semantic meaning

Examples:
    ```python
    # Named colormap
    colors = 'viridis'  # Automatic color generation

    # Explicit color list
    colors = ['red', 'blue', 'green', '#FFD700']

    # Component-specific mapping
    colors = {
        'Wind_Turbine': 'skyblue',
        'Solar_Panel': 'gold',
        'Natural_Gas': 'brown',
        'Battery': 'green',
        'Electric_Load': 'darkred'
    }
    ```

Color Format Support:
    - **Named Colors**: 'red', 'blue', 'forestgreen', 'darkorange'
    - **HEX Codes**: '#FF0000', '#0000FF', '#228B22', '#FF8C00'
    - **RGB Tuples**: (255, 0, 0), (0, 0, 255) [Matplotlib only]
    - **RGBA**: 'rgba(255,0,0,0.8)' [Plotly only]

References:
    - HTML Color Names: https://htmlcolorcodes.com/color-names/
    - Matplotlib Colormaps: https://matplotlib.org/stable/tutorials/colors/colormaps.html
    - Plotly Built-in Colorscales: https://plotly.com/python/builtin-colorscales/
"""

PlottingEngine = Literal['plotly', 'matplotlib']
"""Identifier for the plotting engine to use."""


class DimensionHandler:
    """Handles dimension roles for faceting, animation, and filtering in multidimensional plots.

    This class manages how different dimensions of xarray data should be visualized:
    as facets (subplots), animation frames, axes, filters, or aggregations. It provides
    a flexible framework for plotting multidimensional energy system data.

    Key Features:
        **Faceting**: Create subplot grids for comparing across dimension values
        **Animation**: Generate animation frames cycling through dimension values (Plotly)
        **Flexible Roles**: Each dimension can be assigned a specific visualization role
        **Auto-detection**: Intelligent defaults for unspecified dimensions

    Dimension Roles:
        - 'facet': Create subplots in a grid layout (one per dimension value)
        - 'animate': Create animation frames (Plotly only, one frame per value)
        - 'axis': Plot dimension on x or y axis (typically time)
        - 'filter': Select specific value(s) from the dimension
        - 'aggregate': Reduce dimension via aggregation (mean, sum, etc.)

    Examples:
        Basic faceting:

        ```python
        # Facet by scenario (creates 4 subplots if 4 scenarios)
        handler = DimensionHandler(data, facet_by='scenario')
        grid_data, n_rows, n_cols = handler.prepare_facet_grid(facet_cols=2)

        # Facet by both scenario and period (2D grid)
        handler = DimensionHandler(data, facet_by=['scenario', 'period'])
        ```

        Animation:

        ```python
        # Animate over periods
        handler = DimensionHandler(data, animate_by='period')
        frames = handler.prepare_animation_frames()
        ```

        Combined faceting and animation:

        ```python
        # Facet by scenario, animate by period
        handler = DimensionHandler(data, facet_by='scenario', animate_by='period')
        # Each subplot shows a different scenario, all animate through periods
        ```

    Args:
        data: xarray DataArray or Dataset with named dimensions
        facet_by: Dimension(s) to create facets for (str or list of str)
        animate_by: Dimension to animate over (str, Plotly only)
        filter_dims: Dictionary of dimension filters {dim: value}
    """

    def __init__(
        self,
        data: xr.DataArray | xr.Dataset,
        facet_by: str | list[str] | None = None,
        animate_by: str | None = None,
        filter_dims: dict[str, Any] | None = None,
    ):
        """Initialize the dimension handler."""
        self.data = data
        self.facet_by = [facet_by] if isinstance(facet_by, str) else (facet_by or [])
        self.animate_by = animate_by
        self.filter_dims = filter_dims or {}

        # Validate
        self._validate_dimensions()

    def _validate_dimensions(self):
        """Validate that specified dimensions exist in the data."""
        data_dims = set(self.data.dims)

        # Check facet dimensions
        for dim in self.facet_by:
            if dim not in data_dims:
                raise ValueError(f"Facet dimension '{dim}' not found in data. Available: {data_dims}")

        # Check animation dimension
        if self.animate_by and self.animate_by not in data_dims:
            raise ValueError(f"Animation dimension '{self.animate_by}' not found in data. Available: {data_dims}")

        # Check that facet and animate don't overlap
        if self.animate_by and self.animate_by in self.facet_by:
            raise ValueError(f"Dimension '{self.animate_by}' cannot be both faceted and animated")

    def get_filtered_data(self) -> xr.DataArray | xr.Dataset:
        """Apply filter dimensions to the data."""
        if not self.filter_dims:
            return self.data
        return self.data.sel(self.filter_dims)

    def prepare_facet_grid(
        self, facet_cols: int = 3
    ) -> tuple[list[tuple[str, dict, xr.DataArray | xr.Dataset]], int, int]:
        """
        Prepare data slices for facet grid plotting.

        Args:
            facet_cols: Number of columns in the facet grid

        Returns:
            Tuple of:
            - List of (title, selector_dict, data_slice) tuples for each subplot
            - Number of rows in grid
            - Number of columns in grid
        """
        filtered_data = self.get_filtered_data()

        if not self.facet_by:
            return [('', {}, filtered_data)], 1, 1

        # Get all combinations of facet dimension values
        facet_coords = []
        facet_labels = []
        for dim in self.facet_by:
            coords = filtered_data.coords[dim].values
            facet_coords.append(coords)
            facet_labels.append(dim)

        combinations = list(itertools.product(*facet_coords))

        # Calculate grid dimensions
        n_plots = len(combinations)
        n_cols = min(facet_cols, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

        # Create data slices for each combination
        slices = []
        for combo in combinations:
            selector = {dim: val for dim, val in zip(facet_labels, combo, strict=False)}
            data_slice = filtered_data.sel(selector)

            # Create title
            if len(combo) == 1:
                title = f'{combo[0]}'
            else:
                title_parts = [f'{dim}={val}' for dim, val in zip(facet_labels, combo, strict=False)]
                title = ', '.join(title_parts)

            slices.append((title, selector, data_slice))

        return slices, n_rows, n_cols

    def prepare_animation_frames(self) -> list[tuple[Any, dict, xr.DataArray | xr.Dataset]]:
        """
        Prepare data for each animation frame.

        Returns:
            List of (frame_value, selector_dict, data_slice) tuples
        """
        filtered_data = self.get_filtered_data()

        if not self.animate_by:
            return [(None, {}, filtered_data)]

        frames = []
        for value in filtered_data.coords[self.animate_by].values:
            selector = {self.animate_by: value}
            data_slice = filtered_data.sel(selector)
            frames.append((value, selector, data_slice))

        return frames

    def get_facet_and_animation_combinations(
        self, facet_cols: int = 3
    ) -> tuple[list[tuple[str, dict, list[tuple[Any, xr.DataArray | xr.Dataset]]]], int, int]:
        """
        Prepare combined facet and animation data.

        Returns:
            Tuple of:
            - List of (subplot_title, facet_selector, [(frame_value, frame_data), ...]) for each subplot
            - Number of rows in grid
            - Number of columns in grid
        """
        # Get facet grid structure (this internally handles filtering)
        facet_slices, n_rows, n_cols = self.prepare_facet_grid(facet_cols)

        if not self.animate_by:
            # No animation, just return facet slices with single "frame"
            result = []
            for title, selector, data_slice in facet_slices:
                result.append((title, selector, [(None, data_slice)]))
            return result, n_rows, n_cols

        # Prepare animation frames for each facet
        result = []
        for title, facet_selector, facet_data in facet_slices:
            frames = []
            for value in facet_data.coords[self.animate_by].values:
                frame_data = facet_data.sel({self.animate_by: value})
                frames.append((value, frame_data))
            result.append((title, facet_selector, frames))

        return result, n_rows, n_cols


class ColorProcessor:
    """Intelligent color management system for consistent multi-backend visualization.

    This class provides unified color processing across Plotly and Matplotlib backends,
    ensuring consistent visual appearance regardless of the plotting engine used.
    It handles color palette generation, named colormap translation, and intelligent
    color cycling for complex datasets with many categories.

    Key Features:
        **Backend Agnostic**: Automatic color format conversion between engines
        **Palette Management**: Support for named colormaps, custom palettes, and color lists
        **Intelligent Cycling**: Smart color assignment for datasets with many categories
        **Fallback Handling**: Graceful degradation when requested colormaps are unavailable
        **Energy System Colors**: Built-in palettes optimized for energy system visualization

    Color Input Types:
        - **Named Colormaps**: 'viridis', 'plasma', 'portland', 'tab10', etc.
        - **Color Lists**: ['red', 'blue', 'green'] or ['#FF0000', '#0000FF', '#00FF00']
        - **Label Dictionaries**: {'Generator': 'red', 'Storage': 'blue', 'Load': 'green'}

    Examples:
        Basic color processing:

        ```python
        # Initialize for Plotly backend
        processor = ColorProcessor(engine='plotly', default_colormap='viridis')

        # Process different color specifications
        colors = processor.process_colors('plasma', ['Gen1', 'Gen2', 'Storage'])
        colors = processor.process_colors(['red', 'blue', 'green'], ['A', 'B', 'C'])
        colors = processor.process_colors({'Wind': 'skyblue', 'Solar': 'gold'}, ['Wind', 'Solar', 'Gas'])

        # Switch to Matplotlib
        processor = ColorProcessor(engine='matplotlib')
        mpl_colors = processor.process_colors('tab10', component_labels)
        ```

        Energy system visualization:

        ```python
        # Specialized energy system palette
        energy_colors = {
            'Natural_Gas': '#8B4513',  # Brown
            'Electricity': '#FFD700',  # Gold
            'Heat': '#FF4500',  # Red-orange
            'Cooling': '#87CEEB',  # Sky blue
            'Hydrogen': '#E6E6FA',  # Lavender
            'Battery': '#32CD32',  # Lime green
        }

        processor = ColorProcessor('plotly')
        flow_colors = processor.process_colors(energy_colors, flow_labels)
        ```

    Args:
        engine: Plotting backend ('plotly' or 'matplotlib'). Determines output color format.
        default_colormap: Fallback colormap when requested palettes are unavailable.
            Common options: 'viridis', 'plasma', 'tab10', 'portland'.

    """

    def __init__(self, engine: PlottingEngine = 'plotly', default_colormap: str = 'viridis'):
        """Initialize the color processor with specified backend and defaults."""
        if engine not in ['plotly', 'matplotlib']:
            raise TypeError(f'engine must be "plotly" or "matplotlib", but is {engine}')
        self.engine = engine
        self.default_colormap = default_colormap

    def _generate_colors_from_colormap(self, colormap_name: str, num_colors: int) -> list[Any]:
        """
        Generate colors from a named colormap.

        Args:
            colormap_name: Name of the colormap
            num_colors: Number of colors to generate

        Returns:
            list of colors in the format appropriate for the engine
        """
        if self.engine == 'plotly':
            try:
                colorscale = px.colors.get_colorscale(colormap_name)
            except PlotlyError as e:
                logger.error(f"Colorscale '{colormap_name}' not found in Plotly. Using {self.default_colormap}: {e}")
                colorscale = px.colors.get_colorscale(self.default_colormap)

            # Generate evenly spaced points
            color_points = [i / (num_colors - 1) for i in range(num_colors)] if num_colors > 1 else [0]
            return px.colors.sample_colorscale(colorscale, color_points)

        else:  # matplotlib
            try:
                cmap = plt.get_cmap(colormap_name, num_colors)
            except ValueError as e:
                logger.error(f"Colormap '{colormap_name}' not found in Matplotlib. Using {self.default_colormap}: {e}")
                cmap = plt.get_cmap(self.default_colormap, num_colors)

            return [cmap(i) for i in range(num_colors)]

    def _handle_color_list(self, colors: list[str], num_labels: int) -> list[str]:
        """
        Handle a list of colors, cycling if necessary.

        Args:
            colors: list of color strings
            num_labels: Number of labels that need colors

        Returns:
            list of colors matching the number of labels
        """
        if len(colors) == 0:
            logger.error(f'Empty color list provided. Using {self.default_colormap} instead.')
            return self._generate_colors_from_colormap(self.default_colormap, num_labels)

        if len(colors) < num_labels:
            logger.warning(
                f'Not enough colors provided ({len(colors)}) for all labels ({num_labels}). Colors will cycle.'
            )
            # Cycle through the colors
            color_iter = itertools.cycle(colors)
            return [next(color_iter) for _ in range(num_labels)]
        else:
            # Trim if necessary
            if len(colors) > num_labels:
                logger.warning(
                    f'More colors provided ({len(colors)}) than labels ({num_labels}). Extra colors will be ignored.'
                )
            return colors[:num_labels]

    def _handle_color_dict(self, colors: dict[str, str], labels: list[str]) -> list[str]:
        """
        Handle a dictionary mapping labels to colors.

        Args:
            colors: Dictionary mapping labels to colors
            labels: list of labels that need colors

        Returns:
            list of colors in the same order as labels
        """
        if len(colors) == 0:
            logger.warning(f'Empty color dictionary provided. Using {self.default_colormap} instead.')
            return self._generate_colors_from_colormap(self.default_colormap, len(labels))

        # Find missing labels
        missing_labels = sorted(set(labels) - set(colors.keys()))
        if missing_labels:
            logger.warning(
                f'Some labels have no color specified: {missing_labels}. Using {self.default_colormap} for these.'
            )

            # Generate colors for missing labels
            missing_colors = self._generate_colors_from_colormap(self.default_colormap, len(missing_labels))

            # Create a copy to avoid modifying the original
            colors_copy = colors.copy()
            for i, label in enumerate(missing_labels):
                colors_copy[label] = missing_colors[i]
        else:
            colors_copy = colors

        # Create color list in the same order as labels
        return [colors_copy[label] for label in labels]

    def process_colors(
        self,
        colors: ColorType,
        labels: list[str],
        return_mapping: bool = False,
    ) -> list[Any] | dict[str, Any]:
        """
        Process colors for the specified labels.

        Args:
            colors: Color specification (colormap name, list of colors, or label-to-color mapping)
            labels: list of data labels that need colors assigned
            return_mapping: If True, returns a dictionary mapping labels to colors;
                           if False, returns a list of colors in the same order as labels

        Returns:
            Either a list of colors or a dictionary mapping labels to colors
        """
        if len(labels) == 0:
            logger.error('No labels provided for color assignment.')
            return {} if return_mapping else []

        # Process based on type of colors input
        if isinstance(colors, str):
            color_list = self._generate_colors_from_colormap(colors, len(labels))
        elif isinstance(colors, list):
            color_list = self._handle_color_list(colors, len(labels))
        elif isinstance(colors, dict):
            color_list = self._handle_color_dict(colors, labels)
        else:
            logger.error(
                f'Unsupported color specification type: {type(colors)}. Using {self.default_colormap} instead.'
            )
            color_list = self._generate_colors_from_colormap(self.default_colormap, len(labels))

        # Return either a list or a mapping
        if return_mapping:
            return {label: color_list[i] for i, label in enumerate(labels)}
        else:
            return color_list


def with_plotly(
    data: pd.DataFrame,
    mode: Literal['stacked_bar', 'line', 'area', 'grouped_bar'] = 'stacked_bar',
    colors: ColorType = 'viridis',
    title: str = '',
    ylabel: str = '',
    xlabel: str = 'Time in h',
    fig: go.Figure | None = None,
) -> go.Figure:
    """
    Plot a DataFrame with Plotly, using either stacked bars or stepped lines.

    Args:
        data: A DataFrame containing the data to plot, where the index represents time (e.g., hours),
              and each column represents a separate data series.
        mode: The plotting mode. Use 'stacked_bar' for stacked bar charts, 'line' for stepped lines,
              or 'area' for stacked area charts.
        colors: Color specification, can be:
            - A string with a colorscale name (e.g., 'viridis', 'plasma')
            - A list of color strings (e.g., ['#ff0000', '#00ff00'])
            - A dictionary mapping column names to colors (e.g., {'Column1': '#ff0000'})
        title: The title of the plot.
        ylabel: The label for the y-axis.
        xlabel: The label for the x-axis.
        fig: A Plotly figure object to plot on. If not provided, a new figure will be created.

    Returns:
        A Plotly figure object containing the generated plot.
    """
    if mode not in ('stacked_bar', 'line', 'area', 'grouped_bar'):
        raise ValueError(f"'mode' must be one of {{'stacked_bar','line','area', 'grouped_bar'}}, got {mode!r}")
    if data.empty:
        return go.Figure()

    processed_colors = ColorProcessor(engine='plotly').process_colors(colors, list(data.columns))

    fig = fig if fig is not None else go.Figure()

    if mode == 'stacked_bar':
        for i, column in enumerate(data.columns):
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data[column],
                    name=column,
                    marker=dict(
                        color=processed_colors[i], line=dict(width=0, color='rgba(0,0,0,0)')
                    ),  # Transparent line with 0 width
                )
            )

        fig.update_layout(
            barmode='relative',
            bargap=0,  # No space between bars
            bargroupgap=0,  # No space between grouped bars
        )
    if mode == 'grouped_bar':
        for i, column in enumerate(data.columns):
            fig.add_trace(go.Bar(x=data.index, y=data[column], name=column, marker=dict(color=processed_colors[i])))

        fig.update_layout(
            barmode='group',
            bargap=0.2,  # No space between bars
            bargroupgap=0,  # space between grouped bars
        )
    elif mode == 'line':
        for i, column in enumerate(data.columns):
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[column],
                    mode='lines',
                    name=column,
                    line=dict(shape='hv', color=processed_colors[i]),
                )
            )
    elif mode == 'area':
        data = data.copy()
        data[(data > -1e-5) & (data < 1e-5)] = 0  # Preventing issues with plotting
        # Split columns into positive, negative, and mixed categories
        positive_columns = list(data.columns[(data >= 0).where(~np.isnan(data), True).all()])
        negative_columns = list(data.columns[(data <= 0).where(~np.isnan(data), True).all()])
        negative_columns = [column for column in negative_columns if column not in positive_columns]
        mixed_columns = list(set(data.columns) - set(positive_columns + negative_columns))

        if mixed_columns:
            logger.error(
                f'Data for plotting stacked lines contains columns with both positive and negative values:'
                f' {mixed_columns}. These can not be stacked, and are printed as simple lines'
            )

        # Get color mapping for all columns
        colors_stacked = {column: processed_colors[i] for i, column in enumerate(data.columns)}

        for column in positive_columns + negative_columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[column],
                    mode='lines',
                    name=column,
                    line=dict(shape='hv', color=colors_stacked[column]),
                    fill='tonexty',
                    stackgroup='pos' if column in positive_columns else 'neg',
                )
            )

        for column in mixed_columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[column],
                    mode='lines',
                    name=column,
                    line=dict(shape='hv', color=colors_stacked[column], dash='dash'),
                )
            )

    # Update layout for better aesthetics
    fig.update_layout(
        title=title,
        yaxis=dict(
            title=ylabel,
            showgrid=True,  # Enable grid lines on the y-axis
            gridcolor='lightgrey',  # Customize grid line color
            gridwidth=0.5,  # Customize grid line width
        ),
        xaxis=dict(
            title=xlabel,
            showgrid=True,  # Enable grid lines on the x-axis
            gridcolor='lightgrey',  # Customize grid line color
            gridwidth=0.5,  # Customize grid line width
        ),
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper background
        font=dict(size=14),  # Increase font size for better readability
    )

    return fig


def with_plotly_faceted(
    data: pd.DataFrame,
    facet_by: str | list[str] | None = None,
    animate_by: str | None = None,
    mode: Literal['stacked_bar', 'line', 'area', 'grouped_bar'] = 'stacked_bar',
    colors: ColorType = 'viridis',
    title: str = '',
    ylabel: str = '',
    xlabel: str = 'Time in h',
    facet_cols: int = 3,
    shared_yaxes: bool = True,
    shared_xaxes: bool = True,
    height_per_row: int | None = None,
    width: int | None = None,
) -> go.Figure:
    """
    Plot a DataFrame with Plotly using facets (subplots) and/or animation for multidimensional data.

    This function extends the basic plotting capabilities by supporting:
    - Faceting: Create subplots for different dimension values
    - Animation: Animate through dimension values (e.g., different periods)
    - Combined: Facet by one dimension while animating another

    Args:
        data: A DataFrame or xarray DataArray/Dataset to plot. If DataFrame, index dimensions
              beyond time should be part of a MultiIndex.
        facet_by: Dimension(s) to create facets for. Creates a subplot grid.
              Can be a single dimension name or list of dimensions.
        animate_by: Dimension to animate over. Only available for Plotly.
              Creates animation frames that cycle through dimension values.
        mode: The plotting mode. Use 'stacked_bar' for stacked bar charts, 'line' for stepped lines,
              'area' for stacked area charts, or 'grouped_bar' for grouped bar charts.
        colors: Color specification (colormap, list, or dict mapping labels to colors).
        title: The main title of the plot.
        ylabel: The label for the y-axis.
        xlabel: The label for the x-axis.
        facet_cols: Number of columns in the facet grid.
        shared_yaxes: Whether subplots share y-axes.
        shared_xaxes: Whether subplots share x-axes.
        height_per_row: Height in pixels for each row of subplots (default: None = auto-sized).
            When None, automatically calculates height to fill ~90% of typical browser viewport.
            Manually specify (e.g., 600) to override auto-sizing.
        width: Total width in pixels (default: None = auto-sized by Plotly).

    Returns:
        A Plotly figure object containing the faceted/animated plot.

    Examples:
        Facet by scenario:

        ```python
        # Create 4 subplots, one for each scenario
        fig = with_plotly_faceted(data, facet_by='scenario', facet_cols=2)
        ```

        Animate by period:

        ```python
        # Create animation cycling through periods
        fig = with_plotly_faceted(data, animate_by='period')
        ```

        Facet and animate:

        ```python
        # 4 subplots (scenarios), each animating through periods
        fig = with_plotly_faceted(data, facet_by='scenario', animate_by='period')
        ```
    """
    from plotly.subplots import make_subplots

    # Convert xarray to appropriate format if needed
    if isinstance(data, (xr.DataArray, xr.Dataset)):
        # Use DimensionHandler to process the data
        handler = DimensionHandler(data, facet_by=facet_by, animate_by=animate_by)

        # Get facet and animation combinations
        subplot_data, n_rows, n_cols = handler.get_facet_and_animation_combinations(facet_cols)

        # Create subplots
        subplot_titles = [title for title, _, _ in subplot_data]
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=subplot_titles if len(subplot_titles) > 1 else None,
            shared_yaxes=shared_yaxes,
            shared_xaxes=shared_xaxes,
            vertical_spacing=0.15 / n_rows if n_rows > 1 else 0.1,
            horizontal_spacing=0.1 / n_cols if n_cols > 1 else 0.1,
        )

        # Get consistent colors for all data variables
        if isinstance(data, xr.Dataset):
            all_vars = list(data.data_vars)
        else:
            all_vars = [data.name or 'data']

        processed_colors = ColorProcessor(engine='plotly').process_colors(colors, all_vars)
        color_map = {var: color for var, color in zip(all_vars, processed_colors, strict=False)}

        # Check if we have animation
        has_animation = animate_by is not None

        if has_animation:
            # Prepare frames for animation
            # First frame: initial state
            for idx, (_, _, frames) in enumerate(subplot_data):
                row = idx // n_cols + 1
                col = idx % n_cols + 1

                # Get first frame data
                _, first_frame_data = frames[0]
                df_frame = (
                    first_frame_data.to_dataframe()
                    if isinstance(first_frame_data, xr.DataArray)
                    else first_frame_data.to_dataframe()
                )

                # Fix MultiIndex if present
                if isinstance(df_frame.index, pd.MultiIndex):
                    if 'time' in df_frame.index.names:
                        levels_to_reset = [name for name in df_frame.index.names if name != 'time' and name is not None]
                        if levels_to_reset:
                            df_frame = df_frame.reset_index(level=levels_to_reset, drop=True)

                # Remove leftover dimension columns
                dims_to_remove = [
                    col for col in df_frame.columns if col in ['scenario', 'period', 'time'] and col not in all_vars
                ]
                if dims_to_remove:
                    df_frame = df_frame.drop(columns=dims_to_remove)

                # Plot first frame
                _add_traces_to_subplot(
                    fig,
                    df_frame,
                    mode,
                    color_map,
                    row,
                    col,
                    show_legend=(idx == 0),  # Only show legend for first subplot
                )

            # Create animation frames
            animation_frames = []
            frame_values = [frame_val for frame_val, _ in subplot_data[0][2]]  # Get frame values from first subplot

            for frame_idx, frame_val in enumerate(frame_values):
                frame_data = []

                for subplot_idx, (_, _, frames) in enumerate(subplot_data):
                    row = subplot_idx // n_cols + 1
                    col = subplot_idx % n_cols + 1

                    _, frame_slice = frames[frame_idx]
                    df_frame = (
                        frame_slice.to_dataframe()
                        if isinstance(frame_slice, xr.DataArray)
                        else frame_slice.to_dataframe()
                    )

                    # Fix MultiIndex if present
                    if isinstance(df_frame.index, pd.MultiIndex):
                        if 'time' in df_frame.index.names:
                            levels_to_reset = [
                                name for name in df_frame.index.names if name != 'time' and name is not None
                            ]
                            if levels_to_reset:
                                df_frame = df_frame.reset_index(level=levels_to_reset, drop=True)

                    # Remove leftover dimension columns
                    dims_to_remove = [
                        col for col in df_frame.columns if col in ['scenario', 'period', 'time'] and col not in all_vars
                    ]
                    if dims_to_remove:
                        df_frame = df_frame.drop(columns=dims_to_remove)

                    # Get trace data for this frame
                    traces = _get_traces_for_data(df_frame, mode, color_map, row, col, show_legend=False)
                    frame_data.extend(traces)

                animation_frames.append(go.Frame(data=frame_data, name=str(frame_val)))

            fig.frames = animation_frames

            # Add play/pause buttons
            fig.update_layout(
                updatemenus=[
                    dict(
                        type='buttons',
                        showactive=False,
                        buttons=[
                            dict(
                                label='Play',
                                method='animate',
                                args=[None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}],
                            ),
                            dict(
                                label='Pause',
                                method='animate',
                                args=[
                                    [None],
                                    {
                                        'frame': {'duration': 0, 'redraw': False},
                                        'mode': 'immediate',
                                        'transition': {'duration': 0},
                                    },
                                ],
                            ),
                        ],
                        x=0.1,
                        y=0,
                        xanchor='right',
                        yanchor='top',
                    )
                ],
                sliders=[
                    dict(
                        steps=[
                            dict(
                                args=[[f.name], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
                                label=str(f.name),
                                method='animate',
                            )
                            for f in animation_frames
                        ],
                        x=0.1,
                        y=0,
                        currentvalue=dict(prefix=f'{animate_by}: ', visible=True, xanchor='left'),
                        len=0.9,
                    )
                ],
            )
        else:
            # No animation, just faceting
            for idx, (_, _, frames) in enumerate(subplot_data):
                row = idx // n_cols + 1
                col = idx % n_cols + 1

                # Get data (only one "frame" without animation)
                _, data_slice = frames[0]

                # Convert to DataFrame properly
                if isinstance(data_slice, xr.DataArray):
                    df = data_slice.to_dataframe()
                else:
                    # For Dataset, convert to DataFrame and reshape if needed
                    df = data_slice.to_dataframe()

                # Clean up DataFrame: ensure only time index and data variables as columns
                # Drop any dimension columns that shouldn't be there
                if isinstance(df.index, pd.MultiIndex):
                    # Keep only 'time' in index
                    if 'time' in df.index.names:
                        levels_to_reset = [name for name in df.index.names if name != 'time' and name is not None]
                        if levels_to_reset:
                            df = df.reset_index(level=levels_to_reset, drop=True)

                # Also remove any leftover dimension columns (they shouldn't be data)
                dims_to_remove = [
                    col for col in df.columns if col in ['scenario', 'period', 'time'] and col not in all_vars
                ]
                if dims_to_remove:
                    df = df.drop(columns=dims_to_remove)

                # Add traces to subplot
                _add_traces_to_subplot(
                    fig,
                    df,
                    mode,
                    color_map,
                    row,
                    col,
                    show_legend=(idx == 0),  # Only show legend for first subplot
                )

        # Calculate intelligent height if not specified
        if height_per_row is None:
            # Auto-size to fill ~90% of a typical browser viewport (900px usable height)
            # Scale based on number of rows to avoid tiny plots with many rows
            if n_rows == 1:
                calculated_height = 800  # Single row: use most of screen
            elif n_rows == 2:
                calculated_height = 900  # Two rows: fill screen
            elif n_rows == 3:
                calculated_height = 1000  # Three rows: slightly taller
            else:
                # More rows: use ~350px per row to keep readable
                calculated_height = min(350 * n_rows, 1400)  # Cap at 1400 for very many rows
        else:
            calculated_height = height_per_row * n_rows

        # Update layout
        layout_kwargs = {
            'title': title,
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'font': dict(size=12),
            'height': calculated_height,
            'showlegend': True,
        }
        if width is not None:
            layout_kwargs['width'] = width

        fig.update_layout(**layout_kwargs)

        # Update axes labels
        for i in range(1, n_rows * n_cols + 1):
            fig.update_xaxes(title_text=xlabel, row=(i - 1) // n_cols + 1, col=(i - 1) % n_cols + 1)
            fig.update_yaxes(title_text=ylabel, row=(i - 1) // n_cols + 1, col=(i - 1) % n_cols + 1)

        return fig

    else:
        # Regular DataFrame without faceting/animation
        # Fall back to regular with_plotly
        return with_plotly(data, mode=mode, colors=colors, title=title, ylabel=ylabel, xlabel=xlabel)


def _add_traces_to_subplot(
    fig: go.Figure, df: pd.DataFrame, mode: str, color_map: dict[str, str], row: int, col: int, show_legend: bool = True
):
    """Helper function to add traces to a subplot."""
    if mode == 'stacked_bar':
        for column in df.columns:
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df[column],
                    name=column,
                    marker=dict(color=color_map.get(column, 'gray'), line=dict(width=0)),
                    showlegend=show_legend,
                    legendgroup=column,
                ),
                row=row,
                col=col,
            )
        fig.update_xaxes(row=row, col=col)
        fig.update_yaxes(row=row, col=col)
        if row == 1 and col == 1:  # Only set barmode once
            fig.update_layout(barmode='relative', bargap=0, bargroupgap=0)

    elif mode == 'grouped_bar':
        for column in df.columns:
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df[column],
                    name=column,
                    marker=dict(color=color_map.get(column, 'gray')),
                    showlegend=show_legend,
                    legendgroup=column,
                ),
                row=row,
                col=col,
            )
        if row == 1 and col == 1:
            fig.update_layout(barmode='group', bargap=0.2, bargroupgap=0)

    elif mode == 'line':
        for column in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[column],
                    mode='lines',
                    name=column,
                    line=dict(shape='hv', color=color_map.get(column, 'gray')),
                    showlegend=show_legend,
                    legendgroup=column,
                ),
                row=row,
                col=col,
            )

    elif mode == 'area':
        df_copy = df.copy()

        # Only filter numeric columns to avoid string comparison errors
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_copy.loc[(df_copy[col] > -1e-5) & (df_copy[col] < 1e-5), col] = 0

        # Separate positive and negative columns (only for numeric data)
        positive_columns = []
        negative_columns = []

        for col in df_copy.columns:
            if col in numeric_cols:
                col_data = df_copy[col].dropna()
                if len(col_data) > 0:
                    if (col_data >= 0).all():
                        positive_columns.append(col)
                    elif (col_data <= 0).all():
                        negative_columns.append(col)
                    # Mixed sign columns are omitted from stacking

        for column in positive_columns + negative_columns:
            fig.add_trace(
                go.Scatter(
                    x=df_copy.index,
                    y=df_copy[column],
                    mode='lines',
                    name=column,
                    line=dict(shape='hv', color=color_map.get(column, 'gray')),
                    fill='tonexty',
                    stackgroup='pos' if column in positive_columns else 'neg',
                    showlegend=show_legend,
                    legendgroup=column,
                ),
                row=row,
                col=col,
            )


def _get_traces_for_data(
    df: pd.DataFrame, mode: str, color_map: dict[str, str], row: int, col: int, show_legend: bool = False
) -> list[go.Bar | go.Scatter]:
    """Helper function to generate trace objects for animation frames."""
    traces = []

    if mode in ['stacked_bar', 'grouped_bar']:
        for column in df.columns:
            traces.append(
                go.Bar(
                    x=df.index,
                    y=df[column],
                    name=column,
                    marker=dict(color=color_map.get(column, 'gray')),
                    showlegend=show_legend,
                    legendgroup=column,
                )
            )
    elif mode in ['line', 'area']:
        for column in df.columns:
            traces.append(
                go.Scatter(
                    x=df.index,
                    y=df[column],
                    mode='lines',
                    name=column,
                    line=dict(shape='hv', color=color_map.get(column, 'gray')),
                    showlegend=show_legend,
                    legendgroup=column,
                )
            )

    return traces


def with_matplotlib(
    data: pd.DataFrame,
    mode: Literal['stacked_bar', 'line'] = 'stacked_bar',
    colors: ColorType = 'viridis',
    title: str = '',
    ylabel: str = '',
    xlabel: str = 'Time in h',
    figsize: tuple[int, int] = (12, 6),
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a DataFrame with Matplotlib using stacked bars or stepped lines.

    Args:
        data: A DataFrame containing the data to plot. The index should represent time (e.g., hours),
              and each column represents a separate data series.
        mode: Plotting mode. Use 'stacked_bar' for stacked bar charts or 'line' for stepped lines.
        colors: Color specification, can be:
            - A string with a colormap name (e.g., 'viridis', 'plasma')
            - A list of color strings (e.g., ['#ff0000', '#00ff00'])
            - A dictionary mapping column names to colors (e.g., {'Column1': '#ff0000'})
        title: The title of the plot.
        ylabel: The ylabel of the plot.
        xlabel: The xlabel of the plot.
        figsize: Specify the size of the figure
        fig: A Matplotlib figure object to plot on. If not provided, a new figure will be created.
        ax: A Matplotlib axes object to plot on. If not provided, a new axes will be created.

    Returns:
        A tuple containing the Matplotlib figure and axes objects used for the plot.

    Notes:
        - If `mode` is 'stacked_bar', bars are stacked for both positive and negative values.
          Negative values are stacked separately without extra labels in the legend.
        - If `mode` is 'line', stepped lines are drawn for each data series.
    """
    if mode not in ('stacked_bar', 'line'):
        raise ValueError(f"'mode' must be one of {{'stacked_bar','line'}} for matplotlib, got {mode!r}")

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    processed_colors = ColorProcessor(engine='matplotlib').process_colors(colors, list(data.columns))

    if mode == 'stacked_bar':
        cumulative_positive = np.zeros(len(data))
        cumulative_negative = np.zeros(len(data))
        width = data.index.to_series().diff().dropna().min()  # Minimum time difference

        for i, column in enumerate(data.columns):
            positive_values = np.clip(data[column], 0, None)  # Keep only positive values
            negative_values = np.clip(data[column], None, 0)  # Keep only negative values
            # Plot positive bars
            ax.bar(
                data.index,
                positive_values,
                bottom=cumulative_positive,
                color=processed_colors[i],
                label=column,
                width=width,
                align='center',
            )
            cumulative_positive += positive_values.values
            # Plot negative bars
            ax.bar(
                data.index,
                negative_values,
                bottom=cumulative_negative,
                color=processed_colors[i],
                label='',  # No label for negative bars
                width=width,
                align='center',
            )
            cumulative_negative += negative_values.values

    elif mode == 'line':
        for i, column in enumerate(data.columns):
            ax.step(data.index, data[column], where='post', color=processed_colors[i], label=column)

    # Aesthetics
    ax.set_xlabel(xlabel, ha='center')
    ax.set_ylabel(ylabel, va='center')
    ax.set_title(title)
    ax.grid(color='lightgrey', linestyle='-', linewidth=0.5)
    ax.legend(
        loc='upper center',  # Place legend at the bottom center
        bbox_to_anchor=(0.5, -0.15),  # Adjust the position to fit below plot
        ncol=5,
        frameon=False,  # Remove box around legend
    )
    fig.tight_layout()

    return fig, ax


def heat_map_matplotlib(
    data: pd.DataFrame,
    color_map: str = 'viridis',
    title: str = '',
    xlabel: str = 'Period',
    ylabel: str = 'Step',
    figsize: tuple[float, float] = (12, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots a DataFrame as a heatmap using Matplotlib. The columns of the DataFrame will be displayed on the x-axis,
    the index will be displayed on the y-axis, and the values will represent the 'heat' intensity in the plot.

    Args:
        data: A DataFrame containing the data to be visualized. The index will be used for the y-axis, and columns will be used for the x-axis.
            The values in the DataFrame will be represented as colors in the heatmap.
        color_map: The colormap to use for the heatmap. Default is 'viridis'. Matplotlib supports various colormaps like 'plasma', 'inferno', 'cividis', etc.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        figsize: The size of the figure to create. Default is (12, 6), which results in a width of 12 inches and a height of 6 inches.

    Returns:
        A tuple containing the Matplotlib `Figure` and `Axes` objects. The `Figure` contains the overall plot, while the `Axes` is the area
        where the heatmap is drawn. These can be used for further customization or saving the plot to a file.

    Notes:
        - The y-axis is flipped so that the first row of the DataFrame is displayed at the top of the plot.
        - The color scale is normalized based on the minimum and maximum values in the DataFrame.
        - The x-axis labels (periods) are placed at the top of the plot.
        - The colorbar is added horizontally at the bottom of the plot, with a label.
    """

    # Get the min and max values for color normalization
    color_bar_min, color_bar_max = data.min().min(), data.max().max()

    # Create the heatmap plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.pcolormesh(data.values, cmap=color_map, shading='auto')
    ax.invert_yaxis()  # Flip the y-axis to start at the top

    # Adjust ticks and labels for x and y axes
    ax.set_xticks(np.arange(len(data.columns)) + 0.5)
    ax.set_xticklabels(data.columns, ha='center')
    ax.set_yticks(np.arange(len(data.index)) + 0.5)
    ax.set_yticklabels(data.index, va='center')

    # Add labels to the axes
    ax.set_xlabel(xlabel, ha='center')
    ax.set_ylabel(ylabel, va='center')
    ax.set_title(title)

    # Position x-axis labels at the top
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_ticks_position('top')

    # Add the colorbar
    sm1 = plt.cm.ScalarMappable(cmap=color_map, norm=plt.Normalize(vmin=color_bar_min, vmax=color_bar_max))
    sm1.set_array([])
    fig.colorbar(sm1, ax=ax, pad=0.12, aspect=15, fraction=0.2, orientation='horizontal')

    fig.tight_layout()

    return fig, ax


def heat_map_plotly(
    data: pd.DataFrame,
    color_map: str = 'viridis',
    title: str = '',
    xlabel: str = 'Period',
    ylabel: str = 'Step',
    categorical_labels: bool = True,
) -> go.Figure:
    """
    Plots a DataFrame as a heatmap using Plotly. The columns of the DataFrame will be mapped to the x-axis,
    and the index will be displayed on the y-axis. The values in the DataFrame will represent the 'heat' in the plot.

    Args:
        data: A DataFrame with the data to be visualized. The index will be used for the y-axis, and columns will be used for the x-axis.
            The values in the DataFrame will be represented as colors in the heatmap.
        color_map: The color scale to use for the heatmap. Default is 'viridis'. Plotly supports various color scales like 'Cividis', 'Inferno', etc.
        title: The title of the heatmap. Default is an empty string.
        xlabel: The label for the x-axis. Default is 'Period'.
        ylabel: The label for the y-axis. Default is 'Step'.
        categorical_labels: If True, the x and y axes are treated as categorical data (i.e., the index and columns will not be interpreted as continuous data).
            Default is True. If False, the axes are treated as continuous, which may be useful for time series or numeric data.

    Returns:
        A Plotly figure object containing the heatmap. This can be further customized and saved
        or displayed using `fig.show()`.

    Notes:
        The color bar is automatically scaled to the minimum and maximum values in the data.
        The y-axis is reversed to display the first row at the top.
    """

    color_bar_min, color_bar_max = data.min().min(), data.max().max()  # Min and max values for color scaling
    # Define the figure
    fig = go.Figure(
        data=go.Heatmap(
            z=data.values,
            x=data.columns,
            y=data.index,
            colorscale=color_map,
            zmin=color_bar_min,
            zmax=color_bar_max,
            colorbar=dict(
                title=dict(text='Color Bar Label', side='right'),
                orientation='h',
                xref='container',
                yref='container',
                len=0.8,  # Color bar length relative to plot
                x=0.5,
                y=0.1,
            ),
        )
    )

    # Set axis labels and style
    fig.update_layout(
        title=title,
        xaxis=dict(title=xlabel, side='top', type='category' if categorical_labels else None),
        yaxis=dict(title=ylabel, autorange='reversed', type='category' if categorical_labels else None),
    )

    return fig


def reshape_to_2d(data_1d: np.ndarray, nr_of_steps_per_column: int) -> np.ndarray:
    """
    Reshapes a 1D numpy array into a 2D array suitable for plotting as a colormap.

    The reshaped array will have the number of rows corresponding to the steps per column
    (e.g., 24 hours per day) and columns representing time periods (e.g., days or months).

    Args:
        data_1d: A 1D numpy array with the data to reshape.
        nr_of_steps_per_column: The number of steps (rows) per column in the resulting 2D array. For example,
            this could be 24 (for hours) or 31 (for days in a month).

    Returns:
        The reshaped 2D array. Each internal array corresponds to one column, with the specified number of steps.
        Each column might represents a time period (e.g., day, month, etc.).
    """

    # Step 1: Ensure the input is a 1D array.
    if data_1d.ndim != 1:
        raise ValueError('Input must be a 1D array')

    # Step 2: Convert data to float type to allow NaN padding
    if data_1d.dtype != np.float64:
        data_1d = data_1d.astype(np.float64)

    # Step 3: Calculate the number of columns required
    total_steps = len(data_1d)
    cols = len(data_1d) // nr_of_steps_per_column  # Base number of columns

    # If there's a remainder, add an extra column to hold the remaining values
    if total_steps % nr_of_steps_per_column != 0:
        cols += 1

    # Step 4: Pad the 1D data to match the required number of rows and columns
    padded_data = np.pad(
        data_1d, (0, cols * nr_of_steps_per_column - total_steps), mode='constant', constant_values=np.nan
    )

    # Step 5: Reshape the padded data into a 2D array
    data_2d = padded_data.reshape(cols, nr_of_steps_per_column)

    return data_2d.T


def heat_map_data_from_df(
    df: pd.DataFrame,
    periods: Literal['YS', 'MS', 'W', 'D', 'h', '15min', 'min'],
    steps_per_period: Literal['W', 'D', 'h', '15min', 'min'],
    fill: Literal['ffill', 'bfill'] | None = None,
) -> pd.DataFrame:
    """
    Reshapes a DataFrame with a DateTime index into a 2D array for heatmap plotting,
    based on a specified sample rate.
    Only specific combinations of `periods` and `steps_per_period` are supported; invalid combinations raise an assertion.

    Args:
        df: A DataFrame with a DateTime index containing the data to reshape.
        periods: The time interval of each period (columns of the heatmap),
            such as 'YS' (year start), 'W' (weekly), 'D' (daily), 'h' (hourly) etc.
        steps_per_period: The time interval within each period (rows in the heatmap),
            such as 'YS' (year start), 'W' (weekly), 'D' (daily), 'h' (hourly) etc.
        fill: Method to fill missing values: 'ffill' for forward fill or 'bfill' for backward fill.

    Returns:
        A DataFrame suitable for heatmap plotting, with rows representing steps within each period
        and columns representing each period.
    """
    assert pd.api.types.is_datetime64_any_dtype(df.index), (
        'The index of the DataFrame must be datetime to transform it properly for a heatmap plot'
    )

    # Define formats for different combinations of `periods` and `steps_per_period`
    formats = {
        ('YS', 'W'): ('%Y', '%W'),
        ('YS', 'D'): ('%Y', '%j'),  # day of year
        ('YS', 'h'): ('%Y', '%j %H:00'),
        ('MS', 'D'): ('%Y-%m', '%d'),  # day of month
        ('MS', 'h'): ('%Y-%m', '%d %H:00'),
        ('W', 'D'): ('%Y-w%W', '%w_%A'),  # week and day of week (with prefix for proper sorting)
        ('W', 'h'): ('%Y-w%W', '%w_%A %H:00'),
        ('D', 'h'): ('%Y-%m-%d', '%H:00'),  # Day and hour
        ('D', '15min'): ('%Y-%m-%d', '%H:%M'),  # Day and minute
        ('h', '15min'): ('%Y-%m-%d %H:00', '%M'),  # minute of hour
        ('h', 'min'): ('%Y-%m-%d %H:00', '%M'),  # minute of hour
    }

    if df.empty:
        raise ValueError('DataFrame is empty.')
    diffs = df.index.to_series().diff().dropna()
    minimum_time_diff_in_min = diffs.min().total_seconds() / 60
    time_intervals = {'min': 1, '15min': 15, 'h': 60, 'D': 24 * 60, 'W': 7 * 24 * 60}
    if time_intervals[steps_per_period] > minimum_time_diff_in_min:
        logger.error(
            f'To compute the heatmap, the data was aggregated from {minimum_time_diff_in_min:.2f} min to '
            f'{time_intervals[steps_per_period]:.2f} min. Mean values are displayed.'
        )

    # Select the format based on the `periods` and `steps_per_period` combination
    format_pair = (periods, steps_per_period)
    if format_pair not in formats:
        raise ValueError(f'{format_pair} is not a valid format. Choose from {list(formats.keys())}')
    period_format, step_format = formats[format_pair]

    df = df.sort_index()  # Ensure DataFrame is sorted by time index

    resampled_data = df.resample(steps_per_period).mean()  # Resample and fill any gaps with NaN

    if fill == 'ffill':  # Apply fill method if specified
        resampled_data = resampled_data.ffill()
    elif fill == 'bfill':
        resampled_data = resampled_data.bfill()

    resampled_data['period'] = resampled_data.index.strftime(period_format)
    resampled_data['step'] = resampled_data.index.strftime(step_format)
    if '%w_%A' in step_format:  # Shift index of strings to ensure proper sorting
        resampled_data['step'] = resampled_data['step'].apply(
            lambda x: x.replace('0_Sunday', '7_Sunday') if '0_Sunday' in x else x
        )

    # Pivot the table so periods are columns and steps are indices
    df_pivoted = resampled_data.pivot(columns='period', index='step', values=df.columns[0])

    return df_pivoted


def plot_network(
    node_infos: dict,
    edge_infos: dict,
    path: str | pathlib.Path | None = None,
    controls: bool
    | list[
        Literal['nodes', 'edges', 'layout', 'interaction', 'manipulation', 'physics', 'selection', 'renderer']
    ] = True,
    show: bool = False,
) -> pyvis.network.Network | None:
    """
    Visualizes the network structure of a FlowSystem using PyVis, using info-dictionaries.

    Args:
        path: Path to save the HTML visualization. `False`: Visualization is created but not saved. `str` or `Path`: Specifies file path (default: 'results/network.html').
        controls: UI controls to add to the visualization. `True`: Enables all available controls. `list`: Specify controls, e.g., ['nodes', 'layout'].
            Options: 'nodes', 'edges', 'layout', 'interaction', 'manipulation', 'physics', 'selection', 'renderer'.
            You can play with these and generate a Dictionary from it that can be applied to the network returned by this function.
            network.set_options()
            https://pyvis.readthedocs.io/en/latest/tutorial.html
        show: Whether to open the visualization in the web browser.
            The calculation must be saved to show it. If no path is given, it defaults to 'network.html'.
    Returns:
        The `Network` instance representing the visualization, or `None` if `pyvis` is not installed.

    Notes:
    - This function requires `pyvis`. If not installed, the function prints a warning and returns `None`.
    - Nodes are styled based on type (e.g., circles for buses, boxes for components) and annotated with node information.
    """
    try:
        from pyvis.network import Network
    except ImportError:
        logger.critical("Plotting the flow system network was not possible. Please install pyvis: 'pip install pyvis'")
        return None

    net = Network(directed=True, height='100%' if controls is False else '800px', font_color='white')

    for node_id, node in node_infos.items():
        net.add_node(
            node_id,
            label=node['label'],
            shape={'Bus': 'circle', 'Component': 'box'}[node['class']],
            color={'Bus': '#393E46', 'Component': '#00ADB5'}[node['class']],
            title=node['infos'].replace(')', '\n)'),
            font={'size': 14},
        )

    for edge in edge_infos.values():
        net.add_edge(
            edge['start'],
            edge['end'],
            label=edge['label'],
            title=edge['infos'].replace(')', '\n)'),
            font={'color': '#4D4D4D', 'size': 14},
            color='#222831',
        )

    # Enhanced physics settings
    net.barnes_hut(central_gravity=0.8, spring_length=50, spring_strength=0.05, gravity=-10000)

    if controls:
        net.show_buttons(filter_=controls)  # Adds UI buttons to control physics settings
    if not show and not path:
        return net
    elif path:
        path = pathlib.Path(path) if isinstance(path, str) else path
        net.write_html(path.as_posix())
    elif show:
        path = pathlib.Path('network.html')
        net.write_html(path.as_posix())

    if show:
        try:
            import webbrowser

            worked = webbrowser.open(f'file://{path.resolve()}', 2)
            if not worked:
                logger.error(f'Showing the network in the Browser went wrong. Open it manually. Its saved under {path}')
        except Exception as e:
            logger.error(
                f'Showing the network in the Browser went wrong. Open it manually. Its saved under {path}: {e}'
            )


def pie_with_plotly(
    data: pd.DataFrame,
    colors: ColorType = 'viridis',
    title: str = '',
    legend_title: str = '',
    hole: float = 0.0,
    fig: go.Figure | None = None,
) -> go.Figure:
    """
    Create a pie chart with Plotly to visualize the proportion of values in a DataFrame.

    Args:
        data: A DataFrame containing the data to plot. If multiple rows exist,
              they will be summed unless a specific index value is passed.
        colors: Color specification, can be:
            - A string with a colorscale name (e.g., 'viridis', 'plasma')
            - A list of color strings (e.g., ['#ff0000', '#00ff00'])
            - A dictionary mapping column names to colors (e.g., {'Column1': '#ff0000'})
        title: The title of the plot.
        legend_title: The title for the legend.
        hole: Size of the hole in the center for creating a donut chart (0.0 to 1.0).
        fig: A Plotly figure object to plot on. If not provided, a new figure will be created.

    Returns:
        A Plotly figure object containing the generated pie chart.

    Notes:
        - Negative values are not appropriate for pie charts and will be converted to absolute values with a warning.
        - If the data contains very small values (less than 1% of the total), they can be grouped into an "Other" category
          for better readability.
        - By default, the sum of all columns is used for the pie chart. For time series data, consider preprocessing.

    """
    if data.empty:
        logger.error('Empty DataFrame provided for pie chart. Returning empty figure.')
        return go.Figure()

    # Create a copy to avoid modifying the original DataFrame
    data_copy = data.copy()

    # Check if any negative values and warn
    if (data_copy < 0).any().any():
        logger.error('Negative values detected in data. Using absolute values for pie chart.')
        data_copy = data_copy.abs()

    # If data has multiple rows, sum them to get total for each column
    if len(data_copy) > 1:
        data_sum = data_copy.sum()
    else:
        data_sum = data_copy.iloc[0]

    # Get labels (column names) and values
    labels = data_sum.index.tolist()
    values = data_sum.values.tolist()

    # Apply color mapping using the unified color processor
    processed_colors = ColorProcessor(engine='plotly').process_colors(colors, labels)

    # Create figure if not provided
    fig = fig if fig is not None else go.Figure()

    # Add pie trace
    fig.add_trace(
        go.Pie(
            labels=labels,
            values=values,
            hole=hole,
            marker=dict(colors=processed_colors),
            textinfo='percent+label+value',
            textposition='inside',
            insidetextorientation='radial',
        )
    )

    # Update layout for better aesthetics
    fig.update_layout(
        title=title,
        legend_title=legend_title,
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper background
        font=dict(size=14),  # Increase font size for better readability
    )

    return fig


def pie_with_matplotlib(
    data: pd.DataFrame,
    colors: ColorType = 'viridis',
    title: str = '',
    legend_title: str = 'Categories',
    hole: float = 0.0,
    figsize: tuple[int, int] = (10, 8),
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Create a pie chart with Matplotlib to visualize the proportion of values in a DataFrame.

    Args:
        data: A DataFrame containing the data to plot. If multiple rows exist,
              they will be summed unless a specific index value is passed.
        colors: Color specification, can be:
            - A string with a colormap name (e.g., 'viridis', 'plasma')
            - A list of color strings (e.g., ['#ff0000', '#00ff00'])
            - A dictionary mapping column names to colors (e.g., {'Column1': '#ff0000'})
        title: The title of the plot.
        legend_title: The title for the legend.
        hole: Size of the hole in the center for creating a donut chart (0.0 to 1.0).
        figsize: The size of the figure (width, height) in inches.
        fig: A Matplotlib figure object to plot on. If not provided, a new figure will be created.
        ax: A Matplotlib axes object to plot on. If not provided, a new axes will be created.

    Returns:
        A tuple containing the Matplotlib figure and axes objects used for the plot.

    Notes:
        - Negative values are not appropriate for pie charts and will be converted to absolute values with a warning.
        - If the data contains very small values (less than 1% of the total), they can be grouped into an "Other" category
          for better readability.
        - By default, the sum of all columns is used for the pie chart. For time series data, consider preprocessing.

    """
    if data.empty:
        logger.error('Empty DataFrame provided for pie chart. Returning empty figure.')
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        return fig, ax

    # Create a copy to avoid modifying the original DataFrame
    data_copy = data.copy()

    # Check if any negative values and warn
    if (data_copy < 0).any().any():
        logger.error('Negative values detected in data. Using absolute values for pie chart.')
        data_copy = data_copy.abs()

    # If data has multiple rows, sum them to get total for each column
    if len(data_copy) > 1:
        data_sum = data_copy.sum()
    else:
        data_sum = data_copy.iloc[0]

    # Get labels (column names) and values
    labels = data_sum.index.tolist()
    values = data_sum.values.tolist()

    # Apply color mapping using the unified color processor
    processed_colors = ColorProcessor(engine='matplotlib').process_colors(colors, labels)

    # Create figure and axis if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Draw the pie chart
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        colors=processed_colors,
        autopct='%1.1f%%',
        startangle=90,
        shadow=False,
        wedgeprops=dict(width=0.5) if hole > 0 else None,  # Set width for donut
    )

    # Adjust the wedgeprops to make donut hole size consistent with plotly
    # For matplotlib, the hole size is determined by the wedge width
    # Convert hole parameter to wedge width
    if hole > 0:
        # Adjust hole size to match plotly's hole parameter
        # In matplotlib, wedge width is relative to the radius (which is 1)
        # For plotly, hole is a fraction of the radius
        wedge_width = 1 - hole
        for wedge in wedges:
            wedge.set_width(wedge_width)

    # Customize the appearance
    # Make autopct text more visible
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_color('white')

    # Set aspect ratio to be equal to ensure a circular pie
    ax.set_aspect('equal')

    # Add title
    if title:
        ax.set_title(title, fontsize=16)

    # Create a legend if there are many segments
    if len(labels) > 6:
        ax.legend(wedges, labels, title=legend_title, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1))

    # Apply tight layout
    fig.tight_layout()

    return fig, ax


def dual_pie_with_plotly(
    data_left: pd.Series,
    data_right: pd.Series,
    colors: ColorType = 'viridis',
    title: str = '',
    subtitles: tuple[str, str] = ('Left Chart', 'Right Chart'),
    legend_title: str = '',
    hole: float = 0.2,
    lower_percentage_group: float = 5.0,
    hover_template: str = '%{label}: %{value} (%{percent})',
    text_info: str = 'percent+label',
    text_position: str = 'inside',
) -> go.Figure:
    """
    Create two pie charts side by side with Plotly, with consistent coloring across both charts.

    Args:
        data_left: Series for the left pie chart.
        data_right: Series for the right pie chart.
        colors: Color specification, can be:
            - A string with a colorscale name (e.g., 'viridis', 'plasma')
            - A list of color strings (e.g., ['#ff0000', '#00ff00'])
            - A dictionary mapping category names to colors (e.g., {'Category1': '#ff0000'})
        title: The main title of the plot.
        subtitles: Tuple containing the subtitles for (left, right) charts.
        legend_title: The title for the legend.
        hole: Size of the hole in the center for creating donut charts (0.0 to 1.0).
        lower_percentage_group: Group segments whose cumulative share is below this percentage (0–100) into "Other".
        hover_template: Template for hover text. Use %{label}, %{value}, %{percent}.
        text_info: What to show on pie segments: 'label', 'percent', 'value', 'label+percent',
                  'label+value', 'percent+value', 'label+percent+value', or 'none'.
        text_position: Position of text: 'inside', 'outside', 'auto', or 'none'.

    Returns:
        A Plotly figure object containing the generated dual pie chart.
    """
    from plotly.subplots import make_subplots

    # Check for empty data
    if data_left.empty and data_right.empty:
        logger.error('Both datasets are empty. Returning empty figure.')
        return go.Figure()

    # Create a subplot figure
    fig = make_subplots(
        rows=1, cols=2, specs=[[{'type': 'pie'}, {'type': 'pie'}]], subplot_titles=subtitles, horizontal_spacing=0.05
    )

    # Process series to handle negative values and apply minimum percentage threshold
    def preprocess_series(series: pd.Series):
        """
        Preprocess a series for pie chart display by handling negative values
        and grouping the smallest parts together if they collectively represent
        less than the specified percentage threshold.

        Args:
            series: The series to preprocess

        Returns:
            A preprocessed pandas Series
        """
        # Handle negative values
        if (series < 0).any():
            logger.error('Negative values detected in data. Using absolute values for pie chart.')
            series = series.abs()

        # Remove zeros
        series = series[series > 0]

        # Apply minimum percentage threshold if needed
        if lower_percentage_group and not series.empty:
            total = series.sum()
            if total > 0:
                # Sort series by value (ascending)
                sorted_series = series.sort_values()

                # Calculate cumulative percentage contribution
                cumulative_percent = (sorted_series.cumsum() / total) * 100

                # Find entries that collectively make up less than lower_percentage_group
                to_group = cumulative_percent <= lower_percentage_group

                if to_group.sum() > 1:
                    # Create "Other" category for the smallest values that together are < threshold
                    other_sum = sorted_series[to_group].sum()

                    # Keep only values that aren't in the "Other" group
                    result_series = series[~series.index.isin(sorted_series[to_group].index)]

                    # Add the "Other" category if it has a value
                    if other_sum > 0:
                        result_series['Other'] = other_sum

                    return result_series

        return series

    data_left_processed = preprocess_series(data_left)
    data_right_processed = preprocess_series(data_right)

    # Get unique set of all labels for consistent coloring
    all_labels = sorted(set(data_left_processed.index) | set(data_right_processed.index))

    # Get consistent color mapping for both charts using our unified function
    color_map = ColorProcessor(engine='plotly').process_colors(colors, all_labels, return_mapping=True)

    # Function to create a pie trace with consistently mapped colors
    def create_pie_trace(data_series, side):
        if data_series.empty:
            return None

        labels = data_series.index.tolist()
        values = data_series.values.tolist()
        trace_colors = [color_map[label] for label in labels]

        return go.Pie(
            labels=labels,
            values=values,
            name=side,
            marker=dict(colors=trace_colors),
            hole=hole,
            textinfo=text_info,
            textposition=text_position,
            insidetextorientation='radial',
            hovertemplate=hover_template,
            sort=True,  # Sort values by default (largest first)
        )

    # Add left pie if data exists
    left_trace = create_pie_trace(data_left_processed, subtitles[0])
    if left_trace:
        left_trace.domain = dict(x=[0, 0.48])
        fig.add_trace(left_trace, row=1, col=1)

    # Add right pie if data exists
    right_trace = create_pie_trace(data_right_processed, subtitles[1])
    if right_trace:
        right_trace.domain = dict(x=[0.52, 1])
        fig.add_trace(right_trace, row=1, col=2)

    # Update layout
    fig.update_layout(
        title=title,
        legend_title=legend_title,
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper background
        font=dict(size=14),
        margin=dict(t=80, b=50, l=30, r=30),
    )

    return fig


def dual_pie_with_matplotlib(
    data_left: pd.Series,
    data_right: pd.Series,
    colors: ColorType = 'viridis',
    title: str = '',
    subtitles: tuple[str, str] = ('Left Chart', 'Right Chart'),
    legend_title: str = '',
    hole: float = 0.2,
    lower_percentage_group: float = 5.0,
    figsize: tuple[int, int] = (14, 7),
    fig: plt.Figure | None = None,
    axes: list[plt.Axes] | None = None,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Create two pie charts side by side with Matplotlib, with consistent coloring across both charts.
    Leverages the existing pie_with_matplotlib function.

    Args:
        data_left: Series for the left pie chart.
        data_right: Series for the right pie chart.
        colors: Color specification, can be:
            - A string with a colormap name (e.g., 'viridis', 'plasma')
            - A list of color strings (e.g., ['#ff0000', '#00ff00'])
            - A dictionary mapping category names to colors (e.g., {'Category1': '#ff0000'})
        title: The main title of the plot.
        subtitles: Tuple containing the subtitles for (left, right) charts.
        legend_title: The title for the legend.
        hole: Size of the hole in the center for creating donut charts (0.0 to 1.0).
        lower_percentage_group: Whether to group small segments (below percentage) into an "Other" category.
        figsize: The size of the figure (width, height) in inches.
        fig: A Matplotlib figure object to plot on. If not provided, a new figure will be created.
        axes: A list of Matplotlib axes objects to plot on. If not provided, new axes will be created.

    Returns:
        A tuple containing the Matplotlib figure and list of axes objects used for the plot.
    """
    # Check for empty data
    if data_left.empty and data_right.empty:
        logger.error('Both datasets are empty. Returning empty figure.')
        if fig is None:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
        return fig, axes

    # Create figure and axes if not provided
    if fig is None or axes is None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Process series to handle negative values and apply minimum percentage threshold
    def preprocess_series(series: pd.Series):
        """
        Preprocess a series for pie chart display by handling negative values
        and grouping the smallest parts together if they collectively represent
        less than the specified percentage threshold.
        """
        # Handle negative values
        if (series < 0).any():
            logger.error('Negative values detected in data. Using absolute values for pie chart.')
            series = series.abs()

        # Remove zeros
        series = series[series > 0]

        # Apply minimum percentage threshold if needed
        if lower_percentage_group and not series.empty:
            total = series.sum()
            if total > 0:
                # Sort series by value (ascending)
                sorted_series = series.sort_values()

                # Calculate cumulative percentage contribution
                cumulative_percent = (sorted_series.cumsum() / total) * 100

                # Find entries that collectively make up less than lower_percentage_group
                to_group = cumulative_percent <= lower_percentage_group

                if to_group.sum() > 1:
                    # Create "Other" category for the smallest values that together are < threshold
                    other_sum = sorted_series[to_group].sum()

                    # Keep only values that aren't in the "Other" group
                    result_series = series[~series.index.isin(sorted_series[to_group].index)]

                    # Add the "Other" category if it has a value
                    if other_sum > 0:
                        result_series['Other'] = other_sum

                    return result_series

        return series

    # Preprocess data
    data_left_processed = preprocess_series(data_left)
    data_right_processed = preprocess_series(data_right)

    # Convert Series to DataFrames for pie_with_matplotlib
    df_left = pd.DataFrame(data_left_processed).T if not data_left_processed.empty else pd.DataFrame()
    df_right = pd.DataFrame(data_right_processed).T if not data_right_processed.empty else pd.DataFrame()

    # Get unique set of all labels for consistent coloring
    all_labels = sorted(set(data_left_processed.index) | set(data_right_processed.index))

    # Get consistent color mapping for both charts using our unified function
    color_map = ColorProcessor(engine='matplotlib').process_colors(colors, all_labels, return_mapping=True)

    # Configure colors for each DataFrame based on the consistent mapping
    left_colors = [color_map[col] for col in df_left.columns] if not df_left.empty else []
    right_colors = [color_map[col] for col in df_right.columns] if not df_right.empty else []

    # Create left pie chart
    if not df_left.empty:
        pie_with_matplotlib(data=df_left, colors=left_colors, title=subtitles[0], hole=hole, fig=fig, ax=axes[0])
    else:
        axes[0].set_title(subtitles[0])
        axes[0].axis('off')

    # Create right pie chart
    if not df_right.empty:
        pie_with_matplotlib(data=df_right, colors=right_colors, title=subtitles[1], hole=hole, fig=fig, ax=axes[1])
    else:
        axes[1].set_title(subtitles[1])
        axes[1].axis('off')

    # Add main title
    if title:
        fig.suptitle(title, fontsize=16, y=0.98)

    # Adjust layout
    fig.tight_layout()

    # Create a unified legend if both charts have data
    if not df_left.empty and not df_right.empty:
        # Remove individual legends
        for ax in axes:
            if ax.get_legend():
                ax.get_legend().remove()

        # Create handles for the unified legend
        handles = []
        labels_for_legend = []

        for label in all_labels:
            color = color_map[label]
            patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label)
            handles.append(patch)
            labels_for_legend.append(label)

        # Add unified legend
        fig.legend(
            handles=handles,
            labels=labels_for_legend,
            title=legend_title,
            loc='lower center',
            bbox_to_anchor=(0.5, 0),
            ncol=min(len(all_labels), 5),  # Limit columns to 5 for readability
        )

        # Add padding at the bottom for the legend
        fig.subplots_adjust(bottom=0.2)

    return fig, axes


def export_figure(
    figure_like: go.Figure | tuple[plt.Figure, plt.Axes],
    default_path: pathlib.Path,
    default_filetype: str | None = None,
    user_path: pathlib.Path | None = None,
    show: bool = True,
    save: bool = False,
) -> go.Figure | tuple[plt.Figure, plt.Axes]:
    """
    Export a figure to a file and or show it.

    Args:
        figure_like: The figure to export. Can be a Plotly figure or a tuple of Matplotlib figure and axes.
        default_path: The default file path if no user filename is provided.
        default_filetype: The default filetype if the path doesnt end with a filetype.
        user_path: An optional user-specified file path.
        show: Whether to display the figure (default: True).
        save: Whether to save the figure (default: False).

    Raises:
        ValueError: If no default filetype is provided and the path doesn't specify a filetype.
        TypeError: If the figure type is not supported.
    """
    filename = user_path or default_path
    filename = filename.with_name(filename.name.replace('|', '__'))
    if filename.suffix == '':
        if default_filetype is None:
            raise ValueError('No default filetype provided')
        filename = filename.with_suffix(default_filetype)

    if isinstance(figure_like, plotly.graph_objs.Figure):
        fig = figure_like
        if filename.suffix != '.html':
            logger.warning(f'To save a Plotly figure, using .html. Adjusting suffix for {filename}')
            filename = filename.with_suffix('.html')

        try:
            is_test_env = 'PYTEST_CURRENT_TEST' in os.environ

            if is_test_env:
                # Test environment: never open browser, only save if requested
                if save:
                    fig.write_html(str(filename))
                # Ignore show flag in tests
            else:
                # Production environment: respect show and save flags
                if save and show:
                    # Save and auto-open in browser
                    plotly.offline.plot(fig, filename=str(filename))
                elif save and not show:
                    # Save without opening
                    fig.write_html(str(filename))
                elif show and not save:
                    # Show interactively without saving
                    fig.show()
                # If neither save nor show: do nothing
        finally:
            # Cleanup to prevent socket warnings
            if hasattr(fig, '_renderer'):
                fig._renderer = None

        return figure_like

    elif isinstance(figure_like, tuple):
        fig, ax = figure_like
        if show:
            # Only show if using interactive backend and not in test environment
            backend = matplotlib.get_backend().lower()
            is_interactive = backend not in {'agg', 'pdf', 'ps', 'svg', 'template'}
            is_test_env = 'PYTEST_CURRENT_TEST' in os.environ

            if is_interactive and not is_test_env:
                plt.show()

        if save:
            fig.savefig(str(filename), dpi=300)
            plt.close(fig)  # Close figure to free memory

        return fig, ax

    raise TypeError(f'Figure type not supported: {type(figure_like)}')
