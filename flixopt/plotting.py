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

from .config import CONFIG

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


def _ensure_dataset(data: xr.Dataset | pd.DataFrame | pd.Series) -> xr.Dataset:
    """Convert DataFrame or Series to Dataset if needed."""
    if isinstance(data, xr.Dataset):
        return data
    elif isinstance(data, pd.DataFrame):
        # Convert DataFrame to Dataset
        return data.to_xarray()
    elif isinstance(data, pd.Series):
        # Convert Series to DataFrame first, then to Dataset
        return data.to_frame().to_xarray()
    else:
        raise TypeError(f'Data must be xr.Dataset, pd.DataFrame, or pd.Series, got {type(data).__name__}')


def _validate_plotting_data(data: xr.Dataset, allow_empty: bool = False) -> None:
    """Validate dataset for plotting (checks for empty data, non-numeric types, etc.)."""
    # Check for empty data
    if not allow_empty and len(data.data_vars) == 0:
        raise ValueError('Empty Dataset provided (no variables). Cannot create plot.')

    # Check if dataset has any data (xarray uses nbytes for total size)
    if all(data[var].size == 0 for var in data.data_vars) if len(data.data_vars) > 0 else True:
        if not allow_empty and len(data.data_vars) > 0:
            raise ValueError('Dataset has zero size. Cannot create plot.')
        if len(data.data_vars) == 0:
            return  # Empty dataset, nothing to validate
        return

    # Check for non-numeric data types
    for var in data.data_vars:
        dtype = data[var].dtype
        if not np.issubdtype(dtype, np.number):
            raise TypeError(
                f"Variable '{var}' has non-numeric dtype '{dtype}'. "
                f'Plotting requires numeric data types (int, float, etc.).'
            )

    # Warn about NaN/Inf values
    for var in data.data_vars:
        if np.isnan(data[var].values).any():
            logger.debug(f"Variable '{var}' contains NaN values which may affect visualization.")
        if np.isinf(data[var].values).any():
            logger.debug(f"Variable '{var}' contains Inf values which may affect visualization.")


def resolve_colors(
    data: xr.Dataset,
    colors: ColorType,
    engine: PlottingEngine = 'plotly',
) -> dict[str, str]:
    """Resolve colors parameter to a dict mapping variable names to colors."""
    # Get variable names from Dataset (always strings and unique)
    labels = list(data.data_vars.keys())

    # If explicit dict provided, use it directly
    if isinstance(colors, dict):
        return colors

    # If string or list, use ColorProcessor (traditional behavior)
    if isinstance(colors, (str, list)):
        processor = ColorProcessor(engine=engine)
        return processor.process_colors(colors, labels, return_mapping=True)

    raise TypeError(f'Wrong type passed to resolve_colors(): {type(colors)}')


def with_plotly(
    data: xr.Dataset | pd.DataFrame | pd.Series,
    mode: Literal['stacked_bar', 'line', 'area', 'grouped_bar'] = 'stacked_bar',
    colors: ColorType = 'viridis',
    title: str = '',
    ylabel: str = '',
    xlabel: str = '',
    facet_by: str | list[str] | None = None,
    animate_by: str | None = None,
    facet_cols: int | None = None,
    shared_yaxes: bool = True,
    shared_xaxes: bool = True,
    trace_kwargs: dict[str, Any] | None = None,
    layout_kwargs: dict[str, Any] | None = None,
    **px_kwargs: Any,
) -> go.Figure:
    """
    Plot data with Plotly using facets (subplots) and/or animation for multidimensional data.

    Uses Plotly Express for convenient faceting and animation with automatic styling.
    For simple plots without faceting, can optionally add to an existing figure.

    Args:
        data: An xarray Dataset, pandas DataFrame, or pandas Series to plot.
        mode: The plotting mode. Use 'stacked_bar' for stacked bar charts, 'line' for lines,
              'area' for stacked area charts, or 'grouped_bar' for grouped bar charts.
        colors: Color specification (colormap, list, or dict mapping labels to colors).
        title: The main title of the plot.
        ylabel: The label for the y-axis.
        xlabel: The label for the x-axis.
        fig: A Plotly figure object to plot on (only for simple plots without faceting).
             If not provided, a new figure will be created.
        facet_by: Dimension(s) to create facets for. Creates a subplot grid.
              Can be a single dimension name or list of dimensions (max 2 for facet_row and facet_col).
              If the dimension doesn't exist in the data, it will be silently ignored.
        animate_by: Dimension to animate over. Creates animation frames.
              If the dimension doesn't exist in the data, it will be silently ignored.
        facet_cols: Number of columns in the facet grid (used when facet_by is single dimension).
        shared_yaxes: Whether subplots share y-axes.
        shared_xaxes: Whether subplots share x-axes.
        trace_kwargs: Optional dict of parameters to pass to fig.update_traces().
                     Use this to customize trace properties (e.g., marker style, line width).
        layout_kwargs: Optional dict of parameters to pass to fig.update_layout().
                      Use this to customize layout properties (e.g., width, height, legend position).
        **px_kwargs: Additional keyword arguments passed to the underlying Plotly Express function
                    (px.bar, px.line, px.area). These override default arguments if provided.

    Returns:
        A Plotly figure object containing the faceted/animated plot.

    Examples:
        Simple plot:

        ```python
        fig = with_plotly(dataset, mode='area', title='Energy Mix')
        ```

        Facet by scenario:

        ```python
        fig = with_plotly(dataset, facet_by='scenario', facet_cols=2)
        ```

        Animate by period:

        ```python
        fig = with_plotly(dataset, animate_by='period')
        ```

        Facet and animate:

        ```python
        fig = with_plotly(dataset, facet_by='scenario', animate_by='period')
        ```
    """
    if mode not in ('stacked_bar', 'line', 'area', 'grouped_bar'):
        raise ValueError(f"'mode' must be one of {{'stacked_bar','line','area', 'grouped_bar'}}, got {mode!r}")

    # Ensure data is a Dataset and validate it
    data = _ensure_dataset(data)
    _validate_plotting_data(data, allow_empty=True)

    # Handle empty data
    if len(data.data_vars) == 0:
        logger.error('"with_plotly() got an empty Dataset.')
        return go.Figure()

    # Handle all-scalar datasets (where all variables have no dimensions)
    # This occurs when all variables are scalar values with dims=()
    if all(len(data[var].dims) == 0 for var in data.data_vars):
        # Create a simple DataFrame with variable names as x-axis
        variables = list(data.data_vars.keys())
        values = [float(data[var].values) for var in data.data_vars]

        # Resolve colors
        color_discrete_map = resolve_colors(data, colors, engine='plotly')
        marker_colors = [color_discrete_map.get(var, '#636EFA') for var in variables]

        # Create simple plot based on mode using go (not px) for better color control
        if mode in ('stacked_bar', 'grouped_bar'):
            fig = go.Figure(data=[go.Bar(x=variables, y=values, marker_color=marker_colors)])
        elif mode == 'line':
            fig = go.Figure(
                data=[
                    go.Scatter(
                        x=variables,
                        y=values,
                        mode='lines+markers',
                        marker=dict(color=marker_colors, size=8),
                        line=dict(color='lightgray'),
                    )
                ]
            )
        elif mode == 'area':
            fig = go.Figure(
                data=[
                    go.Scatter(
                        x=variables,
                        y=values,
                        fill='tozeroy',
                        marker=dict(color=marker_colors, size=8),
                        line=dict(color='lightgray'),
                    )
                ]
            )

        fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel, showlegend=False)
        return fig

    # Convert Dataset to long-form DataFrame for Plotly Express
    # Structure: time, variable, value, scenario, period, ... (all dims as columns)
    dim_names = list(data.dims)
    df_long = data.to_dataframe().reset_index().melt(id_vars=dim_names, var_name='variable', value_name='value')

    # Validate facet_by and animate_by dimensions exist in the data
    available_dims = [col for col in df_long.columns if col not in ['variable', 'value']]

    # Check facet_by dimensions
    if facet_by is not None:
        if isinstance(facet_by, str):
            if facet_by not in available_dims:
                logger.debug(
                    f"Dimension '{facet_by}' not found in data. Available dimensions: {available_dims}. "
                    f'Ignoring facet_by parameter.'
                )
                facet_by = None
        elif isinstance(facet_by, list):
            # Filter out dimensions that don't exist
            missing_dims = [dim for dim in facet_by if dim not in available_dims]
            facet_by = [dim for dim in facet_by if dim in available_dims]
            if missing_dims:
                logger.debug(
                    f'Dimensions {missing_dims} not found in data. Available dimensions: {available_dims}. '
                    f'Using only existing dimensions: {facet_by if facet_by else "none"}.'
                )
            if len(facet_by) == 0:
                facet_by = None

    # Check animate_by dimension
    if animate_by is not None and animate_by not in available_dims:
        logger.debug(
            f"Dimension '{animate_by}' not found in data. Available dimensions: {available_dims}. "
            f'Ignoring animate_by parameter.'
        )
        animate_by = None

    # Setup faceting parameters for Plotly Express
    facet_row = None
    facet_col = None
    if facet_by:
        if isinstance(facet_by, str):
            # Single facet dimension - use facet_col with facet_col_wrap
            facet_col = facet_by
        elif len(facet_by) == 1:
            facet_col = facet_by[0]
        elif len(facet_by) == 2:
            # Two facet dimensions - use facet_row and facet_col
            facet_row = facet_by[0]
            facet_col = facet_by[1]
        else:
            raise ValueError(f'facet_by can have at most 2 dimensions, got {len(facet_by)}')

    # Process colors
    all_vars = df_long['variable'].unique().tolist()
    processed_colors = ColorProcessor(engine='plotly').process_colors(colors, all_vars)
    color_discrete_map = {var: color for var, color in zip(all_vars, processed_colors, strict=True)}

    # Determine which dimension to use for x-axis
    # Collect dimensions used for faceting and animation
    used_dims = set()
    if facet_row:
        used_dims.add(facet_row)
    if facet_col:
        used_dims.add(facet_col)
    if animate_by:
        used_dims.add(animate_by)

    # Find available dimensions for x-axis (not used for faceting/animation)
    x_candidates = [d for d in available_dims if d not in used_dims]

    # Use 'time' if available, otherwise use the first available dimension
    if 'time' in x_candidates:
        x_dim = 'time'
    elif len(x_candidates) > 0:
        x_dim = x_candidates[0]
    else:
        # Fallback: use the first dimension (shouldn't happen in normal cases)
        x_dim = available_dims[0] if available_dims else 'time'

    # Create plot using Plotly Express based on mode
    common_args = {
        'data_frame': df_long,
        'x': x_dim,
        'y': 'value',
        'color': 'variable',
        'facet_row': facet_row,
        'facet_col': facet_col,
        'animation_frame': animate_by,
        'color_discrete_map': color_discrete_map,
        'title': title,
        'labels': {'value': ylabel, x_dim: xlabel, 'variable': ''},
    }

    # Add facet_col_wrap for single facet dimension
    if facet_col and not facet_row:
        common_args['facet_col_wrap'] = facet_cols

    # Allow callers to pass any px.* keyword args (e.g., category_orders, range_x/y)
    if px_kwargs:
        common_args.update(px_kwargs)

    if mode == 'stacked_bar':
        fig = px.bar(**common_args)
        fig.update_traces(marker_line_width=0)
        fig.update_layout(barmode='relative', bargap=0, bargroupgap=0)
    elif mode == 'grouped_bar':
        fig = px.bar(**common_args)
        fig.update_layout(barmode='group', bargap=0.2, bargroupgap=0)
    elif mode == 'line':
        fig = px.line(**common_args, line_shape='hv')  # Stepped lines
    elif mode == 'area':
        # Use Plotly Express to create the area plot (preserves animation, legends, faceting)
        fig = px.area(**common_args, line_shape='hv')

        # Classify each variable based on its values
        variable_classification = {}
        for var in all_vars:
            var_data = df_long[df_long['variable'] == var]['value']
            var_data_clean = var_data[(var_data < -1e-5) | (var_data > 1e-5)]

            if len(var_data_clean) == 0:
                variable_classification[var] = 'zero'
            else:
                has_pos, has_neg = (var_data_clean > 0).any(), (var_data_clean < 0).any()
                variable_classification[var] = (
                    'mixed' if has_pos and has_neg else ('negative' if has_neg else 'positive')
                )

        # Log warning for mixed variables
        mixed_vars = [v for v, c in variable_classification.items() if c == 'mixed']
        if mixed_vars:
            logger.warning(f'Variables with both positive and negative values: {mixed_vars}. Plotted as dashed lines.')

        all_traces = list(fig.data)
        for frame in fig.frames:
            all_traces.extend(frame.data)

        for trace in all_traces:
            cls = variable_classification.get(trace.name, None)
            # Only stack positive and negative, not mixed or zero
            trace.stackgroup = cls if cls in ('positive', 'negative') else None

            if cls in ('positive', 'negative'):
                # Stacked area: add opacity to avoid hiding layers, remove line border
                if hasattr(trace, 'line') and trace.line.color:
                    trace.fillcolor = trace.line.color
                    trace.line.width = 0
            elif cls == 'mixed':
                # Mixed variables: show as dashed line, not stacked
                if hasattr(trace, 'line'):
                    trace.line.width = 2
                    trace.line.dash = 'dash'
                if hasattr(trace, 'fill'):
                    trace.fill = None

    # Update axes to share if requested (Plotly Express already handles this, but we can customize)
    if not shared_yaxes:
        fig.update_yaxes(matches=None)
    if not shared_xaxes:
        fig.update_xaxes(matches=None)

    # Apply user-provided trace and layout customizations
    if trace_kwargs:
        fig.update_traces(**trace_kwargs)
    if layout_kwargs:
        fig.update_layout(**layout_kwargs)

    return fig


def with_matplotlib(
    data: xr.Dataset | pd.DataFrame | pd.Series,
    mode: Literal['stacked_bar', 'line'] = 'stacked_bar',
    colors: ColorType = 'viridis',
    title: str = '',
    ylabel: str = '',
    xlabel: str = 'Time in h',
    figsize: tuple[int, int] = (12, 6),
    plot_kwargs: dict[str, Any] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot data with Matplotlib using stacked bars or stepped lines.

    Args:
        data: An xarray Dataset, pandas DataFrame, or pandas Series to plot. After conversion to DataFrame,
              the index represents time and each column represents a separate data series (variables).
        mode: Plotting mode. Use 'stacked_bar' for stacked bar charts or 'line' for stepped lines.
        colors: Color specification. Can be:
            - A colormap name (e.g., 'turbo', 'plasma')
            - A list of color strings (e.g., ['#ff0000', '#00ff00'])
            - A dict mapping column names to colors (e.g., {'Column1': '#ff0000'})
        title: The title of the plot.
        ylabel: The ylabel of the plot.
        xlabel: The xlabel of the plot.
        figsize: Specify the size of the figure (width, height) in inches.
        plot_kwargs: Optional dict of parameters to pass to ax.bar() or ax.step() plotting calls.
                    Use this to customize plot properties (e.g., linewidth, alpha, edgecolor).

    Returns:
        A tuple containing the Matplotlib figure and axes objects used for the plot.

    Notes:
        - If `mode` is 'stacked_bar', bars are stacked for both positive and negative values.
          Negative values are stacked separately without extra labels in the legend.
        - If `mode` is 'line', stepped lines are drawn for each data series.
    """
    if mode not in ('stacked_bar', 'line'):
        raise ValueError(f"'mode' must be one of {{'stacked_bar','line'}} for matplotlib, got {mode!r}")

    # Ensure data is a Dataset and validate it
    data = _ensure_dataset(data)
    _validate_plotting_data(data, allow_empty=True)

    # Create new figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Initialize plot_kwargs if not provided
    if plot_kwargs is None:
        plot_kwargs = {}

    # Handle all-scalar datasets (where all variables have no dimensions)
    # This occurs when all variables are scalar values with dims=()
    if all(len(data[var].dims) == 0 for var in data.data_vars):
        # Create simple bar/line plot with variable names as x-axis
        variables = list(data.data_vars.keys())
        values = [float(data[var].values) for var in data.data_vars]

        # Resolve colors
        color_discrete_map = resolve_colors(data, colors, engine='matplotlib')
        colors_list = [color_discrete_map.get(var, '#808080') for var in variables]

        # Create plot based on mode
        if mode == 'stacked_bar':
            ax.bar(variables, values, color=colors_list, **plot_kwargs)
        elif mode == 'line':
            ax.plot(
                variables,
                values,
                marker='o',
                color=colors_list[0] if len(set(colors_list)) == 1 else None,
                **plot_kwargs,
            )
            # If different colors, plot each point separately
            if len(set(colors_list)) > 1:
                ax.clear()
                for i, (var, val) in enumerate(zip(variables, values, strict=False)):
                    ax.plot([i], [val], marker='o', color=colors_list[i], label=var, **plot_kwargs)
                ax.set_xticks(range(len(variables)))
                ax.set_xticklabels(variables)

        ax.set_xlabel(xlabel, ha='center')
        ax.set_ylabel(ylabel, va='center')
        ax.set_title(title)
        ax.grid(color='lightgrey', linestyle='-', linewidth=0.5, axis='y')
        fig.tight_layout()

        return fig, ax

    # Resolve colors first (includes validation)
    color_discrete_map = resolve_colors(data, colors, engine='matplotlib')

    # Convert Dataset to DataFrame for matplotlib plotting (naturally wide-form)
    df = data.to_dataframe()

    # Get colors in column order
    processed_colors = [color_discrete_map.get(str(col), '#808080') for col in df.columns]

    if mode == 'stacked_bar':
        cumulative_positive = np.zeros(len(df))
        cumulative_negative = np.zeros(len(df))
        width = df.index.to_series().diff().dropna().min()  # Minimum time difference

        for i, column in enumerate(df.columns):
            positive_values = np.clip(df[column], 0, None)  # Keep only positive values
            negative_values = np.clip(df[column], None, 0)  # Keep only negative values
            # Plot positive bars
            ax.bar(
                df.index,
                positive_values,
                bottom=cumulative_positive,
                color=processed_colors[i],
                label=column,
                width=width,
                align='center',
                **plot_kwargs,
            )
            cumulative_positive += positive_values.values
            # Plot negative bars
            ax.bar(
                df.index,
                negative_values,
                bottom=cumulative_negative,
                color=processed_colors[i],
                label='',  # No label for negative bars
                width=width,
                align='center',
                **plot_kwargs,
            )
            cumulative_negative += negative_values.values

    elif mode == 'line':
        for i, column in enumerate(df.columns):
            ax.step(df.index, df[column], where='post', color=processed_colors[i], label=column, **plot_kwargs)

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


def reshape_data_for_heatmap(
    data: xr.DataArray,
    reshape_time: tuple[Literal['YS', 'MS', 'W', 'D', 'h', '15min', 'min'], Literal['W', 'D', 'h', '15min', 'min']]
    | Literal['auto']
    | None = 'auto',
    facet_by: str | list[str] | None = None,
    animate_by: str | None = None,
    fill: Literal['ffill', 'bfill'] | None = 'ffill',
) -> xr.DataArray:
    """
    Reshape data for heatmap visualization, handling time dimension intelligently.

    This function decides whether to reshape the 'time' dimension based on the reshape_time parameter:
    - 'auto': Automatically reshapes if only 'time' dimension would remain for heatmap
    - Tuple: Explicitly reshapes time with specified parameters
    - None: No reshaping (returns data as-is)

    All non-time dimensions are preserved during reshaping.

    Args:
        data: DataArray to reshape for heatmap visualization.
        reshape_time: Reshaping configuration:
                     - 'auto' (default): Auto-reshape if needed based on facet_by/animate_by
                     - Tuple (timeframes, timesteps_per_frame): Explicit time reshaping
                     - None: No reshaping
        facet_by: Dimension(s) used for faceting (used in 'auto' decision).
        animate_by: Dimension used for animation (used in 'auto' decision).
        fill: Method to fill missing values: 'ffill' or 'bfill'. Default is 'ffill'.

    Returns:
        Reshaped DataArray. If time reshaping is applied, 'time' dimension is replaced
        by 'timestep' and 'timeframe'. All other dimensions are preserved.

    Examples:
        Auto-reshaping:

        ```python
        # Will auto-reshape because only 'time' remains after faceting/animation
        data = reshape_data_for_heatmap(data, reshape_time='auto', facet_by='scenario', animate_by='period')
        ```

        Explicit reshaping:

        ```python
        # Explicitly reshape to daily pattern
        data = reshape_data_for_heatmap(data, reshape_time=('D', 'h'))
        ```

        No reshaping:

        ```python
        # Keep data as-is
        data = reshape_data_for_heatmap(data, reshape_time=None)
        ```
    """
    # If no time dimension, return data as-is
    if 'time' not in data.dims:
        return data

    # Handle None (disabled) - return data as-is
    if reshape_time is None:
        return data

    # Determine timeframes and timesteps_per_frame based on reshape_time parameter
    if reshape_time == 'auto':
        # Check if we need automatic time reshaping
        facet_dims_used = []
        if facet_by:
            facet_dims_used = [facet_by] if isinstance(facet_by, str) else list(facet_by)
        if animate_by:
            facet_dims_used.append(animate_by)

        # Get dimensions that would remain for heatmap
        potential_heatmap_dims = [dim for dim in data.dims if dim not in facet_dims_used]

        # Auto-reshape if only 'time' dimension remains
        if len(potential_heatmap_dims) == 1 and potential_heatmap_dims[0] == 'time':
            logger.debug(
                "Auto-applying time reshaping: Only 'time' dimension remains after faceting/animation. "
                "Using default timeframes='D' and timesteps_per_frame='h'. "
                "To customize, use reshape_time=('D', 'h') or disable with reshape_time=None."
            )
            timeframes, timesteps_per_frame = 'D', 'h'
        else:
            # No reshaping needed
            return data
    elif isinstance(reshape_time, tuple):
        # Explicit reshaping
        timeframes, timesteps_per_frame = reshape_time
    else:
        raise ValueError(f"reshape_time must be 'auto', a tuple like ('D', 'h'), or None. Got: {reshape_time}")

    # Validate that time is datetime
    if not np.issubdtype(data.coords['time'].dtype, np.datetime64):
        raise ValueError(f'Time dimension must be datetime-based, got {data.coords["time"].dtype}')

    # Define formats for different combinations
    formats = {
        ('YS', 'W'): ('%Y', '%W'),
        ('YS', 'D'): ('%Y', '%j'),  # day of year
        ('YS', 'h'): ('%Y', '%j %H:00'),
        ('MS', 'D'): ('%Y-%m', '%d'),  # day of month
        ('MS', 'h'): ('%Y-%m', '%d %H:00'),
        ('W', 'D'): ('%Y-w%W', '%w_%A'),  # week and day of week
        ('W', 'h'): ('%Y-w%W', '%w_%A %H:00'),
        ('D', 'h'): ('%Y-%m-%d', '%H:00'),  # Day and hour
        ('D', '15min'): ('%Y-%m-%d', '%H:%M'),  # Day and minute
        ('h', '15min'): ('%Y-%m-%d %H:00', '%M'),  # minute of hour
        ('h', 'min'): ('%Y-%m-%d %H:00', '%M'),  # minute of hour
    }

    format_pair = (timeframes, timesteps_per_frame)
    if format_pair not in formats:
        raise ValueError(f'{format_pair} is not a valid format. Choose from {list(formats.keys())}')
    period_format, step_format = formats[format_pair]

    # Check if resampling is needed
    if data.sizes['time'] > 1:
        # Use NumPy for more efficient timedelta computation
        time_values = data.coords['time'].values  # Already numpy datetime64[ns]
        # Calculate differences and convert to minutes
        time_diffs = np.diff(time_values).astype('timedelta64[s]').astype(float) / 60.0
        if time_diffs.size > 0:
            min_time_diff_min = np.nanmin(time_diffs)
            time_intervals = {'min': 1, '15min': 15, 'h': 60, 'D': 24 * 60, 'W': 7 * 24 * 60}
            if time_intervals[timesteps_per_frame] > min_time_diff_min:
                logger.warning(
                    f'Resampling data from {min_time_diff_min:.2f} min to '
                    f'{time_intervals[timesteps_per_frame]:.2f} min. Mean values are displayed.'
                )

    # Resample along time dimension
    resampled = data.resample(time=timesteps_per_frame).mean()

    # Apply fill if specified
    if fill == 'ffill':
        resampled = resampled.ffill(dim='time')
    elif fill == 'bfill':
        resampled = resampled.bfill(dim='time')

    # Create period and step labels
    time_values = pd.to_datetime(resampled.coords['time'].values)
    period_labels = time_values.strftime(period_format)
    step_labels = time_values.strftime(step_format)

    # Handle special case for weekly day format
    if '%w_%A' in step_format:
        step_labels = pd.Series(step_labels).replace('0_Sunday', '7_Sunday').values

    # Add period and step as coordinates
    resampled = resampled.assign_coords(
        {
            'timeframe': ('time', period_labels),
            'timestep': ('time', step_labels),
        }
    )

    # Convert to multi-index and unstack
    resampled = resampled.set_index(time=['timeframe', 'timestep'])
    result = resampled.unstack('time')

    # Ensure timestep and timeframe come first in dimension order
    # Get other dimensions
    other_dims = [d for d in result.dims if d not in ['timestep', 'timeframe']]

    # Reorder: timestep, timeframe, then other dimensions
    result = result.transpose('timestep', 'timeframe', *other_dims)

    return result


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


def preprocess_data_for_pie(
    data: xr.Dataset | pd.DataFrame | pd.Series,
    lower_percentage_threshold: float = 5.0,
) -> pd.Series:
    """
    Preprocess data for pie chart display.

    Groups items that are individually below the threshold percentage into an "Other" category.
    Converts various input types to a pandas Series for uniform handling.

    Args:
        data: Input data (xarray Dataset, DataFrame, or Series)
        lower_percentage_threshold: Percentage threshold - items below this are grouped into "Other"

    Returns:
        Processed pandas Series with small items grouped into "Other"
    """
    # Convert to Series
    if isinstance(data, xr.Dataset):
        # Sum all dimensions for each variable to get total values
        values = {}
        for var in data.data_vars:
            var_data = data[var]
            if len(var_data.dims) > 0:
                total_value = float(var_data.sum().item())
            else:
                total_value = float(var_data.item())

            # Handle negative values
            if total_value < 0:
                logger.warning(f'Negative value for {var}: {total_value}. Using absolute value.')
                total_value = abs(total_value)

            values[var] = total_value

        series = pd.Series(values)

    elif isinstance(data, pd.DataFrame):
        # Sum across all columns if DataFrame
        series = data.sum(axis=0)
        # Handle negative values
        negative_mask = series < 0
        if negative_mask.any():
            logger.warning(f'Negative values found: {series[negative_mask].to_dict()}. Using absolute values.')
            series = series.abs()

    else:  # pd.Series
        series = data.copy()
        # Handle negative values
        negative_mask = series < 0
        if negative_mask.any():
            logger.warning(f'Negative values found: {series[negative_mask].to_dict()}. Using absolute values.')
            series = series.abs()

    # Only keep positive values
    series = series[series > 0]

    if series.empty or lower_percentage_threshold <= 0:
        return series

    # Calculate percentages
    total = series.sum()
    percentages = (series / total) * 100

    # Find items below and above threshold
    below_threshold = series[percentages < lower_percentage_threshold]
    above_threshold = series[percentages >= lower_percentage_threshold]

    # Only group if there are at least 2 items below threshold
    if len(below_threshold) > 1:
        # Create new series with items above threshold + "Other"
        result = above_threshold.copy()
        result['Other'] = below_threshold.sum()
        return result

    return series


def dual_pie_with_plotly(
    data_left: xr.Dataset | pd.DataFrame | pd.Series,
    data_right: xr.Dataset | pd.DataFrame | pd.Series,
    colors: ColorType = 'viridis',
    title: str = '',
    subtitles: tuple[str, str] = ('Left Chart', 'Right Chart'),
    legend_title: str = '',
    hole: float = 0.2,
    lower_percentage_group: float = 5.0,
    text_info: str = 'percent+label',
    text_position: str = 'inside',
    hover_template: str = '%{label}: %{value} (%{percent})',
) -> go.Figure:
    """
    Create two pie charts side by side with Plotly.

    Args:
        data_left: Data for the left pie chart. Variables are summed across all dimensions.
        data_right: Data for the right pie chart. Variables are summed across all dimensions.
        colors: Color specification (colorscale name, list of colors, or dict mapping)
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
        Plotly Figure object
    """
    # Preprocess data to Series
    left_series = preprocess_data_for_pie(data_left, lower_percentage_group)
    right_series = preprocess_data_for_pie(data_right, lower_percentage_group)

    # Extract labels and values
    left_labels = left_series.index.tolist()
    left_values = left_series.values.tolist()

    right_labels = right_series.index.tolist()
    right_values = right_series.values.tolist()

    # Get all unique labels for consistent coloring
    all_labels = sorted(set(left_labels) | set(right_labels))

    # Create color map
    color_map = ColorProcessor(engine='plotly').process_colors(colors, all_labels, return_mapping=True)

    # Create figure
    fig = go.Figure()

    # Add left pie
    if left_labels:
        fig.add_trace(
            go.Pie(
                labels=left_labels,
                values=left_values,
                name=subtitles[0],
                marker=dict(colors=[color_map.get(label, '#636EFA') for label in left_labels]),
                hole=hole,
                textinfo=text_info,
                textposition=text_position,
                hovertemplate=hover_template,
                domain=dict(x=[0, 0.48]),
            )
        )

    # Add right pie
    if right_labels:
        fig.add_trace(
            go.Pie(
                labels=right_labels,
                values=right_values,
                name=subtitles[1],
                marker=dict(colors=[color_map.get(label, '#636EFA') for label in right_labels]),
                hole=hole,
                textinfo=text_info,
                textposition=text_position,
                hovertemplate=hover_template,
                domain=dict(x=[0.52, 1]),
            )
        )

    # Update layout
    fig.update_layout(
        title=title,
        legend_title=legend_title,
        margin=dict(t=80, b=50, l=30, r=30),
    )

    return fig


def dual_pie_with_matplotlib(
    data_left: xr.Dataset | pd.DataFrame | pd.Series,
    data_right: xr.Dataset | pd.DataFrame | pd.Series,
    colors: ColorType = 'viridis',
    title: str = '',
    subtitles: tuple[str, str] = ('Left Chart', 'Right Chart'),
    legend_title: str = '',
    hole: float = 0.2,
    lower_percentage_group: float = 5.0,
    figsize: tuple[int, int] = (14, 7),
) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Create two pie charts side by side with Matplotlib.

    Args:
        data_left: Data for the left pie chart.
        data_right: Data for the right pie chart.
        colors: Color specification (colormap name, list of colors, or dict mapping)
        title: The main title of the plot.
        subtitles: Tuple containing the subtitles for (left, right) charts.
        legend_title: The title for the legend.
        hole: Size of the hole in the center for creating donut charts (0.0 to 1.0).
        lower_percentage_group: Whether to group small segments (below percentage) into an "Other" category.
        figsize: The size of the figure (width, height) in inches.

    Returns:
        Tuple of (Figure, list of Axes)
    """
    # Preprocess data to Series
    left_series = preprocess_data_for_pie(data_left, lower_percentage_group)
    right_series = preprocess_data_for_pie(data_right, lower_percentage_group)

    # Extract labels and values
    left_labels = left_series.index.tolist()
    left_values = left_series.values.tolist()

    right_labels = right_series.index.tolist()
    right_values = right_series.values.tolist()

    # Get all unique labels for consistent coloring
    all_labels = sorted(set(left_labels) | set(right_labels))

    # Create color map
    color_map = ColorProcessor(engine='matplotlib').process_colors(colors, all_labels, return_mapping=True)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    def draw_pie(ax, labels, values, subtitle):
        """Draw a single pie chart."""
        if not labels:
            ax.set_title(subtitle)
            ax.axis('off')
            return

        chart_colors = [color_map[label] for label in labels]

        # Draw pie
        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            colors=chart_colors,
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops=dict(width=1 - hole) if hole > 0 else None,
        )

        # Style text
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_color('white')
            autotext.set_weight('bold')

        ax.set_aspect('equal')
        ax.set_title(subtitle, fontsize=14, pad=20)

    # Draw both pies
    draw_pie(axes[0], left_labels, left_values, subtitles[0])
    draw_pie(axes[1], right_labels, right_values, subtitles[1])

    # Add main title
    if title:
        fig.suptitle(title, fontsize=16, y=0.98)

    # Create unified legend
    if left_labels or right_labels:
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[label], markersize=10)
            for label in all_labels
        ]

        fig.legend(
            handles=handles,
            labels=all_labels,
            title=legend_title,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.02),
            ncol=min(len(all_labels), 5),
        )

        fig.subplots_adjust(bottom=0.15)

    fig.tight_layout()

    return fig, axes


def heatmap_with_plotly(
    data: xr.DataArray,
    colors: ColorType = 'viridis',
    title: str = '',
    facet_by: str | list[str] | None = None,
    animate_by: str | None = None,
    facet_cols: int = 3,
    reshape_time: tuple[Literal['YS', 'MS', 'W', 'D', 'h', '15min', 'min'], Literal['W', 'D', 'h', '15min', 'min']]
    | Literal['auto']
    | None = 'auto',
    fill: Literal['ffill', 'bfill'] | None = 'ffill',
    **imshow_kwargs: Any,
) -> go.Figure:
    """
    Plot a heatmap visualization using Plotly's imshow with faceting and animation support.

    This function creates heatmap visualizations from xarray DataArrays, supporting
    multi-dimensional data through faceting (subplots) and animation. It automatically
    handles dimension reduction and data reshaping for optimal heatmap display.

    Automatic Time Reshaping:
        If only the 'time' dimension remains after faceting/animation (making the data 1D),
        the function automatically reshapes time into a 2D format using default values
        (timeframes='D', timesteps_per_frame='h'). This creates a daily pattern heatmap
        showing hours vs days.

    Args:
        data: An xarray DataArray containing the data to visualize. Should have at least
              2 dimensions, or a 'time' dimension that can be reshaped into 2D.
        colors: Color specification (colormap name, list, or dict). Common options:
                'viridis', 'plasma', 'RdBu', 'portland'.
        title: The main title of the heatmap.
        facet_by: Dimension to create facets for. Creates a subplot grid.
                  Can be a single dimension name or list (only first dimension used).
                  Note: px.imshow only supports single-dimension faceting.
                  If the dimension doesn't exist in the data, it will be silently ignored.
        animate_by: Dimension to animate over. Creates animation frames.
                    If the dimension doesn't exist in the data, it will be silently ignored.
        facet_cols: Number of columns in the facet grid (used with facet_by).
        reshape_time: Time reshaping configuration:
                     - 'auto' (default): Automatically applies ('D', 'h') if only 'time' dimension remains
                     - Tuple like ('D', 'h'): Explicit time reshaping (days vs hours)
                     - None: Disable time reshaping (will error if only 1D time data)
        fill: Method to fill missing values when reshaping time: 'ffill' or 'bfill'. Default is 'ffill'.
        **imshow_kwargs: Additional keyword arguments to pass to plotly.express.imshow.
                        Common options include:
                        - aspect: 'auto', 'equal', or a number for aspect ratio
                        - zmin, zmax: Minimum and maximum values for color scale
                        - labels: Dict to customize axis labels

    Returns:
        A Plotly figure object containing the heatmap visualization.

    Examples:
        Simple heatmap:

        ```python
        fig = heatmap_with_plotly(data_array, colors='RdBu', title='Temperature Map')
        ```

        Facet by scenario:

        ```python
        fig = heatmap_with_plotly(data_array, facet_by='scenario', facet_cols=2)
        ```

        Animate by period:

        ```python
        fig = heatmap_with_plotly(data_array, animate_by='period')
        ```

        Automatic time reshaping (when only time dimension remains):

        ```python
        # Data with dims ['time', 'scenario', 'period']
        # After faceting and animation, only 'time' remains -> auto-reshapes to (timestep, timeframe)
        fig = heatmap_with_plotly(data_array, facet_by='scenario', animate_by='period')
        ```

        Explicit time reshaping:

        ```python
        fig = heatmap_with_plotly(data_array, facet_by='scenario', animate_by='period', reshape_time=('W', 'D'))
        ```
    """
    # Handle empty data
    if data.size == 0:
        return go.Figure()

    # Apply time reshaping using the new unified function
    data = reshape_data_for_heatmap(
        data, reshape_time=reshape_time, facet_by=facet_by, animate_by=animate_by, fill=fill
    )

    # Get available dimensions
    available_dims = list(data.dims)

    # Validate and filter facet_by dimensions
    if facet_by is not None:
        if isinstance(facet_by, str):
            if facet_by not in available_dims:
                logger.debug(
                    f"Dimension '{facet_by}' not found in data. Available dimensions: {available_dims}. "
                    f'Ignoring facet_by parameter.'
                )
                facet_by = None
        elif isinstance(facet_by, list):
            missing_dims = [dim for dim in facet_by if dim not in available_dims]
            facet_by = [dim for dim in facet_by if dim in available_dims]
            if missing_dims:
                logger.debug(
                    f'Dimensions {missing_dims} not found in data. Available dimensions: {available_dims}. '
                    f'Using only existing dimensions: {facet_by if facet_by else "none"}.'
                )
            if len(facet_by) == 0:
                facet_by = None

    # Validate animate_by dimension
    if animate_by is not None and animate_by not in available_dims:
        logger.debug(
            f"Dimension '{animate_by}' not found in data. Available dimensions: {available_dims}. "
            f'Ignoring animate_by parameter.'
        )
        animate_by = None

    # Determine which dimensions are used for faceting/animation
    facet_dims = []
    if facet_by:
        facet_dims = [facet_by] if isinstance(facet_by, str) else facet_by
    if animate_by:
        facet_dims.append(animate_by)

    # Get remaining dimensions for the heatmap itself
    heatmap_dims = [dim for dim in available_dims if dim not in facet_dims]

    if len(heatmap_dims) < 2:
        # Handle single-dimension case by adding variable name as a dimension
        if len(heatmap_dims) == 1:
            # Get the variable name, or use a default
            var_name = data.name if data.name else 'value'

            # Expand the DataArray by adding a new dimension with the variable name
            data = data.expand_dims({'variable': [var_name]})

            # Update available dimensions
            available_dims = list(data.dims)
            heatmap_dims = [dim for dim in available_dims if dim not in facet_dims]

            logger.debug(f'Only 1 dimension remaining for heatmap. Added variable dimension: {var_name}')
        else:
            # No dimensions at all - cannot create a heatmap
            logger.error(
                f'Heatmap requires at least 1 dimension. '
                f'After faceting/animation, {len(heatmap_dims)} dimension(s) remain: {heatmap_dims}'
            )
            return go.Figure()

    # Setup faceting parameters for Plotly Express
    # Note: px.imshow only supports facet_col, not facet_row
    facet_col_param = None
    if facet_by:
        if isinstance(facet_by, str):
            facet_col_param = facet_by
        elif len(facet_by) == 1:
            facet_col_param = facet_by[0]
        elif len(facet_by) >= 2:
            # px.imshow doesn't support facet_row, so we can only facet by one dimension
            # Use the first dimension and warn about the rest
            facet_col_param = facet_by[0]
            logger.warning(
                f'px.imshow only supports faceting by a single dimension. '
                f'Using {facet_by[0]} for faceting. Dimensions {facet_by[1:]} will be ignored. '
                f'Consider using animate_by for additional dimensions.'
            )

    # Create the imshow plot - px.imshow can work directly with xarray DataArrays
    common_args = {
        'img': data,
        'color_continuous_scale': colors if isinstance(colors, str) else 'viridis',
        'title': title,
    }

    # Add faceting if specified
    if facet_col_param:
        common_args['facet_col'] = facet_col_param
        if facet_cols:
            common_args['facet_col_wrap'] = facet_cols

    # Add animation if specified
    if animate_by:
        common_args['animation_frame'] = animate_by

    # Merge in additional imshow kwargs
    common_args.update(imshow_kwargs)

    try:
        fig = px.imshow(**common_args)
    except Exception as e:
        logger.error(f'Error creating imshow plot: {e}. Falling back to basic heatmap.')
        # Fallback: create a simple heatmap without faceting
        fallback_args = {
            'img': data.values,
            'color_continuous_scale': colors if isinstance(colors, str) else 'viridis',
            'title': title,
        }
        fallback_args.update(imshow_kwargs)
        fig = px.imshow(**fallback_args)

    return fig


def heatmap_with_matplotlib(
    data: xr.DataArray,
    colors: ColorType = 'viridis',
    title: str = '',
    figsize: tuple[float, float] = (12, 6),
    reshape_time: tuple[Literal['YS', 'MS', 'W', 'D', 'h', '15min', 'min'], Literal['W', 'D', 'h', '15min', 'min']]
    | Literal['auto']
    | None = 'auto',
    fill: Literal['ffill', 'bfill'] | None = 'ffill',
    vmin: float | None = None,
    vmax: float | None = None,
    imshow_kwargs: dict[str, Any] | None = None,
    cbar_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a heatmap visualization using Matplotlib's imshow.

    This function creates a basic 2D heatmap from an xarray DataArray using matplotlib's
    imshow function. For multi-dimensional data, only the first two dimensions are used.

    Args:
        data: An xarray DataArray containing the data to visualize. Should have at least
              2 dimensions. If more than 2 dimensions exist, additional dimensions will
              be reduced by taking the first slice.
        colors: Color specification. Should be a colormap name (e.g., 'turbo', 'RdBu').
        title: The title of the heatmap.
        figsize: The size of the figure (width, height) in inches.
        reshape_time: Time reshaping configuration:
                     - 'auto' (default): Automatically applies ('D', 'h') if only 'time' dimension
                     - Tuple like ('D', 'h'): Explicit time reshaping (days vs hours)
                     - None: Disable time reshaping
        fill: Method to fill missing values when reshaping time: 'ffill' or 'bfill'. Default is 'ffill'.
        vmin: Minimum value for color scale. If None, uses data minimum.
        vmax: Maximum value for color scale. If None, uses data maximum.
        imshow_kwargs: Optional dict of parameters to pass to ax.imshow().
                      Use this to customize image properties (e.g., interpolation, aspect).
        cbar_kwargs: Optional dict of parameters to pass to plt.colorbar().
                    Use this to customize colorbar properties (e.g., orientation, label).
        **kwargs: Additional keyword arguments passed to ax.imshow().
                 Common options include:
                 - interpolation: 'nearest', 'bilinear', 'bicubic', etc.
                 - alpha: Transparency level (0-1)
                 - extent: [left, right, bottom, top] for axis limits

    Returns:
        A tuple containing the Matplotlib figure and axes objects used for the plot.

    Notes:
        - Matplotlib backend doesn't support faceting or animation. Use plotly engine for those features.
        - The y-axis is automatically inverted to display data with origin at top-left.
        - A colorbar is added to show the value scale.

    Examples:
        ```python
        fig, ax = heatmap_with_matplotlib(data_array, colors='RdBu', title='Temperature')
        plt.savefig('heatmap.png')
        ```

        Time reshaping:

        ```python
        fig, ax = heatmap_with_matplotlib(data_array, reshape_time=('D', 'h'))
        ```
    """
    # Initialize kwargs if not provided
    if imshow_kwargs is None:
        imshow_kwargs = {}
    if cbar_kwargs is None:
        cbar_kwargs = {}

    # Merge any additional kwargs into imshow_kwargs
    # This allows users to pass imshow options directly
    imshow_kwargs.update(kwargs)

    # Handle empty data
    if data.size == 0:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax

    # Apply time reshaping using the new unified function
    # Matplotlib doesn't support faceting/animation, so we pass None for those
    data = reshape_data_for_heatmap(data, reshape_time=reshape_time, facet_by=None, animate_by=None, fill=fill)

    # Handle single-dimension case by adding variable name as a dimension
    if isinstance(data, xr.DataArray) and len(data.dims) == 1:
        var_name = data.name if data.name else 'value'
        data = data.expand_dims({'variable': [var_name]})
        logger.debug(f'Only 1 dimension in data. Added variable dimension: {var_name}')

    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Extract data values
    # If data has more than 2 dimensions, we need to reduce it
    if isinstance(data, xr.DataArray):
        # Get the first 2 dimensions
        dims = list(data.dims)
        if len(dims) > 2:
            logger.warning(
                f'Data has {len(dims)} dimensions: {dims}. '
                f'Only the first 2 will be used for the heatmap. '
                f'Use the plotly engine for faceting/animation support.'
            )
            # Select only the first 2 dimensions by taking first slice of others
            selection = {dim: 0 for dim in dims[2:]}
            data = data.isel(selection)

        values = data.values
        x_labels = data.dims[1] if len(data.dims) > 1 else 'x'
        y_labels = data.dims[0] if len(data.dims) > 0 else 'y'
    else:
        values = data
        x_labels = 'x'
        y_labels = 'y'

    # Process colormap
    cmap = colors if isinstance(colors, str) else 'viridis'

    # Create the heatmap using imshow with user customizations
    imshow_defaults = {'cmap': cmap, 'aspect': 'auto', 'origin': 'upper', 'vmin': vmin, 'vmax': vmax}
    imshow_defaults.update(imshow_kwargs)  # User kwargs override defaults
    im = ax.imshow(values, **imshow_defaults)

    # Add colorbar with user customizations
    cbar_defaults = {'ax': ax, 'orientation': 'horizontal', 'pad': 0.1, 'aspect': 15, 'fraction': 0.05}
    cbar_defaults.update(cbar_kwargs)  # User kwargs override defaults
    cbar = plt.colorbar(im, **cbar_defaults)

    # Set colorbar label if not overridden by user
    if 'label' not in cbar_kwargs:
        cbar.set_label('Value')

    # Set labels and title
    ax.set_xlabel(str(x_labels).capitalize())
    ax.set_ylabel(str(y_labels).capitalize())
    ax.set_title(title)

    # Apply tight layout
    fig.tight_layout()

    return fig, ax


def export_figure(
    figure_like: go.Figure | tuple[plt.Figure, plt.Axes],
    default_path: pathlib.Path,
    default_filetype: str | None = None,
    user_path: pathlib.Path | None = None,
    show: bool = True,
    save: bool = False,
    dpi: int = 300,
) -> go.Figure | tuple[plt.Figure, plt.Axes]:
    """
    Export a figure to a file and or show it.

    Args:
        figure_like: The figure to export. Can be a Plotly figure or a tuple of Matplotlib figure and axes.
        default_path: The default file path if no user filename is provided.
        default_filetype: The default filetype if the path doesnt end with a filetype.
        user_path: An optional user-specified file path.
        show: Whether to display the figure. If None, uses CONFIG.Plotting.default_show (default: None).
        save: Whether to save the figure (default: False).
        dpi: DPI (dots per inch) for saving Matplotlib figures. If None, uses CONFIG.Plotting.default_dpi.

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
            fig.savefig(str(filename), dpi=dpi)
            plt.close(fig)  # Close figure to free memory

        return fig, ax

    raise TypeError(f'Figure type not supported: {type(figure_like)}')
