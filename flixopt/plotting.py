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

import logging
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

from .color_processing import ColorType, process_colors
from .config import CONFIG

if TYPE_CHECKING:
    import pyvis

logger = logging.getLogger('flixopt')

# Define the colors for the 'portland' colorscale in matplotlib
_portland_colors = [
    [12 / 255, 51 / 255, 131 / 255],  # Dark blue
    [10 / 255, 136 / 255, 186 / 255],  # Light blue
    [242 / 255, 211 / 255, 56 / 255],  # Yellow
    [242 / 255, 143 / 255, 56 / 255],  # Orange
    [217 / 255, 30 / 255, 30 / 255],  # Red
]

# Check if the colorscale already exists before registering it
if hasattr(plt, 'colormaps'):  # Matplotlib >= 3.7
    registry = plt.colormaps
    if 'portland' not in registry:
        registry.register(mcolors.LinearSegmentedColormap.from_list('portland', _portland_colors))
else:  # Matplotlib < 3.7
    if 'portland' not in [c for c in plt.colormaps()]:
        plt.register_cmap(name='portland', cmap=mcolors.LinearSegmentedColormap.from_list('portland', _portland_colors))


PlottingEngine = Literal['plotly', 'matplotlib']
"""Identifier for the plotting engine to use."""


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


def with_plotly(
    data: xr.Dataset | pd.DataFrame | pd.Series,
    mode: Literal['stacked_bar', 'line', 'area', 'grouped_bar'] = 'stacked_bar',
    colors: ColorType | None = None,
    title: str = '',
    ylabel: str = '',
    xlabel: str = '',
    facet_by: str | list[str] | None = None,
    animate_by: str | None = None,
    facet_cols: int | None = None,
    shared_yaxes: bool = True,
    shared_xaxes: bool = True,
    **px_kwargs: Any,
) -> go.Figure:
    """
    Plot data with Plotly using facets (subplots) and/or animation for multidimensional data.

    Uses Plotly Express for convenient faceting and animation with automatic styling.

    Args:
        data: An xarray Dataset, pandas DataFrame, or pandas Series to plot.
        mode: The plotting mode. Use 'stacked_bar' for stacked bar charts, 'line' for lines,
              'area' for stacked area charts, or 'grouped_bar' for grouped bar charts.
        colors: Color specification (colorscale, list, or dict mapping labels to colors).
        title: The main title of the plot.
        ylabel: The label for the y-axis.
        xlabel: The label for the x-axis.
        facet_by: Dimension(s) to create facets for. Creates a subplot grid.
              Can be a single dimension name or list of dimensions (max 2 for facet_row and facet_col).
              If the dimension doesn't exist in the data, it will be silently ignored.
        animate_by: Dimension to animate over. Creates animation frames.
              If the dimension doesn't exist in the data, it will be silently ignored.
        facet_cols: Number of columns in the facet grid (used when facet_by is single dimension).
        shared_yaxes: Whether subplots share y-axes.
        shared_xaxes: Whether subplots share x-axes.
        **px_kwargs: Additional keyword arguments passed to the underlying Plotly Express function
                    (px.bar, px.line, px.area). These override default arguments if provided.
                    Examples: range_x=[0, 100], range_y=[0, 50], category_orders={...}, line_shape='linear'

    Returns:
        A Plotly figure object containing the faceted/animated plot. You can further customize
        the returned figure using Plotly's methods (e.g., fig.update_traces(), fig.update_layout()).

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

        Customize with Plotly Express kwargs:

        ```python
        fig = with_plotly(dataset, range_y=[0, 100], line_shape='linear')
        ```

        Further customize the returned figure:

        ```python
        fig = with_plotly(dataset, mode='line')
        fig.update_traces(line={'width': 5, 'dash': 'dot'})
        fig.update_layout(template='plotly_dark', width=1200, height=600)
        ```
    """
    if colors is None:
        colors = CONFIG.Plotting.default_qualitative_colorscale

    if mode not in ('stacked_bar', 'line', 'area', 'grouped_bar'):
        raise ValueError(f"'mode' must be one of {{'stacked_bar','line','area', 'grouped_bar'}}, got {mode!r}")

    # Apply CONFIG defaults if not explicitly set
    if facet_cols is None:
        facet_cols = CONFIG.Plotting.default_facet_cols

    # Ensure data is a Dataset and validate it
    data = _ensure_dataset(data)
    _validate_plotting_data(data, allow_empty=True)

    # Handle empty data
    if len(data.data_vars) == 0:
        logger.error('with_plotly() got an empty Dataset.')
        return go.Figure()

    # Handle all-scalar datasets (where all variables have no dimensions)
    # This occurs when all variables are scalar values with dims=()
    if all(len(data[var].dims) == 0 for var in data.data_vars):
        # Create a simple DataFrame with variable names as x-axis
        variables = list(data.data_vars.keys())
        values = [float(data[var].values) for var in data.data_vars]

        # Resolve colors
        color_discrete_map = process_colors(
            colors, variables, default_colorscale=CONFIG.Plotting.default_qualitative_colorscale
        )
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
        else:
            raise ValueError('"mode" must be one of "stacked_bar", "grouped_bar", "line", "area"')

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
    color_discrete_map = process_colors(
        colors, all_vars, default_colorscale=CONFIG.Plotting.default_qualitative_colorscale
    )

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

    # Add mode-specific defaults (before px_kwargs so they can be overridden)
    if mode in ('line', 'area'):
        common_args['line_shape'] = 'hv'  # Stepped lines by default

    # Allow callers to pass any px.* keyword args (e.g., category_orders, range_x/y, line_shape)
    # These will override the defaults set above
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
        fig = px.line(**common_args)
    elif mode == 'area':
        # Use Plotly Express to create the area plot (preserves animation, legends, faceting)
        fig = px.area(**common_args)

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

    return fig


def with_matplotlib(
    data: xr.Dataset | pd.DataFrame | pd.Series,
    mode: Literal['stacked_bar', 'line'] = 'stacked_bar',
    colors: ColorType | None = None,
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
            - A colorscale name (e.g., 'turbo', 'plasma')
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
    if colors is None:
        colors = CONFIG.Plotting.default_qualitative_colorscale

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
        color_discrete_map = process_colors(
            colors, variables, default_colorscale=CONFIG.Plotting.default_qualitative_colorscale
        )
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
    color_discrete_map = process_colors(
        colors, list(data.data_vars), default_colorscale=CONFIG.Plotting.default_qualitative_colorscale
    )

    # Convert Dataset to DataFrame for matplotlib plotting (naturally wide-form)
    df = data.to_dataframe()

    # Get colors in column order
    processed_colors = [color_discrete_map.get(str(col), '#808080') for col in df.columns]

    if mode == 'stacked_bar':
        cumulative_positive = np.zeros(len(df))
        cumulative_negative = np.zeros(len(df))

        # Robust bar width: handle datetime-like, numeric, and single-point indexes
        if len(df.index) > 1:
            delta = pd.Index(df.index).to_series().diff().dropna().min()
            if hasattr(delta, 'total_seconds'):  # datetime-like
                width = delta.total_seconds() / 86400.0  # Matplotlib date units = days
            else:
                width = float(delta)
        else:
            width = 0.8  # reasonable default for a single bar

        for i, column in enumerate(df.columns):
            # Fill NaNs to avoid breaking stacking math
            series = df[column].fillna(0)
            positive_values = np.clip(series, 0, None)  # Keep only positive values
            negative_values = np.clip(series, None, 0)  # Keep only negative values
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

            if not isinstance(path, pathlib.Path):
                raise RuntimeError('path must be set when show=True')
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
    colors: ColorType | None = None,
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
    if colors is None:
        colors = CONFIG.Plotting.default_qualitative_colorscale

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
    color_map = process_colors(colors, all_labels, default_colorscale=CONFIG.Plotting.default_qualitative_colorscale)

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
    colors: ColorType | None = None,
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
        colors: Color specification (colorscale name, list of colors, or dict mapping)
        title: The main title of the plot.
        subtitles: Tuple containing the subtitles for (left, right) charts.
        legend_title: The title for the legend.
        hole: Size of the hole in the center for creating donut charts (0.0 to 1.0).
        lower_percentage_group: Whether to group small segments (below percentage) into an "Other" category.
        figsize: The size of the figure (width, height) in inches.

    Returns:
        Tuple of (Figure, list of Axes)
    """
    if colors is None:
        colors = CONFIG.Plotting.default_qualitative_colorscale

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

    # Create color map (process_colors always returns a dict)
    color_map = process_colors(colors, all_labels, default_colorscale=CONFIG.Plotting.default_qualitative_colorscale)

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


def heatmap_with_plotly_v2(
    data: xr.DataArray,
    colors: ColorType | None = None,
    title: str = '',
    facet_col: str | None = None,
    animation_frame: str | None = None,
    facet_col_wrap: int | None = None,
    **imshow_kwargs: Any,
) -> go.Figure:
    """
    Plot a heatmap using Plotly's imshow.

    Data should be prepared with dims in order: (y_axis, x_axis, [facet_col], [animation_frame]).
    Use reshape_data_for_heatmap() to prepare time-series data before calling this.

    Args:
        data: DataArray with 2-4 dimensions. First two are heatmap axes.
        colors: Colorscale name ('viridis', 'plasma', etc.).
        title: Plot title.
        facet_col: Dimension name for subplot columns (3rd dim).
        animation_frame: Dimension name for animation (4th dim).
        facet_col_wrap: Max columns before wrapping (only if < n_facets).
        **imshow_kwargs: Additional args for px.imshow.

    Returns:
        Plotly Figure object.
    """
    if data.size == 0:
        return go.Figure()

    colors = colors or CONFIG.Plotting.default_sequential_colorscale
    facet_col_wrap = facet_col_wrap or CONFIG.Plotting.default_facet_cols

    imshow_args: dict[str, Any] = {
        'img': data,
        'color_continuous_scale': colors,
        'title': title,
        **imshow_kwargs,
    }

    if facet_col and facet_col in data.dims:
        imshow_args['facet_col'] = facet_col
        if facet_col_wrap < data.sizes[facet_col]:
            imshow_args['facet_col_wrap'] = facet_col_wrap

    if animation_frame and animation_frame in data.dims:
        imshow_args['animation_frame'] = animation_frame

    return px.imshow(**imshow_args)


def heatmap_with_plotly(
    data: xr.DataArray,
    colors: ColorType | None = None,
    title: str = '',
    facet_by: str | list[str] | None = None,
    animate_by: str | None = None,
    facet_cols: int | None = None,
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
        colors: Color specification (colorscale name, list, or dict). Common options:
                'turbo', 'plasma', 'RdBu', 'portland'.
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
        # Data with dims ['time', 'period','scenario']
        # After faceting and animation, only 'time' remains -> auto-reshapes to (timestep, timeframe)
        fig = heatmap_with_plotly(data_array, facet_by='scenario', animate_by='period')
        ```

        Explicit time reshaping:

        ```python
        fig = heatmap_with_plotly(data_array, facet_by='scenario', animate_by='period', reshape_time=('W', 'D'))
        ```
    """
    if colors is None:
        colors = CONFIG.Plotting.default_sequential_colorscale

    # Apply CONFIG defaults if not explicitly set
    if facet_cols is None:
        facet_cols = CONFIG.Plotting.default_facet_cols

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
        'color_continuous_scale': colors,
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
            'color_continuous_scale': colors,
            'title': title,
        }
        fallback_args.update(imshow_kwargs)
        fig = px.imshow(**fallback_args)

    return fig


def heatmap_with_matplotlib(
    data: xr.DataArray,
    colors: ColorType | None = None,
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
        colors: Color specification. Should be a colorscale name (e.g., 'turbo', 'RdBu').
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
    if colors is None:
        colors = CONFIG.Plotting.default_sequential_colorscale

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

    # Create the heatmap using imshow with user customizations
    imshow_defaults = {'cmap': colors, 'aspect': 'auto', 'origin': 'upper', 'vmin': vmin, 'vmax': vmax}
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
    show: bool | None = None,
    save: bool = False,
    dpi: int | None = None,
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
    # Apply CONFIG defaults if not explicitly set
    if show is None:
        show = CONFIG.Plotting.default_show

    if dpi is None:
        dpi = CONFIG.Plotting.default_dpi

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
            # Respect show and save flags (tests should set CONFIG.Plotting.default_show=False)
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
            # Only show if using interactive backend (tests should set CONFIG.Plotting.default_show=False)
            backend = matplotlib.get_backend().lower()
            is_interactive = backend not in {'agg', 'pdf', 'ps', 'svg', 'template'}

            if is_interactive:
                plt.show()

        if save:
            fig.savefig(str(filename), dpi=dpi)
            plt.close(fig)  # Close figure to free memory

        return fig, ax

    raise TypeError(f'Figure type not supported: {type(figure_like)}')
