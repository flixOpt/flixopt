# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 09:43:09 2021
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden
"""
import logging
from typing import Literal, Tuple, Union, Optional, List

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline

logger = logging.getLogger('flixOpt')


def with_plotly(data: pd.DataFrame,
                mode: Literal['bar', 'line'] = 'bar',
                colors: Union[List[str], str] = 'viridis',
                fig: Optional[go.Figure] = None,
                show: bool = False) -> go.Figure:
    """
    Plot a DataFrame with Plotly, using either stacked bars or stepped lines.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing the data to plot, where the index represents
        time (e.g., hours), and each column represents a separate data series.
    mode : {'bar', 'line'}, default='bar'
        The plotting mode. Use 'bar' for stacked bar charts or 'line' for
        stepped lines.
    colors : List[str], str, default='viridis'
        A List of colors (as str) or a name of a colorscale (e.g., 'viridis', 'plasma') to use for
        coloring the data series.
    fig : go.Figure, optional
        A Plotly figure object to plot on. If not provided, a new figure
        will be created.

    Returns
    -------
    go.Figure
        A Plotly figure object containing the generated plot.

    Notes
    -----
    - If `mode` is 'bar', bars are stacked for each data series.
    - If `mode` is 'line', a stepped line is drawn for each data series.
    - The legend is positioned below the plot for a cleaner layout when many
      data series are present.

    Examples
    --------
    >>> fig = with_plotly(data, mode='bar', colorscale='plasma')
    >>> fig.show()
    """
    if isinstance(colors, str):
        colorscale = px.colors.get_colorscale(colors)
        colors = px.colors.sample_colorscale(
            colorscale,
            [i / (len(data.columns) - 1) for i in range(len(data.columns))] if len(data.columns) > 1 else [0])

    assert len(colors) == len(data.columns), (f'The number of colors does not match the provided data columns. '
                                              f'{len(colors)=}; {len(colors)=}')
    fig = fig if fig is not None else go.Figure()

    if mode == 'bar':
        for i, column in enumerate(data.columns):
            fig.add_trace(go.Bar(
                x=data.index,
                y=data[column],
                name=column,
                marker=dict(color=colors[i]),
            ))

        fig.update_layout(
            barmode='relative' if mode == 'bar' else None,
            bargap=0,  # No space between bars
            bargroupgap=0,  # No space between groups of bars
        )
    elif mode == 'line':
        for i, column in enumerate(data.columns):
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[column],
                mode='lines',
                name=column,
                line=dict(shape='hv', color=colors[i]),
            ))
    else:
        raise ValueError(f'mode must be either "bar" or "line"')

    # Update layout for better aesthetics
    fig.update_layout(
        yaxis=dict(
            showgrid=True,  # Enable grid lines on the y-axis
            gridcolor='lightgrey',  # Customize grid line color
            gridwidth=0.5,  # Customize grid line width
        ),
        xaxis=dict(
            title='Time in h',
            showgrid=True,  # Enable grid lines on the x-axis
            gridcolor='lightgrey',  # Customize grid line color
            gridwidth=0.5  # Customize grid line width
        ),
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper background
        font=dict(size=14),  # Increase font size for better readability
        legend=dict(
            orientation="h",  # Horizontal legend
            yanchor="bottom",
            y=-0.3,  # Adjusts how far below the plot it appears
            xanchor="center",
            x=0.5,
            title_text=None  # Removes legend title for a cleaner look
        )
    )

    if show:
        plotly.offline.plot(fig)
    return fig


def with_matplotlib(data: pd.DataFrame,
                    mode: Literal['bar', 'line'] = 'bar',
                    colors: Union[List[str], str] = 'viridis',
                    figsize: Tuple[int, int] = (12, 6),
                    fig: Optional[plt.Figure] = None,
                    ax: Optional[plt.Axes] = None,
                    show: bool = False) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a DataFrame with Matplotlib using stacked bars or stepped lines.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing the data to plot. The index should represent
        time (e.g., hours), and each column represents a separate data series.
    mode : {'bar', 'line'}, default='bar'
        Plotting mode. Use 'bar' for stacked bar charts or 'line' for stepped lines.
    colors : List[str], str, default='viridis'
        A List of colors (as str) or a name of a colorscale (e.g., 'viridis', 'plasma') to use for
        coloring the data series.
    figsize: Tuple[int, int], optional
        Specify the size of the figure
    fig : plt.Figure, optional
        A Matplotlib figure object to plot on. If not provided, a new figure
        will be created.
    ax : plt.Axes, optional
        A Matplotlib axes object to plot on. If not provided, a new axes
        will be created.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        A tuple containing the Matplotlib figure and axes objects used for the plot.

    Notes
    -----
    - If `mode` is 'bar', bars are stacked for both positive and negative values.
      Negative values are stacked separately without extra labels in the legend.
    - If `mode` is 'line', stepped lines are drawn for each data series.
    - The legend is placed below the plot to accommodate multiple data series.

    Examples
    --------
    >>> fig, ax = with_matplotlib(data, mode='bar', colorscale='plasma')
    >>> plt.show()
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if isinstance(colors, str):
        cmap = plt.get_cmap(colors, len(data.columns))
        colors = [cmap(i) for i in range(len(data.columns))]
    assert len(colors) == len(data.columns), (f'The number of colors does not match the provided data columns. '
                                              f'{len(colors)=}; {len(colors)=}')

    if mode == 'bar':
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
                color=colors[i],
                label=column,
                width=width,
                align='edge'
            )
            cumulative_positive += positive_values.values
            # Plot negative bars
            ax.bar(
                data.index,
                negative_values,
                bottom=cumulative_negative,
                color=colors[i],
                label="",  # No label for negative bars
                width=width,
                align='edge'
            )
            cumulative_negative += negative_values.values

    elif mode == 'line':
        for i, column in enumerate(data.columns):
            ax.step(
                data.index,
                data[column],
                where='post',
                color=colors[i],
                label=column
            )
    else:
        raise ValueError(f"mode must be either 'bar' or 'line'")

    # Aesthetics
    ax.set_xlabel('Time in h', fontsize=14)
    ax.grid(color='lightgrey', linestyle='-', linewidth=0.5)
    ax.legend(
        loc='upper center',  # Place legend at the bottom center
        bbox_to_anchor=(0.5, -0.15),  # Adjust the position to fit below plot
        ncol=5,
        frameon=False  # Remove box around legend
    )
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def heat_map_matplotlib(data: pd.DataFrame,
                        color_map: str = 'viridis',
                        figsize: Tuple[float, float] = (12, 6)) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots a DataFrame as a heatmap using Matplotlib. The columns of the DataFrame will be displayed on the x-axis,
    the index will be displayed on the y-axis, and the values will represent the 'heat' intensity in the plot.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing the data to be visualized. The index will be used for the y-axis, and columns will be used for the x-axis.
        The values in the DataFrame will be represented as colors in the heatmap.
    color_map : str, optional
        The colormap to use for the heatmap. Default is 'viridis'. Matplotlib supports various colormaps like 'plasma', 'inferno', 'cividis', etc.
    figsize : tuple of float, optional
        The size of the figure to create. Default is (12, 6), which results in a width of 12 inches and a height of 6 inches.

    Returns
    -------
    tuple of (plt.Figure, plt.Axes)
        A tuple containing the Matplotlib `Figure` and `Axes` objects. The `Figure` contains the overall plot, while the `Axes` is the area
        where the heatmap is drawn. These can be used for further customization or saving the plot to a file.

    Notes
    -----
    - The y-axis is flipped so that the first row of the DataFrame is displayed at the top of the plot.
    - The color scale is normalized based on the minimum and maximum values in the DataFrame.
    - The x-axis labels (periods) are placed at the top of the plot.
    - The colorbar is added horizontally at the bottom of the plot, with a label.
    """

    # Get the min and max values for color normalization
    color_bar_min, color_bar_max = data.min().min(), data.max().max()

    # Create the heatmap plot
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.pcolormesh(data.values, cmap=color_map)
    ax.invert_yaxis()  # Flip the y-axis to start at the top

    # Adjust ticks and labels for x and y axes
    ax.set_xticks(np.arange(len(data.columns)) + 0.5)
    ax.set_xticklabels(data.columns, ha='center')
    ax.set_yticks(np.arange(len(data.index)) + 0.5)
    ax.set_yticklabels(data.index, va='center')

    # Add labels to the axes
    ax.set_xlabel("Period", ha='center')
    ax.set_ylabel("Step", va='center')

    # Position x-axis labels at the top
    ax.xaxis.set_label_position("top")
    ax.xaxis.set_ticks_position("top")

    # Add the colorbar
    sm1 = plt.cm.ScalarMappable(cmap=color_map, norm=plt.Normalize(vmin=color_bar_min, vmax=color_bar_max))
    sm1._A = []
    cb1 = fig.colorbar(sm1, ax=ax, pad=0.12, aspect=15, fraction=0.2, orientation='horizontal')

    fig.tight_layout()
    return fig, ax


def heat_map_plotly(data: pd.DataFrame,
                    color_map: str = 'viridis',
                    categorical_labels: bool = True) -> go.Figure:
    """
    Plots a DataFrame as a heatmap using Plotly. The columns of the DataFrame will be mapped to the x-axis,
    and the index will be displayed on the y-axis. The values in the DataFrame will represent the 'heat' in the plot.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame with the data to be visualized. The index will be used for the y-axis, and columns will be used for the x-axis.
        The values in the DataFrame will be represented as colors in the heatmap.
    color_map : str, optional
        The color scale to use for the heatmap. Default is 'viridis'. Plotly supports various color scales like 'Cividis', 'Inferno', etc.
    categorical_labels : bool, optional
        If True, the x and y axes are treated as categorical data (i.e., the index and columns will not be interpreted as continuous data).
        Default is True. If False, the axes are treated as continuous, which may be useful for time series or numeric data.

    Returns
    -------
    go.Figure
        A Plotly figure object containing the heatmap. This can be further customized and saved
        or displayed using `fig.show()`.

    Notes
    -----
    The color bar is automatically scaled to the minimum and maximum values in the data.
    The y-axis is reversed to display the first row at the top.
    """

    color_bar_min, color_bar_max = data.min().min(), data.max().max()  # Min and max values for color scaling
    # Define the figure
    fig = go.Figure(data=go.Heatmap(
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
            y=0.1
        ),
    ))

    # Set axis labels and style
    fig.update_layout(
        xaxis=dict(title='Period', side='top', type='category' if categorical_labels else None),
        yaxis=dict(title='Step', autorange='reversed', type='category' if categorical_labels else None)
    )

    return fig


def reshape_to_2d(data_1d: np.ndarray, nr_of_steps_per_column: int) -> np.ndarray:
    """
    Reshapes a 1D numpy array into a 2D array suitable for plotting as a colormap.

    The reshaped array will have the number of rows corresponding to the steps per column
    (e.g., 24 hours per day) and columns representing time periods (e.g., days or months).

    Parameters
    ----------
    data_1d : np.ndarray
        A 1D numpy array with the data to reshape.

    nr_of_steps_per_column : int
        The number of steps (rows) per column in the resulting 2D array. For example,
        this could be 24 (for hours) or 31 (for days in a month).

    Returns
    -------
    np.ndarray
        The reshaped 2D array. Each internal array corresponds to one column, with the specified number of steps.
        Each column might represents a time period (e.g., day, month, etc.).
    """

    # Step 1: Ensure the input is a 1D array.
    if data_1d.ndim != 1:
        raise ValueError("Input must be a 1D array")

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
    padded_data = np.pad(data_1d, (0, cols * nr_of_steps_per_column - total_steps), mode='constant',
                         constant_values=np.nan)

    # Step 5: Reshape the padded data into a 2D array
    data_2d = padded_data.reshape(cols, nr_of_steps_per_column)

    return data_2d.T


def heat_map_data_from_df(df: pd.DataFrame,
                          periods: Literal['YS', 'MS', 'W', 'D', 'h', '15min', 'min'],
                          steps_per_period: Literal['W', 'D', 'h', '15min', 'min'],
                          fill: Optional[Literal['ffill', 'bfill']] = None) -> pd.DataFrame:
    """
    Reshapes a DataFrame with a DateTime index into a 2D array for heatmap plotting,
    based on a specified sample rate.
    If a non-valid combination of periods and steps per period is used, falls back to numerical indices

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame with a DateTime index containing the data to reshape.
    periods : str
        The time interval of each period (columns of the heatmap),
        such as 'YS' (year start), 'W' (weekly), 'D' (daily), 'h' (hourly) etc.
    steps_per_period : str
        The time interval within each period (rows in the heatmap),
        such as 'YS' (year start), 'W' (weekly), 'D' (daily), 'h' (hourly) etc.
    fill : str, optional
        Method to fill missing values: 'ffill' for forward fill or 'bfill' for backward fill.

    Returns
    -------
    pd.DataFrame
        A DataFrame suitable for heatmap plotting, with rows representing steps within each period
        and columns representing each period.
    """

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
        ('D', '15min'): ('%Y-%m-%d', '%H:%MM'),  # Day and hour
        ('h', '15min'): ('%Y-%m-%d %H:00', '%M'),  # minute of hour
        ('h', 'min'): ('%Y-%m-%d %H:00', '%M'),  # minute of hour
    }
    # Select the format based on the `periods` and `steps_per_period` combination
    format_pair = (periods, steps_per_period)
    assert format_pair in formats, f'{format_pair} is not a valid format. Choose from {list(formats.keys())}'
    period_format, step_format = formats[format_pair]

    df = df.sort_index()  # Ensure DataFrame is sorted by time index

    resampled_data = df.resample(steps_per_period).mean()  # Resample and fill any gaps with NaN

    if fill == 'ffill':  # Apply fill method if specified
        resampled_data = resampled_data.ffill()
    elif fill == 'bfill':
        resampled_data = resampled_data.bfill()

    resampled_data['period'] = resampled_data.index.strftime(period_format)
    resampled_data['step'] = resampled_data.index.strftime(step_format)
    if '%w_%A' in step_format:  # SHift index of strings to ensure proper sorting
        resampled_data['step'] = resampled_data['step'].apply(lambda x: x.replace('0_Sunday', '7_Sunday') if '0_Sunday' in x else x)

    # Pivot the table so periods are columns and steps are indices
    df_pivoted = resampled_data.pivot(columns='period', index='step', values='value')

    return df_pivoted
