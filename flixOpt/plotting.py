"""
This module contains the plotting functionality of the flixOpt framework.
It provides high level functions to plot data with plotly and matplotlib.
It's meant to be used in results.py, but is designed to be used by the end user as well.
"""

import logging
import pathlib
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline

if TYPE_CHECKING:
    import pyvis

logger = logging.getLogger('flixOpt')


def with_plotly(
    data: pd.DataFrame,
    mode: Literal['bar', 'line', 'area'] = 'area',
    colors: Union[List[str], str] = 'viridis',
    title: str = '',
    ylabel: str = '',
    fig: Optional[go.Figure] = None,
    show: bool = False,
    save: bool = False,
    path: Union[str, pathlib.Path] = 'temp-plot.html',
) -> go.Figure:
    """
    Plot a DataFrame with Plotly, using either stacked bars or stepped lines.

    Args:
        data: A DataFrame containing the data to plot, where the index represents time (e.g., hours), and each column represents a separate data series.
        mode: The plotting mode. Use 'bar' for stacked bar charts or 'line' for stepped lines.
        colors: A List of colors (as str) or a name of a colorscale (e.g., 'viridis', 'plasma') to use for coloring the data series.
        title: The title of the plot.
        ylabel: The label for the y-axis.
        fig: A Plotly figure object to plot on. If not provided, a new figure will be created.
        show: Wether to show the figure after creation. (This includes saving the figure)
        save: Wether to save the figure after creation (without showing)
        path: Path to save the figure.

    Returns:
        A Plotly figure object containing the generated plot.

    Notes:
        - If `mode` is 'bar', bars are stacked for each data series.
        - If `mode` is 'line', a stepped line is drawn for each data series.
        - The legend is positioned below the plot for a cleaner layout when many data series are present.

    Examples:
        >>> fig = with_plotly(data, mode='bar', colorscale='plasma')
        >>> fig.show()
    """
    assert mode in ['bar', 'line', 'area'], f"'mode' must be one of {['bar', 'line', 'area']}"
    if data.empty:
        return go.Figure()
    if isinstance(colors, str):
        colorscale = px.colors.get_colorscale(colors)
        colors = px.colors.sample_colorscale(
            colorscale,
            [i / (len(data.columns) - 1) for i in range(len(data.columns))] if len(data.columns) > 1 else [0],
        )

    assert len(colors) == len(data.columns), (
        f'The number of colors does not match the provided data columns. {len(colors)=}; {len(colors)=}'
    )
    fig = fig if fig is not None else go.Figure()

    if mode == 'bar':
        for i, column in enumerate(data.columns):
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data[column],
                    name=column,
                    marker=dict(color=colors[i]),
                )
            )

        fig.update_layout(
            barmode='relative' if mode == 'bar' else None,
            bargap=0,  # No space between bars
            bargroupgap=0,  # No space between groups of bars
        )
    elif mode == 'line':
        for i, column in enumerate(data.columns):
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[column],
                    mode='lines',
                    name=column,
                    line=dict(shape='hv', color=colors[i]),
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
            logger.warning(
                f'Data for plotting stacked lines contains columns with both positive and negative values:'
                f' {mixed_columns}. These can not be stacked, and are printed as simple lines'
            )

        colors_stacked = {column: colors[i] for i, column in enumerate(data.columns)}

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
            title='Time in h',
            showgrid=True,  # Enable grid lines on the x-axis
            gridcolor='lightgrey',  # Customize grid line color
            gridwidth=0.5,  # Customize grid line width
        ),
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper background
        font=dict(size=14),  # Increase font size for better readability
        legend=dict(
            orientation='h',  # Horizontal legend
            yanchor='bottom',
            y=-0.3,  # Adjusts how far below the plot it appears
            xanchor='center',
            x=0.5,
            title_text=None,  # Removes legend title for a cleaner look
        ),
    )

    if isinstance(path, pathlib.Path):
        path = path.as_posix()
    if show:
        plotly.offline.plot(fig, filename=path)
    elif save:  # If show, the file is saved anyway
        fig.write_html(path)
    return fig


def with_matplotlib(
    data: pd.DataFrame,
    mode: Literal['bar', 'line'] = 'bar',
    colors: Union[List[str], str] = 'viridis',
    figsize: Tuple[int, int] = (12, 6),
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    show: bool = False,
    path: Optional[Union[str, pathlib.Path]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a DataFrame with Matplotlib using stacked bars or stepped lines.

    Args:
        data: A DataFrame containing the data to plot. The index should represent time (e.g., hours), and each column represents a separate data series.
        mode: Plotting mode. Use 'bar' for stacked bar charts or 'line' for stepped lines.
        colors: A List of colors (as str) or a name of a colorscale (e.g., 'viridis', 'plasma') to use for coloring the data series.
        figsize: Specify the size of the figure
        fig: A Matplotlib figure object to plot on. If not provided, a new figure will be created.
        ax: A Matplotlib axes object to plot on. If not provided, a new axes will be created.
        show: Wether to show the figure after creation.
        path: Path to save the figure to.

    Returns:
        A tuple containing the Matplotlib figure and axes objects used for the plot.

    Notes:
        - If `mode` is 'bar', bars are stacked for both positive and negative values.
          Negative values are stacked separately without extra labels in the legend.
        - If `mode` is 'line', stepped lines are drawn for each data series.
        - The legend is placed below the plot to accommodate multiple data series.

    Examples:
        >>> fig, ax = with_matplotlib(data, mode='bar', colorscale='plasma')
        >>> plt.show()
    """
    assert mode in ['bar', 'line'], f"'mode' must be one of {['bar', 'line']} for matplotlib"

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if isinstance(colors, str):
        cmap = plt.get_cmap(colors, len(data.columns))
        colors = [cmap(i) for i in range(len(data.columns))]
    assert len(colors) == len(data.columns), (
        f'The number of colors does not match the provided data columns. {len(colors)=}; {len(colors)=}'
    )

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
                align='center',
            )
            cumulative_positive += positive_values.values
            # Plot negative bars
            ax.bar(
                data.index,
                negative_values,
                bottom=cumulative_negative,
                color=colors[i],
                label='',  # No label for negative bars
                width=width,
                align='center',
            )
            cumulative_negative += negative_values.values

    elif mode == 'line':
        for i, column in enumerate(data.columns):
            ax.step(data.index, data[column], where='post', color=colors[i], label=column)

    # Aesthetics
    ax.set_xlabel('Time in h', fontsize=14)
    ax.grid(color='lightgrey', linestyle='-', linewidth=0.5)
    ax.legend(
        loc='upper center',  # Place legend at the bottom center
        bbox_to_anchor=(0.5, -0.15),  # Adjust the position to fit below plot
        ncol=5,
        frameon=False,  # Remove box around legend
    )
    fig.tight_layout()

    if show:
        plt.show()
    if path is not None:
        fig.savefig(path, dpi=300)

    return fig, ax


def heat_map_matplotlib(
    data: pd.DataFrame,
    color_map: str = 'viridis',
    figsize: Tuple[float, float] = (12, 6),
    show: bool = False,
    path: Optional[Union[str, pathlib.Path]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots a DataFrame as a heatmap using Matplotlib. The columns of the DataFrame will be displayed on the x-axis,
    the index will be displayed on the y-axis, and the values will represent the 'heat' intensity in the plot.

    Args:
        data: A DataFrame containing the data to be visualized. The index will be used for the y-axis, and columns will be used for the x-axis.
            The values in the DataFrame will be represented as colors in the heatmap.
        color_map: The colormap to use for the heatmap. Default is 'viridis'. Matplotlib supports various colormaps like 'plasma', 'inferno', 'cividis', etc.
        figsize: The size of the figure to create. Default is (12, 6), which results in a width of 12 inches and a height of 6 inches.
        show: Wether to show the figure after creation.
        path: Path to save the figure to.

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
    ax.pcolormesh(data.values, cmap=color_map)
    ax.invert_yaxis()  # Flip the y-axis to start at the top

    # Adjust ticks and labels for x and y axes
    ax.set_xticks(np.arange(len(data.columns)) + 0.5)
    ax.set_xticklabels(data.columns, ha='center')
    ax.set_yticks(np.arange(len(data.index)) + 0.5)
    ax.set_yticklabels(data.index, va='center')

    # Add labels to the axes
    ax.set_xlabel('Period', ha='center')
    ax.set_ylabel('Step', va='center')

    # Position x-axis labels at the top
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_ticks_position('top')

    # Add the colorbar
    sm1 = plt.cm.ScalarMappable(cmap=color_map, norm=plt.Normalize(vmin=color_bar_min, vmax=color_bar_max))
    sm1._A = []
    fig.colorbar(sm1, ax=ax, pad=0.12, aspect=15, fraction=0.2, orientation='horizontal')

    fig.tight_layout()
    if show:
        plt.show()
    if path is not None:
        fig.savefig(path, dpi=300)

    return fig, ax


def heat_map_plotly(
    data: pd.DataFrame,
    color_map: str = 'viridis',
    title: str = '',
    xlabel: str = 'Periods',
    ylabel: str = 'Step',
    categorical_labels: bool = True,
    show: bool = False,
    save: bool = False,
    path: Union[str, pathlib.Path] = 'temp-plot.html',
) -> go.Figure:
    """
    Plots a DataFrame as a heatmap using Plotly. The columns of the DataFrame will be mapped to the x-axis,
    and the index will be displayed on the y-axis. The values in the DataFrame will represent the 'heat' in the plot.

    Args:
        data: A DataFrame with the data to be visualized. The index will be used for the y-axis, and columns will be used for the x-axis.
            The values in the DataFrame will be represented as colors in the heatmap.
        color_map: The color scale to use for the heatmap. Default is 'viridis'. Plotly supports various color scales like 'Cividis', 'Inferno', etc.
        categorical_labels: If True, the x and y axes are treated as categorical data (i.e., the index and columns will not be interpreted as continuous data).
            Default is True. If False, the axes are treated as continuous, which may be useful for time series or numeric data.
        show: Wether to show the figure after creation. (This includes saving the figure)
        save: Wether to save the figure after creation (without showing)
        path: Path to save the figure.

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

    if isinstance(path, pathlib.Path):
        path = path.as_posix()
    if show:
        plotly.offline.plot(fig, filename=path)
    elif save:  # If show, the file is saved anyway
        fig.write_html(path)

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
    fill: Optional[Literal['ffill', 'bfill']] = None,
) -> pd.DataFrame:
    """
    Reshapes a DataFrame with a DateTime index into a 2D array for heatmap plotting,
    based on a specified sample rate.
    If a non-valid combination of periods and steps per period is used, falls back to numerical indices

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
        'The index of the Dataframe must be datetime to transfrom it properly for a heatmap plot'
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
        ('D', '15min'): ('%Y-%m-%d', '%H:%MM'),  # Day and hour
        ('h', '15min'): ('%Y-%m-%d %H:00', '%M'),  # minute of hour
        ('h', 'min'): ('%Y-%m-%d %H:00', '%M'),  # minute of hour
    }

    minimum_time_diff_in_min = df.index.to_series().diff().min().total_seconds() / 60  # Smallest time_diff in minutes
    time_intervals = {'min': 1, '15min': 15, 'h': 60, 'D': 24 * 60, 'W': 7 * 24 * 60}
    if time_intervals[steps_per_period] > minimum_time_diff_in_min:
        time_intervals[steps_per_period]
        logger.warning(
            f'To compute the heatmap, the data was aggregated from {minimum_time_diff_in_min:.2f} min to '
            f'{time_intervals[steps_per_period]:.2f} min. Mean values are displayed.'
        )

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
        resampled_data['step'] = resampled_data['step'].apply(
            lambda x: x.replace('0_Sunday', '7_Sunday') if '0_Sunday' in x else x
        )

    # Pivot the table so periods are columns and steps are indices
    df_pivoted = resampled_data.pivot(columns='period', index='step', values=df.columns[0])

    return df_pivoted


def plot_network(
    node_infos: dict,
    edge_infos: dict,
    path: Optional[Union[str, pathlib.Path]] = None,
    controls: Union[
        bool,
        List[Literal['nodes', 'edges', 'layout', 'interaction', 'manipulation', 'physics', 'selection', 'renderer']],
    ] = True,
    show: bool = False,
) -> Optional['pyvis.network.Network']:
    """
    Visualizes the network structure of a FlowSystem using PyVis, using info-dictionaries.

    Args:
        path: Path to save the HTML visualization. `False`: Visualization is created but not saved. `str` or `Path`: Specifies file path (default: 'results/network.html').
        controls: UI controls to add to the visualization. `True`: Enables all available controls. `List`: Specify controls, e.g., ['nodes', 'layout'].
            Options: 'nodes', 'edges', 'layout', 'interaction', 'manipulation', 'physics', 'selection', 'renderer'.
            You can play with these and generate a Dictionary from it that can be applied to the network returned by this function.
            network.set_options()
            https://pyvis.readthedocs.io/en/latest/tutorial.html
        show: Whether to open the visualization in the web browser.
            The calculation must be saved to show it. If no path is given, it defaults to 'network.html'.
    Returns:
        The `Network` instance representing the visualization, or `None` if `pyvis` is not installed.

    Usage:
    - Visualize and open the network with default options:
      >>> self.plot_network()

    - Save the visualization with opening:
      >>> self.plot_network(show=True)

    - Visualize with custom controls and path:
      >>> self.plot_network(path='output/custom_network.html', controls=['nodes', 'layout'])

    Notes:
    - This function requires `pyvis`. If not installed, the function prints a warning and returns `None`.
    - Nodes are styled based on type (e.g., circles for buses, boxes for components) and annotated with node information.
    """
    try:
        from pyvis.network import Network
    except ImportError:
        logger.warning("Please install pyvis to visualize the network: 'pip install pyvis'")
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
                logger.warning(
                    f'Showing the network in the Browser went wrong. Open it manually. Its saved under {path}'
                )
        except Exception as e:
            logger.warning(
                f'Showing the network in the Browser went wrong. Open it manually. Its saved under {path}: {e}'
            )


def pie_with_plotly(
        data: pd.DataFrame,
        colors: Union[List[str], str] = 'viridis',
        title: str = '',
        legend_title: str = '',
        hole: float = 0.0,
        fig: Optional[go.Figure] = None,
        show: bool = False,
        save: bool = False,
        path: Union[str, pathlib.Path] = 'temp-plot.html',
) -> go.Figure:
    """
    Create a pie chart with Plotly to visualize the proportion of values in a DataFrame.

    Args:
        data: A DataFrame containing the data to plot. If multiple rows exist,
              they will be summed unless a specific index value is passed.
        colors: A List of colors (as str) or a name of a colorscale (e.g., 'viridis', 'plasma')
                to use for coloring the pie segments.
        title: The title of the plot.
        legend_title: The title for the legend.
        hole: Size of the hole in the center for creating a donut chart (0.0 to 1.0).
        fig: A Plotly figure object to plot on. If not provided, a new figure will be created.
        show: Whether to show the figure after creation.
        save: Whether to save the figure after creation (without showing).
        path: Path to save the figure.

    Returns:
        A Plotly figure object containing the generated pie chart.

    Notes:
        - Negative values are not appropriate for pie charts and will be converted to absolute values with a warning.
        - If the data contains very small values (less than 1% of the total), they can be grouped into an "Other" category
          for better readability.
        - By default, the sum of all columns is used for the pie chart. For time series data, consider preprocessing.

    Examples:
        >>> fig = pie_with_plotly(data, colorscale='Pastel')
        >>> fig.show()
    """
    if data.empty:
        logger.warning("Empty DataFrame provided for pie chart. Returning empty figure.")
        return go.Figure()

    # Create a copy to avoid modifying the original DataFrame
    data_copy = data.copy()

    # Check if any negative values and warn
    if (data_copy < 0).any().any():
        logger.warning("Negative values detected in data. Using absolute values for pie chart.")
        data_copy = data_copy.abs()

    # If data has multiple rows, sum them to get total for each column
    if len(data_copy) > 1:
        data_sum = data_copy.sum()
    else:
        data_sum = data_copy.iloc[0]

    # Get labels (column names) and values
    labels = data_sum.index.tolist()
    values = data_sum.values.tolist()

    # Apply color mapping
    if isinstance(colors, str):
        colorscale = px.colors.get_colorscale(colors)
        colors = px.colors.sample_colorscale(
            colorscale,
            [i / (len(labels) - 1) for i in range(len(labels))] if len(labels) > 1 else [0],
        )

    # Create figure if not provided
    fig = fig if fig is not None else go.Figure()

    # Add pie trace
    fig.add_trace(
        go.Pie(
            labels=labels,
            values=values,
            hole=hole,
            marker=dict(colors=colors),
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
        font=dict(size=14),             # Increase font size for better readability
    )

    if isinstance(path, pathlib.Path):
        path = path.as_posix()
    if show:
        plotly.offline.plot(fig, filename=path)
    elif save:  # If show, the file is saved anyway
        fig.write_html(path)

    return fig


def pie_with_matplotlib(
        data: pd.DataFrame,
        colors: Union[List[str], str] = 'viridis',
        title: str = '',
        figsize: Tuple[int, int] = (10, 8),
        autopct: str = '%1.1f%%',
        startangle: int = 90,
        shadow: bool = False,
        is_donut: bool = False,
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
        show: bool = False,
        path: Optional[Union[str, pathlib.Path]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a pie chart with Matplotlib to visualize the proportion of values in a DataFrame.

    Args:
        data: A DataFrame containing the data to plot. If multiple rows exist,
              they will be summed unless a specific index value is passed.
        colors: A List of colors (as str) or a name of a colorscale (e.g., 'viridis', 'plasma')
                to use for coloring the pie segments.
        title: The title of the plot.
        figsize: The size of the figure (width, height) in inches.
        autopct: String format for the percentage display on wedges.
        startangle: Starting angle for the first wedge in degrees.
        shadow: Whether to draw the pie with a shadow beneath it.
        is_donut: If True, creates a donut chart by adding a white circle in the center.
        fig: A Matplotlib figure object to plot on. If not provided, a new figure will be created.
        ax: A Matplotlib axes object to plot on. If not provided, a new axes will be created.
        show: Whether to show the figure after creation.
        path: Path to save the figure to.

    Returns:
        A tuple containing the Matplotlib figure and axes objects used for the plot.

    Notes:
        - Negative values are not appropriate for pie charts and will be converted to absolute values with a warning.
        - If the data contains very small values (less than 1% of the total), they can be grouped into an "Other" category
          for better readability.
        - By default, the sum of all columns is used for the pie chart. For time series data, consider preprocessing.

    Examples:
        >>> fig, ax = pie_with_matplotlib(data, colorscale='viridis', is_donut=True)
        >>> plt.show()
    """
    if data.empty:
        logger.warning("Empty DataFrame provided for pie chart. Returning empty figure.")
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        return fig, ax

    # Create a copy to avoid modifying the original DataFrame
    data_copy = data.copy()

    # Check if any negative values and warn
    if (data_copy < 0).any().any():
        logger.warning("Negative values detected in data. Using absolute values for pie chart.")
        data_copy = data_copy.abs()

    # If data has multiple rows, sum them to get total for each column
    if len(data_copy) > 1:
        data_sum = data_copy.sum()
    else:
        data_sum = data_copy.iloc[0]

    # Get labels (column names) and values
    labels = data_sum.index.tolist()
    values = data_sum.values.tolist()

    # Apply color mapping
    if isinstance(colors, str):
        cmap = plt.get_cmap(colors, len(labels))
        colors = [cmap(i) for i in range(len(labels))]

    # Create figure and axis if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Draw the pie chart
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        colors=colors,
        autopct=autopct,
        startangle=startangle,
        shadow=shadow,
        wedgeprops=dict(width=0.5) if is_donut else None,  # Set width for donut
    )

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
        ax.legend(
            wedges,
            labels,
            title="Categories",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1)
        )

    # Apply tight layout
    fig.tight_layout()

    # Show or save
    if show:
        plt.show()
    if path is not None:
        fig.savefig(path, dpi=300, bbox_inches='tight')

    return fig, ax


def dual_pie_with_plotly(
    data_left: pd.Series,
    data_right: pd.Series,
    colors: Union[List[str], str] = 'viridis',
    title: str = '',
    subtitles: Tuple[str, str] = ('Left Chart', 'Right Chart'),
    legend_title: str = '',
    hole: float = 0.2,
    lower_percentage_group: float = 5.0,
    hover_template: str = '%{label}: %{value} (%{percent})',
    text_info: str = 'percent+label',
    text_position: str = 'inside',
    show: bool = False,
    save: bool = False,
    path: Union[str, pathlib.Path] = 'temp-plot.html',
) -> go.Figure:
    """
    Create two pie charts side by side with Plotly, with consistent coloring across both charts.

    Args:
        data_left: Series for the left pie chart.
        data_right: Series for the right pie chart.
        colors: A List of colors (as str) or a name of a colorscale (e.g., 'viridis', 'plasma')
                to use for coloring the pie segments.
        title: The main title of the plot.
        subtitles: Tuple containing the subtitles for (left, right) charts.
        legend_title: The title for the legend.
        hole: Size of the hole in the center for creating donut charts (0.0 to 100).
        lower_percentage_group: Whether to group small segments (below percentage (0...1)) into an "Other" category.
        hover_template: Template for hover text. Use %{label}, %{value}, %{percent}.
        text_info: What to show on pie segments: 'label', 'percent', 'value', 'label+percent',
                  'label+value', 'percent+value', 'label+percent+value', or 'none'.
        text_position: Position of text: 'inside', 'outside', 'auto', or 'none'.
        show: Whether to show the figure after creation.
        save: Whether to save the figure after creation (without showing).
        path: Path to save the figure.

    Returns:
        A Plotly figure object containing the generated dual pie chart.
    """
    from plotly.subplots import make_subplots
    import itertools

    # Check for empty data
    if data_left.empty and data_right.empty:
        logger.warning('Both datasets are empty. Returning empty figure.')
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
            logger.warning(f'Negative values detected in data. Using absolute values for pie chart.')
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

    # Generate consistent color mapping
    if isinstance(colors, str):
        colorscale = px.colors.get_colorscale(colors)
        color_list = px.colors.sample_colorscale(
            colorscale,
            [i / (len(all_labels) - 1) for i in range(len(all_labels))] if len(all_labels) > 1 else [0],
        )
        color_map = {label: color_list[i] for i, label in enumerate(all_labels)}
    else:
        # If colors is a list, create a cycling iterator
        color_iter = itertools.cycle(colors)
        color_map = {label: next(color_iter) for label in all_labels}

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
            marker_colors=trace_colors,
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
        legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5, font=dict(size=12)),
    )

    # Handle file saving and display
    if isinstance(path, pathlib.Path):
        path = path.as_posix()
    if show:
        plotly.offline.plot(fig, filename=path)
    elif save:
        fig.write_html(path)

    return fig
