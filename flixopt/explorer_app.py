# FlixOpt Results Explorer App

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr
import plotly.express as px
import plotly.graph_objects as go
import io
import tempfile
from typing import Dict, List, Optional, Union, Tuple, Any

from flixopt import plotting


def plot_heatmap(
    data: pd.DataFrame,
    name: str,
    timeframes: str,
    timesteps: str,
    color_map: str,
) -> go.Figure:
    fig = plotting.heat_map_plotly(
        plotting.heat_map_data_from_df(data, timeframes, timesteps, 'ffill'),
        title=name,
        color_map=color_map,
        xlabel=f'timeframe [{timeframes}]',
        ylabel=f'timesteps [{timesteps}]',
    )
    fig.update_layout(
        margin=dict(l=50, r=100, t=50, b=50),  # Extra space for colorbar
        coloraxis_colorbar=dict(
            lenmode='fraction',
            len=0.8,
            title='Scale',
            tickvals=[0, 5, 10],  # Force ticks at min, mid, max
            ticktext=['0 (Min)', '5', '10 (Max)'],  # Custom labels
        ),  # Make colorbar bigger
    )
    return fig


def get_dimension_selector(dataset: xr.Dataset, dim: str, container: Any) -> Optional[Union[slice, List]]:
    """Creates UI elements to select values or ranges for a specific dimension.

    Args:
        dataset: The dataset containing the dimension.
        dim: The dimension name.
        container: The streamlit container to render UI elements in.

    Returns:
        The selected filter for this dimension (either a slice or list of values).
        Returns None if no selection is made.
    """
    container.write(f'**{dim}** (size: {len(dataset[dim])})')

    # Get the values for this dimension
    values = dataset[dim].values

    # Check if we have no data to work with
    if len(values) == 0:
        container.warning(f"Dimension '{dim}' is empty")
        return None

    # Determine the data type of the dimension
    first_val = values[0]

    # Create unique keys for all widgets based on dimension name
    # This prevents duplicate widget ID errors
    widget_key_base = f'dim_selector_{dim}'

    # Case 1: Small number of values - always use multiselect regardless of type
    if len(values) <= 5:
        # For datetime64, convert to readable datetime objects
        if isinstance(first_val, np.datetime64):
            values = [pd.to_datetime(str(val)) for val in values]

        selected = container.multiselect(
            f'Select {dim} values', options=values, default=[values[0]], key=f'{widget_key_base}_small_multiselect'
        )
        if selected:
            return selected
        return None

    # Case 2: Datetime values - use date picker
    elif isinstance(first_val, np.datetime64):
        date_min = pd.to_datetime(str(dataset[dim].min().values))
        date_max = pd.to_datetime(str(dataset[dim].max().values))
        start_date, end_date = container.date_input(
            f'Select {dim} range',
            value=(date_min, min(date_min + pd.Timedelta(days=30), date_max)),
            min_value=date_min,
            max_value=date_max,
            key=f'{widget_key_base}_date_input',
        )
        return slice(str(start_date), str(end_date))

    # Case 3: String values (categorical data) - use multiselect with limiting features
    elif isinstance(first_val, str):
        # For string values, provide a way to select multiple values
        # First, get unique values and sort them
        unique_values = sorted(list(set(values)))

        # If we have too many unique values, provide a selection mechanism
        if len(unique_values) > 20:
            container.warning(f"Dimension '{dim}' has {len(unique_values)} unique string values. Showing first 20.")

            # Option to show all values or search
            show_all = container.checkbox(
                f"Show all values for '{dim}'", value=False, key=f'{widget_key_base}_show_all'
            )

            if show_all:
                # Show all values but provide a text search to filter
                search_term = container.text_input(f"Filter values for '{dim}'", '', key=f'{widget_key_base}_search')

                if search_term:
                    # Filter values that contain the search term
                    filtered_values = [val for val in unique_values if search_term.lower() in str(val).lower()]
                    if not filtered_values:
                        container.warning(f"No values matching '{search_term}'")
                        return None

                    unique_values = filtered_values
            else:
                # Just show the first 20 values
                unique_values = unique_values[:20]

        # Display multiselect with available values
        selected = container.multiselect(
            f'Select {dim} values',
            options=unique_values,
            default=[unique_values[0]] if unique_values else [],
            key=f'{widget_key_base}_str_multiselect',
        )

        if selected:
            return selected
        return None

    # Case 4: Numeric values - use slider
    elif np.issubdtype(type(first_val), np.number) or isinstance(first_val, (int, float)):
        try:
            min_val = float(dataset[dim].min().values)
            max_val = float(dataset[dim].max().values)

            # Check for identical min/max values
            if min_val == max_val:
                container.info(f"All values in dimension '{dim}' are identical: {min_val}")
                return None

            # Determine appropriate step size
            range_size = max_val - min_val
            if range_size < 1:
                step = range_size / 100
            elif range_size < 10:
                step = 0.1
            elif range_size < 100:
                step = 1
            else:
                step = range_size / 100

            # Round values for better UI
            range_val = container.slider(
                f'Select {dim} range',
                min_value=min_val,
                max_value=max_val,
                value=(min_val, min(min_val + range_size / 10, max_val)),
                step=step,
                key=f'{widget_key_base}_slider',
            )
            return slice(range_val[0], range_val[1])
        except Exception as e:
            container.error(f"Error creating slider for '{dim}': {e}")
            return None

    # Case 5: Unknown/Unhandled type - fallback to multiselect with first 20 values
    else:
        container.warning(f"Dimension '{dim}' has an unusual data type. Using simple selection.")

        # Limit to first 20 values to avoid overwhelming the UI
        display_values = list(values)[:20]

        selected = container.multiselect(
            f'Select {dim} values (first 20 shown)',
            options=display_values,
            default=[display_values[0]] if display_values else [],
            key=f'{widget_key_base}_fallback_multiselect',
        )

        if selected:
            return selected
        return None


def filter_and_aggregate(
    dataset: xr.Dataset, var_name: str, filters: Dict[str, Union[slice, List]], agg_dims: List[str], agg_method: str
) -> xr.DataArray:
    """Filters and aggregates a variable from the dataset.

    Args:
        dataset: The dataset containing the variable.
        var_name: Name of the variable to process.
        filters: Dictionary of dimension filters.
        agg_dims: Dimensions to aggregate over.
        agg_method: Aggregation method (mean, sum, etc.).

    Returns:
        Filtered and aggregated data.
    """
    # Get the variable
    variable = dataset[var_name]

    # Filter the data
    if filters:
        filtered_data = variable.sel(**filters)
    else:
        filtered_data = variable

    # Apply aggregation if selected
    if agg_dims and agg_method:
        if agg_method == 'mean':
            filtered_data = filtered_data.mean(dim=agg_dims)
        elif agg_method == 'sum':
            filtered_data = filtered_data.sum(dim=agg_dims)
        elif agg_method == 'min':
            filtered_data = filtered_data.min(dim=agg_dims)
        elif agg_method == 'max':
            filtered_data = filtered_data.max(dim=agg_dims)
        elif agg_method == 'std':
            filtered_data = filtered_data.std(dim=agg_dims)

    return filtered_data


def resample_time_data(data: xr.DataArray, freq: str) -> xr.DataArray:
    """Resamples a DataArray along its time dimension.

    Args:
        data: The xarray DataArray containing a time dimension.
        freq: The resampling frequency string (e.g., 'D', 'M', 'Y', '5min').

    Returns:
        The resampled DataArray.
    """
    # Find the time dimension name
    time_dims = [dim for dim in data.dims if dim in ['time', 't'] or (isinstance(data[dim].values[0], np.datetime64))]

    if not time_dims:
        # No time dimension found
        return data

    time_dim = time_dims[0]

    try:
        # Resample the data - default aggregation is mean
        resampled_data = data.resample({time_dim: freq}).mean()
        return resampled_data
    except Exception as e:
        print(f'Error resampling data: {e}')
        return data


def get_time_aggregation_ui(container: Any, data: xr.DataArray) -> Tuple[bool, Optional[str]]:
    """Creates UI elements for time-based aggregation options.

    Args:
        container: The streamlit container to render UI elements in.
        data: The xarray DataArray to check for time dimensions.

    Returns:
        A tuple containing:
            - Boolean indicating if time resampling should be applied
            - The selected resampling frequency (or None if no resampling)
    """
    # Find time dimensions
    time_dims = [
        dim
        for dim in data.dims
        if dim in ['time', 't', 'date', 'datetime']
        or (len(data[dim]) > 0 and isinstance(data[dim].values[0], np.datetime64))
    ]

    if not time_dims:
        # No time dimensions
        return False, None

    # Create unique key base for all widgets
    key_base = f'time_resampling_{time_dims[0]}'

    # Show time resampling options
    container.subheader('Time Resampling')
    time_dim = time_dims[0]

    # Check if the dimension has enough elements to be worth resampling
    min_elements_for_resampling = 5
    if len(data[time_dim]) < min_elements_for_resampling:
        container.info(f"Time dimension '{time_dim}' has too few elements for resampling.")
        return False, None

    # Get the time range for display
    try:
        # Convert to pandas datetime for safe handling of different datetime formats
        time_values = data[time_dim].values
        if isinstance(time_values[0], str):
            # Try to convert string dates to datetime
            time_values = pd.to_datetime(time_values)

        start_time = pd.to_datetime(time_values[0])
        end_time = pd.to_datetime(time_values[-1])
        time_range = end_time - start_time

        # Show time range information
        container.write(f'Time range: {start_time.date()} to {end_time.date()} ({time_range.days} days)')
    except Exception as e:
        container.warning(f'Error determining time range: {e}')
        # Even if there's an error showing the range, we can still offer resampling
        time_range = pd.Timedelta(days=365)  # Assume a 1-year range as default

    # Determine appropriate resampling options based on the time range
    resampling_options = []

    try:
        days = time_range.days

        # Always include options for hourly/daily data
        if days >= 2:
            resampling_options.extend(['H', 'D'])

        # For data spanning more than a week
        if days > 7:
            resampling_options.extend(['W'])

        # For data spanning more than a month
        if days > 30:
            resampling_options.extend(['M'])

        # For data spanning more than a year
        if days > 365:
            resampling_options.extend(['Q', 'Y'])
    except:
        # Fallback options if we can't determine from time range
        resampling_options = ['H', 'D', 'W', 'M']

    # Ensure we have at least some options
    if not resampling_options:
        resampling_options = ['H', 'D', 'W', 'M']

    # Create friendly names for UI
    freq_map = {'H': 'Hour', 'D': 'Day', 'W': 'Week', 'M': 'Month', 'Q': 'Quarter', 'Y': 'Year'}

    friendly_options = [freq_map.get(opt, opt) for opt in resampling_options]

    # Add "None" option for no resampling
    resampling_options = ['none'] + resampling_options
    friendly_options = ['None (original data)'] + friendly_options

    # Add "Custom" option
    resampling_options.append('custom')
    friendly_options.append('Custom frequency string')

    # Create the selection widget
    use_resampling = container.checkbox('Enable time resampling', value=False, key=f'{key_base}_enable')

    if use_resampling:
        selected_freq_name = container.selectbox(
            'Resample to:', options=friendly_options, key=f'{key_base}_freq_select'
        )

        # Map back to actual frequency string
        selected_index = friendly_options.index(selected_freq_name)
        selected_freq = resampling_options[selected_index]

        if selected_freq == 'none':
            return False, None
        elif selected_freq == 'custom':
            # Provide information about pandas frequency strings
            with container.expander('Frequency string help', key=f'{key_base}_help_expander'):
                container.write("""
                **Pandas frequency strings examples:**
                - '5min': 5 minutes
                - '2H': 2 hours
                - '1D': 1 day
                - '1W': 1 week
                - '2W-MON': Biweekly on Monday
                - '1M': 1 month
                - '1Q': 1 quarter
                - '1A' or '1Y': 1 year
                - '3A': 3 years

                You can also use combinations like '1D12H' for 1 day and 12 hours.
                """)

            # Allow user to input a custom frequency string
            custom_freq = container.text_input(
                'Enter custom frequency string:',
                value='1D',  # Default to daily
                help="Enter a pandas frequency string like '5min', '2H', '1D', '1W', '1M'",
                key=f'{key_base}_custom_input',
            )

            if custom_freq:
                # Validate the frequency string
                try:
                    # Try to create a sample resampling to validate the string
                    test_dates = pd.date_range('2020-01-01', periods=3, freq='D')
                    test_series = pd.Series(range(3), index=test_dates)
                    test_series.resample(custom_freq).mean()

                    # If we get here, the frequency string is valid
                    # Show information about what resampling will do
                    try:
                        n_points_before = len(data[time_dim])

                        # Convert string dates to datetime if needed for resampling preview
                        if isinstance(data[time_dim].values[0], str):
                            # Create a temporary copy with datetime index for preview
                            temp_data = data.copy()
                            temp_data.coords[time_dim] = pd.to_datetime(temp_data[time_dim].values)
                            resampled = temp_data.resample({time_dim: custom_freq}).mean()
                        else:
                            resampled = data.resample({time_dim: custom_freq}).mean()

                        n_points_after = len(resampled[time_dim])
                        container.info(f'Resampling will change data points from {n_points_before} to {n_points_after}')
                    except Exception as e:
                        container.warning(f'Cannot preview resampling effect: {str(e)}')

                    return True, custom_freq
                except Exception as e:
                    container.error(f'Invalid frequency string: {str(e)}')
                    return False, None
            else:
                return False, None
        else:
            # Show information about what resampling will do
            try:
                n_points_before = len(data[time_dim])

                # Convert string dates to datetime if needed for resampling preview
                if len(data[time_dim]) > 0 and isinstance(data[time_dim].values[0], str):
                    # Create a temporary copy with datetime index for preview
                    temp_data = data.copy()
                    temp_data.coords[time_dim] = pd.to_datetime(temp_data[time_dim].values)
                    resampled = temp_data.resample({time_dim: selected_freq}).mean()
                else:
                    resampled = data.resample({time_dim: selected_freq}).mean()

                n_points_after = len(resampled[time_dim])
                container.info(
                    f'Resampling will change data points from {n_points_before} to {n_points_after}',
                    key=f'{key_base}_info',
                )
            except Exception as e:
                container.warning(f'Cannot preview resampling effect: {str(e)}', key=f'{key_base}_warning')

            return True, selected_freq
    else:
        return False, None

def create_plotly_plot(
    data: xr.DataArray, plot_type: str, var_name: str, title: Optional[str] = None, x_dim: Optional[str] = None
) -> go.Figure:
    """Creates a plotly plot based on the selected data and plot type.

    Args:
        data: The filtered/aggregated data array to plot.
        plot_type: Type of plot to create (Line, Stacked Bar, Grouped Bar, or Heatmap).
        var_name: Name of the selected variable.
        title: Plot title.
        x_dim: Dimension to use for x-axis in line plots.

    Returns:
        Plotly figure object.
    """
    # Check if we have valid data to plot
    if data is None:
        return go.Figure().update_layout(
            title='No data to plot',
            annotations=[
                dict(
                    text='No valid data found for plotting. Check your selections.',
                    showarrow=False,
                    xref='paper',
                    yref='paper',
                    x=0.5,
                    y=0.5,
                )
            ],
        )

    # Get dimensions of the data array
    dims = list(data.dims)

    # Create different plot types based on dimensions and selection
    if plot_type == 'Line':
        # Line plot
        if len(dims) == 1:
            # Simple line plot for 1D data
            x_values = data[dims[0]].values
            y_values = data.values

            fig = px.line(x=x_values, y=y_values, labels={'x': dims[0], 'y': var_name}, title=title)

        elif len(dims) >= 2 and x_dim is not None:
            # Multiple lines for higher dimensional data
            # Convert to dataframe for easy plotting
            df = data.to_dataframe().reset_index()

            # Group by the x dimension
            group_dims = [d for d in dims if d != x_dim]

            if len(group_dims) == 0:
                # If no grouping dimensions, just plot a single line
                fig = px.line(df, x=x_dim, y=var_name, title=title)
            else:
                # Create a plot with a line for each unique combination of group dimensions
                fig = px.line(
                    df,
                    x=x_dim,
                    y=var_name,
                    color=group_dims[0] if len(group_dims) == 1 else None,  # Use first group dim for color
                    facet_col=group_dims[1] if len(group_dims) > 1 else None,  # Use second group dim for faceting
                    title=title,
                )
        else:
            # Not enough dimensions for line plot
            fig = go.Figure().update_layout(
                title='Cannot create Line plot',
                annotations=[
                    dict(
                        text='Need at least one dimension for Line plot',
                        showarrow=False,
                        xref='paper',
                        yref='paper',
                        x=0.5,
                        y=0.5,
                    )
                ],
            )

    elif plot_type == 'Stacked Bar':
        if len(dims) >= 2:
            # Convert to dataframe
            df = data.to_dataframe().reset_index()

            # For stacked bar, need a category dimension and a value dimension
            if x_dim is not None and x_dim in dims:
                # Use the selected x dimension
                x = x_dim
                # Get another dimension for stacking
                stack_dim = next((d for d in dims if d != x_dim), None)

                if stack_dim:
                    fig = px.bar(df, x=x, y=var_name, color=stack_dim, barmode='stack', title=title)
                else:
                    # No dimension to stack
                    fig = px.bar(df, x=x, y=var_name, title=title)
            else:
                # Default to first dimension for x-axis
                x = dims[0]
                stack_dim = dims[1] if len(dims) > 1 else None

                fig = px.bar(df, x=x, y=var_name, color=stack_dim, barmode='stack', title=title)
        elif len(dims) == 1:
            # Single dimension bar plot
            df = data.to_dataframe().reset_index()

            fig = px.bar(df, x=dims[0], y=var_name, title=title)
        else:
            # Not enough dimensions
            fig = go.Figure().update_layout(
                title='Cannot create Stacked Bar plot',
                annotations=[
                    dict(
                        text='Need at least one dimension for Stacked Bar plot',
                        showarrow=False,
                        xref='paper',
                        yref='paper',
                        x=0.5,
                        y=0.5,
                    )
                ],
            )

    elif plot_type == 'Grouped Bar':
        if len(dims) >= 2:
            # Convert to dataframe
            df = data.to_dataframe().reset_index()

            # For grouped bar, need a category dimension and a group dimension
            if x_dim is not None and x_dim in dims:
                # Use the selected x dimension
                x = x_dim
                # Get another dimension for grouping
                group_dim = next((d for d in dims if d != x_dim), None)

                if group_dim:
                    fig = px.bar(df, x=x, y=var_name, color=group_dim, barmode='group', title=title)
                else:
                    # No dimension to group
                    fig = px.bar(df, x=x, y=var_name, title=title)
            else:
                # Default to first dimension for x-axis
                x = dims[0]
                group_dim = dims[1] if len(dims) > 1 else None

                fig = px.bar(df, x=x, y=var_name, color=group_dim, barmode='group', title=title)
        elif len(dims) == 1:
            # Single dimension bar plot
            df = data.to_dataframe().reset_index()

            fig = px.bar(df, x=dims[0], y=var_name, title=title)
        else:
            # Not enough dimensions
            fig = go.Figure().update_layout(
                title='Cannot create Grouped Bar plot',
                annotations=[
                    dict(
                        text='Need at least one dimension for Grouped Bar plot',
                        showarrow=False,
                        xref='paper',
                        yref='paper',
                        x=0.5,
                        y=0.5,
                    )
                ],
            )

    elif plot_type == 'Heatmap' and len(dims) >= 2:
        # Heatmap for 2D data
        if len(dims) > 2:
            # If more than 2 dimensions, need to select which dimensions to use
            if x_dim is not None and x_dim in dims:
                # Use x_dim and find another dimension
                dim1 = x_dim
                dim2 = next((d for d in dims if d != x_dim), None)

                # Need to aggregate other dimensions
                agg_dims = [d for d in dims if d != dim1 and d != dim2]
                if agg_dims:
                    # Aggregate other dimensions using mean
                    data = data.mean(dim=agg_dims)
            else:
                # Use first two dimensions
                dim1, dim2 = dims[:2]

                # Aggregate other dimensions if needed
                if len(dims) > 2:
                    agg_dims = dims[2:]
                    data = data.mean(dim=agg_dims)
        else:
            dim1, dim2 = dims

        # Create heatmap
        fig = px.imshow(
            data.values,
            x=data[dim1].values,
            y=data[dim2].values,
            labels=dict(x=dim1, y=dim2, color=var_name),
            title=title,
            color_continuous_scale='Viridis',
        )
    else:
        # Default empty plot with warning
        fig = go.Figure().update_layout(
            title='Cannot create plot',
            annotations=[
                dict(
                    text=f'Cannot create {plot_type} plot with the current data dimensions',
                    showarrow=False,
                    xref='paper',
                    yref='paper',
                    x=0.5,
                    y=0.5,
                )
            ],
        )

    # Common layout settings
    fig.update_layout(
        height=600,
        width=800,
        margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )

    return fig


def download_data(filtered_data: xr.DataArray, var_name: str, download_format: str, container: Any) -> None:
    """Creates download buttons for the filtered data.

    Args:
        filtered_data: The filtered data to download.
        var_name: Name of the variable.
        download_format: Format to download (CSV, NetCDF, Excel).
        container: Streamlit container to place the download button.
    """
    if download_format == 'CSV':
        csv = filtered_data.to_dataframe().reset_index().to_csv(index=False)
        container.download_button(label='Download CSV', data=csv, file_name=f'{var_name}_filtered.csv', mime='text/csv')
    elif download_format == 'NetCDF':
        # Create temp file for netCDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nc') as tmp:
            filtered_data.to_netcdf(tmp.name)
            with open(tmp.name, 'rb') as f:
                container.download_button(
                    label='Download NetCDF',
                    data=f.read(),
                    file_name=f'{var_name}_filtered.nc',
                    mime='application/x-netcdf',
                )
    elif download_format == 'Excel':
        # Create in-memory Excel file
        buffer = io.BytesIO()
        filtered_data.to_dataframe().reset_index().to_excel(buffer, index=False)
        buffer.seek(0)

        container.download_button(
            label='Download Excel',
            data=buffer,
            file_name=f'{var_name}_filtered.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        )


def xarray_explorer_component(dataset: xr.Dataset, container: Any = None) -> xr.DataArray:
    """A reusable Streamlit component that creates an xarray dataset explorer.

    This component allows users to interactively explore an xarray Dataset by
    selecting variables, filtering dimensions, and creating visualizations.

    Args:
        dataset: The xarray Dataset to explore.
        container: The Streamlit container to render the explorer in.
            If None, renders in the current Streamlit app context.

    Returns:
        The filtered/selected data for the selected variable.
    """
    # If no container is provided, use the current Streamlit context
    if container is None:
        container = st

    # Dataset information
    with container.expander('Dataset Overview', key="dataset_overview_expander"):
        container.write('### Dataset Metadata')
        container.write(dataset.attrs)

        container.write('### Dimensions')
        container.write(pd.DataFrame({'Dimension': list(dataset.dims.keys()), 'Size': list(dataset.dims.values())}))

        container.write('### Variables')
        var_info = []
        for var_name, var in dataset.variables.items():
            var_info.append(
                {
                    'Variable': var_name,
                    'Dimensions': ', '.join(var.dims),
                    'Shape': str(var.shape),
                    'Type': str(var.dtype),
                }
            )
        container.dataframe(pd.DataFrame(var_info), key="var_info_dataframe")

    # Variable selection - single variable only
    container.subheader('Variable Selection')
    selected_var = container.selectbox(
        'Select variable to explore',
        list(dataset.data_vars),
        key="variable_selector"
    )

    # Get the variable
    variable = dataset[selected_var]

    # Display variable info
    container.write(f'**Variable shape:** {variable.shape}')
    container.write(f'**Variable dimensions:** {variable.dims}')
    dims = list(variable.dims)

    # Create column layout
    col1, col2 = container.columns([1, 2])

    with col1:
        container.subheader('Query Parameters')

        # Set filters for each dimension
        filters = {}
        for dim in dims:
            dim_filter = get_dimension_selector(dataset, dim, container)
            if dim_filter is not None:
                filters[dim] = dim_filter

        # Aggregation options
        container.subheader('Aggregation Options')
        agg_dims = container.multiselect(
            'Dimensions to aggregate',
            dims,
            key="agg_dims_selector"
        )
        agg_method = container.selectbox(
            'Aggregation method',
            ['mean', 'sum', 'min', 'max', 'std'],
            key="agg_method_selector"
        )

        # Check if data has time dimension and add time resampling UI
        use_time_resampling, resampling_freq = get_time_aggregation_ui(container, variable)

        # Plot type selection - limited to the requested types
        container.subheader('Plot Settings')
        plot_type = container.selectbox(
            'Plot type',
            ['Line', 'Stacked Bar', 'Grouped Bar', 'Heatmap'],
            key="plot_type_selector"
        )

        if plot_type in ['Line', 'Stacked Bar', 'Grouped Bar']:
            remaining_dims = [d for d in dims if d not in agg_dims]
            if remaining_dims:
                x_dim = container.selectbox(
                    'X axis dimension',
                    remaining_dims,
                    key="x_dim_selector"
                )
            else:
                x_dim = None
        else:
            x_dim = None

    # Filter and aggregate the selected variable
    filtered_data = filter_and_aggregate(dataset, selected_var, filters, agg_dims, agg_method)

    # Apply time resampling if requested
    if use_time_resampling and resampling_freq:
        container.info(f'Applying time resampling with frequency: {resampling_freq}')
        filtered_data = resample_time_data(filtered_data, resampling_freq)

    # Display the visualizations
    with col2:
        container.subheader('Visualization')

        # Create the plot
        plot_title = f'{selected_var} {plot_type} Plot'
        fig = create_plotly_plot(filtered_data, plot_type, selected_var, title=plot_title, x_dim=x_dim)

        # Show the plot
        container.plotly_chart(fig, use_container_width=True, key="main_plot")

        # Data preview
        with container.expander('Data Preview', key="data_preview_expander"):
            container.dataframe(filtered_data.to_dataframe(), key="filtered_data_preview")

        # Download options
        container.subheader('Download Options')
        download_format = container.selectbox(
            'Download format',
            ['CSV', 'NetCDF', 'Excel'],
            key="download_format_selector"
        )

        if container.button('Download filtered data', key="download_button"):
            download_data(filtered_data, selected_var, download_format, container)

    return filtered_data

def explore_results_app(results):
    """
    Main function to explore calculation results

    Args:
        results: A CalculationResults object to explore
    """
    # Set page config
    st.set_page_config(
        page_title="FlixOpt Results Explorer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Create sidebar for navigation
    st.sidebar.title("FlixOpt Results Explorer")
    pages = ["Overview", "Components", "Buses", "Effects", "Explorer", "Effects DS"]
    selected_page = st.sidebar.radio("Navigation", pages)

    # Overview page
    if selected_page == "Overview":
        st.title("Calculation Overview")

        # Model information
        st.header("Model Information")
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Name:** {results.name}")
            st.write(f"**Folder:** {results.folder}")
            st.write(f"**Time Steps:** {len(results.timesteps_extra)}")
            if len(results.timesteps_extra) > 0:
                st.write(f"**Time Range:** {results.timesteps_extra[0]} to {results.timesteps_extra[-1]}")

        with col2:
            st.write(f"**Components:** {len(results.components)}")
            st.write(f"**Buses:** {len(results.buses)}")
            st.write(f"**Effects:** {len(results.effects)}")
            st.write(f"**Storage Components:** {len(results.storages)}")

        # Results summary
        st.header('Results Summary')
        tabs = st.tabs(list(results.summary.keys()))
        for i, key in enumerate(results.summary.keys()):
            with tabs[i]:
                if isinstance(results.summary[key], dict):
                    st.json(results.summary[key])
                else:
                    st.write(results.summary[key])

        # Network visualization
        st.header("Network Structure")
        tabs = st.tabs(["Component Connections", "Nodes", "Edges"])

        # Show component connections
        with tabs[0]:
            connections_data = []

            for comp_name, comp in results.flow_system.components.items():
                for flow_name, flow in comp.flows.items():
                    connections_data.append({
                        "Component": comp_name,
                        "Flow": flow_name,
                        "Direction": "from" if flow_name in comp.inputs else "to",
                        "Bus": flow.bus,
                    })

            st.dataframe(pd.DataFrame(connections_data))

        network_infos = results.flow_system.network_infos()
        with tabs[1]:
            st.json(network_infos[0])

        with tabs[2]:
            st.json(network_infos[1])


    # Components page
    elif selected_page == "Components":
        st.title("Components")

        # Component selector
        component_names = list(results.components.keys())

        # Allow grouping by storage/non-storage
        show_storage_first = st.checkbox("Show storage components first", value=True)

        if show_storage_first:
            storage_components = [comp.label for comp in results.storages]
            non_storage_components = [name for name in component_names if name not in storage_components]
            sorted_components = storage_components + non_storage_components
        else:
            sorted_components = sorted(component_names)

        component_name = st.selectbox("Select a component:", sorted_components)

        if component_name:
            component = results.components[component_name]

            st.header(f"Component: {component_name}")
            if component.is_storage:
                st.info("This is a storage component")

            # Component tabs
            tabs = st.tabs(["Node Balance", "All Variables"])

            # Node Balance tab
            with tabs[0]:
                try:
                    st.subheader("Node Balance")

                    scenario = (
                        st.selectbox(f'Select a scenario: {results.scenarios[0]}', list(results.scenarios))
                        if results.scenarios is not None
                        else None
                    )

                    # Use built-in plotting method
                    if component.is_storage:
                        fig = component.plot_charge_state(show=False, save=False, scenario=scenario)
                    else:
                        fig = component.plot_node_balance(show=False, save=False, scenario=scenario)

                    st.plotly_chart(fig, theme='streamlit', use_container_width=True)

                    # Also show as dataframe if requested
                    if st.checkbox("Show Data Table"):
                        if component.is_storage:
                            node_balance = component.node_balance_with_charge_state()
                        else:
                            node_balance = component.node_balance()

                        if scenario:
                            st.dataframe(node_balance.sel(scenario=scenario).to_pandas())
                        else:
                            st.dataframe(node_balance.to_pandas())

                except Exception as e:
                    st.error(f"Error displaying the node balance: {e}")

            # Variables tab
            with tabs[1]:
                # Use the reusable function
                xarray_explorer_component(component.solution)

    # Buses page
    elif selected_page == "Buses":
        st.title("Buses")

        # Bus selector
        bus_names = list(results.buses.keys())
        bus_name = st.selectbox("Select a bus:", sorted(bus_names))

        if bus_name:
            bus = results.buses[bus_name]

            st.header(f"Bus: {bus_name}")

            # Bus tabs
            tabs = st.tabs(["Node Balance", "All Variables"])

            # Node Balance tab
            with tabs[0]:
                try:
                    st.subheader("Node Balance")

                    scenario = (
                        st.selectbox(f'Select a scenario: {results.scenarios[0]}', list(results.scenarios))
                        if results.scenarios is not None
                        else None
                    )

                    # Use built-in plotting method
                    fig = bus.plot_node_balance(show=False, save=False, scenario=scenario)
                    st.plotly_chart(fig, theme=None, use_container_width=True)

                    # Also show as dataframe if requested
                    if st.checkbox("Show Data Table"):
                        if scenario:
                            df = bus.node_balance().sel(scenario=scenario).to_pandas()
                        else:
                            df = bus.node_balance().to_pandas()
                        st.dataframe(df)

                except Exception as e:
                    st.error(f"Error displaying the node balance: {e}")

            # Variables tab
            with tabs[1]:
                # Use the reusable function
                xarray_explorer_component(bus.solution)

    # Effects page
    elif selected_page == "Effects":
        st.title("Effects")

        # Effect selector
        effect_names = list(results.effects.keys())
        effect_name = st.selectbox("Select an effect:", sorted(effect_names), index=0)
        effect = results.effects[effect_name]

        st.header(f"Effect: {effect_name}")

        xarray_explorer_component(effect.solution)

    elif selected_page == "Explorer":
        st.title("Explorer")
        xarray_explorer_component(results.solution)

    elif selected_page == "Effects DS":
        st.title('Effects Dataset')
        tabs = st.tabs(["total", "invest", "operation"])

        with tabs[0]:
            xarray_explorer_component(results.effects_per_component('total'))
        with tabs[1]:
            xarray_explorer_component(results.effects_per_component('invest'))
        with tabs[2]:
            xarray_explorer_component(results.effects_per_component('operation'))


def run_explorer_from_file(folder, name):
    """
    Run the explorer by loading results from a file

    Args:
        folder: Folder path containing the calculation results
        name: Name of the calculation
    """
    # Import the relevant modules
    try:
        # Try different import approaches
        try:
            # First try standard import
            try:
                from flixopt.results import CalculationResults
            except ImportError:
                from flixopt.results import CalculationResults
        except ImportError:
            # Add potential module paths
            for path in [os.getcwd(), os.path.dirname(os.path.abspath(__file__))]:
                if path not in sys.path:
                    sys.path.append(path)

            # Try again with modified path
            try:
                from flixopt.results import CalculationResults
            except ImportError:
                from flixopt.results import CalculationResults

        # Load from file
        results = CalculationResults.from_file(folder, name)
        explore_results_app(results)
    except Exception as e:
        st.error(f"Error loading calculation results: {e}")
        st.stop()

# Entry point for module execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FlixOpt Results Explorer')
    parser.add_argument('folder', type=str, help='Results folder path')
    parser.add_argument('name', type=str, help='Calculation name')
    args = parser.parse_args()

    run_explorer_from_file(args.folder, args.name)
