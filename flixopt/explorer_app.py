# FlixOpt Results Explorer App

import argparse
import os
import sys
import io
import tempfile
from typing import Dict, List, Optional, Union, Tuple, Any, Callable, cast, TypeVar
import traceback
import inspect

import numpy as np
import functools
import pandas as pd
import streamlit as st
import xarray as xr
import plotly.express as px
import plotly.graph_objects as go

T = TypeVar('T')


def show_traceback(
    return_original_input: bool = False, include_args: bool = True, container: Optional[Any] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    A decorator that shows the full traceback in Streamlit when an exception occurs.

    Args:
        return_original_input: If True and the first argument is not None, return it on error.
                              Useful for data processing functions to return original data.
        include_args: If True, show function arguments in the error details.
        container: Optional Streamlit container to display errors in.
                  If None, uses st directly.

    Usage:
        @show_traceback()
        def my_function(data, param1, param2):
            # Your code here

        # Or with custom options:
        @show_traceback(return_original_input=True, include_args=False)
        def process_data(data, options):
            # Your code here

    Returns:
        The decorated function
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get display container
            display = container if container is not None else st

            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Show error message
                display.error(f'⚠️ Error in {func.__name__}: {str(e)}')

                # Create an expander for detailed error info
                with display.expander('See detailed traceback'):
                    # Show the full traceback
                    display.code(traceback.format_exc(), language='python')

                    # Show function info if requested
                    if include_args:
                        display.markdown('**Function Information:**')

                        # Try to get source code
                        try:
                            display.code(inspect.getsource(func), language='python')
                        except:
                            display.warning('Could not retrieve function source code.')

                        # Show arguments
                        display.markdown('**Function Arguments:**')

                        # Safely represent args
                        safe_args = []
                        for arg in args:
                            try:
                                repr_arg = repr(arg)
                                if len(repr_arg) > 200:  # Truncate long representations
                                    repr_arg = repr_arg[:200] + '...'
                                safe_args.append(repr_arg)
                            except:
                                safe_args.append('[Representation failed]')

                        # Safely represent kwargs
                        safe_kwargs = {}
                        for k, v in kwargs.items():
                            try:
                                repr_v = repr(v)
                                if len(repr_v) > 200:  # Truncate long representations
                                    repr_v = repr_v[:200] + '...'
                                safe_kwargs[k] = repr_v
                            except:
                                safe_kwargs[k] = '[Representation failed]'

                        # Display args and kwargs
                        display.text(f'Args: {safe_args}')
                        display.text(f'Kwargs: {safe_kwargs}')

                # Also log to console/stderr for server logs
                print(f'Exception in {func.__name__}:', file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

                # Determine what to return on error
                if return_original_input and args and args[0] is not None:
                    # Return the first argument (usually the data being processed)
                    return args[0]
                else:
                    # Return None as default
                    return None

        return cast(Callable[..., T], wrapper)

    return decorator


@show_traceback()
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


@show_traceback()
def display_data_info(data: Union[xr.Dataset, xr.DataArray], container: Optional[Any] = None) -> None:
    """
    Display basic information about an xarray object.

    Args:
        data: xarray.Dataset or xarray.DataArray
        container: Streamlit container to render in (if None, uses st directly)
    """
    if container is None:
        container = st

    # Show dimensions and their sizes
    container.write('**Dimensions:**')
    dim_df = pd.DataFrame({'Dimension': list(data.sizes.keys()), 'Size': list(data.sizes.values())})
    container.dataframe(dim_df)

    # For Dataset, show variables
    if isinstance(data, xr.Dataset):
        container.write('**Variables:**')
        var_info = []
        for var_name, var in data.variables.items():
            var_info.append({'Variable': var_name, 'Dimensions': ', '.join(var.dims), 'Type': str(var.dtype)})
        container.dataframe(pd.DataFrame(var_info))

    # Show coordinates
    if data.coords:
        container.write('**Coordinates:**')
        coord_info = []
        for coord_name, coord in data.coords.items():
            coord_info.append({'Coordinate': coord_name, 'Dimensions': ', '.join(coord.dims), 'Type': str(coord.dtype)})
        container.dataframe(pd.DataFrame(coord_info))

    # Show attributes
    if data.attrs:
        container.write('**Attributes:**')
        container.json(data.attrs)


@show_traceback()
def display_variable_stats(array: xr.DataArray, container: Optional[Any] = None) -> None:
    """
    Display basic statistics for a DataArray if it's numeric.

    Args:
        array: xarray.DataArray to compute stats for
        container: Streamlit container to render in (if None, uses st directly)
    """
    if container is None:
        container = st

    try:
        if np.issubdtype(array.dtype, np.number):
            stats_cols = container.columns(4)
            stats_cols[0].metric('Min', float(array.min().values))
            stats_cols[1].metric('Max', float(array.max().values))
            stats_cols[2].metric('Mean', float(array.mean().values))
            stats_cols[3].metric('Std', float(array.std().values))
    except:
        pass


@show_traceback()
def aggregate_dimensions(
    array: xr.DataArray, agg_dims: List[str], agg_method: str, container: Optional[Any] = None
) -> xr.DataArray:
    """
    Aggregate a DataArray over specified dimensions using a specified method.

    Args:
        array: xarray.DataArray to aggregate
        agg_dims: List of dimension names to aggregate over
        agg_method: Aggregation method ('mean', 'sum', 'min', 'max', 'std', 'median')
        container: Streamlit container for displaying messages

    Returns:
        Aggregated DataArray
    """
    if container is None:
        container = st

    # Filter out any dimensions that don't exist in the array
    valid_agg_dims = [dim for dim in agg_dims if dim in array.dims]

    # If there are no valid dimensions to aggregate over, just return the original array
    if not valid_agg_dims:
        return array

    # Apply the selected aggregation method
    try:
        if agg_method == 'mean':
            result = array.mean(dim=valid_agg_dims)
        elif agg_method == 'sum':
            result = array.sum(dim=valid_agg_dims)
        elif agg_method == 'min':
            result = array.min(dim=valid_agg_dims)
        elif agg_method == 'max':
            result = array.max(dim=valid_agg_dims)
        elif agg_method == 'std':
            result = array.std(dim=valid_agg_dims)
        elif agg_method == 'median':
            result = array.median(dim=valid_agg_dims)
        elif agg_method == 'var':
            result = array.var(dim=valid_agg_dims)
        else:
            container.warning(f"Unknown aggregation method: {agg_method}. Using 'mean' instead.")
            result = array.mean(dim=valid_agg_dims)

        # If the aggregation removed all dimensions, ensure result has correct shape
        if len(result.dims) == 0:
            # Convert scalar result to 0D DataArray
            result = xr.DataArray(result.values, name=array.name, attrs=array.attrs)

        return result
    except Exception as e:
        container.error(f'Error during aggregation: {str(e)}')
        return array  # Return original array if aggregation fails


@show_traceback()
def plot_scalar(array: xr.DataArray, container: Optional[Any] = None) -> None:
    """
    Plot a scalar (0-dimensional) DataArray.

    Args:
        array: xarray.DataArray with 0 dimensions
        container: Streamlit container to render in (if None, uses st directly)
    """
    if container is None:
        container = st

    container.metric('Value', float(array.values))


@show_traceback()
def plot_1d(array: xr.DataArray, var_name: str, container: Optional[Any] = None) -> None:
    """
    Plot a 1-dimensional DataArray with multiple plot type options.

    Args:
        array: xarray.DataArray with 1 dimension
        var_name: Name of the variable being plotted
        container: Streamlit container to render in (if None, uses st directly)
    """
    if container is None:
        container = st

    dim = list(array.dims)[0]

    # Add plot type selector
    plot_type = container.selectbox('Plot type:', ['Line', 'Bar', 'Histogram', 'Area'], key=f'plot_type_1d_{var_name}')

    # Create figure based on selected plot type
    if plot_type == 'Line':
        fig = px.line(
            x=array[dim].values, y=array.values, labels={'x': dim, 'y': var_name}, title=f'{var_name} by {dim}'
        )
    elif plot_type == 'Bar':
        df = pd.DataFrame({dim: array[dim].values, 'value': array.values})
        fig = px.bar(df, x=dim, y='value', labels={'value': var_name}, title=f'{var_name} by {dim}')
    elif plot_type == 'Histogram':
        fig = px.histogram(
            x=array.values,
            nbins=min(30, len(array) // 2) if len(array) > 2 else 10,
            labels={'x': var_name},
            title=f'Distribution of {var_name}',
        )
    elif plot_type == 'Area':
        df = pd.DataFrame({dim: array[dim].values, 'value': array.values})
        fig = px.area(df, x=dim, y='value', labels={'value': var_name}, title=f'{var_name} by {dim}')

    # Show the plot
    container.plotly_chart(fig, use_container_width=True)

    # For 1D data, we can also offer some basic statistics
    if container.checkbox('Show statistics', key=f'show_stats_{var_name}'):
        try:
            stats = pd.DataFrame(
                {
                    'Statistic': ['Min', 'Max', 'Mean', 'Median', 'Std', 'Sum'],
                    'Value': [
                        float(array.min().values),
                        float(array.max().values),
                        float(array.mean().values),
                        float(np.median(array.values)),
                        float(array.std().values),
                        float(array.sum().values),
                    ],
                }
            )
            container.dataframe(stats, use_container_width=True)
        except Exception as e:
            container.warning(f'Could not compute statistics: {str(e)}')

@show_traceback()
def plot_nd(array: xr.DataArray, var_name: str, container: Optional[Any] = None) -> Tuple[xr.DataArray, Optional[Dict]]:
    """
    Plot a multi-dimensional DataArray with interactive dimension selectors.
    Supports multiple plot types and dimension aggregation.

    Args:
        array: xarray.DataArray with 2+ dimensions
        var_name: Name of the variable being plotted
        container: Streamlit container to render in (if None, uses st directly)

    Returns:
        Tuple of (sliced array, selection dictionary)
    """
    if container is None:
        container = st

    dims = list(array.dims)

    # Aggregation options
    container.subheader('Dimension Handling')

    # Define columns for the UI layout
    col1, col2 = container.columns(2)

    with col1:
        # Multi-select for dimensions to aggregate
        agg_dims = st.multiselect(
            'Dimensions to aggregate:',
            dims,
            default=[],
            help='Select dimensions to aggregate (reduce) using the method selected below',
        )

        # Aggregation method selection
        agg_method = st.selectbox(
            'Aggregation method:',
            ['mean', 'sum', 'min', 'max', 'std', 'median', 'var'],
            index=0,
            help='Method used to aggregate over the selected dimensions',
        )

    # Apply aggregation if dimensions were selected
    if agg_dims:
        orig_dims = dims.copy()
        array = aggregate_dimensions(array, agg_dims, agg_method, container)

        # Update the list of available dimensions after aggregation
        dims = list(array.dims)

        # Show information about the aggregation
        removed_dims = [dim for dim in orig_dims if dim not in dims]
        if removed_dims:
            msg = f'Applied {agg_method} aggregation over: {", ".join(removed_dims)}'
            container.info(msg)

    # If no dimensions left after aggregation, show scalar result
    if len(dims) == 0:
        plot_scalar(array, container)
        return array, None

    # If one dimension left after aggregation, use 1D plotting
    if len(dims) == 1:
        plot_1d(array, var_name, container)
        return array, None

    # Visualization options for 2+ dimensions
    container.subheader('Visualization Settings')

    # Choose which dimension to put on x-axis
    with col2:
        x_dim = st.selectbox('X dimension:', dims, index=0)

        # Choose which dimension to put on y-axis if we have at least 2 dimensions
        remaining_dims = [d for d in dims if d != x_dim]
        y_dim = None
        if len(remaining_dims) > 0:
            y_dim_options = ['None'] + remaining_dims
            y_dim_selection = st.selectbox('Y dimension:', y_dim_options, index=1)
            if y_dim_selection != 'None':
                y_dim = y_dim_selection

        # Add plot type selector
        plot_types = ['Heatmap', 'Line', 'Stacked Bar', 'Grouped Bar']
        if y_dim is None:
            # Remove heatmap option if there's no Y dimension
            plot_types = [pt for pt in plot_types if pt != 'Heatmap']
            default_idx = 0  # Default to Line for 1D
        else:
            default_idx = 0  # Default to Heatmap for 2D

        plot_type = st.selectbox('Plot type:', plot_types, index=default_idx)

    # If we have more than the selected dimensions, let user select values for other dimensions
    container.subheader('Other Dimension Values')

    # Calculate which dimensions need slicers
    slice_dims = [d for d in dims if d not in ([x_dim] if y_dim is None else [x_dim, y_dim])]
    slice_indexes = {}

    # Create sliders in a more compact layout if there are many dimensions
    if len(slice_dims) > 0:
        if len(slice_dims) <= 3:
            # For a few dimensions, use columns
            cols = container.columns(len(slice_dims))
            for i, dim in enumerate(slice_dims):
                dim_size = array.sizes[dim]
                with cols[i]:
                    slice_indexes[dim] = st.slider(
                        f'{dim}', 0, dim_size - 1, dim_size // 2, help=f'Select position along {dim} dimension'
                    )
        else:
            # For many dimensions, use a more compact layout
            with container.expander('Select values for other dimensions', expanded=True):
                for dim in slice_dims:
                    dim_size = array.sizes[dim]
                    slice_indexes[dim] = st.slider(
                        f'{dim}', 0, dim_size - 1, dim_size // 2, help=f'Select position along {dim} dimension'
                    )

    # Create slice dictionary for selection
    slice_dict = {dim: slice_indexes[dim] for dim in slice_dims}

    # Select the data to plot
    if slice_dims:
        array_slice = array.isel(slice_dict)
    else:
        array_slice = array

    # Visualization depends on the selected plot type and dimensions
    container.subheader('Plot')

    if y_dim is not None:
        # 2D visualization
        if plot_type == 'Heatmap':
            # Heatmap visualization
            fig = px.imshow(
                array_slice.transpose(y_dim, x_dim).values,
                x=array_slice[x_dim].values,
                y=array_slice[y_dim].values,
                color_continuous_scale='viridis',
                labels={'x': x_dim, 'y': y_dim, 'color': var_name},
            )
            fig.update_layout(height=500)
        elif plot_type == 'Line':
            # Line plot with multiple lines (one per y-dimension value)
            fig = go.Figure()

            # Convert to dataframe for easier plotting
            df = array_slice.to_dataframe(name='value').reset_index()

            # Group by y-dimension for multiple lines
            for y_val in array_slice[y_dim].values:
                df_subset = df[df[y_dim] == y_val]
                fig.add_trace(
                    go.Scatter(x=df_subset[x_dim], y=df_subset['value'], mode='lines', name=f'{y_dim}={y_val}')
                )

            fig.update_layout(
                height=500,
                title=f'{var_name} by {x_dim} and {y_dim}',
                xaxis_title=x_dim,
                yaxis_title=var_name,
                legend_title=y_dim,
            )
        elif plot_type == 'Stacked Bar':
            # Stacked bar chart
            # Convert to dataframe for easier plotting
            df = array_slice.to_dataframe(name='value').reset_index()

            fig = px.bar(
                df,
                x=x_dim,
                y='value',
                color=y_dim,
                barmode='stack',
                labels={'value': var_name, x_dim: x_dim, y_dim: y_dim},
            )
            fig.update_layout(height=500)
        elif plot_type == 'Grouped Bar':
            # Grouped bar chart
            # Convert to dataframe for easier plotting
            df = array_slice.to_dataframe(name='value').reset_index()

            fig = px.bar(
                df,
                x=x_dim,
                y='value',
                color=y_dim,
                barmode='group',
                labels={'value': var_name, x_dim: x_dim, y_dim: y_dim},
            )
            fig.update_layout(height=500)
    else:
        # 1D visualization after slicing (no y_dim)
        if plot_type == 'Line':
            fig = px.line(x=array_slice[x_dim].values, y=array_slice.values, labels={'x': x_dim, 'y': var_name})
        elif plot_type in ['Stacked Bar', 'Grouped Bar']:  # Both are the same for 1D
            # Create a dataframe for the bar chart
            df = pd.DataFrame({x_dim: array_slice[x_dim].values, 'value': array_slice.values})

            fig = px.bar(df, x=x_dim, y='value', labels={'value': var_name})

    container.plotly_chart(fig, use_container_width=True)
    return array_slice, slice_dict

@show_traceback()
def display_data_preview(array: xr.DataArray, container: Optional[Any] = None) -> pd.DataFrame:
    """
    Display a preview of the data as a dataframe.

    Args:
        array: xarray.DataArray to preview
        container: Streamlit container to render in (if None, uses st directly)

    Returns:
        DataFrame containing the preview data
    """
    if container is None:
        container = st

    try:
        # Limit to first 1000 elements for performance
        preview_data = array
        total_size = np.prod(preview_data.shape)

        if total_size > 1000:
            container.warning(f'Data is large ({total_size} elements). Showing first 1000 elements.')
            # Create a slice dict to get first elements from each dimension
            preview_slice = {}
            remaining = 1000
            for dim in preview_data.dims:
                dim_size = preview_data.sizes[dim]
                take = min(dim_size, max(1, int(remaining ** (1 / len(preview_data.dims)))))
                preview_slice[dim] = slice(0, take)
                remaining = remaining // take

            preview_data = preview_data.isel(preview_slice)

        # Convert to dataframe and display
        df = preview_data.to_dataframe()
        container.dataframe(df)
        return df
    except Exception as e:
        container.error(f'Could not convert to dataframe: {str(e)}')
        return pd.DataFrame()


@show_traceback()
def xarray_explorer(
    data: Union[xr.Dataset, xr.DataArray],
    custom_plotters: Optional[Dict[str, Callable]] = None,
    container: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    A modular xarray explorer for both DataArrays and Datasets.

    Args:
        data: xarray.Dataset or xarray.DataArray
        custom_plotters: Dictionary of custom plotting functions by dimension.
                        Keys are 'scalar', '1d', and 'nd'.
        title: Title for the explorer
        container: Streamlit container to render in (if None, uses st directly)

    Returns:
        Dictionary containing information about the current state:
        - 'data': Original xarray data
        - 'selected_array': Currently selected/displayed array
        - 'selected_var': Name of selected variable
        - 'sliced_array': Array after slicing (for multi-dimensional arrays)
        - 'slice_dict': Dictionary of dimension slices applied
    """
    if container is None:
        container = st

    # Determine if we're working with Dataset or DataArray
    is_dataset = isinstance(data, xr.Dataset)

    # Variable selection for Dataset or direct visualization for DataArray
    if is_dataset:
        # Variable selection
        selected_var = container.selectbox('Select variable:', list(data.data_vars))
        array_to_plot = data[selected_var]
    else:
        # If DataArray, use directly
        array_to_plot = data
        selected_var = data.name if data.name else 'Data'

    # Initialize result dictionary
    result = {
        'data': data,
        'selected_array': array_to_plot,
        'selected_var': selected_var,
        'sliced_array': None,
        'slice_dict': None,
    }

    # Visualization in right column
    container.subheader('Visualization')

    # Determine available visualization options based on dimensions
    dims = list(array_to_plot.dims)
    ndim = len(dims)

    # Get the appropriate plotter function
    plotters = {'scalar': plot_scalar, '1d': plot_1d, 'nd': plot_nd}

    # Override with custom plotters if provided
    if custom_plotters:
        plotters.update(custom_plotters)

    # Different visualization options based on dimensionality
    if ndim == 0:
        # Scalar value
        plotters['scalar'](array_to_plot, container)
    elif ndim == 1:
        # 1D data
        plotters['1d'](array_to_plot, selected_var, container)
    else:
        # 2D+ data
        sliced_array, slice_dict = plotters['nd'](array_to_plot, selected_var, container)
        result['sliced_array'] = sliced_array
        result['slice_dict'] = slice_dict

    # Data preview section
    container.subheader('Data Preview')
    display_data_preview(array_to_plot, container)

    # Download options
    container.subheader('Download Options')
    download_format = container.selectbox('Download format', ['CSV', 'NetCDF', 'Excel'])

    if container.button('Download filtered data'):
        download_data(
            array_to_plot if result['sliced_array'] is None else result['sliced_array'],
            selected_var,
            download_format,
            container,
        )


    container.subheader('Data Information')
    display_data_info(data, container)

    # Display variable information
    container.subheader(f'Variable: {selected_var}')
    display_variable_stats(array_to_plot, container)

    return result


# Example of a custom plotter
@show_traceback()
def custom_heatmap_plotter(
    array: xr.DataArray, var_name: str, container: Optional[Any] = None
) -> Tuple[xr.DataArray, Optional[Dict]]:
    """
    A custom plotter for multi-dimensional arrays that uses a different color scheme.

    Args:
        array: xarray.DataArray with 2+ dimensions
        var_name: Name of the variable being plotted
        container: Streamlit container to render in (if None, uses st directly)

    Returns:
        Tuple of (sliced array, selection dictionary)
    """
    if container is None:
        container = st

    # You can reuse much of the code from plot_nd but customize the actual plotting
    dims = list(array.dims)

    container.write('Select dimensions to visualize:')

    viz_cols = container.columns(2)

    with viz_cols[0]:
        # Choose which dimension to put on x-axis
        x_dim = st.selectbox('X dimension:', dims, index=0, key='custom_x_dim')

        # Choose which dimension to put on y-axis
        remaining_dims = [d for d in dims if d != x_dim]
        y_dim = st.selectbox(
            'Y dimension:', remaining_dims, index=0 if len(remaining_dims) > 0 else None, key='custom_y_dim'
        )

    # If we have more than 2 dimensions, let user select values for other dimensions
    with viz_cols[1]:
        # Setup sliders for other dimensions
        slice_dims = [d for d in dims if d not in [x_dim, y_dim]]
        slice_indexes = {}

        for dim in slice_dims:
            dim_size = array.sizes[dim]
            slice_indexes[dim] = st.slider(
                f'Position in {dim} dimension', 0, dim_size - 1, dim_size // 2, key=f'custom_{dim}_slider'
            )

    # Create slice dictionary for selection
    slice_dict = {dim: slice_indexes[dim] for dim in slice_dims}

    # Select the data to plot
    if slice_dims:
        array_slice = array.isel(slice_dict)
    else:
        array_slice = array

    # Visualization depends on whether we have 1 or 2 dimensions selected
    if y_dim:
        # 2D visualization: heatmap with CUSTOM COLORS and LAYOUT
        fig = px.imshow(
            array_slice.transpose(y_dim, x_dim).values,
            x=array_slice[x_dim].values,
            y=array_slice[y_dim].values,
            color_continuous_scale='Plasma',  # Different color scale
            labels={'x': x_dim, 'y': y_dim, 'color': var_name},
        )

        # Customize layout
        fig.update_layout(
            height=600,  # Taller
            margin=dict(l=50, r=50, t=50, b=50),  # More margin
            coloraxis_colorbar=dict(
                title=var_name,
                thicknessmode='pixels',
                thickness=20,
                lenmode='pixels',
                len=400,
                outlinewidth=1,
                outlinecolor='black',
                borderwidth=1,
            ),
        )

        container.plotly_chart(fig, use_container_width=True)
    else:
        # 1D visualization after slicing - with CUSTOM LINE STYLE
        fig = px.line(x=array_slice[x_dim].values, y=array_slice.values, labels={'x': x_dim, 'y': var_name})

        # Customize the line
        fig.update_traces(line=dict(width=3, dash='dash', color='darkred'))

        container.plotly_chart(fig, use_container_width=True)

    return array_slice, slice_dict



@show_traceback()
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
    pages = ["Overview", "Components", "Buses", "Effects", "Flows DS", "Effects DS", "Explorer"]
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
                xarray_explorer(component.solution)

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
                xarray_explorer(bus.solution)

    # Effects page
    elif selected_page == "Effects":
        st.title("Effects")

        # Effect selector
        effect_names = list(results.effects.keys())
        effect_name = st.selectbox("Select an effect:", sorted(effect_names), index=0)
        effect = results.effects[effect_name]

        st.header(f"Effect: {effect_name}")

        xarray_explorer(effect.solution)

    elif selected_page == "Flows DS":
        st.title('Flow Rates Dataset')
        mode = st.selectbox("Select a mode", ['Flow Rates', 'Flow Hours'])
        if mode == 'Flow Hours':
            xarray_explorer(results.flow_hours())
        else:
            xarray_explorer(results.flow_rates())

    elif selected_page == 'Effects DS':
        st.title('Effects Dataset')
        mode = st.selectbox("Select a mode", ['total', 'invest', 'operation'])
        xarray_explorer(results.effects_per_component(mode))

    elif selected_page == "Explorer":
        st.title("Explore all variable results")
        xarray_explorer(results.solution)


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
