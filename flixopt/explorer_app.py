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
def xarray_explorer(data: Union[xr.Dataset, xr.DataArray]):
    """
    A simple xarray explorer for both DataArrays and Datasets.
    Just pass your xarray object to this function.

    Args:
        data: xarray.Dataset or xarray.DataArray
    """
    # Determine if we're working with Dataset or DataArray
    is_dataset = isinstance(data, xr.Dataset)

    # Variable selection for Dataset or direct visualization for DataArray
    if is_dataset:
        # Variable selection
        selected_var = st.selectbox("Select variable:", list(data.data_vars))
        array_to_plot = data[selected_var]
    else:
        # If DataArray, use directly
        array_to_plot = data
        selected_var = data.name if data.name else "Data"

    # Visualization section
    st.subheader("Visualization")

    # Determine available visualization options based on dimensions
    dims = list(array_to_plot.dims)
    ndim = len(dims)

    # Different visualization options based on dimensionality
    if ndim == 0:
        # Scalar value
        st.metric("Value", float(array_to_plot.values))

    elif ndim == 1:
        # 1D data: line plot
        fig = px.line(x=array_to_plot[dims[0]].values,
                      y=array_to_plot.values,
                      labels={"x": dims[0], "y": selected_var})
        st.plotly_chart(fig, use_container_width=True)

        # Also show histogram
        fig2 = px.histogram(x=array_to_plot.values, nbins=30,
                            labels={"x": selected_var})
        st.plotly_chart(fig2, use_container_width=True)

    elif ndim >= 2:
        # For high dimensional data, let user select dimensions to plot
        st.write("Select dimensions to visualize:")

        viz_cols = st.columns(2)

        with viz_cols[0]:
            # Choose which dimension to put on x-axis
            x_dim = st.selectbox("X dimension:", dims, index=0)

            # Choose which dimension to put on y-axis
            remaining_dims = [d for d in dims if d != x_dim]
            y_dim = st.selectbox("Y dimension:", remaining_dims,
                                 index=0 if len(remaining_dims) > 0 else None)

        # If we have more than 2 dimensions, let user select values for other dimensions
        with viz_cols[1]:
            # Setup sliders for other dimensions
            slice_dims = [d for d in dims if d not in [x_dim, y_dim]]
            slice_indexes = {}

            for dim in slice_dims:
                dim_size = array_to_plot.sizes[dim]  # Use sizes instead of dims
                slice_indexes[dim] = st.slider(f"Position in {dim} dimension",
                                               0, dim_size-1, dim_size//2)

        # Create slice dictionary for selection
        slice_dict = {dim: slice_indexes[dim] for dim in slice_dims}

        # Select the data to plot
        if slice_dims:
            array_slice = array_to_plot.isel(slice_dict)
        else:
            array_slice = array_to_plot

        # Visualization depends on whether we have 1 or 2 dimensions selected
        if y_dim:
            # 2D visualization: heatmap
            fig = px.imshow(array_slice.transpose(y_dim, x_dim).values,
                            x=array_slice[x_dim].values,
                            y=array_slice[y_dim].values,
                            color_continuous_scale="viridis",
                            labels={"x": x_dim, "y": y_dim, "color": selected_var})

            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            # 1D visualization after slicing
            fig = px.line(x=array_slice[x_dim].values,
                          y=array_slice.values,
                          labels={"x": x_dim, "y": selected_var})
            st.plotly_chart(fig, use_container_width=True)

    # Data preview section
    st.subheader("Data Preview")

    # Convert to dataframe for display
    try:
        # Limit to first 1000 elements for performance
        preview_data = array_to_plot
        total_size = np.prod(preview_data.shape)

        if total_size > 1000:
            st.warning(f"Data is large ({total_size} elements). Showing first 1000 elements.")
            # Create a slice dict to get first elements from each dimension
            preview_slice = {}
            remaining = 1000
            for dim in preview_data.dims:
                dim_size = preview_data.sizes[dim]  # Use sizes instead of dims
                take = min(dim_size, max(1, int(remaining**(1/len(preview_data.dims)))))
                preview_slice[dim] = slice(0, take)
                remaining = remaining // take

            preview_data = preview_data.isel(preview_slice)

        # Convert to dataframe and display
        df = preview_data.to_dataframe()
        st.dataframe(df)
    except Exception as e:
        st.error(f"Could not convert to dataframe: {str(e)}")

    # Download options
    st.subheader('Download Options')
    download_format = st.selectbox(
        'Download format', ['CSV', 'NetCDF', 'Excel']
    )

    if st.button('Download filtered data'):
        download_data(array_to_plot, selected_var, download_format)

    # Display basic information
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Data Information")

        # Show dimensions and their sizes - using sizes instead of dims
        st.write("**Dimensions:**")
        dim_df = pd.DataFrame({
            "Dimension": list(data.sizes.keys()),
            "Size": list(data.sizes.values())
        })
        st.dataframe(dim_df)

        # For Dataset, show variables
        if is_dataset:
            st.write("**Variables:**")
            var_info = []
            for var_name, var in data.variables.items():
                var_info.append({
                    "Variable": var_name,
                    "Dimensions": ", ".join(var.dims),
                    "Type": str(var.dtype)
                })
            st.dataframe(pd.DataFrame(var_info))

        # Show coordinates
        if data.coords:
            st.write("**Coordinates:**")
            coord_info = []
            for coord_name, coord in data.coords.items():
                coord_info.append({
                    "Coordinate": coord_name,
                    "Dimensions": ", ".join(coord.dims),
                    "Type": str(coord.dtype)
                })
            st.dataframe(pd.DataFrame(coord_info))

        # Show attributes
        if data.attrs:
            st.write("**Attributes:**")
            st.json(data.attrs)

    # Display variable information
    with col2:
        st.subheader(f"Variable: {selected_var}")

        # Display basic stats if numeric
        try:
            if np.issubdtype(array_to_plot.dtype, np.number):
                stats_cols = st.columns(4)
                stats_cols[0].metric("Min", float(array_to_plot.min().values))
                stats_cols[1].metric("Max", float(array_to_plot.max().values))
                stats_cols[2].metric("Mean", float(array_to_plot.mean().values))
                stats_cols[3].metric("Std", float(array_to_plot.std().values))
        except:
            pass


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
