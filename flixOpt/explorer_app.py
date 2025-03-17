# FlixOpt Results Explorer App

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from flixOpt import plotting

# Parse command line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FlixOpt Results Explorer')
    parser.add_argument('folder', type=str, help='Results folder path')
    parser.add_argument('name', type=str, help='Calculation name')
    args = parser.parse_args()

    results_folder = args.folder
    results_name = args.name
else:
    # Default values when imported as module
    results_folder = "."
    results_name = "results"

# Set page config
st.set_page_config(
    page_title="FlixOpt Results Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper function to capture plotly figures
def get_plotly_fig(plot_func, *args, **kwargs):
    """Capture a plotly figure from a plotting function"""
    # Add default parameters to ensure the function returns the figure without showing it
    kwargs['show'] = False
    kwargs['save'] = False

    # Call the plotting function
    return plot_func(*args, **kwargs)

# Cache the calculation loading
@st.cache_resource
def get_calculation_results(folder, name):
    # Import the relevant modules
    try:
        # Try different import approaches
        try:
            # First try standard import
            try:
                from flixopt.results import CalculationResults
            except ImportError:
                from flixOpt.results import CalculationResults
        except ImportError:
            # Add potential module paths
            for path in [os.getcwd(), os.path.dirname(os.path.abspath(__file__))]:
                if path not in sys.path:
                    sys.path.append(path)

            # Try again with modified path
            try:
                from flixopt.results import CalculationResults
            except ImportError:
                from flixOpt.results import CalculationResults

        # Load from file
        return CalculationResults.from_file(folder, name)
    except Exception as e:
        st.error(f"Error loading calculation results: {e}")
        return None

# Load the calculation results
results = get_calculation_results(results_folder, results_name)

if results is None:
    st.error("Failed to load calculation results.")
    st.stop()

# Create sidebar for navigation
st.sidebar.title("FlixOpt Results Explorer")
pages = ["Overview", "Components", "Buses", "Effects", "Variables", "Heatmaps"]
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

    # Additional info
    if hasattr(results, 'infos') and results.infos:
        st.subheader("Additional Information")
        st.json(results.infos)

    # Network info
    if hasattr(results, 'network_infos') and results.network_infos:
        st.subheader("Network Information")
        st.json(results.network_infos)

    # Network visualization
    st.header("Network Structure")

    # Show component connections
    st.subheader("Component Connections")
    connections_data = []

    for comp_name, comp in results.components.items():
        for bus_name in comp.inputs + comp.outputs:
            connections_data.append({
                "Component": comp_name,
                "Bus": bus_name,
                "Type": "Input" if bus_name in comp.inputs else "Output"
            })

    if connections_data:
        st.dataframe(pd.DataFrame(connections_data))

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

                # Use built-in plotting method
                if component.is_storage:
                    fig = get_plotly_fig(component.plot_charge_state)
                else:
                    fig = get_plotly_fig(component.plot_flow_rates)

                st.plotly_chart(fig, use_container_width=True)

                # Also show as dataframe if requested
                if st.checkbox("Show Data Table"):
                    if component.is_storage:
                        flow_rates = component.charge_state_and_flow_rates().to_dataframe()
                    else:
                        flow_rates = component.flow_rates().to_dataframe()
                    st.dataframe(flow_rates)
            except Exception as e:
                st.error(f"Error displaying the node balance: {e}")

        # Variables tab
        with tabs[1]:
            st.title("Model Variables")

            # Add a filter option
            variable_filter = st.text_input("Filter variables by name:")

            # Get all variables and apply filter
            all_variables = list(component.variables)

            if variable_filter:
                filtered_variables = [v for v in all_variables if variable_filter.lower() in v.lower()]
            else:
                filtered_variables = all_variables

            # Heatmap options in a single row
            show_heatmap_col, heatmap_col1, heatmap_col2, heatmap_col3 = st.columns(4)
            with show_heatmap_col:
                show_heatmap = st.checkbox('Show as heatmap', value=False)
            with heatmap_col1:
                timeframes = st.selectbox(
                    'Timeframes',
                    ['YS', 'MS', 'W', 'D', 'h', '15min', 'min'],
                    index=3,  # Default to "D"
                )
            with heatmap_col2:
                timesteps = st.selectbox(
                    'Timesteps',
                    ['W', 'D', 'h', '15min', 'min'],
                    index=2,  # Default to "h"
                )
            with heatmap_col3:
                color_map = st.selectbox(
                    'Colormap',
                    ['portland', 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'RdBu', 'Blues', 'YlOrRd'],
                    index=0,
                )

            st.write(f"Showing {len(filtered_variables)} of {len(all_variables)} variables")

            # Display all filtered variables directly
            for var_name in filtered_variables:
                try:
                    var = component.variables[var_name]
                    var_solution = var.solution

                    # Check if this is a time-based variable
                    if 'time' in var_solution.dims:
                        if show_heatmap:
                            try:
                                # Create heatmap using var_solution
                                heatmap_data = plotting.heat_map_data_from_df(
                                    var_solution.to_dataframe(var_name),
                                    timeframes,
                                    timesteps,
                                    'ffill'
                                )

                                fig = plotting.heat_map_plotly(
                                    heatmap_data,
                                    title=var_name,
                                    color_map=color_map,
                                    xlabel=f'timeframe [{timeframes}]',
                                    ylabel=f'timesteps [{timesteps}]'
                                )

                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error creating heatmap: {e}")
                        else:
                            # Regular time series plot
                            fig = get_plotly_fig(plotting.with_plotly, data=var_solution.to_dataframe(), mode='area', title=f'Variable: {var_name}')
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)

                        show_datatable = st.checkbox(f'Show data table', key=f'datatable_{var_name}', value=False)
                        if show_datatable:
                            st.dataframe(var_solution.to_dataframe())

                    else:
                        # Show scalar value
                        st.write(f"{var_name}: {var_solution.values}")
                except Exception as e:
                    st.error(f"Error displaying variable {var_name}: {e}")

# Buses page
elif selected_page == "Buses":
    st.title("Buses")

    # Bus selector
    bus_names = list(results.buses.keys())
    bus_name = st.selectbox("Select a bus:", sorted(bus_names))

    if bus_name:
        bus = results.buses[bus_name]

        st.header(f"Bus: {bus_name}")

        # Show Node Balance
        try:
            st.subheader("Node Balance")

            # Use built-in plotting method
            fig = get_plotly_fig(bus.plot_flow_rates)
            df = bus.flow_rates().to_dataframe()
            st.plotly_chart(fig, use_container_width=True)

            # Also show as dataframe if requested
            if st.checkbox("Show Data Table"):
                st.dataframe(df)
        except Exception as e:
            st.error(f"Error displaying the node balance: {e}")

        # Show inputs and outputs
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Inputs")
            for input_name in bus.inputs:
                st.write(f"- {input_name}")

        with col2:
            st.subheader("Outputs")
            for output_name in bus.outputs:
                st.write(f"- {output_name}")

# Effects page
elif selected_page == "Effects":
    st.title("Effects")

    # Effect selector
    effect_names = list(results.effects.keys())

    if effect_names:
        effect_name = st.selectbox("Select an effect:", sorted(effect_names))

        if effect_name:
            effect = results.effects[effect_name]

            st.header(f"Effect: {effect_name}")

            # List variables
            st.subheader("Variables")

            for var_name in effect._variables:
                try:
                    var = effect.variables[var_name]

                    # Create an expander for each variable
                    with st.expander(f"Variable: {var_name}"):
                        var_solution = var.solution

                        # Check if this is a time-based variable
                        if 'time' in var_solution.dims:
                            # Plot as heatmap toggle
                            show_heatmap = st.checkbox("Show as heatmap", key=f"heatmap_{var_name}", value=False)

                            if show_heatmap:
                                # Heatmap options in a single row
                                heatmap_col1, heatmap_col2, heatmap_col3 = st.columns(3)
                                with heatmap_col1:
                                    timeframes = st.selectbox(
                                        "Timeframes",
                                        ["YS", "MS", "W", "D", "h", "15min", "min"],
                                        index=3,  # Default to "D"
                                        key=f"timeframes_{var_name}"
                                    )
                                with heatmap_col2:
                                    timesteps = st.selectbox(
                                        "Timesteps",
                                        ["W", "D", "h", "15min", "min"],
                                        index=2,  # Default to "h"
                                        key=f"timesteps_{var_name}"
                                    )
                                with heatmap_col3:
                                    color_map = st.selectbox(
                                        "Colormap",
                                        ["portland", "viridis", "plasma", "inferno", "magma", "cividis", "RdBu", "Blues", "YlOrRd"],
                                        index=0,
                                        key=f"colormap_{var_name}"
                                    )

                                try:
                                    # Create heatmap using var_solution
                                    heatmap_data = plotting.heat_map_data_from_df(
                                        var_solution.to_dataframe(var_name),
                                        timeframes,
                                        timesteps,
                                        'ffill'
                                    )

                                    fig = plotting.heat_map_plotly(
                                        heatmap_data,
                                        title=var_name,
                                        color_map=color_map,
                                        xlabel=f'timeframe [{timeframes}]',
                                        ylabel=f'timesteps [{timesteps}]'
                                    )

                                    st.plotly_chart(fig, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Error creating heatmap: {e}")
                            else:
                                # Regular time series plot
                                df = var_solution.to_dataframe()

                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=df.index,
                                    y=df[var_name],
                                    mode='lines',
                                    name=var_name
                                ))

                                fig.update_layout(
                                    title=f"{var_name} Time Series",
                                    xaxis_title="Time",
                                    yaxis_title="Value",
                                    height=300
                                )

                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            # Show scalar value
                            st.write(f"Value: {var_solution.values}")
                except Exception as e:
                    st.error(f"Error displaying variable {var_name}: {e}")

            # List shares
            connected_elements = set()
            for var_name in effect._variables:
                if '->' in var_name:
                    elem = var_name.split('->')[0]
                    connected_elements.add(elem)

            if connected_elements:
                st.subheader("Element Shares")

                for elem in sorted(connected_elements):
                    with st.expander(f"Shares from {elem}"):
                        try:
                            shares = effect.get_shares_from(elem)

                            # Plot shares if time-based
                            time_shares = [s for s in shares if 'time' in shares[s].solution.dims]

                            if time_shares:
                                df = pd.DataFrame()
                                for share_name in time_shares:
                                    share_df = shares[share_name].solution.to_dataframe()
                                    df[share_name] = share_df[share_name]

                                fig = go.Figure()
                                for col in df.columns:
                                    fig.add_trace(go.Scatter(
                                        x=df.index,
                                        y=df[col],
                                        mode='lines',
                                        name=col
                                    ))

                                fig.update_layout(
                                    title=f"Shares from {elem}",
                                    xaxis_title="Time",
                                    yaxis_title="Share",
                                    height=400
                                )

                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                # Display as simple table
                                share_data = []
                                for share_name in shares:
                                    share_data.append({
                                        "Share": share_name,
                                        "Value": float(shares[share_name].solution.values.flatten()[0])
                                    })

                                if share_data:
                                    st.table(pd.DataFrame(share_data))
                        except Exception as e:
                            st.error(f"Error displaying shares from {elem}: {e}")
    else:
        st.info("No effects available in this calculation.")

# Variables page
elif selected_page == "Variables":
    st.title("Model Variables")

    # Add a filter option
    variable_filter = st.text_input("Filter variables by name:")

    # Get all variables and apply filter
    all_variables = list(results.model.variables)

    if variable_filter:
        filtered_variables = [v for v in all_variables if variable_filter.lower() in v.lower()]
    else:
        filtered_variables = all_variables

    st.write(f"Showing {len(filtered_variables)} of {len(all_variables)} variables")

    # Variable selection
    variable_name = st.selectbox("Select a variable:", filtered_variables)

    if variable_name:
        try:
            variable = results.model.variables[variable_name]
            var_solution = variable.solution

            st.header(f"Variable: {variable_name}")

            # Visualization based on dimensionality
            if 'time' in var_solution.dims:
                # Plot as heatmap toggle
                show_heatmap = st.checkbox("Show as heatmap", value=False)

                if show_heatmap:
                    # Heatmap options in a single row
                    heatmap_cols = st.columns(3)
                    with heatmap_cols[0]:
                        timeframes = st.selectbox(
                            "Timeframes",
                            ["YS", "MS", "W", "D", "h", "15min", "min"],
                            index=3  # Default to "D"
                        )
                    with heatmap_cols[1]:
                        timesteps = st.selectbox(
                            "Timesteps",
                            ["W", "D", "h", "15min", "min"],
                            index=2  # Default to "h"
                        )
                    with heatmap_cols[2]:
                        color_map = st.selectbox(
                            "Colormap",
                            ["portland", "viridis", "plasma", "inferno", "magma", "cividis", "RdBu", "Blues", "YlOrRd"],
                            index=0
                        )

                    try:
                        # Create heatmap using results.plot_heatmap
                        fig = get_plotly_fig(
                            results.plot_heatmap,
                            variable=variable_name,
                            heatmap_timeframes=timeframes,
                            heatmap_timesteps_per_frame=timesteps,
                            color_map=color_map
                        )

                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating heatmap: {e}")
                else:
                    # Regular time series plot
                    try:
                        fig = get_plotly_fig(
                            plotting.with_plotly,
                            data=var_solution.to_dataframe(),
                            mode='area',
                            title=f'Variable: {variable_name}',
                        )
                        fig.update_layout(height=300)

                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f'Error displaying variable {variable_name}: {e}')

                df = var_solution.to_dataframe()

                if st.checkbox("Show data as table"):
                    st.dataframe(df)
            else:
                # Non-time series data
                st.subheader("Values")
                st.write(var_solution.values.item())

            # Basic info
            st.subheader("Stats")
            st.write(f"**Dimensions:** {', '.join(var_solution.dims)}")
            st.write(f"**Shape:** {var_solution.shape}")

        except Exception as e:
            st.error(f"Error displaying variable {variable_name}: {e}")

# Heatmaps page
elif selected_page == "Heatmaps":
    st.title("Heatmap Generator")

    # Get time-based variables
    time_vars = [var_name for var_name, var in results.model.variables.items()
                 if 'time' in var.solution.dims]

    # Variable selection
    variable_name = st.selectbox("Select a variable:", time_vars)

    if variable_name:
        # Configure heatmap settings in one row
        st.subheader("Heatmap Settings")

        # All options in a single row
        cols = st.columns(3)

        with cols[0]:
            timeframes = st.selectbox(
                "Timeframe grouping:",
                ["YS", "MS", "W", "D", "h", "15min", "min"],
                index=3  # Default to "D"
            )

        with cols[1]:
            timesteps = st.selectbox(
                "Timesteps per frame:",
                ["W", "D", "h", "15min", "min"],
                index=2  # Default to "h"
            )

        with cols[2]:
            color_map = st.selectbox(
                "Color map:",
                ["portland", "viridis", "plasma", "inferno", "magma", "cividis", "RdBu", "Blues", "YlOrRd"],
                index=0
            )

        # Generate button
        if st.button("Generate Heatmap"):
            try:
                st.subheader(f"Heatmap for {variable_name}")

                # Use the built-in heatmap function
                fig = get_plotly_fig(
                    results.plot_heatmap,
                    variable=variable_name,
                    heatmap_timeframes=timeframes,
                    heatmap_timesteps_per_frame=timesteps,
                    color_map=color_map
                )

                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating heatmap: {e}")
