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
import xarray as xr

from flixopt import plotting
from flixopt.results import filter_dataset


# Reusable function to display variables
def display_dataset(ds: xr.Dataset, prefix=""):
    """
    Display variables from a dictionary with options for visualization

    Args:
        ds: Dataset to display
        prefix: Prefix for widget keys to avoid collisions
    """
    # Add a filter option
    filter_contains = st.text_input("Filter variables by name:", key=f"{prefix}_filter")

    filtered_ds = filter_dataset(
        ds,
        contains=filter_contains if filter_contains else None,
    )

    # Heatmap options in a single row
    show_heatmap_col, heatmap_col1, heatmap_col2, heatmap_col3 = st.columns(4)
    with show_heatmap_col:
        show_heatmap = st.checkbox('Show as heatmap', value=False, key=f"{prefix}_heatmap")
    with heatmap_col1:
        timeframes = st.selectbox(
            'Timeframes',
            ['YS', 'MS', 'W', 'D', 'h', '15min', 'min'],
            index=3,  # Default to "D"
            key=f"{prefix}_timeframes"
        )
    with heatmap_col2:
        timesteps = st.selectbox(
            'Timesteps',
            ['W', 'D', 'h', '15min', 'min'],
            index=2,  # Default to "h"
            key=f"{prefix}_timesteps"
        )
    with heatmap_col3:
        color_map = st.selectbox(
            'Colormap',
            ['portland', 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'RdBu', 'Blues', 'YlOrRd'],
            index=0,
            key=f"{prefix}_colormap"
        )

    st.write(f"Showing {len(filtered_ds)} of {len(ds)} variables")

    # Display all filtered variables directly
    for name, da in filtered_ds.data_vars.items():
        try:

            # Add a divider for each variable
            st.markdown(f"### {name}")

            # Check if this is a time-based variable
            if len(da.dims) > 0:
                if len(da.dims) == 2:
                    data = da.to_pandas()
                else:
                    data = da.to_dataframe()
                if show_heatmap:
                    try:
                        # Create heatmap using var_solution
                        heatmap_data = plotting.heat_map_data_from_df(
                            data,
                            timeframes,
                            timesteps,
                            'ffill'
                        )

                        fig = plotting.heat_map_plotly(
                            data,
                            title=name,
                            color_map=color_map,
                            xlabel=f'timeframe [{timeframes}]',
                            ylabel=f'timesteps [{timesteps}]'
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

                        st.plotly_chart(fig, theme=None, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating heatmap: {e}")
                else:
                    # Regular time series plot
                    fig = plotting.with_plotly(
                        data,
                        style='stacked_bar',
                        title=f'Variable: {name}',
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, theme='streamlit', use_container_width=True)

                show_datatable = st.checkbox('Show data table', key=f'{prefix}_datatable_{name}', value=False)
                if show_datatable:
                    st.dataframe(data)

            else:
                # Show scalar value
                st.write(f"Value: {da.item()}")
        except Exception as e:
            st.error(f"Error displaying variable {name}: {e}")


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
    pages = ["Overview", "Components", "Buses", "Effects"]
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

                    # Use built-in plotting method
                    if component.is_storage:
                        fig = component.plot_charge_state(show=False, save=False)
                    else:
                        fig = component.plot_node_balance(show=False, save=False)

                    st.plotly_chart(fig, theme='streamlit', use_container_width=True)

                    # Also show as dataframe if requested
                    if st.checkbox("Show Data Table"):
                        if component.is_storage:
                            node_balance = component.node_balance_with_charge_state().to_dataframe()
                        else:
                            node_balance = component.node_balance().to_dataframe()
                        st.dataframe(node_balance)
                except Exception as e:
                    st.error(f"Error displaying the node balance: {e}")

            # Variables tab
            with tabs[1]:
                # Use the reusable function
                display_dataset(component.solution, prefix=f"comp_{component_name}")

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

                    # Use built-in plotting method
                    fig = bus.plot_node_balance(show=False, save=False)
                    st.plotly_chart(fig, theme=None, use_container_width=True)

                    # Also show as dataframe if requested
                    if st.checkbox("Show Data Table"):
                        df = bus.node_balance().to_dataframe()
                        st.dataframe(df)

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
                except Exception as e:
                    st.error(f"Error displaying the node balance: {e}")

            # Variables tab
            with tabs[1]:
                # Use the reusable function
                display_dataset(bus.solution, prefix=f"bus_{bus_name}")

    # Effects page
    elif selected_page == "Effects":
        st.title("Effects")

        # Effect selector
        effect_names = list(results.effects.keys())
        effect_name = st.selectbox("Select an effect:", sorted(effect_names), index=0)
        effect = results.effects[effect_name]

        st.header(f"Effect: {effect_name}")

        display_dataset(effect.solution, prefix=f"effect_{effect_name}")


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
