import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import tempfile
import webbrowser
import subprocess
import os
import atexit
import shutil
import sys
from typing import Union, Optional, Literal, Dict, List

def explore_results(self, port: int = 8501):
    """
    Launch a Streamlit app to explore the calculation results.
    
    This method creates a temporary Streamlit script that directly references
    the current CalculationResults object and launches it in a browser.
    
    Args:
        port: Port to use for the Streamlit server
        
    Returns:
        subprocess.Popen: The running Streamlit process
    """
    # Create a temporary file for the Streamlit app
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as f:
        app_path = f.name
        app_code = f"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
from pathlib import Path

# Add the current directory to path so we can import the calculation results
sys.path.append(r"{os.getcwd()}")

# Set page config
st.set_page_config(
    page_title="FlixOpt Results Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache the calculation loading
@st.cache_resource
def get_calculation_results():
    # Import the relevant modules
    try:
        # Load the model from saved files (more reliable)
        from flixOpt.results import CalculationResults
        return CalculationResults.from_file(r"{self.folder}", "{self.name}")
    except Exception as e:
        st.error(f"Error loading calculation results: {{e}}")
        return None

# Load the calculation results
results = get_calculation_results()

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
        st.write(f"**Name:** {{results.name}}")
        st.write(f"**Time Steps:** {{len(results.timesteps_extra)}}")
        if len(results.timesteps_extra) > 0:
            st.write(f"**Time Range:** {{results.timesteps_extra[0]}} to {{results.timesteps_extra[-1]}}")
    
    with col2:
        st.write(f"**Components:** {{len(results.components)}}")
        st.write(f"**Buses:** {{len(results.buses)}}")
        st.write(f"**Effects:** {{len(results.effects)}}")
    
    # Additional info
    if hasattr(results, 'infos') and results.infos:
        st.subheader("Additional Information")
        st.json(results.infos)
    
    # Network visualization (simplified)
    st.header("Network Structure")
    
    # Create a list of components and buses for network diagram
    nodes = []
    edges = []
    
    # Add components as nodes
    for comp_name in results.components:
        nodes.append({{"id": comp_name, "type": "component"}})
    
    # Add buses as nodes
    for bus_name in results.buses:
        nodes.append({{"id": bus_name, "type": "bus"}})
    
    # Show component connections
    st.subheader("Component Connections")
    connections_data = []
    
    for comp_name, comp in results.components.items():
        for bus_name in comp.inputs + comp.outputs:
            connections_data.append({{"Component": comp_name, "Bus": bus_name, 
                                   "Type": "Input" if bus_name in comp.inputs else "Output"}})
    
    if connections_data:
        st.dataframe(pd.DataFrame(connections_data))

# Components page
elif selected_page == "Components":
    st.title("Components")
    
    # Component selector
    component_names = list(results.components.keys())
    component_name = st.selectbox("Select a component:", component_names)
    
    if component_name:
        component = results.components[component_name]
        
        st.header(f"Component: {{component_name}}")
        
        # Component tabs
        tabs = st.tabs(["Flow Rates", "Charge State (if storage)", "Variables"])
        
        # Flow Rates tab
        with tabs[0]:
            try:
                flow_rates = component.flow_rates(with_last_timestep=True).to_dataframe()
                
                if not flow_rates.empty:
                    st.subheader("Flow Rates")
                    fig = go.Figure()
                    
                    for column in flow_rates.columns:
                        fig.add_trace(go.Scatter(
                            x=flow_rates.index,
                            y=flow_rates[column],
                            mode='lines',
                            name=column
                        ))
                    
                    fig.update_layout(
                        title=f"Flow Rates for {{component_name}}",
                        xaxis_title="Time",
                        yaxis_title="Flow Rate",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Also show as dataframe if requested
                    if st.checkbox("Show flow rates as table"):
                        st.dataframe(flow_rates)
                else:
                    st.info("No flow rates data available for this component.")
            except Exception as e:
                st.error(f"Error displaying flow rates: {{e}}")
        
        # Charge State tab
        with tabs[1]:
            if component.is_storage:
                try:
                    st.subheader("Charge State")
                    
                    # Get charge state data
                    charge_state = component.charge_state.solution.to_dataframe()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=charge_state.index,
                        y=charge_state.values.flatten(),
                        mode='lines',
                        name='Charge State'
                    ))
                    
                    fig.update_layout(
                        title=f"Charge State for {{component_name}}",
                        xaxis_title="Time",
                        yaxis_title="Charge",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show statistics
                    st.subheader("Charge State Statistics")
                    charge_vals = charge_state.values.flatten()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Minimum", f"{{charge_vals.min():.2f}}")
                    col2.metric("Maximum", f"{{charge_vals.max():.2f}}")
                    col3.metric("Average", f"{{charge_vals.mean():.2f}}")
                    col4.metric("Final", f"{{charge_vals[-1]:.2f}}")
                    
                    # Also show as dataframe if requested
                    if st.checkbox("Show charge state as table"):
                        st.dataframe(charge_state)
                except Exception as e:
                    st.error(f"Error displaying charge state: {{e}}")
            else:
                st.info(f"Component {{component_name}} is not a storage component.")
        
        # Variables tab
        with tabs[2]:
            st.subheader("Component Variables")
            
            for var_name in component._variables:
                try:
                    var = component.variables[var_name]
                    
                    # Create an expander for each variable
                    with st.expander(f"Variable: {{var_name}}"):
                        var_solution = var.solution
                        
                        # Check if this is a time-based variable
                        if 'time' in var_solution.dims:
                            # Plot time series
                            df = var_solution.to_dataframe()
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=df.index,
                                y=df[var_name],
                                mode='lines',
                                name=var_name
                            ))
                            
                            fig.update_layout(
                                title=f"{{var_name}} Time Series",
                                xaxis_title="Time",
                                yaxis_title="Value",
                                height=300
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            # Show scalar value
                            st.write(f"Value: {{var_solution.values}}")
                except Exception as e:
                    st.error(f"Error displaying variable {{var_name}}: {{e}}")

# Buses page
elif selected_page == "Buses":
    st.title("Buses")
    
    # Bus selector
    bus_names = list(results.buses.keys())
    bus_name = st.selectbox("Select a bus:", bus_names)
    
    if bus_name:
        bus = results.buses[bus_name]
        
        st.header(f"Bus: {{bus_name}}")
        
        # Show flow rates
        try:
            flow_rates = bus.flow_rates(with_last_timestep=True).to_dataframe()
            
            if not flow_rates.empty:
                st.subheader("Flow Rates")
                fig = go.Figure()
                
                for column in flow_rates.columns:
                    fig.add_trace(go.Scatter(
                        x=flow_rates.index,
                        y=flow_rates[column],
                        mode='lines',
                        name=column
                    ))
                
                fig.update_layout(
                    title=f"Flow Rates for {{bus_name}}",
                    xaxis_title="Time",
                    yaxis_title="Flow Rate",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate and show balance
                st.subheader("Flow Balance")
                
                inputs = [col for col in flow_rates.columns if col in bus.inputs]
                outputs = [col for col in flow_rates.columns if col in bus.outputs]
                
                balance_df = pd.DataFrame(index=flow_rates.index)
                
                if inputs:
                    balance_df['Total Input'] = flow_rates[inputs].sum(axis=1)
                else:
                    balance_df['Total Input'] = 0
                    
                if outputs:
                    balance_df['Total Output'] = flow_rates[outputs].sum(axis=1)
                else:
                    balance_df['Total Output'] = 0
                
                balance_df['Net Flow'] = balance_df['Total Input'] + balance_df['Total Output']
                
                fig = go.Figure()
                for column in balance_df.columns:
                    fig.add_trace(go.Scatter(
                        x=balance_df.index,
                        y=balance_df[column],
                        mode='lines',
                        name=column
                    ))
                
                fig.update_layout(
                    title=f"Flow Balance for {{bus_name}}",
                    xaxis_title="Time",
                    yaxis_title="Flow Rate",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Also show as dataframe if requested
                if st.checkbox("Show flow data as table"):
                    st.dataframe(flow_rates)
            else:
                st.info("No flow rates data available for this bus.")
        except Exception as e:
            st.error(f"Error displaying flow rates: {{e}}")
        
        # Show inputs and outputs
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Inputs")
            for input_name in bus.inputs:
                st.write(f"- {{input_name}}")
        
        with col2:
            st.subheader("Outputs")
            for output_name in bus.outputs:
                st.write(f"- {{output_name}}")

# Effects page
elif selected_page == "Effects":
    st.title("Effects")
    
    # Effect selector
    effect_names = list(results.effects.keys())
    
    if effect_names:
        effect_name = st.selectbox("Select an effect:", effect_names)
        
        if effect_name:
            effect = results.effects[effect_name]
            
            st.header(f"Effect: {{effect_name}}")
            
            # List variables
            st.subheader("Variables")
            
            for var_name in effect._variables:
                try:
                    var = effect.variables[var_name]
                    
                    # Create an expander for each variable
                    with st.expander(f"Variable: {{var_name}}"):
                        var_solution = var.solution
                        
                        # Check if this is a time-based variable
                        if 'time' in var_solution.dims:
                            # Plot time series
                            df = var_solution.to_dataframe()
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=df.index,
                                y=df[var_name],
                                mode='lines',
                                name=var_name
                            ))
                            
                            fig.update_layout(
                                title=f"{{var_name}} Time Series",
                                xaxis_title="Time",
                                yaxis_title="Value",
                                height=300
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            # Show scalar value
                            st.write(f"Value: {{var_solution.values}}")
                except Exception as e:
                    st.error(f"Error displaying variable {{var_name}}: {{e}}")
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
    
    st.write(f"Showing {{len(filtered_variables)}} of {{len(all_variables)}} variables")
    
    # Variable selection
    variable_name = st.selectbox("Select a variable:", filtered_variables)
    
    if variable_name:
        try:
            variable = results.model.variables[variable_name]
            var_solution = variable.solution
            
            st.header(f"Variable: {{variable_name}}")
            
            # Basic info
            st.subheader("Information")
            st.write(f"**Dimensions:** {{', '.join(var_solution.dims)}}")
            st.write(f"**Shape:** {{var_solution.shape}}")
            
            # Visualization based on dimensionality
            if 'time' in var_solution.dims:
                st.subheader("Time Series")
                
                df = var_solution.to_dataframe()
                
                # Simple case: just time dimension
                if len(df.columns) == 1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df[variable_name],
                        mode='lines',
                        name=variable_name
                    ))
                    
                    fig.update_layout(
                        title=f"{{variable_name}} Time Series",
                        xaxis_title="Time",
                        yaxis_title="Value",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Also show as dataframe if requested
                    if st.checkbox("Show data as table"):
                        st.dataframe(df)
                else:
                    # Multi-dimensional
                    st.write("This variable has multiple dimensions. Choose visualization type:")
                    
                    viz_type = st.radio(
                        "Visualization type:",
                        ["Line chart (all dimensions)", "Heatmap", "Raw data table"]
                    )
                    
                    if viz_type == "Line chart (all dimensions)":
                        fig = go.Figure()
                        
                        # Limited to first 20 dimensions to avoid overloading
                        columns_to_plot = list(df.columns)[:20]
                        
                        if len(df.columns) > 20:
                            st.warning(f"Variable has {{len(df.columns)}} dimensions. Showing only first 20.")
                        
                        for column in columns_to_plot:
                            fig.add_trace(go.Scatter(
                                x=df.index,
                                y=df[column],
                                mode='lines',
                                name=str(column)
                            ))
                        
                        fig.update_layout(
                            title=f"{{variable_name}} Time Series (Multiple Dimensions)",
                            xaxis_title="Time",
                            yaxis_title="Value",
                            height=600
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    elif viz_type == "Heatmap":
                        # Try to use the built-in heatmap function
                        try:
                            fig = results.plot_heatmap(
                                variable=variable_name,
                                show=False,
                                save=False
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error creating heatmap: {{e}}")
                    
                    elif viz_type == "Raw data table":
                        st.dataframe(df)
            else:
                # Non-time series data
                st.subheader("Values")
                st.write(var_solution.values)
        except Exception as e:
            st.error(f"Error displaying variable {{variable_name}}: {{e}}")

# Heatmaps page
elif selected_page == "Heatmaps":
    st.title("Heatmap Generator")
    
    # Get time-based variables
    time_vars = [var_name for var_name, var in results.model.variables.items() 
                if 'time' in var.solution.dims]
    
    # Variable selection
    variable_name = st.selectbox("Select a variable:", time_vars)
    
    if variable_name:
        # Configure heatmap settings
        st.subheader("Heatmap Settings")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            timeframes = st.selectbox(
                "Timeframe grouping:",
                ["YS", "MS", "W", "D", "h", "15min", "min"],
                index=2  # Default to "W"
            )
        
        with col2:
            timesteps = st.selectbox(
                "Timesteps per frame:",
                ["W", "D", "h", "15min", "min"],
                index=2  # Default to "h"
            )
        
        with col3:
            color_map = st.selectbox(
                "Color map:",
                ["portland", "viridis", "plasma", "inferno", "magma", "cividis", "RdBu", "Blues", "YlOrRd"],
                index=0
            )
        
        # Generate button
        if st.button("Generate Heatmap"):
            try:
                st.subheader(f"Heatmap for {{variable_name}}")
                
                fig = results.plot_heatmap(
                    variable=variable_name,
                    heatmap_timeframes=timeframes,
                    heatmap_timesteps_per_frame=timesteps,
                    color_map=color_map,
                    show=False,
                    save=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating heatmap: {{e}}")
"""
        
        f.write(app_code.encode('utf-8'))
    
    # Construct the command to run Streamlit
    cmd = [sys.executable, "-m", "streamlit", "run", app_path, "--server.port", str(port)]
    
    # Launch the Streamlit app
    process = subprocess.Popen(cmd)
    
    # Open browser
    webbrowser.open(f'http://localhost:{port}')
    
    # Set up cleanup of temporary file when the process exits
    def cleanup_temp_file():
        try:
            os.unlink(app_path)
            print(f"Cleaned up temporary app file: {app_path}")
        except Exception as e:
            print(f"Error cleaning up temporary app file: {e}")
    
    atexit.register(cleanup_temp_file)
    
    print(f"Streamlit app launched on port {port}. Press Ctrl+C to stop the app.")
    
    return process
