"""
This script shows how to use the flixopt framework to model a super minimalistic energy system.
"""

import numpy as np
import pandas as pd
from rich.pretty import pprint

import sys
sys.path.append("C:/Florian/Studium/RES/2025SoSe/Studienarbeit/code/flixopt")
import flixopt as fx

if __name__ == '__main__':
    # --- Define the Flow System, that will hold all elements, and the time steps you want to model ---
    timesteps = pd.date_range('2020-01-01', periods=24, freq='h')
    flow_system = fx.FlowSystem(timesteps)

    # --- Define Thermal Load Profile ---
    # Load profile (e.g., kW) for heating demand over time
    thermal_load_profile = np.array([80, 80, 80, 80, 80, 80, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 80, 80, 80, 80, 80, 80])
    #thermal_load_profile = np.array([100, 100, 100, 100, 100, 100, 120, 120, 120, 100, 100, 100, 100, 100, 100, 80, 80, 80, 100, 100, 100, 100, 100, 100])
    #thermal_load_profile = np.array([80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80])

    # --- Define Energy Buses ---
    # These are balancing nodes (inputs=outputs) and balance the different energy carriers your system
    flow_system.add_elements(fx.Bus('District Heating'), fx.Bus('Natural Gas'))

    # --- Define Objective Effect (Cost) ---
    # Cost effect representing the optimization objective (minimizing costs)
    cost_effect = fx.Effect('costs', '€', 'Cost', is_standard=True, is_objective=True)

    # --- Define Flow System Components ---
    # Boiler component with thermal output (heat) and fuel input (gas)
    boiler1 = fx.linear_converters.Boiler(
        'Boiler1',
        eta=0.5,
        Q_th=fx.Flow(label='Thermal Output', bus='District Heating', size=100),
        Q_fu=fx.Flow(label='Fuel Input', bus='Natural Gas'),
    )
    boiler2 = fx.linear_converters.Boiler(
        'Boiler2',
        eta=1/3,
        Q_th=fx.Flow(label='Thermal Output', bus='District Heating', size=50),
        Q_fu=fx.Flow(label='Fuel Input', bus='Natural Gas'),
    )

    # Heat load component with a fixed thermal demand profile
    heat_load = fx.DSMSinkTS(
        'DSM Sink Heat Demand',
        sink=fx.Flow(label='Heat Load', bus='District Heating', size=150),
        initial_demand=thermal_load_profile,
        forward_timeshift=3,
        backward_timeshift=3,
        maximum_flow_surplus_per_hour=20,
        maximum_flow_deficit_per_hour=-20,
    )

    # Gas source component with cost-effect per flow hour
    gas_source = fx.Source(
        'Natural Gas Tariff',
        source=fx.Flow(label='Gas Flow', bus='Natural Gas', size=1000, effects_per_flow_hour=0.04),  # 0.04 €/kWh
    )

    # --- Build the Flow System ---
    # Add all components and effects to the system
    flow_system.add_elements(cost_effect, boiler1, boiler2, heat_load, gas_source)

    # --- Define, model and solve a Calculation ---
    calculation = fx.FullCalculation('Simulation1', flow_system)
    calculation.do_modeling()
    #calculation.solve(fx.solvers.HighsSolver(0.01, 60))
    calculation.solve(fx.solvers.GurobiSolver(0.01, 60))

    # --- Analyze Results ---
    # Access the results of an element
    df1 = calculation.results['costs'].filter_solution('time').to_dataframe()

    # Create a custom plot showing node balance, initial demand and surplus/deficit
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from flixopt import plotting

    # Get the data
    node_balance = calculation.results['District Heating'].node_balance(with_last_timestep=True).to_dataframe()
    dsm_results = calculation.results['DSM Sink Heat Demand']
    
    # Get initial demand from the flow system's time series collection
    initial_demand = calculation.flow_system.time_series_collection.time_series_data[f'{dsm_results.label}|initial_demand'].active_data.to_dataframe(name='initial_demand')
    
    # Get surplus and deficit from the solution
    surplus = dsm_results.solution[f'{dsm_results.label}|surplus'].to_dataframe()
    
    # Get the number of timesteps from the component's model
    timesteps_backward = calculation.flow_system.components['DSM Sink Heat Demand'].timesteps_backward
    timesteps_forward = calculation.flow_system.components['DSM Sink Heat Demand'].timesteps_forward
    
    # For timeshift DSM, deficit is split into pre and post timesteps
    # Initialize deficit DataFrames with zeros
    deficit_pre = pd.DataFrame(0, index=surplus.index, columns=['deficit_pre'])
    deficit_post = pd.DataFrame(0, index=surplus.index, columns=['deficit_post'])
    
    # Sum up all pre and post deficits
    for i in range(1, timesteps_backward + 1):
        pre_df = dsm_results.solution[f'{dsm_results.label}|deficit_pre_{i}'].to_dataframe()
        deficit_pre['deficit_pre'] += pre_df.values.flatten()
    
    for i in range(1, timesteps_forward + 1):
        post_df = dsm_results.solution[f'{dsm_results.label}|deficit_post_{i}'].to_dataframe()
        deficit_post['deficit_post'] += post_df.values.flatten()
    
    # Combine deficits
    deficit = pd.DataFrame(0, index=surplus.index, columns=['deficit'])
    deficit['deficit'] = deficit_pre['deficit_pre'] + deficit_post['deficit_post']

    # Create figure with area plot for node balance
    fig = plotting.with_plotly(
        node_balance,
        mode='area',
        colors='viridis',
        title='District Heating Node Balance with DSM Surplus/Deficit',
        ylabel='Power [kW]',
        xlabel='Time'
    )

    # Get colors from viridis for the surplus/deficit
    import plotly.express as px
    viridis_colors = px.colors.sample_colorscale('viridis', 8)
    surplus_color = viridis_colors[4]  # Use a blue-ish color from viridis
    deficit_color = viridis_colors[2]  # Use a violette-ish color from viridis
    cumulated_color = viridis_colors[3]  # Use another color from viridis for cumulated flow

    # Add initial demand with step lines (no interpolation)
    fig.add_trace(
        go.Scatter(
            x=initial_demand.index,
            y=initial_demand['initial_demand'],
            name='Initial Demand',
            line=dict(dash='dash', color='black', shape='hv'),  # 'hv' for horizontal-vertical steps
            mode='lines'  
        )
    )

    # Add surplus and deficit as area plots with similar style to node balance
    fig.add_trace(
        go.Scatter(
            x=surplus.index,
            y=surplus.values.flatten(),
            name='Surplus',
            fill='tonexty',  # Fill to the next trace
            line=dict(color=surplus_color, width=1, shape='hv'),  # Thin line for the area border
            mode='lines',
            stackgroup='one'  # Stack with other traces in the same group
        )
    )

    fig.add_trace(
        go.Scatter(
            x=deficit.index,
            y=deficit['deficit'],
            name='Deficit',
            fill='tonexty',  # Fill to the next trace
            line=dict(color=deficit_color, width=1, shape='hv'),  # Thin line for the area border
            mode='lines',
            stackgroup='two'  # Stack with other traces in the same group
        )
    )

    # Get cumulated flow deviation from the model
    cumulated_flow = dsm_results.solution[f'{dsm_results.label}|cumulated_flow_deviation'].to_dataframe()

    # Add cumulated flow deviation as diamonds on secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=cumulated_flow.index,
            y=cumulated_flow.values.flatten(),
            name='Cumulated Flow Deviation',
            mode='markers',
            marker=dict(
                color=cumulated_color,
                size=10,
                symbol='diamond',
                line=dict(width=1, color='black')
            ),
            yaxis='y2'
        )
    )

    # Update layout to include secondary y-axis
    fig.update_layout(
        hovermode='x unified',
        yaxis=dict(
            range=[-1.2*max(0, node_balance.max().max(), -cumulated_flow.min().min(), -deficit.min().min()), 1.2*max(node_balance.max().max(), surplus.max().max(), cumulated_flow.max().max(), initial_demand.max().max())],
            showgrid=True
        ),
        yaxis2=dict(
            title='Cumulated Flow [kWh]',
            overlaying='y',
            side='right',
            showgrid=False,
            range=[-1.2*max(0, node_balance.max().max(), -cumulated_flow.min().min(), -deficit.min().min()), 1.2*max(node_balance.max().max(), surplus.max().max(), cumulated_flow.max().max(), initial_demand.max().max())]
        )
    )

    # Show the plot
    fig.show()

    # Original plots
    #calculation.results['District Heating'].plot_node_balance_pie()
    #calculation.results['District Heating'].plot_node_balance()

    # Save the DSM Sink Heat Demand solution dataset to a CSV file
    calculation.results['DSM Sink Heat Demand'].solution.to_dataframe().to_csv('results/DSM_Sink_Heat_Demand_results.csv')
    calculation.results.solution.to_dask_dataframe().to_csv('results/results.csv')

    # Save results to a file
    df2 = calculation.results['District Heating'].node_balance().to_dataframe()
    #df2.to_csv('results/District Heating.csv')  # Save results to csv

    # Print infos to the console.
    pprint(calculation.summary)
