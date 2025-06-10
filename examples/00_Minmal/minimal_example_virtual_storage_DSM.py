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
    heat_load = fx.DSMSink(
        'DSM Sink Heat Demand',
        sink=fx.Flow(label='Heat Load', bus='District Heating', size=150),
        initial_demand=thermal_load_profile,
        virtual_capacity_in_flow_hours=100,
        maximum_relative_virtual_charging_rate = 0.2,
        maximum_relative_virtual_discharging_rate = -0.2,
        #relative_loss_per_hour_positive_charge_state = 0.05,
        #relative_loss_per_hour_negative_charge_state = 0.05,
        initial_charge_state = 'lastValueOfSim',
        #penalty_costs_positive_charge_states=0,
        #penalty_costs_negative_charge_states=0.01,
        allow_mixed_charge_states=False,
        forward_timeshift = 3,
        backward_timeshift = 3
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
    calculation.solve(fx.solvers.GurobiSolver(0.001, 60))

    # --- Analyze Results ---
    # Access the results of an element
    df1 = calculation.results['costs'].filter_solution('time').to_dataframe()

    # Create a custom plot showing node balance, initial demand and charge states
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from flixopt import plotting

    # Get the data
    node_balance = calculation.results['District Heating'].node_balance(with_last_timestep=True).to_dataframe()
    dsm_results = calculation.results['DSM Sink Heat Demand']
    
    # Get initial demand from the flow system's time series collection
    initial_demand = calculation.flow_system.time_series_collection.time_series_data[f'{dsm_results.label}|initial_demand'].active_data.to_dataframe(name='initial_demand')
    
    # Get charge states from the solution
    positive_charge = dsm_results.solution[f'{dsm_results.label}|positive_charge_state'].to_dataframe()
    negative_charge = dsm_results.solution[f'{dsm_results.label}|negative_charge_state'].to_dataframe()

    # Create figure with secondary y-axis using the same style as node balance
    fig = plotting.with_plotly(
        node_balance,
        mode='area',
        colors='viridis',
        title='District Heating Node Balance with DSM Charge States',
        ylabel='Power [kW]',
        xlabel='Time'
    )

    # Get colors from viridis for the charge states
    import plotly.express as px
    # Use a more muted color scale
    viridis_colors = px.colors.sample_colorscale('viridis', 4)
    positive_color = viridis_colors[1]  # Use a blue-ish color from viridis
    negative_color = viridis_colors[0]  # Use a violette-ish color from viridis

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

    # Add charge states as bars on secondary y-axis with reduced opacity
    fig.add_trace(
        go.Bar(
            x=positive_charge.index,
            y=positive_charge.values.flatten(),
            name='Positive Charge State',
            marker=dict(color=positive_color, opacity=0.7),  # Add opacity for less saturation
            yaxis='y2'
        )
    )

    fig.add_trace(
        go.Bar(
            x=negative_charge.index,
            y=negative_charge.values.flatten(),
            name='Negative Charge State',
            marker=dict(color=negative_color, opacity=0.7),  # Add opacity for less saturation
            yaxis='y2'
        )
    )

    # Update layout for secondary y-axis and bar styling
    fig.update_layout(
        yaxis2=dict(
            title='Charge State [kWh]',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        hovermode='x unified',
        bargap=0,  # No gap between bars
        bargroupgap=0  # No gap between bar groups
    )

    # Show the plot
    fig.show()

    # Original plots
    #calculation.results['District Heating'].plot_node_balance_pie()
    #calculation.results['District Heating'].plot_node_balance()

    # Save the DSM Sink Heat Demand solution dataset to a CSV file
    calculation.results['DSM Sink Heat Demand'].solution.to_dataframe().to_csv('results/DSM_Sink_Heat_Demand_results.csv')

    # Save results to a file
    df2 = calculation.results['District Heating'].node_balance().to_dataframe()
    #df2.to_csv('results/District Heating.csv')  # Save results to csv

    # Print infos to the console.
    pprint(calculation.summary)
