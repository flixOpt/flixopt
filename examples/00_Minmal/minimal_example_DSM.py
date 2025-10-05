"""
This script shows how to use the flixopt framework to model a super minimalistic energy system.
"""

import numpy as np
import pandas as pd
from rich.pretty import pprint

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
    heat_load = fx.DSMSink(
        'DSM Sink Heat Demand',
        sink=fx.Flow(label='Heat Load', bus='District Heating', size=150),
        initial_demand=thermal_load_profile,
        maximum_cumulated_deficit = -50,
        maximum_cumulated_surplus = 50,
        maximum_flow_deficit = -20,
        maximum_flow_surplus = 20,
        relative_loss_per_hour_positive_charge_state = 0.05,
        relative_loss_per_hour_negative_charge_state = 0.05,
        penalty_costs_positive_charge_states=0,
        penalty_costs_negative_charge_states=0.01,
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

    # Original plots
    #calculation.results['District Heating'].plot_node_balance_pie()
    calculation.results['District Heating'].plot_node_balance()
    calculation.results['DSM Sink Heat Demand'].plot_DSM_sink()

    # Save the DSM Sink Heat Demand solution dataset to a CSV file
    calculation.results['DSM Sink Heat Demand'].solution.to_dataframe().to_csv('results/DSM_Sink_Heat_Demand_results.csv')

    # Save results to a file
    df2 = calculation.results['District Heating'].node_balance().to_dataframe()
    #df2.to_csv('results/District Heating.csv')  # Save results to csv

    # Print infos to the console.
    pprint(calculation.summary)
