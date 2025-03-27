"""
This script shows how to use the flixopt framework to model a super minimalistic energy system.
"""

import numpy as np
import pandas as pd
from rich.pretty import pprint

import flixopt as fx

if __name__ == '__main__':
    # --- Define the Flow System, that will hold all elements, and the time steps you want to model ---
    timesteps = pd.date_range('2020-01-01', periods=3, freq='h')
    flow_system = fx.FlowSystem(timesteps)

    # --- Define Thermal Load Profile ---
    # Load profile (e.g., kW) for heating demand over time
    thermal_load_profile = np.array([30, 0, 20])

    # --- Define Energy Buses ---
    # These are balancing nodes (inputs=outputs) and balance the different energy carriers your system
    flow_system.add_elements(fx.Bus('District Heating'), fx.Bus('Natural Gas'))

    # --- Define Objective Effect (Cost) ---
    # Cost effect representing the optimization objective (minimizing costs)
    cost_effect = fx.Effect('costs', '€', 'Cost', is_standard=True, is_objective=True)

    # --- Define Flow System Components ---
    # Boiler component with thermal output (heat) and fuel input (gas)
    boiler = fx.linear_converters.Boiler(
        'Boiler',
        eta=0.5,
        Q_th=fx.Flow(label='Thermal Output', bus='District Heating', size=50),
        Q_fu=fx.Flow(label='Fuel Input', bus='Natural Gas'),
    )

    # Heat load component with a fixed thermal demand profile
    heat_load = fx.Sink(
        'Heat Demand',
        sink=fx.Flow(label='Thermal Load', bus='District Heating', size=1, fixed_relative_profile=thermal_load_profile),
    )

    # Gas source component with cost-effect per flow hour
    gas_source = fx.Source(
        'Natural Gas Tariff',
        source=fx.Flow(label='Gas Flow', bus='Natural Gas', size=1000, effects_per_flow_hour=0.04),  # 0.04 €/kWh
    )

    # --- Build the Flow System ---
    # Add all components and effects to the system
    flow_system.add_elements(cost_effect, boiler, heat_load, gas_source)

    # --- Define, model and solve a Calculation ---
    calculation = fx.FullCalculation('Simulation1', flow_system)
    calculation.do_modeling()
    calculation.solve(fx.solvers.HighsSolver(0.01, 60))

    # --- Analyze Results ---
    # Access the results of an element
    df1 = calculation.results['costs'].filter_solution('time').to_dataframe()

    # Plot the results of a specific element
    calculation.results['District Heating'].plot_node_balance_pie()
    calculation.results['District Heating'].plot_node_balance()

    # Save results to a file
    df2 = calculation.results['District Heating'].node_balance().to_dataframe()
    # df2.to_csv('results/District Heating.csv')  # Save results to csv

    # Print infos to the console.
    pprint(calculation.summary)
