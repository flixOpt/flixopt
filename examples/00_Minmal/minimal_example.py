"""
This script shows how to use the flixopt framework to model a super minimalistic energy system.
"""

import numpy as np
from rich.pretty import pprint

import flixopt as fx

if __name__ == '__main__':
    # --- Define Thermal Load Profile ---
    # Load profile (e.g., kW) for heating demand over time
    thermal_load_profile = np.array([30, 0, 20])
    datetime_series = fx.create_datetime_array('2020-01-01', 3, 'h')

    # --- Define Energy Buses ---
    # These represent the different energy carriers in the system
    electricity_bus = fx.Bus('Electricity')
    heat_bus = fx.Bus('District Heating')
    fuel_bus = fx.Bus('Natural Gas')

    # --- Define Objective Effect (Cost) ---
    # Cost effect representing the optimization objective (minimizing costs)
    cost_effect = fx.Effect('costs', '€', 'Cost', is_standard=True, is_objective=True)

    # --- Define Flow System Components ---
    # Boiler component with thermal output (heat) and fuel input (gas)
    boiler = fx.linear_converters.Boiler(
        'Boiler',
        eta=0.5,
        Q_th=fx.Flow(label='Thermal Output', bus=heat_bus, size=50),
        Q_fu=fx.Flow(label='Fuel Input', bus=fuel_bus),
    )

    # Heat load component with a fixed thermal demand profile
    heat_load = fx.Sink(
        'Heat Demand',
        sink=fx.Flow(label='Thermal Load', bus=heat_bus, size=1, fixed_relative_profile=thermal_load_profile),
    )

    # Gas source component with cost-effect per flow hour
    gas_source = fx.Source(
        'Natural Gas Tariff',
        source=fx.Flow(label='Gas Flow', bus=fuel_bus, size=1000, effects_per_flow_hour=0.04),  # 0.04 €/kWh
    )

    # --- Build the Flow System ---
    # Add all components and effects to the system
    flow_system = fx.FlowSystem(datetime_series)
    flow_system.add_elements(cost_effect, boiler, heat_load, gas_source)

    # --- Define and Run Calculation ---
    calculation = fx.FullCalculation('Simulation1', flow_system)
    calculation.do_modeling()

    # --- Solve the Calculation and Save Results ---
    calculation.solve(fx.solvers.HighsSolver(), save_results=True)

    # --- Load and Analyze Results ---
    # Load results and plot the operation of the District Heating Bus
    results = fx.results.CalculationResults(calculation.name, folder='results')
    results.plot_operation('District Heating', 'area')

    # Print results to the console. Check Results in file or perform more plotting
    pprint(calculation.results())
    pprint('Look into .yaml and .json file for results')
    pprint(calculation.system_model.main_results)
