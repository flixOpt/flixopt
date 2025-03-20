"""
THis script shows how to use the flixOpt framework to model a simple energy system.
"""

import numpy as np
from rich.pretty import pprint  # Used for pretty printing

import flixOpt as fx

if __name__ == '__main__':
    # --- Create Time Series Data ---
    # Heat demand profile (e.g., kW) over time and corresponding power prices
    heat_demand_per_h = np.array([30, 0, 90, 110, 110, 20, 20, 20, 20])
    power_prices = 1 / 1000 * np.array([80, 80, 80, 80, 80, 80, 80, 80, 80])

    # Create datetime array starting from '2020-01-01' for the given time period
    time_series = fx.create_datetime_array('2020-01-01', len(heat_demand_per_h))

    # --- Define Energy Buses ---
    # These represent nodes, where the used medias are balanced (electricity, heat, and gas)
    # Medias are used to validating connections, and to define colors for plotting.
    Strom = fx.Bus(label='Strom', medium=fx.Medium('electricity', 'kWh', categories=[fx.MediumCategories.electricity]))
    Fernwaerme = fx.Bus(label='Fernwärme', medium=fx.Medium('heat', 'kWh', categories=[fx.MediumCategories.heat]))
    Gas = fx.Bus(label='Gas', medium=fx.Medium('gas', 'kWh', categories=[fx.MediumCategories.fuel]))

    # --- Define Effects (Objective and CO2 Emissions) ---
    # Cost effect: used as the optimization objective --> minimizing costs
    costs = fx.Effect(
        label='costs',
        unit='€',
        description='Kosten',
        is_standard=True,  # standard effect: no explicit value needed for costs
        is_objective=True,  # Minimizing costs as the optimization objective
    )

    # CO2 emissions effect with an associated cost impact
    CO2 = fx.Effect(
        label='CO2',
        unit='kg',
        description='CO2_e-Emissionen',
        specific_share_to_other_effects_operation={costs: 0.2},
        maximum_operation_per_hour=1000,  # Max CO2 emissions per hour
    )

    # --- Define Flow System Components ---
    # Boiler: Converts fuel (gas) into thermal energy (heat)
    boiler = fx.linear_converters.Boiler(
        label='Boiler',
        eta=0.5,
        Q_th=fx.Flow(label='Q_th', bus=Fernwaerme, size=50, relative_minimum=0.1, relative_maximum=1),
        Q_fu=fx.Flow(label='Q_fu', bus=Gas),
    )

    # Combined Heat and Power (CHP): Generates both electricity and heat from fuel
    chp = fx.linear_converters.CHP(
        label='CHP',
        eta_th=0.5,
        eta_el=0.4,
        P_el=fx.Flow('P_el', bus=Strom, size=60, relative_minimum=5 / 60),
        Q_th=fx.Flow('Q_th', bus=Fernwaerme),
        Q_fu=fx.Flow('Q_fu', bus=Gas),
    )

    # Storage: Energy storage system with charging and discharging capabilities
    storage = fx.Storage(
        label='Storage',
        charging=fx.Flow('Q_th_load', bus=Fernwaerme, size=1000),
        discharging=fx.Flow('Q_th_unload', bus=Fernwaerme, size=1000),
        capacity_in_flow_hours=fx.InvestParameters(fix_effects=20, fixed_size=30, optional=False),
        initial_charge_state=0,  # Initial storage state: empty
        relative_maximum_charge_state=1 / 100 * np.array([80, 70, 80, 80, 80, 80, 80, 80, 80, 80]),
        eta_charge=0.9,
        eta_discharge=1,  # Efficiency factors for charging/discharging
        relative_loss_per_hour=0.08,  # 8% loss per hour. Absolute loss depends on current charge state
        prevent_simultaneous_charge_and_discharge=True,  # Prevent charging and discharging at the same time
    )

    # Heat Demand Sink: Represents a fixed heat demand profile
    heat_sink = fx.Sink(
        label='Heat Demand',
        sink=fx.Flow(label='Q_th_Last', bus=Fernwaerme, size=1, fixed_relative_profile=heat_demand_per_h),
    )

    # Gas Source: Gas tariff source with associated costs and CO2 emissions
    gas_source = fx.Source(
        label='Gastarif',
        source=fx.Flow(label='Q_Gas', bus=Gas, size=1000, effects_per_flow_hour={costs: 0.04, CO2: 0.3}),
    )

    # Power Sink: Represents the export of electricity to the grid
    power_sink = fx.Sink(
        label='Einspeisung', sink=fx.Flow(label='P_el', bus=Strom, effects_per_flow_hour=-1 * power_prices)
    )

    # --- Build the Flow System ---
    # Create the flow system and add all defined components and effects
    flow_system = fx.FlowSystem(time_series=time_series)
    flow_system.add_elements(costs, CO2, boiler, storage, chp, heat_sink, gas_source, power_sink)

    # Visualize the flow system for validation purposes
    flow_system.visualize_network()

    # --- Define and Run Calculation ---
    # Create a calculation object to model the Flow System
    calculation = fx.FullCalculation(name='Sim1', flow_system=flow_system)
    calculation.do_modeling()  # Translate the model to a solvable form, creating equations and Variables

    # --- Solve the Calculation and Save Results ---
    calculation.solve(fx.solvers.HighsSolver(), save_results=True)

    # --- Load and Analyze Results ---
    # Load the results and plot the operation of the District Heating Bus
    results = fx.results.CalculationResults(calculation.name, folder='results')
    results.change_colors({
        'Boiler': 'darkorange',
        'CHP': 'blue',
        'Storage': 'green',
        'Heat Demand': 'orange',
        'Gastarif': 'yellow',
        'Einspeisung': 'lightblue'
    })
    results.plot_operation('Fernwärme', 'area')
    results.plot_storage('Storage')
    results.plot_operation('Fernwärme', 'bar')
    results.plot_operation('Fernwärme', 'line')
    results.plot_operation('CHP__Q_th', 'line')
    results.plot_operation('CHP__Q_th', 'heatmap')

    # Convert the results for the storage component to a dataframe, save and display
    df = results.to_dataframe('Storage')
    df.to_csv('results/Storage.csv')
    print(df)

    pprint(results.all_results)
