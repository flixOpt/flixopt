"""
This script shows how to use the flixopt framework to model a simple energy system.
"""

import numpy as np
import pandas as pd

import flixopt as fx

if __name__ == '__main__':
    fx.CONFIG.exploring()

    # --- Create Time Series Data ---
    # Heat demand profile (e.g., kW) over time and corresponding power prices
    heat_demand_per_h = np.array([30, 0, 90, 110, 110, 20, 20, 20, 20])
    power_prices = 1 / 1000 * np.array([80, 80, 80, 80, 80, 80, 80, 80, 80])

    # Create datetime array starting from '2020-01-01' for the given time period
    timesteps = pd.date_range('2020-01-01', periods=len(heat_demand_per_h), freq='h')
    flow_system = fx.FlowSystem(timesteps=timesteps)

    # --- Define Energy Buses ---
    # These represent nodes, where the used medias are balanced (electricity, heat, and gas)
    flow_system.add_elements(fx.Bus(label='Strom'), fx.Bus(label='Fernwärme'), fx.Bus(label='Gas'))

    # --- Define Effects (Objective and CO2 Emissions) ---
    # Cost effect: used as the optimization objective --> minimizing costs
    costs = fx.Effect(
        label='costs',
        unit='€',
        description='Kosten',
        is_standard=True,  # standard effect: no explicit value needed for costs
        is_objective=True,  # Minimizing costs as the optimization objective
        share_from_temporal={'CO2': 0.2},
    )

    # CO2 emissions effect with an associated cost impact
    CO2 = fx.Effect(
        label='CO2',
        unit='kg',
        description='CO2_e-Emissionen',
        maximum_per_hour=1000,  # Max CO2 emissions per hour
    )

    # --- Define Flow System Components ---
    # Boiler: Converts fuel (gas) into thermal energy (heat)
    boiler = fx.linear_converters.Boiler(
        label='Boiler',
        thermal_efficiency=0.5,
        thermal_flow=fx.Flow(label='Q_th', bus='Fernwärme', size=50, relative_minimum=0.1, relative_maximum=1),
        fuel_flow=fx.Flow(label='Q_fu', bus='Gas'),
    )

    # Combined Heat and Power (CHP): Generates both electricity and heat from fuel
    chp = fx.linear_converters.CHP(
        label='CHP',
        thermal_efficiency=0.5,
        electrical_efficiency=0.4,
        electrical_flow=fx.Flow('P_el', bus='Strom', size=60, relative_minimum=5 / 60),
        thermal_flow=fx.Flow('Q_th', bus='Fernwärme'),
        fuel_flow=fx.Flow('Q_fu', bus='Gas'),
    )

    # Storage: Energy storage system with charging and discharging capabilities
    storage = fx.Storage(
        label='Storage',
        charging=fx.Flow('Q_th_load', bus='Fernwärme', size=1000),
        discharging=fx.Flow('Q_th_unload', bus='Fernwärme', size=1000),
        capacity_in_flow_hours=fx.InvestParameters(effects_of_investment=20, fixed_size=30, mandatory=True),
        initial_charge_state=0,  # Initial storage state: empty
        relative_maximum_charge_state=1 / 100 * np.array([80, 70, 80, 80, 80, 80, 80, 80, 80]),
        relative_maximum_final_charge_state=0.8,
        eta_charge=0.9,
        eta_discharge=1,  # Efficiency factors for charging/discharging
        relative_loss_per_hour=0.08,  # 8% loss per hour. Absolute loss depends on current charge state
        prevent_simultaneous_charge_and_discharge=True,  # Prevent charging and discharging at the same time
    )

    # Heat Demand Sink: Represents a fixed heat demand profile
    heat_sink = fx.Sink(
        label='Heat Demand',
        inputs=[fx.Flow(label='Q_th_Last', bus='Fernwärme', size=1, fixed_relative_profile=heat_demand_per_h)],
    )

    # Gas Source: Gas tariff source with associated costs and CO2 emissions
    gas_source = fx.Source(
        label='Gastarif',
        outputs=[
            fx.Flow(label='Q_Gas', bus='Gas', size=1000, effects_per_flow_hour={costs.label: 0.04, CO2.label: 0.3})
        ],
    )

    # Power Sink: Represents the export of electricity to the grid
    power_sink = fx.Sink(
        label='Einspeisung', inputs=[fx.Flow(label='P_el', bus='Strom', effects_per_flow_hour=-1 * power_prices)]
    )

    # --- Build the Flow System ---
    # Add all defined components and effects to the flow system
    flow_system.add_elements(costs, CO2, boiler, storage, chp, heat_sink, gas_source, power_sink)

    # Visualize the flow system for validation purposes
    flow_system.topology.plot()

    # --- Define and Solve Optimization ---
    flow_system.optimize(fx.solvers.HighsSolver(mip_gap=0, time_limit_seconds=30))

    # --- Analyze Results ---
    # Plotting through statistics accessor - returns PlotResult with .data and .figure
    flow_system.statistics.plot.balance('Fernwärme')
    flow_system.statistics.plot.balance('Storage')
    flow_system.statistics.plot.heatmap('CHP(Q_th)|flow_rate')
    flow_system.statistics.plot.heatmap('Storage|charge_state')

    # Access data as xarray Datasets
    print(flow_system.statistics.flow_rates)
    print(flow_system.statistics.charge_states)

    # Duration curve and effects analysis
    flow_system.statistics.plot.duration_curve('Boiler(Q_th)|flow_rate')
    print(flow_system.statistics.temporal_effects)
