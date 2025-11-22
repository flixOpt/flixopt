"""
This script shows how to use the flixopt framework to model a simple energy system.
"""

import numpy as np
import pandas as pd

import flixopt as fx

if __name__ == '__main__':
    fx.CONFIG.exploring()

    # Create datetime array starting from '2020-01-01' for one week
    timesteps = pd.date_range('2020-01-01', periods=24 * 7, freq='h')
    scenarios = pd.Index(['Base Case', 'High Demand'])
    periods = pd.Index([2020, 2021, 2022])

    # --- Create Time Series Data ---
    # Realistic daily patterns: morning/evening peaks, night/midday lows
    np.random.seed(42)
    n_hours = len(timesteps)

    # Heat demand: 24-hour patterns (kW) for Base Case and High Demand scenarios
    base_daily_pattern = np.array(
        [22, 20, 18, 18, 20, 25, 40, 70, 95, 110, 85, 65, 60, 58, 62, 68, 75, 88, 105, 125, 130, 122, 95, 35]
    )
    high_daily_pattern = np.array(
        [28, 25, 22, 22, 24, 30, 52, 88, 118, 135, 105, 80, 75, 72, 75, 82, 92, 108, 128, 148, 155, 145, 115, 48]
    )

    # Tile and add variation
    base_demand = np.tile(base_daily_pattern, n_hours // 24 + 1)[:n_hours] * (
        1 + np.random.uniform(-0.05, 0.05, n_hours)
    )
    high_demand = np.tile(high_daily_pattern, n_hours // 24 + 1)[:n_hours] * (
        1 + np.random.uniform(-0.07, 0.07, n_hours)
    )

    heat_demand_per_h = pd.DataFrame({'Base Case': base_demand, 'High Demand': high_demand}, index=timesteps)

    # Power prices: hourly factors (night low, peak high) and period escalation (2020-2022)
    hourly_price_factors = np.array(
        [
            0.70,
            0.65,
            0.62,
            0.60,
            0.62,
            0.70,
            0.95,
            1.15,
            1.30,
            1.25,
            1.10,
            1.00,
            0.95,
            0.90,
            0.88,
            0.92,
            1.00,
            1.10,
            1.25,
            1.40,
            1.35,
            1.20,
            0.95,
            0.80,
        ]
    )
    period_base_prices = np.array([0.075, 0.095, 0.135])  # €/kWh for 2020, 2021, 2022

    price_series = np.zeros((n_hours, 3))
    for period_idx, base_price in enumerate(period_base_prices):
        price_series[:, period_idx] = (
            np.tile(hourly_price_factors, n_hours // 24 + 1)[:n_hours]
            * base_price
            * (1 + np.random.uniform(-0.03, 0.03, n_hours))
        )

    power_prices = price_series.mean(axis=0)

    # Scenario weights: probability of each scenario occurring
    # Base Case: 60% probability, High Demand: 40% probability
    scenario_weights = np.array([0.6, 0.4])

    flow_system = fx.FlowSystem(
        timesteps=timesteps, periods=periods, scenarios=scenarios, scenario_weights=scenario_weights
    )

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
        share_from_temporal={'CO2': 0.2},  # Carbon price: 0.2 €/kg CO2 (e.g., carbon tax)
    )

    # CO2 emissions effect with constraint
    # Maximum of 1000 kg CO2/hour represents a regulatory or voluntary emissions limit
    CO2 = fx.Effect(
        label='CO2',
        unit='kg',
        description='CO2_e-Emissionen',
        maximum_per_hour=1000,  # Regulatory emissions limit: 1000 kg CO2/hour
    )

    # --- Define Flow System Components ---
    # Boiler: Converts fuel (gas) into thermal energy (heat)
    # Modern condensing gas boiler with realistic efficiency
    boiler = fx.linear_converters.Boiler(
        label='Boiler',
        thermal_efficiency=0.92,  # Realistic efficiency for modern condensing gas boiler (92%)
        thermal_flow=fx.Flow(
            label='Q_th',
            bus='Fernwärme',
            size=50,
            relative_minimum=0.1,
            relative_maximum=1,
            on_off_parameters=fx.OnOffParameters(),
        ),
        fuel_flow=fx.Flow(label='Q_fu', bus='Gas'),
    )

    # Combined Heat and Power (CHP): Generates both electricity and heat from fuel
    # Modern CHP unit with realistic efficiencies (total efficiency ~88%)
    chp = fx.linear_converters.CHP(
        label='CHP',
        thermal_efficiency=0.48,  # Realistic thermal efficiency (48%)
        electrical_efficiency=0.40,  # Realistic electrical efficiency (40%)
        electrical_flow=fx.Flow(
            'P_el', bus='Strom', size=60, relative_minimum=5 / 60, on_off_parameters=fx.OnOffParameters()
        ),
        thermal_flow=fx.Flow('Q_th', bus='Fernwärme'),
        fuel_flow=fx.Flow('Q_fu', bus='Gas'),
    )

    # Storage: Thermal energy storage system with charging and discharging capabilities
    # Realistic thermal storage parameters (e.g., insulated hot water tank)
    storage = fx.Storage(
        label='Storage',
        charging=fx.Flow('Q_th_load', bus='Fernwärme', size=1000),
        discharging=fx.Flow('Q_th_unload', bus='Fernwärme', size=1000),
        capacity_in_flow_hours=fx.InvestParameters(effects_of_investment=20, fixed_size=30, mandatory=True),
        initial_charge_state=0,  # Initial storage state: empty
        relative_maximum_final_charge_state=np.array([0.8, 0.5, 0.1]),
        eta_charge=0.95,  # Realistic charging efficiency (~95%)
        eta_discharge=0.98,  # Realistic discharging efficiency (~98%)
        relative_loss_per_hour=np.array([0.008, 0.015]),  # Realistic thermal losses: 0.8-1.5% per hour
        prevent_simultaneous_charge_and_discharge=True,  # Prevent charging and discharging at the same time
    )

    # Heat Demand Sink: Represents a fixed heat demand profile
    heat_sink = fx.Sink(
        label='Heat Demand',
        inputs=[fx.Flow(label='Q_th_Last', bus='Fernwärme', size=1, fixed_relative_profile=heat_demand_per_h)],
    )

    # Gas Source: Gas tariff source with associated costs and CO2 emissions
    # Realistic gas prices varying by period (reflecting 2020-2022 energy crisis)
    # 2020: 0.04 €/kWh, 2021: 0.06 €/kWh, 2022: 0.11 €/kWh
    gas_prices_per_period = np.array([0.04, 0.06, 0.11])

    # CO2 emissions factor for natural gas: ~0.202 kg CO2/kWh (realistic value)
    gas_co2_emissions = 0.202

    gas_source = fx.Source(
        label='Gastarif',
        outputs=[
            fx.Flow(
                label='Q_Gas',
                bus='Gas',
                size=1000,
                effects_per_flow_hour={costs.label: gas_prices_per_period, CO2.label: gas_co2_emissions},
            )
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
    flow_system.plot_network()

    # --- Define and Run Calculation ---
    # Create a calculation object to model the Flow System
    optimization = fx.Optimization(name='Sim1', flow_system=flow_system)
    optimization.do_modeling()  # Translate the model to a solvable form, creating equations and Variables

    # --- Solve the Calculation and Save Results ---
    optimization.solve(fx.solvers.HighsSolver(mip_gap=0, time_limit_seconds=30))

    optimization.results.setup_colors(
        {
            'CHP': 'red',
            'Greys': ['Gastarif', 'Einspeisung', 'Heat Demand'],
            'Storage': 'blue',
            'Boiler': 'orange',
        }
    )

    optimization.results.plot_heatmap('CHP(Q_th)|flow_rate')

    # --- Analyze Results ---
    optimization.results['Fernwärme'].plot_node_balance(mode='stacked_bar')
    optimization.results.plot_heatmap('CHP(Q_th)|flow_rate')
    optimization.results['Storage'].plot_charge_state()
    optimization.results['Fernwärme'].plot_node_balance_pie(select={'period': 2020, 'scenario': 'Base Case'})

    # Convert the results for the storage component to a dataframe and display
    df = optimization.results['Storage'].node_balance_with_charge_state()

    # Save results to file for later usage
    optimization.results.to_file()
